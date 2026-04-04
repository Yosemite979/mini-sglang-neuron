from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import uvicorn
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from minisgl.core import SamplingParams
from minisgl.env import ENV
from minisgl.message import (
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    TokenizeMsg,
    UserReply,
)
from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .args import ServerArgs
from .tool_parser import parse_tool_calls

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
_MAX_REQ_PER_MIN = None

logger = init_logger(__name__, "FrontendAPI")

_GLOBAL_STATE = None


def get_global_state() -> FrontendManager:
    global _GLOBAL_STATE
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    return _GLOBAL_STATE


def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]:
    if isinstance(msg, BatchFrontendMsg):
        result = []
        for reply in msg.data:
            assert isinstance(reply, UserReply)
            result.append(reply)
        return result
    assert isinstance(msg, UserReply)
    return [msg]


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int
    ignore_eos: bool = False


class FunctionDef(BaseModel):
    name: str
    description: str = ""
    parameters: Optional[Dict[str, Any]] = None


class ToolDef(BaseModel):
    type: str = "function"
    function: FunctionDef


class ToolCallFunction(BaseModel):
    name: str
    arguments: str


class ToolCallObj(BaseModel):
    id: str = ""
    type: str = "function"
    function: ToolCallFunction


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCallObj]] = None
    tool_call_id: Optional[str] = None


class OpenAICompletionRequest(BaseModel):
    model: str
    prompt: Optional[str] = None
    messages: Optional[List[Message]] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    n: int = 1
    stream: bool = False
    stop: List[str] = []
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    ignore_eos: bool = False
    tools: Optional[List[ToolDef]] = None
    tool_choice: Union[str, Dict, None] = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mini-sglang"
    root: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = Field(default_factory=list)


def _sampling_params_from_request(req: OpenAICompletionRequest) -> SamplingParams:
    return SamplingParams(
        ignore_eos=req.ignore_eos,
        max_tokens=req.max_tokens if req.max_tokens is not None else ENV.SHELL_MAX_TOKENS.value,
        temperature=(
            req.temperature if req.temperature is not None else ENV.SHELL_TEMPERATURE.value
        ),
        top_k=req.top_k if req.top_k is not None else ENV.SHELL_TOP_K.value,
        top_p=req.top_p if req.top_p is not None else ENV.SHELL_TOP_P.value,
    )


def _tools_for_template(req: OpenAICompletionRequest) -> Optional[List[Dict]]:
    if not req.tools or req.tool_choice == "none":
        return None
    return [t.model_dump() for t in req.tools]


@dataclass
class FrontendManager:
    config: ServerArgs
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    uid_counter: int = 0
    initialized: bool = False
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)

    def new_user(self) -> int:
        uid = self.uid_counter
        self.uid_counter += 1
        self.ack_map[uid] = []
        self.event_map[uid] = asyncio.Event()
        return uid

    async def listen(self):
        while True:
            msg = await self.recv_tokenizer.get()
            for msg in _unwrap_msg(msg):
                assert msg.uid in self.ack_map
                self.ack_map[msg.uid].append(msg)
                self.event_map[msg.uid].set()

    def _create_listener_once(self):
        if not self.initialized:
            asyncio.create_task(self.listen())
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        self._create_listener_once()
        await self.send_tokenizer.put(msg)

    async def wait_for_ack(self, uid: int):
        event = self.event_map[uid]
        while True:
            await event.wait()
            event.clear()
            pending = self.ack_map[uid]
            self.ack_map[uid] = []
            ack = None
            for ack in pending:
                yield ack
            if ack and ack.finished:
                break
        del self.ack_map[uid]
        del self.event_map[uid]

    async def collect_full_response(self, uid: int) -> str:
        full = []
        async for ack in self.wait_for_ack(uid):
            if ack.incremental_output:
                full.append(ack.incremental_output)
        return "".join(full)

    async def stream_generate(self, uid: int):
        async for ack in self.wait_for_ack(uid):
            yield f"data: {ack.incremental_output}\n".encode()
            if ack.finished:
                break
        yield "data: [DONE]\n".encode()

    async def stream_chat_completions(self, uid: int, has_tools: bool = False):
        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        if has_tools:
            full_text = await self.collect_full_response(uid)
            normal_text, tool_calls = parse_tool_calls(full_text)

            if tool_calls:
                msg = {"role": "assistant", "content": normal_text, "tool_calls": tool_calls}
                resp = {
                    "id": cmpl_id, "object": "chat.completion", "created": created,
                    "model": self.config.model_path,
                    "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
                yield f"data: {json.dumps(resp)}\n\n".encode()
                yield b"data: [DONE]\n\n"
                return
            else:
                delta = {"role": "assistant", "content": full_text}
                chunk = {
                    "id": cmpl_id, "object": "chat.completion.chunk", "created": created,
                    "model": self.config.model_path,
                    "choices": [{"delta": delta, "index": 0, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(chunk)}\n\n".encode()
                yield b"data: [DONE]\n\n"
                return

        first_chunk = True
        async for ack in self.wait_for_ack(uid):
            delta = {}
            if first_chunk:
                delta["role"] = "assistant"
                first_chunk = False
            if ack.incremental_output:
                delta["content"] = ack.incremental_output
            chunk = {
                "id": cmpl_id, "object": "chat.completion.chunk", "created": created,
                "model": self.config.model_path,
                "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()
            if ack.finished:
                break

        end_chunk = {
            "id": cmpl_id, "object": "chat.completion.chunk", "created": created,
            "model": self.config.model_path,
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    async def abort_user(self, uid: int):
        await asyncio.sleep(0.1)
        if uid in self.ack_map:
            del self.ack_map[uid]
        if uid in self.event_map:
            del self.event_map[uid]

    def shutdown(self):
        self.send_tokenizer.stop()
        self.recv_tokenizer.stop()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()


app = FastAPI(title="MiniSGL API Server", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.post("/generate")
@limiter.limit(lambda: f"{_MAX_REQ_PER_MIN}/minute"
               if _MAX_REQ_PER_MIN else "1000/minute")
async def generate(request: Request, req: GenerateRequest):
    logger.debug("Received generate request %s", req)
    state = get_global_state()
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(uid=uid, text=req.prompt, sampling_params=SamplingParams(
            ignore_eos=req.ignore_eos, max_tokens=req.max_tokens))
    )
    return StreamingResponse(
        state.stream_generate(uid), media_type="text/event-stream",
        background=BackgroundTask(lambda: state.abort_user(uid)),
    )


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    return {"status": "ok"}


@app.post("/v1/chat/completions")
@limiter.limit(lambda: f"{_MAX_REQ_PER_MIN}/minute"
               if _MAX_REQ_PER_MIN else "1000/minute")
async def v1_completions(request: Request, req: OpenAICompletionRequest):
    state = get_global_state()
    has_tools = bool(req.tools and req.tool_choice != "none")

    if req.messages:
        prompt = [msg.model_dump(exclude_none=True) for msg in req.messages]
    else:
        assert req.prompt is not None
        prompt = req.prompt

    tools_for_tpl = _tools_for_template(req) if has_tools else None

    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(
            uid=uid, text=prompt,
            sampling_params=_sampling_params_from_request(req),
            tools=tools_for_tpl,
        )
    )

    if not req.stream:
        full_text = await state.collect_full_response(uid)
        normal_text, tool_calls = parse_tool_calls(full_text)
        cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        if tool_calls:
            msg = {"role": "assistant", "content": normal_text, "tool_calls": tool_calls}
            finish = "tool_calls"
        else:
            msg = {"role": "assistant", "content": full_text}
            finish = "stop"
        return JSONResponse({
            "id": cmpl_id, "object": "chat.completion", "created": int(time.time()),
            "model": state.config.model_path,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    return StreamingResponse(
        state.stream_chat_completions(uid, has_tools=has_tools),
        media_type="text/event-stream",
        background=BackgroundTask(lambda: state.abort_user(uid)),
    )


@app.get("/v1/models")
async def available_models():
    state = get_global_state()
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])


async def shell_completion(req: OpenAICompletionRequest):
    state = get_global_state()
    assert req.messages is not None
    prompt = [msg.model_dump(exclude_none=True) for msg in req.messages]
    uid = state.new_user()
    await state.send_one(
        TokenizeMsg(uid=uid, text=prompt, sampling_params=_sampling_params_from_request(req))
    )
    return StreamingResponse(
        state.stream_generate(uid), media_type="text/event-stream",
        background=BackgroundTask(lambda: state.abort_user(uid)),
    )


async def async_input(prompt=""):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: input(prompt))


async def shell():
    commands = ["/exit", "/reset"]
    completer = WordCompleter(commands)
    session = PromptSession("$ ", completer=completer)
    try:
        history: List[Tuple[str, str]] = []
        while True:
            cmd = (await session.prompt_async()).strip()
            if cmd == "":
                continue
            if cmd.startswith("/"):
                if cmd == "/exit":
                    return
                if cmd == "/reset":
                    history = []
                    continue
                raise ValueError(f"Unknown command: {cmd}")
            history_messages: List[Message] = []
            for user_msg, assistant_msg in history:
                history_messages.append(Message(role="user", content=user_msg))
                history_messages.append(Message(role="assistant", content=assistant_msg))
            req = OpenAICompletionRequest(
                model="",
                messages=history_messages + [Message(role="user", content=cmd)],
                max_tokens=ENV.SHELL_MAX_TOKENS.value,
                top_k=ENV.SHELL_TOP_K.value,
                top_p=ENV.SHELL_TOP_P.value,
                temperature=ENV.SHELL_TEMPERATURE.value,
                stream=True,
            )
            cur_msg = ""
            async for chunk in (await shell_completion(req)).body_iterator:
                msg = chunk.decode()
                assert msg.startswith("data: "), msg
                msg = msg[6:]
                assert msg.endswith("\n"), msg
                msg = msg[:-1]
                if msg == "[DONE]":
                    continue
                cur_msg += msg
                print(msg, end="", flush=True)
            print("", flush=True)
            history.append((cmd, cur_msg))
    except EOFError:
        pass
    finally:
        print("Exiting shell...")
        await asyncio.sleep(0.1)
        get_global_state().shutdown()
        import psutil
        parent = psutil.Process()
        for child in parent.children(recursive=True):
            child.kill()


def run_api_server(config: ServerArgs, start_backend: Callable[[], None], run_shell: bool) -> None:
    global _GLOBAL_STATE
    global _MAX_REQ_PER_MIN
    _MAX_REQ_PER_MIN = config.max_req_per_min

    if run_shell:
        assert not config.use_dummy_weight
    host = config.server_host
    port = config.server_port
    assert _GLOBAL_STATE is None
    _GLOBAL_STATE = FrontendManager(
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr, create=True, decoder=BaseFrontendMsg.decoder,
        ),
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr, create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )
    start_backend()
    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        uvicorn.run(app, host=host, port=port)
    else:
        asyncio.run(shell())
