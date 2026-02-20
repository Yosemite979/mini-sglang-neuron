from __future__ import annotations

import collections
import contextlib
import hashlib
import logging
import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import re
import torch
import torch.nn as nn
from neuronx_distributed_inference.models.config import NeuronConfig, OnDeviceSamplingConfig
from neuronx_distributed_inference.utils.constants import MODEL_TYPES
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config
from transformers import AutoModelForCausalLM, PretrainedConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NeuronLoadConfig:
    model_path: str
    hf_config: PretrainedConfig
    tp_degree: int
    max_batch_size: int
    max_model_len: int
    block_size: int
    num_blocks: int
    max_extend_tokens: int
    override_neuron_config: Dict[str, Any] | None = None
    compile_kwargs: Dict[str, Any] | None = None


def _camel_to_kebab(name: str) -> str:
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1-\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", s1).lower()


def _get_architecture(hf_config: PretrainedConfig) -> str:
    architectures = getattr(hf_config, "architectures", None)
    if architectures:
        return architectures[0]
    model_type = getattr(hf_config, "model_type", None)
    if model_type:
        return f"{model_type.capitalize()}ForCausalLM"
    raise ValueError("Unable to determine model architecture from HF config.")


def _get_neuron_model_cls(architecture: str):
    try:
        if "For" in architecture:
            model, task = architecture.split("For", 1)
            if task == "ConditionalGeneration":
                task = "CausalLM"
            model, task = model.lower(), _camel_to_kebab(task)

            if model == "qwen3moe":
                model = "qwen3_moe"

            if architecture == "LlavaForConditionalGeneration":
                model = "pixtral"

            return MODEL_TYPES[model][task]
        raise KeyError
    except KeyError as exc:
        raise ValueError(
            f"Model {architecture} is not supported on Neuron. Supported: {list(MODEL_TYPES.keys())}"
        ) from exc


class NeuronModelBase(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.hf_config = config
        self.model: nn.Module
        self.neuron_config: NeuronConfig
        self.is_reorder_needed: bool = True
        self.architecture: str

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def load_weights(self, model_name_or_path: str, architecture: str, **kwargs):
        raise NotImplementedError

    def compile(self, compiled_model_path: str, **kwargs):
        if not hasattr(self, "model"):
            raise RuntimeError("Neuron model is not initialized. Call load_weights() first.")
        return self.model.compile(compiled_model_path, **kwargs)

    def load(self, compiled_model_path: str, **kwargs):
        if not hasattr(self, "model"):
            raise RuntimeError("Neuron model is not initialized. Call load_weights() first.")
        return self.model.load(compiled_model_path, **kwargs)

    @contextmanager
    def _reordered(self, input_block_ids: torch.Tensor, **inputs):
        if self.is_reorder_needed:
            sorted_ids, sorted_indices = torch.sort(input_block_ids)
            reordered_inputs = self._sort_inputs(inputs, sorted_indices)
            def restore(output: torch.Tensor) -> torch.Tensor:
                if sorted_ids.shape[0] != 1:
                    return torch.index_select(output, 0, torch.argsort(sorted_indices))
                return output

            yield sorted_ids, reordered_inputs, restore
        else:
            yield input_block_ids, inputs, lambda x: x

    @staticmethod
    def _sort_inputs(inputs: dict[str, Any], sorted_indices: torch.Tensor) -> dict[str, Any]:
        sorted_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if v.shape[0] > 0 and v.shape[0] == sorted_indices.shape[0]:
                    sorted_inputs[k] = torch.index_select(v, 0, sorted_indices)
                else:
                    sorted_inputs[k] = v
            elif isinstance(v, list):
                sorted_inputs[k] = [v[i.item()] for i in sorted_indices]
            else:
                sorted_inputs[k] = v
        return sorted_inputs

    def _load_weights_common(self, model_name_or_path: str, neuronx_model_cls, **kwargs):
        logger.error(f"xinux - {kwargs["neuron_config"]=}") 
        neuron_config = neuronx_model_cls.get_neuron_config_cls()(**kwargs["neuron_config"])
        config = neuronx_model_cls.get_config_cls()(
            neuron_config, load_config=load_pretrained_config(model_name_or_path)
        )
        hashed_config = hashlib.md5(config.to_json_string().encode("utf-8")).hexdigest()
        compiled_model_path = self._get_compiled_model_path(model_name_or_path, hashed_config)
        try:
            self._load_compiled_model(compiled_model_path, neuronx_model_cls)
            return True, compiled_model_path, config
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
            logger.warning("Unable to find precompiled artifacts: %s", exc)
            return False, compiled_model_path, config

    def _get_compiled_model_path(self, model_name_or_path: str, hashed_config: str) -> str:
        if os.getenv("NEURON_COMPILED_ARTIFACTS"):
            path = Path(os.getenv("NEURON_COMPILED_ARTIFACTS", ""))
            path.mkdir(parents=True, exist_ok=True)
            return str(path)
        if os.path.exists(model_name_or_path):
            path = Path(model_name_or_path) / "neuron-compiled-artifacts" / hashed_config
        else:
            path = Path("local-models") / model_name_or_path / "neuron-compiled-artifacts" / hashed_config
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _load_compiled_model(self, compiled_model_path: str, neuronx_model_cls):
        self.model = neuronx_model_cls(compiled_model_path)
        self.model.load(compiled_model_path)
        logger.info("Loaded pre-compiled Neuron model from %s", compiled_model_path)

    def _save_pretrained_model(self, model_name: str) -> str:
        hf_model = AutoModelForCausalLM.from_pretrained(model_name)
        saved_path = os.path.join("local-models", model_name)
        hf_model.save_pretrained(saved_path)
        return saved_path

    @staticmethod
    @contextlib.contextmanager
    def _exclusive_compile_lock(compiled_path: str):
        # Neuron compilation artifacts are written to a directory that is commonly shared
        # across tensor-parallel ranks. Guard compilation to avoid multi-process races
        # (partial artifacts, deleted files, etc.).
        import fcntl

        lock_path = Path(compiled_path) / ".compile.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _clear_compiled_dir(compiled_path: str, *, keep_files: set[str] | None = None) -> None:
        keep_files = keep_files or set()
        compiled_dir = Path(compiled_path)
        if not compiled_dir.exists():
            return
        for entry in compiled_dir.iterdir():
            if entry.name in keep_files:
                continue
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                with contextlib.suppress(FileNotFoundError):
                    entry.unlink()

    def _compile_and_load_model(
        self,
        model_path: str,
        neuronx_model_cls,
        config,
        compiled_path: str,
        *,
        compile_kwargs: Dict[str, Any] | None = None,
    ):
        compile_kwargs = dict(compile_kwargs or {})

        with self._exclusive_compile_lock(compiled_path):
            # Another rank/process may have finished compilation while we were waiting.
            try:
                self._load_compiled_model(compiled_path, neuronx_model_cls)
                return
            except (FileNotFoundError, ValueError, RuntimeError, OSError):
                pass

            self._clear_compiled_dir(compiled_path, keep_files={".compile.lock"})
            self.model = neuronx_model_cls(model_path, config)
            if getattr(config.neuron_config, "quantized", False):
                neuronx_model_cls.save_quantized_state_dict(model_path, config)
            self.model.compile(compiled_path, **compile_kwargs)
            self.model.load(compiled_path)


class NeuronCausalLM(NeuronModelBase):
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        input_block_ids: torch.Tensor,
        block_tables: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        #logger.error(f"xinux - {block_tables.shape=}, {input_ids.shape=}, {position_ids.shape=}")
        #logger.error(f"xinux - {kwargs['full_context_lens'].shape=}, {kwargs['computed_context_lens'].shape=}")
        #logger.error(f"xinux - {kwargs['full_context_lens']=}, {kwargs['computed_context_lens']=}")

        with self._reordered(
            input_block_ids,
            input_ids=input_ids,
            position_ids=position_ids,
            block_tables=block_tables,
            **kwargs,
        ) as (sorted_ids, inputs, restore):
            output = self.model(
                inputs["input_ids"],
                attention_mask=None,
                seq_ids=sorted_ids,
                block_table=inputs["block_tables"],
                **{k: v for k, v in inputs.items() if k not in ["input_ids", "block_tables"]},
            )
            logits = output.logits if hasattr(output, "logits") else output
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            return restore(logits)

    def load_weights(self, model_name_or_path: str, architecture: str, **kwargs):
        neuronx_model_cls = _get_neuron_model_cls(architecture)
        success, compiled_model_path, config = self._load_weights_common(
            model_name_or_path, neuronx_model_cls, **kwargs
        )
        if not success:
            if not os.path.exists(model_name_or_path):
                model_name_or_path = self._save_pretrained_model(model_name_or_path)
            self._compile_and_load_model(
                model_name_or_path,
                neuronx_model_cls,
                config,
                compiled_model_path,
                compile_kwargs=kwargs.get("compile_kwargs"),
            )
        self.neuron_config = config.neuron_config
        self.architecture = architecture
        return success, compiled_model_path

    def init_for_compile(self, model_name_or_path: str, architecture: str, **kwargs):
        neuronx_model_cls = _get_neuron_model_cls(architecture)
        neuron_config = neuronx_model_cls.get_neuron_config_cls()(**kwargs["neuron_config"])
        config = neuronx_model_cls.get_config_cls()(
            neuron_config, load_config=load_pretrained_config(model_name_or_path)
        )
        if not os.path.exists(model_name_or_path):
            model_name_or_path = self._save_pretrained_model(model_name_or_path)
        self.model = neuronx_model_cls(model_name_or_path, config)
        self.neuron_config = config.neuron_config
        self.architecture = architecture


def _default_neuron_config(load_cfg: NeuronLoadConfig) -> Dict[str, Any]:
    """
    neuron_config: Dict[str, Any] = {
        "tp_degree": load_cfg.tp_degree,
        "ctx_batch_size": 1,
        "batch_size": load_cfg.max_batch_size,
        "max_context_length": load_cfg.max_model_len,
        "max_new_tokens": load_cfg.max_extend_tokens,
        "pa_block_size": 16, #load_cfg.block_size,
        "pa_num_blocks": load_cfg.num_blocks,
        "is_block_kv_layout": False,
        "is_prefix_caching": False,
        #"chunked_prefill_config": None,
        "attn_kernel_enabled": False,
        "output_logits": True,
        "on_device_sampling_config": OnDeviceSamplingConfig(dynamic=True, deterministic=False),
        "seq_len": load_cfg.max_model_len, 
    }

    """
    neuron_config: Dict[str, Any] = {
        "tp_degree": load_cfg.tp_degree,
        #"ctx_batch_size": 1,
        "batch_size": load_cfg.max_batch_size,
        "max_context_length": load_cfg.max_model_len,
        "max_new_tokens": load_cfg.max_extend_tokens,
        "pa_block_size": 1, #load_cfg.block_size,
        "pa_num_blocks": load_cfg.num_blocks,
        "is_block_kv_layout": True,
        "is_prefix_caching": True,
        #"chunked_prefill_config": None,
        #"is_continuous_batching": (load_cfg.max_batch_size>1),
        "attn_kernel_enabled": False,
        "output_logits": True,
        "on_device_sampling_config": OnDeviceSamplingConfig(dynamic=True, deterministic=False),
        "seq_len": load_cfg.max_model_len, 
    }
    if load_cfg.override_neuron_config:
        neuron_config.update(load_cfg.override_neuron_config)
    return neuron_config


def get_neuron_model(load_cfg: NeuronLoadConfig, *, init_only: bool = False) -> NeuronCausalLM:
    architecture = _get_architecture(load_cfg.hf_config)
    model = NeuronCausalLM(load_cfg.hf_config)
    neuron_config = _default_neuron_config(load_cfg)
    logger.error(f"xinux - {neuron_config}")
    if init_only:
        model.init_for_compile(
            load_cfg.model_path,
            architecture=architecture,
            neuron_config=neuron_config,
            override_neuron_config=load_cfg.override_neuron_config or {},
        )
    else:
        model.load_weights(
            load_cfg.model_path,
            architecture=architecture,
            neuron_config=neuron_config,
            override_neuron_config=load_cfg.override_neuron_config or {},
            compile_kwargs=load_cfg.compile_kwargs or {},
        )
    return model.eval()
