from __future__ import annotations

from .base import BaseLLMModel
from .config import ModelConfig, RotaryConfig
from .weight import load_hf_weight


def create_model(model_path: str, model_config: ModelConfig) -> BaseLLMModel:
    model_name = model_path.lower()
    if "llama" in model_name:
        #from .llama import NeuLlamaForCausalLM

        #return LlamaForCausalLM(model_config)

        from neuronx_distributed_inference.models.llama.modeling_llama import NeuronLlamaForCausalLM

        return NeuronLlamaForCausalLM(model_config)

    elif "qwen3" in model_name:
        #from .qwen3 import Qwen3ForCausalLM

        #return Qwen3ForCausalLM(model_config)
        from neuronx_distributed_inference.models.qwen3.modeling_qwen3 import NeuronQwen3ForCausalLM

        return NeuronQwen3ForCausalLM(model_config)
    else:
        raise ValueError(f"Unsupported model: {model_path}")

__all__ = ["BaseLLMModel", "load_hf_weight", "create_model", "ModelConfig", "RotaryConfig"]
