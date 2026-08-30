from adjaxt.config import *
from adjaxt.layers import *
from adjaxt.sharding import *
from adjaxt.model_maps import *
from huggingface_hub import snapshot_download, hf_hub_download
from typing import Callable

#=====================================================================================================
#Qwen3 MoE

def qwen3_moe_model_load(cfg: Qwen3MoEModelConfig, ckpt_path: str, token: Optional[str] = None):
    """
    Loads weights into a Qwen3 MoE JAX model from either a local path 
    or directly from a Hugging Face repository repo_id.
    """
    # 1. Resolve checkpoint path (download from HF hub if not a local folder/file)
    if not os.path.exists(ckpt_path):
        resolved_path = snapshot_download(
            repo_id=ckpt_path,
            allow_patterns=["*.safetensors", "*.json"],
            token=token,
        )
    else:
        resolved_path = ckpt_path

    # 2. Extract loop/stack dimension limits dynamically from cfg
    dim_sizes = {
        "i": cfg.num_decoder_blocks,
        "e": cfg.moe_layer_conf.moe_block_conf.num_experts,
    }

    # 3. Load flat safetensors and reconstruct the nested JAX tree
    weights = load_checkpoint(
        checkpoint_path=resolved_path,
        weight_map=QWEN3_MOE_WEIGHT_MAP,
        dim_sizes=dim_sizes,
    )

    # 4. Handle tied word embeddings if configured
    if cfg.tie_word_embeddings:
        weights["lm_head"] = weights["embeds"].T

    return weights

def _get_act_fn(act_name: str) -> Callable[[jax.Array], jax.Array]:
    act_map = {
        "silu": jax.nn.silu,
        "swish": jax.nn.swish,
        "gelu": jax.nn.gelu,
        "relu": jax.nn.relu,
    }
    if act_name not in act_map:
        raise ValueError(f"Unsupported activation Callable: {act_name}")
    return act_map[act_name]

def create_qwen3_moe_config(
    hf_config: dict,
    attn_implementation: StandardAttnImplementation = StandardAttnImplementation.XLA,
) -> Qwen3MoEModelConfig:
    """Builds an MoE Qwen3 configuration dataclass from a parsed HF config.json dictionary."""
    hidden_size = hf_config["hidden_size"]
    num_heads = hf_config["num_attention_heads"]
    num_kv_heads = hf_config.get("num_key_value_heads", num_heads)
    head_dim = hf_config.get("head_dim", hidden_size // num_heads)
    rms_eps = hf_config.get("rms_norm_eps", 1e-6)
    moe_intermediate_size = hf_config.get("moe_intermediate_size", hf_config.get("intermediate_size"))
    num_layers = hf_config["num_hidden_layers"]
    vocab_size = hf_config["vocab_size"]
    rope_theta = float(hf_config.get("rope_theta", 1000000.0))
    max_pos = hf_config.get("max_position_embeddings", 32768)
    num_experts = hf_config.get("num_experts", hf_config.get("num_local_experts", 64))
    top_k = hf_config.get("num_experts_per_tok", 8)
    act_fn = _get_act_fn(hf_config.get("hidden_act", "silu"))

    gqa_conf = GQAAttnConfig(
        implementation=attn_implementation,
        num_kv_groups=num_heads // num_kv_heads,
        is_causal=True,
    )
    q_rms_conf = RMSNormConfig(dim=head_dim, eps=rms_eps)
    k_rms_conf = RMSNormConfig(dim=head_dim, eps=rms_eps)

    attn_conf = Qwen3AttnConfig(
        gqa_conf=gqa_conf,
        q_rms_conf=q_rms_conf,
        k_rms_conf=k_rms_conf,
        d=hidden_size,
        num_heads=num_heads,
        head_dim=head_dim,
        rope_theta=rope_theta,
        max_position_embeddings=max_pos,
        n_layers=num_layers,
    )

    mlp_conf = Qwen3MLPConfig(
        act_fn=act_fn,
        in_dim=hidden_size,
        hidden_dim=moe_intermediate_size,
    )

    moe_block_conf = Qwen3MoEBlockConfig(
        mlp_conf=mlp_conf,
        top_k=top_k,
        num_experts=num_experts,
        d_model=hidden_size,
    )

    moe_layer_conf = Qwen3MoELayerConfig(
        input_rms_conf=RMSNormConfig(dim=hidden_size, eps=rms_eps),
        attn_conf=attn_conf,
        post_attn_rms_conf=RMSNormConfig(dim=hidden_size, eps=rms_eps),
        moe_block_conf=moe_block_conf,
    )

    return Qwen3MoEModelConfig(
        moe_layer_conf=moe_layer_conf,
        final_rms_conf=RMSNormConfig(dim=hidden_size, eps=rms_eps),
        num_decoder_blocks=num_layers,
        vocab_size=vocab_size,
        d_model=hidden_size,
        tie_word_embeddings=hf_config.get("tie_word_embeddings", False),
    )


def qwen3_moe_config_from_pretrained(
    model_name_or_path: str,
    attn_implementation: StandardAttnImplementation = StandardAttnImplementation.XLA,
    token: Optional[str] = None,
) -> Qwen3MoEModelConfig:
    """Loads config.json from local disk or HuggingFace Hub and constructs the corresponding dataclass."""
    if os.path.exists(model_name_or_path):
        config_file = (
            model_name_or_path
            if os.path.isfile(model_name_or_path)
            else os.path.join(model_name_or_path, "config.json")
        )
    else:
        config_file = hf_hub_download(
            repo_id=model_name_or_path,
            filename="config.json",
            token=token,
        )

    with open(config_file, "r", encoding="utf-8") as f:
        hf_config = json.load(f)

    return create_qwen3_moe_config(hf_config, attn_implementation)

#=====================================================================================================