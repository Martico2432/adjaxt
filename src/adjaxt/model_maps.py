from adjaxt.sharding import *

QWEN3_MOE_WEIGHT_MAP = ModelWeightMap(
    specs=[
        WeightSpec("embeds", "model.embed_tokens.weight"),
        WeightSpec("norm", "model.norm.weight"),
        WeightSpec("lm_head", "lm_head.weight", transpose=True),
        
        # Layer Norms
        WeightSpec("decoder_blocks.{i}.input_layernorm", "model.layers.{i}.input_layernorm.weight"),
        WeightSpec("decoder_blocks.{i}.post_attn_layernorm", "model.layers.{i}.post_attention_layernorm.weight"),

        # Attention Projections & QK Norms
        WeightSpec("decoder_blocks.{i}.attn.q_proj", "model.layers.{i}.self_attn.q_proj.weight", transpose=True),
        WeightSpec("decoder_blocks.{i}.attn.k_proj", "model.layers.{i}.self_attn.k_proj.weight", transpose=True),
        WeightSpec("decoder_blocks.{i}.attn.v_proj", "model.layers.{i}.self_attn.v_proj.weight", transpose=True),
        WeightSpec("decoder_blocks.{i}.attn.o_proj", "model.layers.{i}.self_attn.o_proj.weight", transpose=True),
        WeightSpec("decoder_blocks.{i}.attn.q_norm", "model.layers.{i}.self_attn.q_norm.weight"),
        WeightSpec("decoder_blocks.{i}.attn.k_norm", "model.layers.{i}.self_attn.k_norm.weight"),

        # MoE Router & Stacked Experts
        WeightSpec("decoder_blocks.{i}.mlp.router", "model.layers.{i}.mlp.gate.weight", transpose=True),
        WeightSpec("decoder_blocks.{i}.mlp.experts.w_gate", "model.layers.{i}.mlp.experts.{e}.gate_proj.weight", transpose=True),
        WeightSpec("decoder_blocks.{i}.mlp.experts.w_up", "model.layers.{i}.mlp.experts.{e}.up_proj.weight", transpose=True),
        WeightSpec("decoder_blocks.{i}.mlp.experts.w_down", "model.layers.{i}.mlp.experts.{e}.down_proj.weight", transpose=True),
    ]
)