# =============================================================================
# GRAPH BUILDER — TDD TEST SUITE
# Tests MUST pass before implementation is considered complete.
# =============================================================================
# Run with: cd backend && python -m pytest app/tests/test_graph_builder.py -v

import pytest
from app.services.graph_builder import ModelGraphBuilder, GraphNode, GraphEdge


# =============================================================================
# FIXTURES — Real HuggingFace config samples
# =============================================================================

GPT2_CONFIG = {
    "model_type": "gpt2",
    "architectures": ["GPT2LMHeadModel"],
    "n_layer": 12,
    "n_embd": 768,
    "n_head": 12,
    "n_positions": 1024,
    "vocab_size": 50257,
    "intermediate_size": 3072,
    "bos_token_id": 50256,
    "eos_token_id": 50256,
}

T5_CONFIG = {
    "model_type": "t5",
    "architectures": ["T5ForConditionalGeneration"],
    "d_model": 512,
    "num_layers": 6,
    "num_heads": 8,
    "d_ff": 2048,
    "vocab_size": 32128,
}

CLIP_CONFIG = {
    "model_type": "clip",
    "architectures": ["CLIPModel"],
    "hidden_size": 512,
    "projection_dim": 512,
    "vision_config": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "patch_size": 32,
        "image_size": 224,
    },
    "text_config": {
        "hidden_size": 512,
        "num_hidden_layers": 12,
        "num_attention_heads": 8,
        "vocab_size": 49408,
    },
}

MIXTRAL_CONFIG = {
    "model_type": "mixtral",
    "architectures": ["MixtralForCausalLM"],
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "intermediate_size": 14336,
    "num_local_experts": 8,
    "num_experts_per_tok": 2,
}

VIT_CONFIG = {
    "model_type": "vit",
    "architectures": ["ViTForImageClassification"],
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "image_size": 224,
    "patch_size": 16,
}

LLAMA_CONFIG = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "intermediate_size": 14336,
    "vocab_size": 128256,
    "rope_theta": 500000.0,
    "sliding_window": 4096,
}

RESNET_CONFIG = {
    "model_type": "resnet",
    "depth": 50,
    "hidden_dim": 2048,
    "num_layers": [3, 4, 6, 3],
}

DEEPSEEK_CONFIG = {
    "model_type": "deepseek_v3",
    "architectures": ["DeepseekV3ForCausalLM"],
    "num_hidden_layers": 8,
    "hidden_size": 7168,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "intermediate_size": 0,
    "num_local_experts": 128,
    "num_experts_per_tok": 8,
    "vocab_size": 129280,
}

PARTIAL_CONFIG = {
    "model_type": "gpt2",
    "n_layer": 12,
    # Missing: n_embd, n_head, vocab_size, etc.
}

UNKNOWN_MODEL_CONFIG = {
    "model_type": "my_undocumented_model",
    "num_layers": 24,
    "hidden_size": 1024,
}


# =============================================================================
# TEST 1: GPT-2 — Decoder-only transformer
# =============================================================================
class TestGPT2:
    def test_model_type_detected(self):
        builder = ModelGraphBuilder(GPT2_CONFIG, "openai-community/gpt2")
        assert builder.model_type == "gpt2"

    def test_stats_extracted(self):
        builder = ModelGraphBuilder(GPT2_CONFIG, "openai-community/gpt2")
        stats = builder.get_stats()
        assert stats["num_layers"] == 12
        assert stats["hidden_size"] == 768
        assert stats["num_heads"] == 12
        assert stats["vocab_size"] == 50257
        assert stats["intermediate_size"] == 3072
        assert stats["max_position_embeddings"] == 1024

    def test_nodes_include_embed_tokens(self):
        builder = ModelGraphBuilder(GPT2_CONFIG, "openai-community/gpt2")
        nodes = builder.get_nodes()
        node_ids = [n.id for n in nodes]
        assert "embed_tokens" in node_ids
        assert "positional_embedding" in node_ids

    def test_nodes_include_decoder_blocks(self):
        builder = ModelGraphBuilder(GPT2_CONFIG, "openai-community/gpt2")
        nodes = builder.get_nodes()
        node_ids = [n.id for n in nodes]
        for i in range(12):
            assert f"block_{i}" in node_ids, f"block_{i} missing"

    def test_nodes_include_final_norm(self):
        builder = ModelGraphBuilder(GPT2_CONFIG, "openai-community/gpt2")
        nodes = builder.get_nodes()
        node_ids = [n.id for n in nodes]
        assert "final_norm" in node_ids
        assert "lm_head" in node_ids

    def test_edges_chain_decoder_blocks(self):
        builder = ModelGraphBuilder(GPT2_CONFIG, "openai-community/gpt2")
        edges = builder.get_edges()
        edge_ids = [(e["from"], e["to"]) for e in edges]
        # Embed → first block
        assert ("embed_tokens", "positional_embedding") in edge_ids
        assert ("positional_embedding", "block_0") in edge_ids
        # Chain of blocks
        for i in range(11):
            assert (f"block_{i}", f"block_{i+1}") in edge_ids
        # Last block → norm → lm_head
        assert ("block_11", "final_norm") in edge_ids
        assert ("final_norm", "lm_head") in edge_ids

    def test_repeated_blocks_detected(self):
        builder = ModelGraphBuilder(GPT2_CONFIG, "openai-community/gpt2")
        result = builder.build()
        assert result["repeated"]["count"] == 12
        assert len(result["repeated"]["node_ids"]) == 12

    def test_graph_serialization(self):
        builder = ModelGraphBuilder(GPT2_CONFIG, "openai-community/gpt2")
        result = builder.build()
        assert "model_id" in result
        assert result["model_id"] == "openai-community/gpt2"
        assert "model_type" in result
        assert result["model_type"] == "gpt2"
        assert "stats" in result
        assert "nodes" in result
        assert "edges" in result
        assert "repeated" in result


# =============================================================================
# TEST 2: GPT-2 — Without intermediate_size
# =============================================================================
class TestGPT2WithoutIntermediate:
    def test_partial_config_does_not_crash(self):
        config = {"model_type": "gpt2", "n_layer": 12, "n_embd": 768, "n_head": 12}
        builder = ModelGraphBuilder(config, "test/model")
        result = builder.build()
        assert len(result["nodes"]) > 0
        assert result["stats"]["intermediate_size"] is None


# =============================================================================
# TEST 3: T5 — Encoder-Decoder
# =============================================================================
class TestT5:
    def test_model_type_detected(self):
        builder = ModelGraphBuilder(T5_CONFIG, "google/t5-small")
        assert builder.model_type == "t5"

    def test_stats_extracted(self):
        builder = ModelGraphBuilder(T5_CONFIG, "google/t5-small")
        stats = builder.get_stats()
        assert stats["num_layers"] == 6
        assert stats["d_model"] == 512
        assert stats["num_heads"] == 8
        assert stats["d_ff"] == 2048

    def test_has_encoder_and_decoder_stages(self):
        builder = ModelGraphBuilder(T5_CONFIG, "google/t5-small")
        result = builder.build()
        node_ids = [n.id for n in result["nodes"]]
        # Encoder blocks
        for i in range(6):
            assert f"encoder_block_{i}" in node_ids, f"encoder_block_{i} missing"
        # Decoder blocks
        for i in range(6):
            assert f"decoder_block_{i}" in node_ids, f"decoder_block_{i} missing"

    def test_encoder_decoder_cross_attention(self):
        builder = ModelGraphBuilder(T5_CONFIG, "google/t5-small")
        edges = builder.get_edges()
        edge_pairs = [(e["from"], e["to"]) for e in edges]
        # Encoder feeds into decoder via cross-attention
        assert ("encoder_block_5", "decoder_block_0") in edge_pairs


# =============================================================================
# TEST 4: CLIP — Multimodal (Vision + Text)
# =============================================================================
class TestCLIP:
    def test_model_type_detected(self):
        builder = ModelGraphBuilder(CLIP_CONFIG, "openai/clip-vit-base-patch32")
        assert builder.model_type == "clip"

    def test_stats_include_vision_config(self):
        builder = ModelGraphBuilder(CLIP_CONFIG, "openai/clip-vit-base-patch32")
        stats = builder.get_stats()
        assert stats["vision_hidden_size"] == 768
        assert stats["vision_layers"] == 12
        assert stats["vision_heads"] == 12

    def test_stats_include_text_config(self):
        builder = ModelGraphBuilder(CLIP_CONFIG, "openai/clip-vit-base-patch32")
        stats = builder.get_stats()
        assert stats["text_hidden_size"] == 512
        assert stats["text_layers"] == 12
        assert stats["text_vocab_size"] == 49408

    def test_has_vision_path(self):
        builder = ModelGraphBuilder(CLIP_CONFIG, "openai/clip-vit-base-patch32")
        result = builder.build()
        node_ids = [n.id for n in result["nodes"]]
        assert "vision_encoder" in node_ids
        assert "vision_projection" in node_ids

    def test_has_text_path(self):
        builder = ModelGraphBuilder(CLIP_CONFIG, "openai/clip-vit-base-patch32")
        result = builder.build()
        node_ids = [n.id for n in result["nodes"]]
        assert "text_encoder" in node_ids
        assert "text_projection" in node_ids

    def test_vision_and_text_converge(self):
        builder = ModelGraphBuilder(CLIP_CONFIG, "openai/clip-vit-base-patch32")
        result = builder.build()
        node_ids = [n.id for n in result["nodes"]]
        # Should have a contrastive head or similar
        assert any("contrastive" in n.id or "head" in n.id for n in result["nodes"])


# =============================================================================
# TEST 5: Mixtral — MoE
# =============================================================================
class TestMixtral:
    def test_model_type_detected(self):
        builder = ModelGraphBuilder(MIXTRAL_CONFIG, "mistralai/Mixtral-8x7B-Instruct-v0.1")
        assert builder.model_type == "mixtral"

    def test_stats_include_moe_fields(self):
        builder = ModelGraphBuilder(MIXTRAL_CONFIG, "mistralai/Mixtral-8x7B-Instruct-v0.1")
        stats = builder.get_stats()
        assert stats["num_local_experts"] == 8
        assert stats["num_experts_per_tok"] == 2

    def test_has_router_node(self):
        builder = ModelGraphBuilder(MIXTRAL_CONFIG, "mistralai/Mixtral-8x7B-Instruct-v0.1")
        result = builder.build()
        node_ids = [n.id for n in result["nodes"]]
        assert "router" in node_ids

    def test_has_expert_nodes(self):
        builder = ModelGraphBuilder(MIXTRAL_CONFIG, "mistralai/Mixtral-8x7B-Instruct-v0.1")
        result = builder.build()
        node_ids = [n.id for n in result["nodes"]]
        for i in range(8):
            assert f"expert_{i}" in node_ids, f"expert_{i} missing"

    def test_router_connects_to_experts(self):
        builder = ModelGraphBuilder(MIXTRAL_CONFIG, "mistralai/Mixtral-8x7B-Instruct-v0.1")
        edges = builder.get_edges()
        edge_pairs = [(e["from"], e["to"]) for e in edges]
        # Router should have edges to experts
        router_to_expert_edges = [(f, t) for f, t in edge_pairs if f == "router" and "expert" in t]
        assert len(router_to_expert_edges) >= 8


# =============================================================================
# TEST 6: ViT — Vision Transformer
# =============================================================================
class TestViT:
    def test_model_type_detected(self):
        builder = ModelGraphBuilder(VIT_CONFIG, "google/vit-base-patch16-224")
        assert builder.model_type == "vit"

    def test_stats_include_vision_fields(self):
        builder = ModelGraphBuilder(VIT_CONFIG, "google/vit-base-patch16-224")
        stats = builder.get_stats()
        assert stats["num_layers"] == 12
        assert stats["hidden_size"] == 768
        assert stats["num_attention_heads"] == 12
        assert stats["image_size"] == 224
        assert stats["patch_size"] == 16

    def test_has_patch_embed(self):
        builder = ModelGraphBuilder(VIT_CONFIG, "google/vit-base-patch16-224")
        result = builder.build()
        node_ids = [n.id for n in result["nodes"]]
        assert "patch_embed" in node_ids

    def test_has_class_token(self):
        builder = ModelGraphBuilder(VIT_CONFIG, "google/vit-base-patch16-224")
        result = builder.build()
        node_ids = [n.id for n in result["nodes"]]
        assert "class_token" in node_ids or "cls_token" in node_ids

    def test_has_mlp_head(self):
        builder = ModelGraphBuilder(VIT_CONFIG, "google/vit-base-patch16-224")
        result = builder.build()
        node_ids = [n.id for n in result["nodes"]]
        assert "mlp_head" in node_ids or "classifier" in node_ids


# =============================================================================
# TEST 7: Llama — Decoder with sliding window
# =============================================================================
class TestLlama:
    def test_model_type_detected(self):
        builder = ModelGraphBuilder(LLAMA_CONFIG, "meta-llama/Llama-3.1-8B-Instruct")
        assert builder.model_type == "llama"

    def test_stats_include_sliding_window(self):
        builder = ModelGraphBuilder(LLAMA_CONFIG, "meta-llama/Llama-3.1-8B-Instruct")
        stats = builder.get_stats()
        assert stats["sliding_window"] == 4096
        assert stats["rope_theta"] == 500000.0

    def test_decoder_blocks_chain(self):
        builder = ModelGraphBuilder(LLAMA_CONFIG, "meta-llama/Llama-3.1-8B-Instruct")
        edges = builder.get_edges()
        edge_pairs = [(e["from"], e["to"]) for e in edges]
        for i in range(31):
            assert (f"block_{i}", f"block_{i+1}") in edge_pairs, f"Missing chain: block_{i} → block_{i+1}"


# =============================================================================
# TEST 8: ResNet — CNN
# =============================================================================
class TestResNet:
    def test_model_type_detected(self):
        builder = ModelGraphBuilder(RESNET_CONFIG, "microsoft/resnet-50")
        assert builder.model_type == "resnet"

    def test_stats_include_depth(self):
        builder = ModelGraphBuilder(RESNET_CONFIG, "microsoft/resnet-50")
        stats = builder.get_stats()
        assert stats["depth"] == 50
        assert stats["hidden_dim"] == 2048


# =============================================================================
# TEST 9: DeepSeek V3 — Large MoE
# =============================================================================
class TestDeepSeekV3:
    def test_deepseek_v3_detected(self):
        builder = ModelGraphBuilder(DEEPSEEK_CONFIG, "deepseek-ai/DeepSeek-V4-Pro")
        assert builder.model_type == "deepseek_v3"

    def test_deepseek_has_128_experts(self):
        builder = ModelGraphBuilder(DEEPSEEK_CONFIG, "deepseek-ai/DeepSeek-V4-Pro")
        stats = builder.get_stats()
        assert stats["num_local_experts"] == 128

    def test_deepseek_has_router(self):
        builder = ModelGraphBuilder(DEEPSEEK_CONFIG, "deepseek-ai/DeepSeek-V4-Pro")
        result = builder.build()
        node_ids = [n.id for n in result["nodes"]]
        assert "router" in node_ids


# =============================================================================
# TEST 10: Partial Config — Graceful degradation
# =============================================================================
class TestPartialConfig:
    def test_does_not_crash_on_missing_fields(self):
        builder = ModelGraphBuilder(PARTIAL_CONFIG, "test/partial")
        result = builder.build()
        assert result["model_type"] == "gpt2"
        assert len(result["nodes"]) > 0
        assert len(result["edges"]) > 0

    def test_missing_fields_are_none(self):
        builder = ModelGraphBuilder(PARTIAL_CONFIG, "test/partial")
        stats = builder.get_stats()
        assert stats.get("vocab_size") is None
        assert stats.get("num_heads") is None


# =============================================================================
# TEST 11: Unknown Model Type — Fallback
# =============================================================================
class TestUnknownModel:
    def test_unknown_model_does_not_crash(self):
        builder = ModelGraphBuilder(UNKNOWN_MODEL_CONFIG, "test/unknown")
        result = builder.build()
        assert "nodes" in result
        assert "edges" in result

    def test_returns_basic_chain(self):
        builder = ModelGraphBuilder(UNKNOWN_MODEL_CONFIG, "test/unknown")
        result = builder.build()
        # Should still produce a chain of blocks
        assert len(result["repeated"]["node_ids"]) == 24


# =============================================================================
# TEST 12: Node Labels — Human-readable
# =============================================================================
class TestNodeLabels:
    def test_embed_label_shows_vocab_and_hidden(self):
        builder = ModelGraphBuilder(GPT2_CONFIG, "openai-community/gpt2")
        result = builder.build()
        embed_node = next(n for n in result["nodes"] if n.id == "embed_tokens")
        assert "50257" in embed_node.label
        assert "768" in embed_node.label

    def test_block_label_shows_layer_number(self):
        builder = ModelGraphBuilder(GPT2_CONFIG, "openai-community/gpt2")
        result = builder.build()
        block_5 = next((n for n in result["nodes"] if n.id == "block_5"), None)
        assert block_5 is not None
        assert "5" in block_5.label


# =============================================================================
# TEST 13: Granularity — Sub-block detail
# =============================================================================
class TestGranularity:
    def test_attention_sub_nodes_available(self):
        builder = ModelGraphBuilder(GPT2_CONFIG, "openai-community/gpt2")
        result = builder.build()
        sub = result["repeated"].get("sub_nodes", [])
        # Sub-nodes should include attention and mlp inside each block
        sub_types = [n.type for n in sub]
        assert "attention" in sub_types
        assert "mlp" in sub_types

    def test_moe_router_as_sub_node(self):
        builder = ModelGraphBuilder(MIXTRAL_CONFIG, "mistralai/Mixtral-8x7B-Instruct-v0.1")
        result = builder.build()
        sub = result["repeated"].get("sub_nodes", [])
        assert any(n.type == "router" for n in sub)
