# backend/app/services/graph_builder.py
# Core graph builder — converts HuggingFace config.json into a visual graph

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class GraphNode:
    id: str
    type: str
    label: str


@dataclass
class GraphEdge:
    from_: str = field(metadata={"alias": "from"})
    to: str = ""


# Field mappings per model_type
# config_field → stats_key
FIELD_MAPS: Dict[str, Dict[str, str]] = {
    "gpt2": {
        "n_layer": "num_layers",
        "n_embd": "hidden_size",
        "n_head": "num_heads",
        "intermediate_size": "intermediate_size",
        "vocab_size": "vocab_size",
        "n_positions": "max_position_embeddings",
    },
    "llama": {
        "num_hidden_layers": "num_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_heads",
        "num_key_value_heads": "num_key_value_heads",
        "intermediate_size": "intermediate_size",
        "vocab_size": "vocab_size",
        "rope_theta": "rope_theta",
        "sliding_window": "sliding_window",
    },
    "mistral": {
        "num_hidden_layers": "num_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_heads",
        "intermediate_size": "intermediate_size",
        "vocab_size": "vocab_size",
        "sliding_window": "sliding_window",
    },
    "qwen2": {
        "num_hidden_layers": "num_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_heads",
        "num_key_value_heads": "num_key_value_heads",
        "intermediate_size": "intermediate_size",
        "vocab_size": "vocab_size",
    },
    "qwen2_vl": {
        "num_hidden_layers": "num_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_heads",
        "num_key_value_heads": "num_key_value_heads",
        "intermediate_size": "intermediate_size",
        "vocab_size": "vocab_size",
    },
    "gemma": {
        "num_hidden_layers": "num_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "intermediate_size": "intermediate_size",
        "vocab_size": "vocab_size",
        "rope_theta": "rope_theta",
        "sliding_window": "sliding_window",
    },
    "gemma3": {
        "num_hidden_layers": "num_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_key_value_heads",
        "intermediate_size": "intermediate_size",
        "vocab_size": "vocab_size",
        "rope_theta": "rope_theta",
        "sliding_window": "sliding_window",
    },
    "phi": {
        "num_hidden_layers": "num_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "intermediate_size": "intermediate_size",
        "vocab_size": "vocab_size",
    },
    "falcon": {
        "num_hidden_layers": "num_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_key_value_heads",
        "intermediate_size": "intermediate_size",
        "vocab_size": "vocab_size",
    },
    # Encoder-decoder
    "t5": {
        "num_layers": "num_layers",
        "d_model": "d_model",
        "num_heads": "num_heads",
        "d_ff": "d_ff",
        "vocab_size": "vocab_size",
    },
    "bart": {
        "num_layers": "num_layers",
        "d_model": "d_model",
        "num_heads": "num_heads",
        "d_ff": "d_ff",
        "vocab_size": "vocab_size",
    },
    # Vision
    "vit": {
        "num_hidden_layers": "num_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "image_size": "image_size",
        "patch_size": "patch_size",
    },
    # Multimodal
    "clip": {
        "vision_hidden_size": "vision_hidden_size",
        "vision_layers": "vision_layers",
        "vision_heads": "vision_heads",
        "vision_patch_size": "vision_patch_size",
        "text_hidden_size": "text_hidden_size",
        "text_layers": "text_layers",
        "text_heads": "text_heads",
        "projection_dim": "projection_dim",
    },
    "llava": {
        "num_hidden_layers": "num_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "vision_layers": "vision_layers",
        "vision_hidden_size": "vision_hidden_size",
    },
    # MoE
    "mixtral": {
        "num_hidden_layers": "num_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_heads",
        "num_key_value_heads": "num_key_value_heads",
        "intermediate_size": "intermediate_size",
        "num_local_experts": "num_local_experts",
        "num_experts_per_tok": "num_experts_per_tok",
    },
    "deepseek_v3": {
        "num_hidden_layers": "num_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_heads",
        "num_key_value_heads": "num_key_value_heads",
        "num_local_experts": "num_local_experts",
        "num_experts_per_tok": "num_experts_per_tok",
        "vocab_size": "vocab_size",
    },
    # CNN
    "resnet": {
        "depth": "depth",
        "hidden_dim": "hidden_dim",
        "num_layers": "num_layers",
    },
}

# Short name → full org/model mapping for popular models
SHORT_ID_MAP: Dict[str, str] = {
    "gpt2": "openai-community/gpt2",
    "t5": "google/t5-small",
    "clip": "openai/clip-vit-base-patch32",
    "vit": "google/vit-base-patch16-224",
    "resnet": "microsoft/resnet-50",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "qwen": "Qwen/Qwen2.5-0.8B",
    "qwen-vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    "phi": "microsoft/Phi-3-mini-4k-instruct",
    "deepseek": "deepseek-ai/DeepSeek-V3",
    "gemma": "google/gemma-2-2b-it",
}


class ModelGraphBuilder:
    def __init__(self, config: Dict[str, Any], model_id: str):
        self.config = config
        self.model_id = model_id
        self.model_type = config.get("model_type", "unknown")
        self.nodes: List[GraphNode] = []
        self.edges: List[Dict[str, str]] = []
        self._build()

    def _get(self, key: str, default: Any = None) -> Any:
        """Get a top-level or nested config key."""
        return self.config.get(key, default)

    def _get_nested(self, path: str, default: Any = None) -> Any:
        """Get a nested config value using dot notation: vision_config.hidden_size"""
        keys = path.split(".")
        val = self.config
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
            if val is None:
                return default
        return val

    def get_stats(self) -> Dict[str, Any]:
        """Extract stats from config using field maps."""
        fm = FIELD_MAPS.get(self.model_type, {})
        stats: Dict[str, Any] = {}

        for config_key, stats_key in fm.items():
            # Try nested path first
            val = self._get_nested(config_key)
            if val is None:
                val = self._get(config_key)
            stats[stats_key] = val

        # CLIP nested configs
        if self.model_type == "clip":
            vc = self._get("vision_config", {})
            tc = self._get("text_config", {})
            if isinstance(vc, dict):
                stats.setdefault("vision_hidden_size", vc.get("hidden_size"))
                stats.setdefault("vision_layers", vc.get("num_hidden_layers"))
                stats.setdefault("vision_heads", vc.get("num_attention_heads"))
                stats.setdefault("vision_patch_size", vc.get("patch_size"))
                stats.setdefault("image_size", vc.get("image_size"))
            if isinstance(tc, dict):
                stats.setdefault("text_hidden_size", tc.get("hidden_size"))
                stats.setdefault("text_layers", tc.get("num_hidden_layers"))
                stats.setdefault("text_heads", tc.get("num_attention_heads"))
                stats.setdefault("text_vocab_size", tc.get("vocab_size"))
            stats["projection_dim"] = self._get("projection_dim")

        # DeepSeek MoE
        if self.model_type == "deepseek_v3":
            stats["num_local_experts"] = self._get("num_local_experts")
            stats["num_experts_per_tok"] = self._get("num_experts_per_tok")

        return stats

    def _add_node(self, id_: str, type_: str, label: str) -> None:
        self.nodes.append(GraphNode(id=id_, type=type_, label=label))

    def _add_edge(self, from_: str, to: str) -> None:
        self.edges.append({"from": from_, "to": to})

    def _chain_edges(self, node_ids: List[str]) -> None:
        """Chain a list of node IDs with directed edges."""
        for i in range(len(node_ids) - 1):
            self._add_edge(node_ids[i], node_ids[i + 1])

    def _build_decoder_only(self) -> None:
        """Build graph for decoder-only transformers (GPT-2, LLaMA, etc.)"""
        stats = self.get_stats()
        num_layers = stats.get("num_layers", 1) or 1
        vocab_size = stats.get("vocab_size") or 0
        hidden_size = stats.get("hidden_size") or 0
        intermediate_size = stats.get("intermediate_size")

        # Embeddings
        if vocab_size and hidden_size:
            self._add_node("embed_tokens", "embedding",
                           f"Token Embeddings ({vocab_size:,}×{hidden_size})")
        else:
            self._add_node("embed_tokens", "embedding", "Token Embeddings")

        pos_id = "positional_embedding"
        max_pos = stats.get("max_position_embeddings")
        if max_pos:
            self._add_node(pos_id, "positional_embedding",
                           f"Positional Embeddings ({max_pos:,}×{hidden_size})")
        else:
            self._add_node(pos_id, "positional_embedding", "Positional Embeddings")

        self._add_edge("embed_tokens", pos_id)

        # Decoder blocks
        block_ids = []
        for i in range(num_layers):
            attn_label = f"Layer {i}: Attention"
            mlp_label = f"Layer {i}: MLP"
            if intermediate_size:
                attn_label += f" ({hidden_size})"
                mlp_label += f" ({intermediate_size})"
            self._add_node(f"block_{i}", "decoder_block", f"Layer {i}")
            block_ids.append(f"block_{i}")

        # Chain blocks
        self._add_edge(pos_id, block_ids[0])
        self._chain_edges(block_ids)

        # Final norm
        self._add_node("final_norm", "layer_norm", "Final LayerNorm")
        self._add_edge(block_ids[-1], "final_norm")

        # LM head
        if vocab_size and hidden_size:
            self._add_node("lm_head", "lm_head",
                           f"LM Head ({vocab_size:,}×{hidden_size})")
        else:
            self._add_node("lm_head", "lm_head", "LM Head")
        self._add_edge("final_norm", "lm_head")

    def _build_encoder_decoder(self) -> None:
        """Build graph for encoder-decoder models (T5, BART)."""
        stats = self.get_stats()
        num_layers = stats.get("num_layers", 6) or 6
        vocab_size = stats.get("vocab_size") or 0
        d_model = stats.get("d_model") or 512
        d_ff = stats.get("d_ff")

        # Encoder embedding
        self._add_node("encoder_embed", "embedding",
                       f"Encoder Embed ({vocab_size:,}×{d_model})")
        self._add_node("encoder_pos", "positional_embedding", "Encoder Positional Embeddings")
        self._add_edge("encoder_embed", "encoder_pos")

        # Encoder blocks
        enc_ids = []
        for i in range(num_layers):
            self._add_node(f"encoder_block_{i}", "encoder_block",
                           f"Encoder Layer {i}" + (f" (d_ff={d_ff})" if d_ff else ""))
            enc_ids.append(f"encoder_block_{i}")
        self._add_edge("encoder_pos", enc_ids[0])
        self._chain_edges(enc_ids)

        # Encoder final norm
        self._add_node("encoder_final_norm", "layer_norm", "Encoder Final Norm")
        self._add_edge(enc_ids[-1], "encoder_final_norm")

        # Decoder embedding + pos
        self._add_node("decoder_embed", "embedding",
                       f"Decoder Embed ({vocab_size:,}×{d_model})")
        self._add_node("decoder_pos", "positional_embedding", "Decoder Positional Embeddings")
        self._add_edge("decoder_embed", "decoder_pos")

        # Decoder blocks with cross-attention
        dec_ids = []
        for i in range(num_layers):
            self._add_node(f"decoder_block_{i}", "decoder_block",
                           f"Decoder Layer {i} (Cross-Attention)")
            dec_ids.append(f"decoder_block_{i}")
        self._add_edge("decoder_pos", dec_ids[0])
        self._chain_edges(dec_ids)

        # Cross-attention from encoder to decoder
        self._add_edge(enc_ids[-1], dec_ids[0])

        # Decoder final norm
        self._add_node("decoder_final_norm", "layer_norm", "Decoder Final Norm")
        self._add_edge(dec_ids[-1], "decoder_final_norm")

        # LM head
        self._add_node("lm_head", "lm_head",
                       f"LM Head ({vocab_size:,}×{d_model})")
        self._add_edge("decoder_final_norm", "lm_head")

    def _build_vision(self) -> None:
        """Build graph for vision transformers (ViT)."""
        stats = self.get_stats()
        num_layers = stats.get("num_layers", 12) or 12
        hidden_size = stats.get("hidden_size") or 768
        patch_size = stats.get("patch_size") or 16
        image_size = stats.get("image_size") or 224

        num_patches = (image_size // patch_size) ** 2

        self._add_node("patch_embed", "patch_embed",
                       f"Patch Embed ({num_patches} patches, {patch_size}px)")
        self._add_node("class_token", "class_token", "CLS Token")
        self._add_node("patch_class_concat", "concat", "Concat Patches + CLS")
        self._add_edge("patch_embed", "patch_class_concat")
        self._add_edge("class_token", "patch_class_concat")

        # Vision transformer layers
        block_ids = []
        for i in range(num_layers):
            self._add_node(f"vit_block_{i}", "vision_block",
                           f"ViT Layer {i} ({hidden_size})")
            block_ids.append(f"vit_block_{i}")
        self._add_edge("patch_class_concat", block_ids[0])
        self._chain_edges(block_ids)

        # Norm + head
        self._add_node("vit_norm", "layer_norm", "Norm")
        self._add_edge(block_ids[-1], "vit_norm")
        self._add_node("mlp_head", "mlp_head",
                       f"MLP Head → {hidden_size}→Classifier")
        self._add_edge("vit_norm", "mlp_head")

    def _build_multimodal(self) -> None:
        """Build graph for multimodal models (CLIP)."""
        stats = self.get_stats()
        vision_layers = stats.get("vision_layers", 12) or 12
        text_layers = stats.get("text_layers", 12) or 12
        vision_hidden = stats.get("vision_hidden_size") or 768
        text_hidden = stats.get("text_hidden_size") or 512
        projection_dim = stats.get("projection_dim") or 512
        vision_patch = stats.get("vision_patch_size") or 32
        image_size = stats.get("image_size") or 224

        # Vision encoder
        self._add_node("vision_embed", "patch_embed",
                       f"Vision Patch Embed ({vision_patch}px, {image_size}px)")
        self._add_node("vision_encoder", "vision_encoder",
                       f"Vision Encoder ({vision_layers} layers)")
        vision_block_ids = [f"vision_block_{i}" for i in range(vision_layers)]
        self._add_edge("vision_embed", "vision_encoder")
        self._add_node("vision_norm", "layer_norm", "Vision Norm")
        self._add_node("vision_projection", "projection",
                       f"Vision Projection ({vision_hidden}→{projection_dim})")
        self._add_edge("vision_encoder", "vision_norm")
        self._add_edge("vision_norm", "vision_projection")

        # Text encoder
        text_vocab = stats.get("text_vocab_size") or 49408
        self._add_node("text_embed", "embedding",
                       f"Text Embed ({text_vocab:,}×{text_hidden})")
        self._add_node("text_encoder", "text_encoder",
                       f"Text Encoder ({text_layers} layers)")
        text_block_ids = [f"text_block_{i}" for i in range(text_layers)]
        self._add_edge("text_embed", "text_encoder")
        self._add_node("text_norm", "layer_norm", "Text Norm")
        self._add_node("text_projection", "projection",
                       f"Text Projection ({text_hidden}→{projection_dim})")
        self._add_edge("text_encoder", "text_norm")
        self._add_edge("text_norm", "text_projection")

        # Contrastive head
        self._add_node("contrastive_head", "contrastive_head",
                       f"Contrastive Head ({projection_dim})")
        self._add_edge("vision_projection", "contrastive_head")
        self._add_edge("text_projection", "contrastive_head")

    def _build_moe(self) -> None:
        """Build graph for Mixture of Experts models (Mixtral, DeepSeek-V3)."""
        stats = self.get_stats()
        num_layers = stats.get("num_layers", 32) or 32
        hidden_size = stats.get("hidden_size") or 4096
        num_experts = stats.get("num_local_experts", 8) or 8
        num_experts_per_tok = stats.get("num_experts_per_tok", 2) or 2

        # Embeddings
        vocab_size = stats.get("vocab_size") or 0
        self._add_node("embed_tokens", "embedding",
                       f"Token Embeddings ({vocab_size:,}×{hidden_size})" if vocab_size else "Token Embeddings")
        self._add_node("positional_embedding", "positional_embedding", "Positional Embeddings")
        self._add_edge("embed_tokens", "positional_embedding")

        # Decoder blocks with MoE
        block_ids = []
        for i in range(num_layers):
            self._add_node(f"block_{i}", "moe_block",
                           f"MoE Layer {i} ({num_experts} experts, top-{num_experts_per_tok})")
            block_ids.append(f"block_{i}")

        self._add_edge("positional_embedding", block_ids[0])
        self._chain_edges(block_ids)

        # Router
        self._add_node("router", "router",
                       f"Router (top-{num_experts_per_tok} from {num_experts} experts)")
        for i in range(num_experts):
            self._add_node(f"expert_{i}", "expert",
                           f"Expert {i} ({hidden_size}→{hidden_size})")
            self._add_edge("router", f"expert_{i}")

        # Final norm
        self._add_node("final_norm", "layer_norm", "Final LayerNorm")
        self._add_edge(block_ids[-1], "final_norm")

        vocab_size = stats.get("vocab_size") or 0
        self._add_node("lm_head", "lm_head",
                       f"LM Head ({vocab_size:,}×{hidden_size})" if vocab_size else "LM Head")
        self._add_edge("final_norm", "lm_head")

    def _build_cnn(self) -> None:
        """Build graph for CNN models (ResNet)."""
        stats = self.get_stats()
        depth = stats.get("depth", 50)
        hidden_dim = stats.get("hidden_dim", 2048)

        self._add_node("conv1", "conv", f"Conv1 (7×7, 64, stride 2)")
        self._add_node("bn1", "batch_norm", "BatchNorm")
        self._add_node("maxpool", "pool", "MaxPool (3×3, stride 2)")
        self._add_edge("conv1", "bn1")
        self._add_edge("bn1", "maxpool")

        # ResNet stages: [3, 4, 6, 3] for ResNet-50
        layer_config = {
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3],
        }
        layers = layer_config.get(depth, [3, 4, 6, 3])
        stage_dims = [256, 512, 1024, 2048]

        current = "maxpool"
        for stage_idx, (num_blocks, dim) in enumerate(zip(layers, stage_dims)):
            stage_name = f"layer{stage_idx + 1}"
            for block_i in range(num_blocks):
                block_name = f"{stage_name}_block{block_i}"
                self._add_node(block_name, "residual_block",
                               f"{block_name} ({dim} channels)")
                self._add_edge(current, block_name)
                current = block_name

        self._add_node("avgpool", "pool", "Average Pool")
        self._add_node("fc", "fc", f"FC ({hidden_dim}→1000)")
        self._add_edge(current, "avgpool")
        self._add_edge("avgpool", "fc")

    def _build_fallback(self) -> None:
        """Fallback for unknown model types."""
        num_layers = self.config.get("num_layers",
                                      self.config.get("n_layer",
                                                      self.config.get("depth", 12)))
        num_layers = num_layers or 12
        hidden_size = self.config.get("hidden_size",
                                       self.config.get("n_embd", 768)) or 768

        self._add_node("embed_tokens", "embedding",
                       f"Token Embeddings ({hidden_size})")
        self._add_node("positional_embedding", "positional_embedding",
                       "Positional Embeddings")

        block_ids = []
        for i in range(num_layers):
            self._add_node(f"block_{i}", "transformer_block",
                           f"Block {i} ({hidden_size})")
            block_ids.append(f"block_{i}")

        self._add_edge("embed_tokens", "positional_embedding")
        self._add_edge("positional_embedding", block_ids[0])
        self._chain_edges(block_ids)
        self._add_node("final_norm", "layer_norm", "Final Norm")
        self._add_edge(block_ids[-1], "final_norm")
        self._add_node("lm_head", "lm_head", f"LM Head ({hidden_size})")
        self._add_edge("final_norm", "lm_head")

    def _build(self) -> None:
        """Route to appropriate builder based on model_type."""
        mt = self.model_type
        if mt in ("gpt2", "llama", "mistral", "qwen2", "qwen2_vl", "gemma", "gemma3", "phi", "falcon"):
            self._build_decoder_only()
        elif mt in ("t5", "bart"):
            self._build_encoder_decoder()
        elif mt in ("vit", "deit", "beit"):
            self._build_vision()
        elif mt in ("clip", "llava"):
            self._build_multimodal()
        elif mt in ("mixtral", "deepseek_v3"):
            self._build_moe()
        elif mt in ("resnet", "convnext"):
            self._build_cnn()
        else:
            self._build_fallback()

    def get_nodes(self) -> List[GraphNode]:
        return self.nodes

    def get_edges(self) -> List[Dict[str, str]]:
        return self.edges

    def build(self) -> Dict[str, Any]:
        """Build the full graph result."""
        repeated_nodes = []
        for n in self.nodes:
            if n.id.startswith("block_") or n.id.startswith("vit_block_"):
                repeated_nodes.append(n.id)

        sub_nodes = [
            GraphNode(id="attention_sub", type="attention", label="Self-Attention"),
            GraphNode(id="mlp_sub", type="mlp", label="MLP / FFN"),
        ]
        if any("expert" in n.id for n in self.nodes):
            sub_nodes.append(GraphNode(id="router_sub", type="router", label="MoE Router"))

        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "stats": self.get_stats(),
            "nodes": [
                {"id": n.id, "type": n.type, "label": n.label}
                for n in self.nodes
            ],
            "edges": self.edges,
            "repeated": {
                "count": len(repeated_nodes),
                "node_ids": repeated_nodes,
                "label": f"Transformer Block ×{len(repeated_nodes)}",
                "sub_nodes": [
                    {"id": n.id, "type": n.type, "label": n.label}
                    for n in sub_nodes
                ],
            },
        }
