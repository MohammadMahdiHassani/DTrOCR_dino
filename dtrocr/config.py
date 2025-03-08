from typing import Optional, Union, Tuple, List, Literal


class DTrOCRConfig:
    def __init__(
        self,
        gpt2_hf_model: str = 'openai-community/gpt2',
        vision_model: str = 'facebook/dino-vits16',  # Changed from vit_hf_model,
        vocab_size: Optional[int] = 50257,
        max_position_embeddings: Optional[int] = 256,
        hidden_size: Optional[int] = 384,  # Changed to match DINO's output dimension
        num_hidden_layers: Optional[int] = 12,
        num_attention_heads: Optional[int] = 12,
        patch_size: Optional[Union[Tuple[int], List[int]]] = (16, 16),      # (height, width)
        image_size: Optional[Union[Tuple[int], List[int]]] = (32, 128),   # (height, width)
        num_channels: Optional[int] = 3,
        resid_pdrop: Optional[float] = 0.1,
        embd_pdrop: Optional[float] = 0.1,
        attn_pdrop: Optional[float] = 0.1,
        layer_norm_epsilon: Optional[float] = 1e-5,
        attn_implementation: Literal['sdpa', 'flash_attention_2'] = 'sdpa'
    ):
        self.gpt2_hf_model = gpt2_hf_model
        self.vision_model = vision_model
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self._attn_implementation = attn_implementation

        # other GPT2 config values
        self.n_inner = None
        self.scale_attn_weights = True
        self.scale_attn_by_inverse_layer_idx = False
        self.reorder_and_upcast_attn = False
        self.add_cross_attention = False
        self.activation_function = "gelu_new"
