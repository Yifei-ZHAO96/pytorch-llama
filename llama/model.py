# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional
import math
from dataclasses import dataclass
from contextvars import ContextVar
import bitsandbytes as bnb

@dataclass
class ModelArgs:
    """
    Model parameter settings.
    
    """
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1 # Later set in the build method
    multiple_of: int = 256 # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Defaults to 1e-6.
            
        """
        super().__init__()
        self.eps = eps
        self.dim = dim
         # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))
        
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
            
        """
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.
            
        """
        return (self.weight * self._norm(x.float())).type_as(x)


def precompute_theta_pos_frequencies(head_dim: int, 
                                     seq_len: int, 
                                     device: str, 
                                     theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials with given dimensions.
    
    This function calculates a frequency tensor with complex exponentials using the given dimension 'head_dim'
    and the sequence length 'seq_len'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        head_dim (int): Dimension of the frequency tensor.
        seq_len (int): Sequence length for precomputing frequencies.
        device (str): Tensor device type.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    """
    # >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    # Build the theta parameter
    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # Shape: (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # Construct the positions (the "m" parameter)
    # Shape: (Seq_Len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product.
    # outer_product, Shape: (Seq_Len) * (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form c = R * exp(m * theta), where R = 1 as follows:
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, 
                            freqs_complex: torch.Tensor, 
                            device: str) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' or key 'xk' tensors using the provided
    frequency tensor 'freqs_complex'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        x (torch.Tensor): Query or Value tensor to apply rotary embeddings.
        freqs_complex (torch.Tensor): Precomputed frequency tensor for complex exponentials.
        device (str): Tensor device type.

    Returns:
        torch.Tensor: modified query or key tensor with rotary embeddings.
        
    """
    # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
    # Two consecutive values will become a single complex number
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
    # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
    # Which results in the rotation of the complex number as shown in the Figure 1 of the paper
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex
    # Convert the complex number back to the real number
    # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat Key or Value for Grouped Multi-Query Attention. 

    Args:
        x (torch.Tensor): Input tensor, Key or Value.
        n_rep (int): Number of repetitions.

    Returns:
        torch.Tensor: Repeated tensor.
    
    """
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

# https://github.com/tloen/llama-int8/blob/ce74669c767e42b5082391dd0cfcb621ba40c7f9/llama/model.py#L74C1-L74C1
class UninitializedLinear(nn.Linear):
    """
    Use torch.nn.Linear class.
    
    """
    def reset_parameters(self) -> None:
        pass


class InferenceQuantizedLinear(bnb.nn.Linear8bitLt):
    """
    Use 8 bits quantized bitsandbytes.nn.Linear8bitLt class.
    
    """
    def __init__(self, *args, **kwargs):
        super().__init__(has_fp16_weights=False, threshold=6.0, *args, **kwargs)

    def reset_parameters(self) -> None:
        pass


# variable context.
default_quantize: ContextVar[bool] = ContextVar("default_quantize", default=False)


def get_linear_class() -> nn.Linear:
    # If 'default_quantize' is True, use 'bnb.nn.Linear8bitLt'.
    if default_quantize.get():
        return InferenceQuantizedLinear
    # Otherwise use 'torch.nn.Linear'.
    return UninitializedLinear

class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.
        
        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_heads_q (int): Number of local query heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (nn.Linear): Linear transformation for queries.
            wk (nn.Linear): Linear transformation for keys.
            wv (nn.Linear): Linear transformation for values.
            wo (nn.Linear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.
    
        """
        super().__init__()
        
        # Indicates the number of heads for the Keys and Values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for the Queries
        self.n_heads_q = args.n_heads
        # Indicates how many times the Keys and Values should be repeated
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicates the dimension of each head, that is, the part of the embedding that each head will be responsible for
        self.head_dim = args.dim // args.n_heads
        
        Linear = get_linear_class()
        self.wq = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
    
    def forward(self, 
                x: torch.Tensor,
                start_pos: int,
                freqs_complex: torch.Tensor,
                masks: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_complex (torch.Tensor): Precomputed frequency tensor.
            masks (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.
            
        """
        batch_size, seq_len, dim = x.shape  # (B, Seq_Len, Dim)
        # Linear transformation
        # (B, Seq_Len, Dim) -> (B, Seq_Len, H_Q * Head_Dim)
        xq = self.wq(x)
        # (B, Seq_Len, Dim) -> (B, Seq_Len, H_KV * Head_Dim)
        xk = self.wk(x)
        # (B, Seq_Len, Dim) -> (B, Seq_Len, H_KV * Head_Dim)
        xv = self.wv(x)
        
        # Reshape
        # (B, Seq_Len, H_Q * Head_Dim) -> (B, Seq_Len, H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, Seq_Len, H_KV * Head_Dim) -> (B, Seq_Len, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, Seq_Len, H_KV * Head_Dim) -> (B, Seq_Len, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply rotary embeddings to Q and K
        # (B, Seq_Len, H_Q * Head_Dim) -> (B, Seq_Len, H_Q, Head_Dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        # (B, Seq_Len, H_KV * Head_Dim) -> (B, Seq_Len, H_KV, Head_Dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)
        
        # K_V cache: replace the entry in the cache
        self.cache_k[:batch_size, start_pos: start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos: start_pos + seq_len] = xv
        
        # Retrieve K, V from cache
        # (B, Seq_Len_KV, H_KV,  Head_Dim)
        keys = self.cache_k[:batch_size, :start_pos + seq_len]
        values = self.cache_v[:batch_size, :start_pos + seq_len]
        
        # Since every group of Q shares the same K and V heads, just repeat the K and V heads for every Q in the same group.
        # (B, Seq_Len_KV, H_KV, Head_Dim) -> (B, Seq_Len_KV, H_Q, Head_Dim)
        keys = repeat_kv(keys, self.n_rep).to(x.device)
        # (B, Seq_Len_KV, H_KV, Head_Dim) -> (B, Seq_Len_KV, H_Q, Head_Dim)
        values = repeat_kv(values, self.n_rep).to(x.device)
        
        # (B, Seq_Len, H_Q, Head_Dim) -> (B, H_Q, Seq_Len, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2)
        
        # Attention operation:  Q @ K.T / sqrt(Dim) * V
        # (B, H_Q, Seq_Len, Head_Dim) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, Seq_Len, Seq_Len_KV)
        scores = torch.matmul(xq, torch.transpose(keys, 2, 3)) / math.sqrt(self.head_dim)
        # Apply masks
        if masks is not None:
            # (B, H_Q, Seq_Len, Seq_Len_KV)
            scores = scores + masks
        # (B, H_Q, Seq_Len, Seq_Len_KV) -> (B, H_Q, Seq_Len, Seq_Len_KV)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        
        # (B, H_Q, Seq_Len, Seq_Len_KV) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, Seq_Len, Head_Dim)
        output = torch.matmul(scores, values)
        # (B, H_Q, Seq_Len, Head_Dim) -> (B, Seq_Len, H_Q, Head_Dim) -> (B, Seq_Len, Dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.wo(output) # (B, Seq_Len, Dim)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        """
        Feed-forward module: self.w2(F.silu(self.w1(x)) * self.w3(x))

        Args:
            args (ModelArgs): Model configuration parameters.
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.
        
        Attributes:
            w1 (nn.Linear): Linear transformation for the first layer.
            w2 (nn.Linear): Linear transformation for the second layer.
            w3 (nn.Linear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # Round the hidden_dim to the nearest multiple of the multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        
        Linear = get_linear_class()
        self.w1 = Linear(args.dim, hidden_dim, bias=False)
        self.w2 = Linear(hidden_dim, args.dim, bias=False)
        self.w3 = Linear(args.dim, hidden_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        w2(Silu(w1(x)) * w3(x))

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
            
        """
        # (B, Seq_Len, Dim) -> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.w1(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.w3(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) -> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) -> (B, Seq_Len, Dim)
        x = self.w2(x)
        
        return x
    
    
class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        """
        Transformer Encoder block: 
        h = x + attention(RMSNorm(x))
        h += ffn(RMSNorm(h))

        Args:
            args (ModelArgs): _description_
        
        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // args.n_heads
        
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        
        # Normalization BEFORE the ATTENTION block
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        # Normalization BEFORE the FEED-FORWARD block
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
    
    def forward(self, 
                x: torch.Tensor, 
                start_pos: int, 
                freqs_complex: torch.Tensor,
                masks: torch.Tensor,
                ) -> torch.Tensor:
        """
        h = x + attention(RMSNorm(x))
        h += ffn(RMSNorm(h))

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_complex (torch.Tensor): Precomputed cosine and sine frequencies.
            masks (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.
            
        """
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex, masks
        )
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) -> (B, Seq_Len, Dim)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        
        return out
    

def convert_linear_to_bnb(float_linear):
    new_layer = InferenceQuantizedLinear(
        float_linear.in_features,
        float_linear.out_features,
        bias=float_linear.bias is not None,
    )
    new_layer._parameters["weight"] = bnb.nn.Int8Params(
        float_linear.weight.data.cpu(),
        requires_grad=False,
        has_fp16_weights=False,
    )
    if float_linear.bias is not None:
        new_layer._parameters["bias"] = float_linear.bias
    return new_layer


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        """
        Initialize a Transformer model.

        Args:
            args (ModelArgs): Model configuration parameters.
        
        Attributes:
            args (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_complex (torch.Tensor): Precomputed cosine and sine frequencies.
            
        """
        super().__init__()
        assert args.vocab_size != -1, "Vocab size must be set"
        self.args = args
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        
        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim, args.norm_eps)
        Linear = get_linear_class()
        self.output = Linear(args.dim, args.vocab_size, bias=False)
        
        self.freqs_complex = precompute_theta_pos_frequencies(
            # Note that self.args.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            args.dim // args.n_heads, args.max_seq_len * 2, args.device)
    
    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        # (B, Seq_Len)
        batch_size, seq_len = tokens.shape
        
        # (B, Seq_Len) -> (B, Seq_Len, Dim)
        x = self.tok_embeddings(tokens)
        
        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos: start_pos+seq_len].to(x.device)
        
        # Masking
        masks = None
        if seq_len > 1:
            masks = torch.full((1, 1, seq_len, seq_len), float('-inf'), device=x.device,
                               )
            masks = torch.triu(masks, diagonal=start_pos + 1).type_as(x)
            
        # Consecutively apply all the encoder layers
        for layer_id, layer in enumerate(self.layers):
            x = layer(x, start_pos, freqs_complex, masks)
        
        x = self.norm(x)
        # (B, Seq_Len, Dim) -> (B, Seq_Len, Vocab_Size)
        return self.output(x[:, -1, :]).float()

    def quantize(self):
        # https://github.com/pytorch/vision/issues/2391#issuecomment-653900218
        def get_layer(model, name):
            layer = model
            for attr in name.split("."):
                layer = getattr(layer, attr)
            return layer

        def set_layer(model, name, layer):
            try:
                attrs, name = name.rsplit(".", 1)
                model = get_layer(model, attrs)
            except ValueError:
                pass
            setattr(model, name, layer)

        linear_layers = {
            k: v for k, v in self.named_modules() if isinstance(v, nn.Linear)
        }

        print("Quantizing", len(linear_layers), "layers")
        for name, layer in linear_layers.items():
            new_layer = convert_linear_to_bnb(layer)
            set_layer(self, name, new_layer)
        self.cuda()