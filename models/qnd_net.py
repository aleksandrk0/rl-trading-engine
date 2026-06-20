# models/qnd_net.py


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gc
import logging
import numpy as np
from typing import Dict, Tuple, Optional, List, Union, Any, Type
from dataclasses import dataclass, field
from collections import defaultdict
from torch.amp import autocast, GradScaler
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from torch.utils.checkpoint import checkpoint
import traceback


logger = logging.getLogger(__name__)

@dataclass
class QNDConfig:
    """Configuration for QND Agent"""
    # Base dimensions (сохраняем оригинальные)
    input_dim: int = 3072
    hidden_dim: int = 3072
    sequence_length: int = 300
    action_dim: int = 3
    n_atoms: int = 51
    
    # Distribution parameters
    v_min: float = -10.0
    v_max: float = 10.0
    gamma: float = 0.99
    
    # Architecture parameters
    num_heads: int = 8
    dropout: float = 0.1
    num_residual_blocks: int = 4



@dataclass
class QNDNetConfig:
    """Optimized QND configuration for RTX 4090"""
    
    # Base dimensions
    input_dim: int = 3072
    hidden_dim: int = 3072  # Optimized for RTX 4090
    state_dim: int = 1024
    sequence_length: int = 300
    num_classes: int = 2
    action_dim: int = 3  # Long, Short, Hold
    num_regimes: int = 5
    
    # Distributional RL params
    n_atoms: int = 101
    v_min: float = -100.0
    v_max: float = 100.0
    
    # Architecture
    num_transformer_layers: int = 12
    num_attention_heads: int = 32
    ffn_dim: int = 8192
    num_conv_layers: int = 4
    
    # Multi-scale temporal analysis
    time_scales: List[int] = field(
        default_factory=lambda: [1, 5, 15, 30, 60, 240, 1440]
    )
    max_sequence_length: int = 1024
    use_rotary_embeddings: bool = True
    
    # Optimization for RTX 4090
    batch_size: int = 512
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    feature_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    
    # Risk modeling
    risk_distortion: float = 0.1
    uncertainty_weight: float = 0.2
    conservative_factor: float = 0.1
    
    # Advanced features
    use_noisy_nets: bool = True
    use_dueling: bool = True
    use_double_q: bool = True
    use_per: bool = True

    def validate(self):
        """Validate configuration"""
        assert self.hidden_dim % self.num_attention_heads == 0, "hidden_dim must be divisible by num_attention_heads"
        assert self.hidden_dim <= 4096, "hidden_dim too large for RTX 4090"
        assert self.batch_size <= 512, "batch_size too large for memory"
        assert all(x > 0 for x in self.time_scales), "time_scales must be positive"
        
        # Memory estimation
        mem_per_batch = (
            self.batch_size * 
            self.sequence_length * 
            self.hidden_dim * 
            4  # bytes per float32
        ) / (1024**3)  # Convert to GB
        
        assert mem_per_batch < 20, f"Estimated memory usage too high: {mem_per_batch:.1f}GB"

    def get_device_config(self) -> Dict[str, Any]:
        """Get device-specific settings"""
        return {
            'allow_tf32': True,
            'cudnn_benchmark': True,
            'cudnn_deterministic': False,
            'num_workers': 8,  # For i7-14700
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 2
        }

    def optimize_for_gpu(self) -> None:
        """Optimize settings for GPU"""
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Get GPU info
            gpu_name = torch.cuda.get_device_name()
            memory_info = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"Optimizing for {gpu_name} with {memory_info:.1f}GB memory")

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding optimized for RTX 4090
    
    Attributes:
        dim (int): Embedding dimension
        max_seq_len (int): Maximum sequence length
        inv_freq (torch.Tensor): Inverse frequency buffer
        sin (torch.Tensor): Sine buffer
        cos (torch.Tensor): Cosine buffer
    """
    
    def __init__(self, dim: int = 32, max_seq_len: int = 512):
        """Initialize embeddings
        
        Args:
            dim: Embedding dimension (must be even)
            max_seq_len: Maximum sequence length supported
        """
        super().__init__()
        
        # Validate dim is even
        if dim % 2 != 0:
            raise ValueError(f"Dimension {dim} must be even")
            
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Initialize frequencies for rotation
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        
        # Create position encodings
        position = torch.arange(max_seq_len).float()
        sinusoid = torch.einsum('i,j->ij', position, freqs)
        
        # Register buffers for CUDA support 
        self.register_buffer('sin', sinusoid.sin())  # [seq_len, dim/2]
        self.register_buffer('cos', sinusoid.cos())  # [seq_len, dim/2]
        
        # Move to appropriate device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        # Log initialization
        logger.info(
            f"Initialized RotaryPositionalEmbedding: "
            f"dim={dim}, max_seq_len={max_seq_len}"
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with dimension checks
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            
        Returns:
            Tuple of (cos, sin) tensors [seq_len, dim/2]
        """
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
            )
            
        # Return cached embeddings for sequence length
        return (
            self.cos[:seq_len],  # [seq_len, dim/2]
            self.sin[:seq_len]   # [seq_len, dim/2]
        )
        
    def rotate_queries_and_keys(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to queries and keys
        
        Args:
            queries: Query tensor [batch, heads, seq_len, dim]
            keys: Key tensor [batch, heads, seq_len, dim]
            
        Returns:
            Tuple of rotated (queries, keys) tensors
        """
        # Get sequence length
        seq_len = queries.size(-2)
        
        # Get embeddings
        cos_emb = self.cos[:seq_len]  # [seq_len, dim/2]
        sin_emb = self.sin[:seq_len]  # [seq_len, dim/2]
        
        # Reshape for broadcasting
        cos_emb = cos_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim/2]
        sin_emb = sin_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, dim/2]
        
        # Split features for rotation
        queries_split = queries.chunk(2, dim=-1)
        keys_split = keys.chunk(2, dim=-1)
        
        # Apply rotation
        queries_rot = torch.cat([
            queries_split[0] * cos_emb - queries_split[1] * sin_emb,
            queries_split[1] * cos_emb + queries_split[0] * sin_emb
        ], dim=-1)
        
        keys_rot = torch.cat([
            keys_split[0] * cos_emb - keys_split[1] * sin_emb,
            keys_split[1] * cos_emb + keys_split[0] * sin_emb
        ], dim=-1)
        
        return queries_rot, keys_rot
        
    def _validate_tensor_device(self, tensor: torch.Tensor) -> None:
        """Validate tensor is on correct device
        
        Args:
            tensor: Input tensor
            
        Raises:
            ValueError if tensor on wrong device
        """
        if tensor.device != self.device:
            raise ValueError(
                f"Tensor on {tensor.device}, "
                f"expected {self.device}"
            )
            
    def _validate_dimensions(self, x: torch.Tensor) -> None:
        """Validate input dimensions
        
        Args:
            x: Input tensor
            
        Raises:
            ValueError if dimensions invalid
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {x.dim()}D")
            
        if x.size(-1) != self.dim:
            raise ValueError(
                f"Feature dimension {x.size(-1)} " 
                f"does not match embedding dim {self.dim}"
            )
            
    def get_max_sequence_length(self) -> int:
        """Get maximum supported sequence length"""
        return self.max_seq_len
        
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"RotaryPositionalEmbedding("
            f"dim={self.dim}, "
            f"max_seq_len={self.max_seq_len})"
        )

class TemporalContextProcessor(nn.Module):
    """Temporal context processor optimized for RTX 4090"""
    def __init__(self):
        super().__init__()
        
        # Fixed dimensions for RTX 4090 
        self.input_dim = 3072
        self.hidden_dim = 512  # Reduced for temporal processing
        self.sequence_length = 300
        self.dropout = 0.1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Fixed temporal convolutions with same padding
        self.temporal_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim // 4,
                    kernel_size=k,
                    padding=(k-1)//2,  # Same padding
                    groups=8  # Grouped convolutions for efficiency
                ),
                nn.BatchNorm1d(self.hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
            for k in [3, 5, 7, 9]  # Multiple scales
        ])
        
        # Fixed output projection
        total_features = (self.hidden_dim // 4) * len(self.temporal_convs)
        self.output_projection = nn.Linear(total_features, self.hidden_dim)
        
        # Move to device
        self.to(self.device)
        
        # Initialize weights
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with proper validation
        
        Args:
            x: Input tensor [batch_size, sequence_length, input_dim]
            
        Returns:
            Output tensor [batch_size, sequence_length, hidden_dim]
        """
        # Validate input dimensions
        batch_size, seq_len, feat_dim = x.shape
        assert seq_len == self.sequence_length, f"Expected seq_len {self.sequence_length}, got {seq_len}"
        assert feat_dim == self.input_dim, f"Expected feat_dim {self.input_dim}, got {feat_dim}"

        # Project input
        h = self.input_projection(x)
        h = h.transpose(1, 2)  # [batch, hidden, seq]

        # Process at multiple scales with padding
        scale_outputs = []
        for conv in self.temporal_convs:
            scale_out = conv(h)  # [batch, hidden//4, seq] 
            scale_outputs.append(scale_out)
            
        # Combine scales
        h = torch.cat(scale_outputs, dim=1)  # [batch, hidden, seq]
        h = h.transpose(1, 2)  # [batch, seq, hidden]
        
        # Final projection
        output = self.output_projection(h)
        
        # Validate output
        assert output.shape == (batch_size, self.sequence_length, self.hidden_dim)
        
        return output
    
    def _init_weights(self) -> None:
        """Initialize weights with custom scaling"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='gelu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='gelu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _validate_init(self) -> None:
        """Validate model initialization"""
        with torch.no_grad():
            try:
                # Test with small batch
                x = torch.randn(2, self.sequence_length, self.input_dim, device=self.device)
                out = self.forward(x)
                
                # Validate output shape
                assert out.shape == (2, self.sequence_length, self.hidden_dim), \
                    f"Output shape mismatch: expected {(2, self.sequence_length, self.hidden_dim)}, got {out.shape}"
                
                # Test memory efficiency
                torch.cuda.empty_cache()
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(f"Peak GPU memory usage: {peak_memory:.2f} GB")
                
            except Exception as e:
                raise RuntimeError(f"Initialization validation failed: {str(e)}")

    def _process_scales(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Process input at different temporal scales
        
        Args:
            x: Input tensor [batch_size, hidden_dim, sequence_length]
        """
        scale_outputs = []
        for conv in self.temporal_convs:
            # Apply convolution with padding
            conv_out = conv(x)
            scale_outputs.append(conv_out)
        return scale_outputs

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in GB"""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**3,
                'cached': torch.cuda.memory_reserved() / 1024**3,
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**3
            }
        return {}

    def reset_memory_stats(self) -> None:
        """Reset memory statistics"""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

class MarketStateEncoder(nn.Module):
    """Market state encoder with LSTM and attention"""
    
    def __init__(self):
        super().__init__()
        # Fixed dimensions and parameters
        self.input_dim = 3072 
        self.hidden_dim = 3072
        self.sequence_length = 300
        self.num_layers = 4
        self.dropout = 0.1
        self.num_heads = 32
        self.batch_size = 256
        
        # Input projection 
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Bidirectional LSTM
        self.encoder = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim // 2,
            num_layers=self.num_layers, 
            dropout=self.dropout,
            bidirectional=True,
            batch_first=True
        )
        
        # Multi-head attention 
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=16,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Market metrics prediction
        self.metrics_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 3)  # volatility, trend, volume
        )
        
        # Initialize weights
        self._init_weights()
        
        if torch.cuda.is_available():
            # Enable TF32
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable compilation if available
            if hasattr(torch, 'compile'):
                self = torch.compile(
                    self,
                    mode='reduce-overhead',
                    fullgraph=True
                )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass
        
        Args:
            x: Input tensor [batch_size, sequence_length, input_dim]
            
        Returns:
            Dict containing:
                output: Encoded features [batch_size, sequence_length, hidden_dim]
                metrics: Market metrics [batch_size, 3]
                attention_weights: Attention weights
                hidden_state: LSTM hidden state
        """
        # Validate input dimensions
        self._validate_dimensions(x)
            
        # Project input
        projected = self.input_projection(x)  # [batch, seq, hidden]
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.encoder(projected)  # [batch, seq, hidden]
        
        # Self attention
        # Prepare attention mask to prevent attending to future tokens
        attn_mask = self._generate_causal_mask(x.size(1)).to(x.device)
        
        attended, attention_weights = self.attention(
            query=lstm_out,
            key=lstm_out,
            value=lstm_out,
            attn_mask=attn_mask,
            need_weights=True
        )  # [batch, seq, hidden]
        
        # Feature transformation
        features = self.feature_transform(attended)  # [batch, seq, hidden]
        
        # Market metrics prediction from final state
        final_state = features[:, -1]  # [batch, hidden]
        metrics = self.metrics_head(final_state)  # [batch, 3]
        
        # Pack outputs
        outputs = {
            'output': features,  # [batch, seq, hidden]
            'metrics': metrics,  # [batch, 3]
            'attention_weights': attention_weights,  # [batch, seq, seq]
            'hidden_state': hidden,  # [num_layers*2, batch, hidden/2]
            'cell_state': cell  # [num_layers*2, batch, hidden/2]
        }
        
        return outputs

    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """Generate causal mask for self-attention"""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return mask

    def _validate_dimensions(self, x: torch.Tensor) -> None:
        """Validate input dimensions"""
        batch_size, seq_len, feat_dim = x.shape
        
        if seq_len != self.sequence_length:
            raise ValueError(f"Expected sequence length {self.sequence_length}, got {seq_len}")
            
        if feat_dim != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {feat_dim}")

    @property
    def device(self):
        """Get current device"""
        return next(self.parameters()).device

class AdaptiveBlock(nn.Module):
    """Enhanced residual block with dual processing paths"""
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        
        # Fast path with ReLU
        self.fast_path = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )
        
        # Slow path with GELU
        self.slow_path = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )
        
        # Dynamic fusion
        self.gate = nn.Sequential(
            nn.Linear(dim, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        fast = self.fast_path(h)
        slow = self.slow_path(h)
        gates = self.gate(h)
        return x + gates[:, 0:1] * fast + gates[:, 1:] * slow

class EnhancedFeatureProcessor(nn.Module):
    """Multi-scale feature processing"""
    def __init__(self, config: QNDConfig):
        super().__init__()
        
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Multi-scale convolutions
        self.temporal_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(config.hidden_dim, config.hidden_dim//4, 
                         kernel_size=k, padding='same'),
                nn.LayerNorm([config.hidden_dim//4, config.sequence_length]),
                nn.GELU()
            ) for k in [3, 7, 15]  # Multiple timeframes
        ])
        
        # Cross attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.residual_blocks = nn.ModuleList([
            AdaptiveBlock(config.hidden_dim)
            for _ in range(config.num_residual_blocks)
        ])
        
        self.output = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial projection
        h = self.input_proj(x)  # [batch, seq, hidden]
        
        # Multi-scale processing
        h_scaled = h.transpose(1, 2)  # [batch, hidden, seq]
        scales = []
        for conv in self.temporal_convs:
            scales.append(conv(h_scaled))
        h_multi = torch.cat(scales, dim=1)  # [batch, hidden, seq]
        h_multi = h_multi.transpose(1, 2)  # [batch, seq, hidden]
        
        # Self attention
        h_attended, _ = self.attention(h_multi, h_multi, h_multi)
        
        # Residual processing
        h = h_attended
        for block in self.residual_blocks:
            h = block(h)
            
        return self.output(h)




class ResidualBlock(nn.Module):
    """Residual block optimized for RTX 4090"""
    
    def __init__(self, dim: int):
        """Initialize residual block
        
        Args:
            dim: Input/output dimension
        """
        super().__init__()
        
        # Two-layer residual block with LayerNorm
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(0.1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection
        
        Args:
            x: Input tensor [batch_size, ..., dim]
            
        Returns:
            Output tensor [batch_size, ..., dim]
        """
        return x + self.layers(x)
    
    def _init_weights(self) -> None:
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Kaiming initialization
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


class PositionScaling(nn.Module):
    """Custom scaling for position sizes"""
    def __init__(self, min_size: float = 0.1, max_size: float = 0.8):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.min_size + (self.max_size - self.min_size) * torch.sigmoid(x)

def _memory_cleanup(self) -> None:
    """Clean up GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
        # Log memory stats
        memory = torch.cuda.memory_allocated() / 1e9
        if memory > 12.0:  # Over 12GB
            logger.warning(f"High memory usage: {memory:.1f}GB")



class QNDAgent(nn.Module):
    """QND Agent with default initialization"""

    def __init__(
        self,
        models: Optional[Dict[str, nn.Module]] = None,
        input_dim: int = 3072,
        hidden_dim: int = 3072,
        action_dim: int = 3,
        n_atoms: int = 51,
        dropout: float = 0.1
    ) -> None:
        """Initialize QND Agent with model dependencies
        
        Args:
            models: Dictionary of required models {'feature': FeatureExtractor,
                                                 'directional': DirectionalPredictor,
                                                 'regime': RegimeDetector}
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            action_dim: Action dimension
            n_atoms: Number of atoms for distribution
            dropout: Dropout rate
        """
        super().__init__()

        # Debug flag and optimizer initialization    
        self.debug = False
        self.optimizer = None
        
        # Store models with validation
        self.models = {}
        if models is not None:
            required_models = {'feature', 'directional', 'regime'}
            if not all(name in models for name in required_models):
                raise ValueError(
                    f"Missing required models. Expected {required_models}, "
                    f"got {set(models.keys())}"
                )
            self.models = models
        
        # Fixed base dimensions for RTX 4090
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.dropout = dropout
    
        # Device initialization
        if torch.cuda.is_available():
            self._device = torch.device('cuda:0')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            self.scaler = GradScaler()
            self.autocast_context = torch.amp.autocast(device_type='cuda')
        else:
            self._device = torch.device('cpu')
            self.scaler = None
            self.autocast_context = nullcontext()



        # Debug flag and memory tracking
        self.debug = False
        self.peak_memory = 0
        self.last_memory = 0
        
        # Fixed base dimensions for RTX 4090
        self.input_dim = 3072
        self.hidden_dim = 3072
        self.sequence_length = 300
        self.action_dim = 3
        self.n_atoms = 51
        self.num_classes = 2

        # ИЗМЕНИТЬ: Сохраняем ссылки на модели
        self.models = models
        if models is None:
            self.models = {}

        # Initialize distributional RL parameters
        self.v_min = -10.0
        self.v_max = 10.0
        delta = (self.v_max - self.v_min) / (self.n_atoms - 1)
        
        # Initialize support vector [n_atoms]
        support = torch.arange(self.n_atoms) * delta + self.v_min
        self.register_buffer('support', support)
        
        # Initialize metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

        
        # Initialize trades tracking
        self.trades = []
        self.current_trade = None
        
        # Initialize trading parameters
        self.initial_balance = 10000.0
        self.position_size = 0.02
        self.pip_value = 10.0
        self.pip_target = 15.0
        self.risk_reward = 1.5
        
        # Trading costs
        self.spread = 0.0002
        self.slippage = 0.0001
        self.commission = 0.0001

        # Добавляем структуры для сделок
        self.trades: List[Dict[str, Any]] = []
        self.trades_buffer_size = 10000  # Оптимизировано под 64GB RAM
        self.trades_dir: Optional[Path] = None
        self.current_trade: Optional[Dict[str, Any]] = None
        
        # Буфер для оптимизации записи
        self.trades_buffer: List[Dict[str, Any]] = []
        
        # Метаданные
        self.trades_metadata = {
            'model_version': '3.5',
            'start_time': None,
            'end_time': None
        }

        if not all(x > 0 for x in [input_dim, hidden_dim, action_dim, n_atoms]):
            raise ValueError("All dimensions must be positive")

        # Основные размерности
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.n_atoms = n_atoms
        self.dropout = dropout
        
        # ДОБАВИТЬ: Loss weights
        self.loss_weights = {
            'classification': 1.0,
            'uncertainty': 0.2,
            'position': 0.3,
            'weights_entropy': 0.1
        }
        
        # ДОБАВЛЕНО: Размерности внешних сигналов
        self.directional_dim = 2
        self.regime_dim = 7

        # ИЗМЕНЕНО: Обновлен входной слой
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # ДОБАВЛЕНО: Fusion layer для объединения сигналов
        self.signal_fusion = nn.Sequential(
            nn.Linear(hidden_dim + self.directional_dim + self.regime_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Расширенная feature_net с residual connections
        self.feature_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ),
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        ])
        
        # ДОБАВЛЕНО: Multi-head attention для обработки сигналов
        self.signal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=16,
            dropout=dropout,
            batch_first=True
        )
        
        # ДОБАВЛЕНО: Context gate для взвешивания сигналов
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 2),
            nn.Softmax(dim=-1)
        )
        
        # Улучшенный LSTM с gradient checkpointing
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )

        # ДОБАВИТЬ: Слои для взвешивания моделей
        self.weight_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.LayerNorm(self.hidden_dim//2),
            nn.GELU(),
            nn.Linear(self.hidden_dim//2, 3)  # weights for 3 models
        )

        # Action network
        self.action_net = nn.Sequential(
            nn.Linear(self.hidden_dim + 9, self.hidden_dim//2),  # 9 = 2(directional) + 7(regime)
            nn.LayerNorm(self.hidden_dim//2),
            nn.GELU(),
            nn.Linear(self.hidden_dim//2, 3)  # 3 actions
        )
        
        # ДОБАВЛЕНО: Автоматическая настройка весов моделей
        self.model_weights = nn.Parameter(torch.ones(3))   # QND, Directional, Regime
        
        # ИЗМЕНЕНО: Расширенный classifier с учетом всех сигналов
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim)
        )

        # ИЗМЕНЕНО: Uncertainty с корректными размерностями
        self.uncertainty_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Sigmoid()
        )
        
        # ИЗМЕНИТЬ: Position sizing с ограничениями
        self.position_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            PositionScaling(min_size=0.1, max_size=0.8)
        )

        # ИЗМЕНЕНО: Value network
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_atoms)
        )

        self.model_weights = nn.Parameter(torch.ones(3))
        self.temperature = 2.0


        # ИЗМЕНЕНО: Добавлена валидация устройства
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Сохраняем параметры для логирования
        self.model_config = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'action_dim': action_dim,
            'n_atoms': n_atoms,
            'dropout': dropout,
            'device': self.device
        }
        
        # Device setup with optimizations
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')

        self.to(self._device)
        self._init_weights()
        
        # Логирование конфигурации
        logger.info(
            f"Initialized QNDAgent:\n"
            f"- Input dim: {input_dim}\n"
            f"- Hidden dim: {hidden_dim}\n"
            f"- Action dim: {action_dim}\n"
            f"- N atoms: {n_atoms}\n"
            f"- Device: {self._device}\n"
            f"- Models: {list(self.models.keys()) if self.models else 'None'}"
        )


    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Set optimizer with validation"""
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"Expected torch.optim.Optimizer, got {type(optimizer)}")
        
        self.optimizer = optimizer
        logger.info(f"Set optimizer: {type(optimizer).__name__}")




    
    def _init_position_net(self) -> None:
        """Initialize position sizing network"""
        self.position_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, 1),
            PositionScaling(min_size=0.1, max_size=0.8)
        )


    def _calculate_metrics(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate comprehensive metrics with trading statistics
        
        Args:
            outputs: Model outputs dictionary containing:
                - logits: [batch_size, num_classes]
                - expert_outputs: [batch_size, num_experts, num_classes] 
                - expert_weights: [batch_size, num_experts]
                - uncertainty: [batch_size, 1]
                - features: [batch_size, hidden_dim]
            batch: Input batch dictionary containing:
                - features: [batch_size, seq_len, input_dim]
                - targets: [batch_size] (optional)
                
        Returns:
            Dict[str, float]: Calculated metrics
        """
        try:
            metrics = {}
            
            def safe_metric(value: Union[torch.Tensor, float]) -> float:
                """Safely convert tensor to float"""
                if isinstance(value, torch.Tensor):
                    return value.detach().cpu().item()
                return float(value)
    
            # Get base predictions
            logits = outputs['logits']  # [batch_size, num_classes]
            predictions = logits.argmax(dim=-1)  # [batch_size]
            probs = F.softmax(logits, dim=-1)  # [batch_size, num_classes]
            
            # Base metrics
            metrics['loss'] = safe_metric(outputs.get('loss', 0.0))
            
            # Get targets
            targets = batch.get('targets')  # [batch_size] or None
    
            # Classification metrics if targets exist
            if targets is not None:
                # Basic counts
                tp = ((predictions == 1) & (targets == 1)).float().sum()
                fp = ((predictions == 1) & (targets == 0)).float().sum()
                tn = ((predictions == 0) & (targets == 0)).float().sum()
                fn = ((predictions == 0) & (targets == 1)).float().sum()
                
                total = tp + tn + fp + fn + 1e-8
                total_trades = tp + fp + fn
                
                # Basic metrics
                metrics.update({
                    'true_positives': safe_metric(tp),
                    'false_positives': safe_metric(fp),
                    'true_negatives': safe_metric(tn),
                    'false_negatives': safe_metric(fn),
                    'total_trades': safe_metric(total_trades),
                    'total_decisions': safe_metric(total),
                    
                    # Classification metrics
                    'accuracy': safe_metric((predictions == targets).float().mean()),
                    'precision': safe_metric(tp / (tp + fp + 1e-8)),
                    'recall': safe_metric(tp / (tp + fn + 1e-8)),
                    'specificity': safe_metric(tn / (tn + fp + 1e-8)),
                    'f1_score': safe_metric(2 * tp / (2 * tp + fp + fn + 1e-8)),
                    
                    # Trading metrics  
                    'active_rate': safe_metric((tp + fp) / total),
                    'accuracy_balanced': safe_metric((tp/(tp+fn+1e-8) + tn/(tn+fp+1e-8))/2),
                    'prediction_rate': safe_metric((tp+fp)/total),
                    'win_rate': safe_metric(tp / (tp + fn + 1e-8)),
                    'loss_rate': safe_metric(fn / (tp + fn + 1e-8))
                })
                
                # Trading efficiency
                if total_trades > 0:
                    metrics.update({
                        'trades_accuracy': safe_metric(tp / total_trades),
                        'missed_trades_ratio': safe_metric(fn / total_trades),
                        'false_trades_ratio': safe_metric(fp / total_trades)
                    })
                    
                # Profit metrics
                if tp > 0 and fp > 0:
                    profit_factor = (tp / fp).clamp(max=10)
                    win_rate = metrics['win_rate']
                    kelly_score = win_rate - ((1 - win_rate) / (safe_metric(profit_factor) + 1e-8))
                    
                    metrics.update({
                        'profit_factor': safe_metric(profit_factor),
                        'kelly_criterion': kelly_score,
                        'profit_efficiency': metrics['trades_accuracy'],
                        'avg_win_loss_ratio': safe_metric((tp/fp).clamp(max=10))
                    })
    
            # Confidence metrics
            confidence = probs.max(dim=-1)[0]  # [batch_size]
            metrics.update({
                'mean_confidence': safe_metric(confidence.mean()),
                'confidence_std': safe_metric(confidence.std()),
                'high_confidence_rate': safe_metric((confidence > 0.8).float().mean())
            })
    
            # Uncertainty metrics
            if 'uncertainty' in outputs:
                uncertainty = outputs['uncertainty']  # [batch_size, 1]
                metrics.update({
                    'uncertainty_mean': safe_metric(uncertainty.mean()),
                    'uncertainty_std': safe_metric(uncertainty.std()),
                    'high_uncertainty_rate': safe_metric((uncertainty > 0.5).float().mean())
                })
    
            # Position sizing metrics
            if 'position_size' in outputs:
                position_size = outputs['position_size']  # [batch_size, 1]
                metrics.update({
                    'position_mean': safe_metric(position_size.mean()),
                    'position_std': safe_metric(position_size.std()),
                    'position_utilization': safe_metric((position_size > 0.1).float().mean())
                })
    
            # Value distribution metrics
            if 'value' in outputs:
                value = outputs['value']  # [batch_size, n_atoms]
                metrics.update({
                    'value_mean': safe_metric(value.mean()),
                    'value_std': safe_metric(value.std())
                })
    
            # Market volatility metrics
            if 'volatility' in outputs:
                volatility = outputs['volatility']  # [batch_size]
                metrics['market_volatility'] = safe_metric(volatility.mean())
    
            # Market regime metrics
            if 'market_regime' in outputs:
                regime = outputs['market_regime'].argmax(dim=-1)  # [batch_size]
                metrics['market_regime'] = safe_metric(regime.float().mean())
    
            # Risk metrics
            if 'risk_score' in outputs:
                risk_score = outputs['risk_score']  # [batch_size, 1]
                metrics['risk_score'] = safe_metric(risk_score.mean())
    
            # Distribution metrics
            metrics.update({
                'pred_entropy': safe_metric(-(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()),
                'logits_mean': safe_metric(logits.mean()),
                'logits_std': safe_metric(logits.std())
            })
    
            # GPU metrics if available
            if torch.cuda.is_available():
                metrics.update({
                    'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,
                    'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,
                    'gpu_utilization': torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                })
    
            return metrics
    
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            if self.debug:
                logger.error(traceback.format_exc())
            return {
                'loss': float('inf'),
                'accuracy': 0.0,
                'f1_score': 0.0,
                'mean_confidence': 0.0
            }



    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Training step with comprehensive metrics
        
        Args:
            batch: Input batch dictionary containing:
                - features: Input tensor [batch_size, sequence_length, input_dim]
                - targets: Optional labels [batch_size]
                
        Returns:
            Dict containing:
                - loss: Training loss
                - metrics: Calculated metrics
                - outputs: Model outputs
        """
        try:
            # Validate optimizer
            if self.optimizer is None:
                raise ValueError("Optimizer not set. Call set_optimizer first.")
    
            # Move data to device efficiently
            features = batch['features'].to(self.device, non_blocking=True)
            targets = batch.get('targets')
            if targets is not None:
                targets = targets.to(self.device, non_blocking=True)
    
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                # Get predictions from feature extractor
                with torch.no_grad():
                    directional_outputs = self.models['directional'](features)
                    directional_probs = F.softmax(directional_outputs['logits'], dim=-1)
    
                    regime_outputs = self.models['regime'](features)
                    regime_probs = F.softmax(regime_outputs['logits'], dim=-1)
    
                # Prepare combined inputs
                qnd_inputs = {
                    'features': features,
                    'directional_probs': directional_probs,
                    'regime_probs': regime_probs
                }
    
                # Forward through QND
                outputs = self(qnd_inputs)
                
                if targets is not None:
                    loss = F.cross_entropy(outputs['logits'], targets)
                else:
                    loss = torch.tensor(0.0, device=self.device, requires_grad=True)
    
                outputs['loss'] = loss
    
            # Calculate all metrics
            with torch.no_grad():
                metrics = self._calculate_metrics(outputs, batch)
                predictions = outputs['logits'].argmax(dim=-1)
                probs = F.softmax(outputs['logits'], dim=-1)
    
                # Memory management
                torch.cuda.empty_cache()
    
                # Detach outputs for metrics
                detached_outputs = {
                    'logits': outputs['logits'].detach(),
                    'predictions': predictions,
                    'uncertainty': outputs['uncertainty'].detach(),
                    'position_size': outputs['position_size'].detach(),
                    'features': outputs['features'].detach(),
                    'model_weights': outputs['model_weights'].detach()
                }
    
                return {
                    'loss': loss,  # Keep tensor with gradients
                    'metrics': metrics,  # Dictionary of float values
                    'outputs': detached_outputs  # Detached tensors
                }
    
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                logger.warning(f"OOM in train_step, trying to recover...")
                return self._get_default_outputs(features.size(0))
            raise e
            
        except Exception as e:
            logger.error(f"Error in QND train_step: {str(e)}")
            if self.debug:
                logger.error(traceback.format_exc())
            return self._get_default_outputs(features.size(0))
    
        finally:
            # Final cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()    




    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Validation step with proper mode setting"""
        try:
            self.eval()
            for model_name in ['directional', 'regime']:
                if model_name in self.models:
                    self.models[model_name].eval()
    
            with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                features = batch['features'].to(self.device)
                targets = batch.get('targets')
                if targets is not None:
                    targets = targets.to(self.device)
    
                directional_outputs = self.models['directional'](features)
                directional_probs = F.softmax(directional_outputs['logits'], dim=-1)
    
                regime_outputs = self.models['regime'](features)
                regime_probs = F.softmax(regime_outputs['logits'], dim=-1)
    
                outputs = self({
                    'features': features,
                    'directional_probs': directional_probs,
                    'regime_probs': regime_probs
                })
    
                if targets is not None:
                    loss = F.cross_entropy(outputs['logits'], targets)
                else:
                    loss = torch.tensor(0.0, device=self.device)
    
                metrics = self._calculate_metrics(outputs, batch)
                
                # ИЗМЕНЕНО: Сохраняем веса моделей отдельно
                weights = outputs['model_weights'].detach().cpu().numpy().tolist()
                
                metrics.update({
                    'loss': loss.item(),
                    'qnd_weight': weights[0],
                    'directional_weight': weights[1],  
                    'regime_weight': weights[2]
                })
    
                return {
                    'metrics': metrics,
                    'outputs': outputs,
                    'model_weights': weights  # веса передаются отдельно
                }
    
        except Exception as e:
            logger.error(f"Error in validate step: {str(e)}")
            if self.debug:
                logger.error(traceback.format_exc())
            return {
                'metrics': {
                    'loss': float('inf'),
                    'accuracy': 0.0
                },
                'outputs': {},
                'model_weights': [1/3, 1/3, 1/3]  # дефолтные веса
            }
    
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()




    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]], **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass with support for both dict and keyword arguments
        
        Args:
            x: Either:
               - Tensor of shape [batch_size, sequence_length, hidden_dim]
               - Dict containing features
            **kwargs: Additional inputs including:
               - directional_probs: [batch_size, 2]
               - regime_probs: [batch_size, 7]
        
        Returns:
            Dict containing model outputs
        """
        try:
            # Convert kwargs to dict input if provided
            if isinstance(x, torch.Tensor) and kwargs:
                x = {
                    'features': x,
                    'directional_probs': kwargs.get('directional_probs'),
                    'regime_probs': kwargs.get('regime_probs')
                }
    
            # Handle different input types
            if isinstance(x, dict):
                # Validate dictionary input
                required_keys = {'features'}
                if not all(k in x for k in required_keys):
                    raise ValueError(f"Missing required keys in input dict. Expected {required_keys}")
                
                features = x['features']
                directional_probs = x.get('directional_probs')
                regime_probs = x.get('regime_probs')
                
            else:
                # Handle tensor input
                features = x
                directional_probs = None
                regime_probs = None
    
            # Get predictions from models if not provided
            if directional_probs is None or regime_probs is None:
                with torch.no_grad():
                    directional_outputs = self.models['directional'](features)
                    regime_outputs = self.models['regime'](features)
                    
                    directional_probs = F.softmax(directional_outputs['logits'], dim=-1)
                    regime_probs = F.softmax(regime_outputs['logits'], dim=-1)
    
            # Validate dimensions
            batch_size = features.size(0)
            if features.dim() != 3 or features.size(-1) != self.hidden_dim:
                raise ValueError(
                    f"Wrong features shape: {features.shape}, "
                    f"expected [batch_size, seq_len, {self.hidden_dim}]"
                )
                
            if directional_probs.size() != (batch_size, 2):
                raise ValueError(
                    f"Wrong directional_probs shape: {directional_probs.shape}, "
                    f"expected [{batch_size}, 2]"
                )
                
            if regime_probs.size() != (batch_size, 7):
                raise ValueError(
                    f"Wrong regime_probs shape: {regime_probs.shape}, "
                    f"expected [{batch_size}, 7]"
                )
    
            # Process features through LSTM
            lstm_out, (hidden, cell) = self.lstm(features)
            final_state = lstm_out[:, -1]  # [batch_size, hidden_dim]
    
            # Get model weights (среднее по батчу)
            model_weights = F.softmax(self.weight_net(final_state), dim=-1)  # [batch_size, 3]
            model_weights = model_weights.mean(dim=0)  # [3]
    
            # Combine features for final prediction
            combined_features = torch.cat([
                final_state,          # [batch_size, hidden_dim]
                directional_probs,    # [batch_size, 2]
                regime_probs         # [batch_size, 7]
            ], dim=-1)
    
            # Generate outputs
            logits = self.action_net(combined_features)         # [batch_size, 3]
            value = self.value_net(final_state)                # [batch_size, 51]
            uncertainty = self.uncertainty_net(final_state)     # [batch_size, 3]
            position_size = torch.sigmoid(self.position_net(final_state))  # [batch_size, 1]
    
            outputs = {
                'logits': logits,                    # [batch_size, 3]
                'value': value,                      # [batch_size, 51]
                'uncertainty': uncertainty,           # [batch_size, 3]
                'position_size': position_size,       # [batch_size, 1]
                'model_weights': model_weights,       # [3]
                'features': final_state              # [batch_size, hidden_dim]
            }
    
            return outputs
    
        except Exception as e:
            logger.error(f"Error in QND forward: {str(e)}")
            raise



    def _validate_outputs(self, outputs: Dict[str, torch.Tensor]) -> None:
        """Validate model outputs
        
        Args:
            outputs: Dictionary of model outputs
            
        Raises:
            ValueError if outputs invalid
        """
        expected_keys = {
            'logits': (self.action_dim,),
            'uncertainty': (self.action_dim,),
            'position_size': (1,),
            'value': (self.n_atoms,),
            'features': (self.hidden_dim,),
            'model_weights': (3,)
        }
        
        for key, expected_shape in expected_keys.items():
            if key not in outputs:
                raise ValueError(f"Missing output: {key}")
                
            output = outputs[key]
            if not isinstance(output, torch.Tensor):
                raise ValueError(f"Output {key} must be tensor")
                
            if output.shape[-len(expected_shape):] != expected_shape:
                raise ValueError(
                    f"Wrong shape for {key}: got {output.shape}, "
                    f"expected *{expected_shape}"
                )
    
    def _transform_directional(self, probs: torch.Tensor) -> torch.Tensor:
       """Transform directional probabilities to action space"""
       batch_size = probs.shape[0]
       transformed = torch.zeros(batch_size, self.action_dim, device=probs.device)
       transformed[:, 0] = probs[:, 0]  # Short
       transformed[:, 2] = probs[:, 1]  # Long
       transformed[:, 1] = 1 - transformed[:, 0] - transformed[:, 2]  # Hold
       return transformed
    
    def _transform_regime(self, probs: torch.Tensor) -> torch.Tensor:
       """Transform regime probabilities to action biases"""
       regime_biases = torch.tensor([
           [0.4, 0.2, 0.4],  # Trend Up
           [0.4, 0.2, 0.4],  # Trend Down 
           [0.2, 0.6, 0.2],  # Range High
           [0.2, 0.6, 0.2],  # Range Low
           [0.3, 0.4, 0.3],  # Breakout Up
           [0.3, 0.4, 0.3],  # Breakout Down
           [0.1, 0.8, 0.1]   # Sideways
       ], device=probs.device)
       
       # Matrix multiply for weighted combination
       action_biases = torch.matmul(probs, regime_biases)
       return action_biases


    def _validate_input_dimensions(self, x: torch.Tensor) -> None:
        """Validate input tensor dimensions"""
        batch_size, seq_len, feat_dim = x.shape
        
        if feat_dim != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, got {feat_dim}"
            )
            
        if seq_len != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, got {seq_len}"
            )

    
    def _get_class_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """Calculate class weights to handle imbalance
        
        Args:
            labels: Ground truth labels [batch_size]
            
        Returns:
            Class weights tensor [num_classes]
        """
        try:
            # Validate input
            if not isinstance(labels, torch.Tensor):
                raise TypeError(f"Expected torch.Tensor, got {type(labels)}")
                
            if labels.dim() != 1:
                raise ValueError(f"Expected 1D tensor, got {labels.dim()}D")
                
            # Calculate class counts with proper device
            counts = torch.bincount(labels.long(), minlength=self.action_dim).float()
            
            # Calculate weights
            total = counts.sum()
            weights = total / (counts + 1e-8)  # Add epsilon to prevent division by zero
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Move to correct device
            weights = weights.to(self._device)
            
            # Validate output
            if not torch.isfinite(weights).all():
                raise ValueError("Weights contain NaN/Inf values")
                
            if weights.shape != (self.action_dim,):
                raise ValueError(
                    f"Wrong weights shape: {weights.shape}, "
                    f"expected ({self.action_dim},)"
                )
                
            return weights
            
        except Exception as e:
            logger.error(f"Error calculating class weights: {str(e)}")
            # Return uniform weights as fallback
            return torch.ones(self.action_dim, device=self._device) / self.action_dim




    def _get_default_outputs(self, batch_size: int) -> Dict[str, Any]:
        """Get default outputs for error cases"""
        return {
            'loss': torch.tensor(0.0, device=self._device),
            'metrics': {
                'loss': 0.0,
                'accuracy': 0.0,
                'grad_norm': 0.0
            },
            'outputs': {
                'logits': torch.zeros(batch_size, self.action_dim, device=self._device),
                'uncertainty': torch.zeros(batch_size, self.action_dim, device=self._device),
                'features': torch.zeros(batch_size, self.hidden_dim, device=self._device)
            }
        }



    def _calculate_trading_metrics(
        self,
        logits: torch.Tensor,
        positions: Optional[torch.Tensor],
        prices: Optional[torch.Tensor],
        uncertainty: torch.Tensor,
        position_size: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate comprehensive trading metrics"""
        metrics = {}
        
        # Get action probabilities
        action_probs = F.softmax(logits, dim=1)
        
        # Basic statistics
        metrics['avg_position_size'] = position_size.mean().item()
        metrics['avg_uncertainty'] = uncertainty.mean().item()
        
        # Action probabilities
        long_probs = action_probs[:, 2]
        short_probs = action_probs[:, 0]
        hold_probs = action_probs[:, 1]
        
        metrics.update({
            'long_ratio': long_probs.mean().item(),
            'short_ratio': short_probs.mean().item(),
            'hold_ratio': hold_probs.mean().item(),
            'high_confidence_ratio': (action_probs.max(dim=1)[0] > 0.7).float().mean().item()
        })
        
        if positions is not None and prices is not None:
            # Calculate returns and positions
            price_changes = (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1]
            position_changes = positions[1:] != positions[:-1]
            
            # Trading activity metrics
            active_positions = (positions != 0).float()
            metrics['position_ratio'] = active_positions.mean().item()
            metrics['trade_frequency'] = position_changes.float().mean().item()
            
            # PnL metrics
            if len(price_changes) > 0:
                # Calculate returns for each position
                position_returns = price_changes * positions[:-1]
                
                # Overall returns
                total_return = position_returns.sum().item()
                metrics['total_return'] = total_return
                metrics['avg_return'] = position_returns.mean().item()
                metrics['return_std'] = position_returns.std().item()
                
                # Success metrics
                profitable_trades = (position_returns > 0).float()
                metrics['win_rate'] = profitable_trades.mean().item()
                
                # Risk metrics
                if metrics['return_std'] > 0:
                    metrics['sharpe_ratio'] = (metrics['avg_return'] / metrics['return_std'])
                else:
                    metrics['sharpe_ratio'] = 0.0
                    
                # Drawdown
                cumulative_returns = torch.cumsum(position_returns, dim=0)
                rolling_max = torch.maximum.accumulate(cumulative_returns)
                drawdowns = (rolling_max - cumulative_returns) / rolling_max
                metrics['max_drawdown'] = drawdowns.max().item()
                
                # Position-specific metrics
                long_returns = position_returns[positions[:-1] == 1]
                short_returns = position_returns[positions[:-1] == -1]
                
                if len(long_returns) > 0:
                    metrics['long_win_rate'] = (long_returns > 0).float().mean().item()
                    metrics['avg_long_return'] = long_returns.mean().item()
                    
                if len(short_returns) > 0:
                    metrics['short_win_rate'] = (short_returns > 0).float().mean().item()
                    metrics['avg_short_return'] = short_returns.mean().item()
                    
                # Risk/Reward
                if metrics['win_rate'] > 0:
                    avg_win = position_returns[position_returns > 0].mean().item()
                    avg_loss = position_returns[position_returns < 0].mean().item()
                    metrics['risk_reward_ratio'] = abs(avg_win / avg_loss) if avg_loss != 0 else 0
                    
                # Trading costs
                total_trades = position_changes.sum().item()
                metrics['trading_costs'] = total_trades * self.trading_cost
                metrics['net_return'] = total_return - metrics['trading_costs']
                
                # Additional metrics
                metrics['avg_trade_duration'] = (position_changes == 0).float().mean().item()
                metrics['profit_factor'] = (profitable_trades.sum() / len(profitable_trades)).item()
                metrics['recovery_factor'] = total_return / metrics['max_drawdown'] if metrics['max_drawdown'] > 0 else 0
        
        return metrics
    
    def _calculate_loss(
        self, 
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculate composite loss with trading penalties"""
        metrics = outputs['metrics']
        
        # Trading performance loss
        trading_loss = 0.0
        if 'net_return' in metrics:
            # Penalize negative returns
            trading_loss -= metrics['net_return'] * 0.3
            
            # Penalize high drawdown
            if metrics['max_drawdown'] > self.max_drawdown_threshold:
                trading_loss += (metrics['max_drawdown'] - self.max_drawdown_threshold) * 0.5
                
            # Penalize low win rate
            if metrics['win_rate'] < self.min_win_rate:
                trading_loss += (self.min_win_rate - metrics['win_rate']) * 0.3
                
            # Penalize poor risk/reward
            if metrics['risk_reward_ratio'] < self.min_risk_reward:
                trading_loss += (self.min_risk_reward - metrics['risk_reward_ratio']) * 0.2
        
        # Classification loss
        if 'labels' in batch:
            class_loss = F.cross_entropy(outputs['logits'], batch['labels'])
        else:
            class_loss = torch.tensor(0.0, device=self.device)
        
        # Uncertainty regularization 
        uncertainty_loss = outputs['uncertainty'].mean() * 0.1
        
        # Position size regularization
        size_loss = torch.mean((outputs['position_size'] - self.target_position_size).pow(2))
        
        # Combine losses
        total_loss = (
            class_loss * 1.0 +
            trading_loss * 0.5 +
            uncertainty_loss * 0.3 +
            size_loss * 0.2
        )
        
        return total_loss




    def _calculate_trade_loss(
        self,
        metrics: Dict[str, float],
        position_size: torch.Tensor,
        uncertainty: torch.Tensor
    ) -> torch.Tensor:
        """Calculate trading-specific loss
        
        Args:
            metrics: Trading metrics dictionary
            position_size: Position sizes [batch_size, 1]
            uncertainty: Model uncertainty [batch_size, action_dim]
            
        Returns:
            Trade loss tensor
        """
        # Position size penalty
        size_penalty = torch.mean((position_size - self.target_position_size).pow(2))
        
        # Uncertainty penalty
        uncertainty_penalty = torch.mean(uncertainty)
        
        # Drawdown penalty
        drawdown_penalty = torch.tensor(metrics.get('max_drawdown', 0.0), 
                                      device=self.device)
        
        # Combine penalties
        trade_loss = (
            size_penalty * 0.3 +
            uncertainty_penalty * 0.3 +
            drawdown_penalty * 0.4
        )
        
        return trade_loss




    def _get_regime_weights(self, regime_probs: torch.Tensor) -> torch.Tensor:
        """Calculate importance weights for different regimes
        
        Args:
            regime_probs: Tensor of shape [batch_size, 7]
            
        Returns:
            Tensor of shape [batch_size]
        """
        regime_importance = torch.tensor([
            1.2,  # Trend Up
            1.2,  # Trend Down
            0.8,  # Range High 
            0.8,  # Range Low
            1.0,  # Breakout Up
            1.0,  # Breakout Down
            0.6   # Sideways
        ], device=regime_probs.device)
        
        weights = torch.matmul(regime_probs, regime_importance.unsqueeze(-1))
        return weights.squeeze(-1)


    
    def _validate_dimensions(self) -> None:
        """Validate model dimensions"""
        test_input = torch.randn(2, self.input_dim).to(self.device)
        try:
            with torch.no_grad():
                _ = self(test_input)
        except Exception as e:
            raise ValueError(f"Model validation failed: {str(e)}")




    def _init_weights(self) -> None:
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
        logger.debug("Model weights initialized")
                

    
       
