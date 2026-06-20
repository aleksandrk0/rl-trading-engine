# C_01_en/models/ensemble_net.py
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from typing import Dict, Tuple, Optional, List, Union, Any, Type
from dataclasses import dataclass, field
from torch.cuda.amp import autocast
import numpy as np
import json
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from models.trade_recorder import TradeRecorder, Trade
from contextlib import nullcontext
import traceback




logger = logging.getLogger(__name__)




class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512):
        """Initialize Rotary Position Embedding
        
        Args:
            dim: Embedding dimension
            max_seq_len: Maximum sequence length (default: 512)
        """
        super().__init__()
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        pos = torch.arange(max_seq_len).type_as(inv_freq)
        sinusoid = torch.einsum('i,j->ij', pos, inv_freq)
        
        self.register_buffer('sin', sinusoid.sin())
        self.register_buffer('cos', sinusoid.cos())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, dim]
            
        Returns:
            Tuple of (sin, cos) tensors for rotary embeddings
        """
        seq_len = x.shape[1]
        return self.cos[:seq_len], self.sin[:seq_len]

@dataclass 
class EnsembleNetConfig:
    """Optimized configuration for RTX 4090"""
    
    # Base dimensions
    input_dim: int = 3072
    hidden_dim: int = 3072  # Increased for RTX 4090
    sequence_length: int = 300
    num_classes: int = 2
    
    # Architecture
    num_experts: int = 6
    expert_dim: int = 1024
    num_attention_heads: int = 32
    ffn_dim: int = 8192
    num_transformer_layers: int = 12
    
    # Temporal processing
    time_scales: List[int] = field(default_factory=lambda: [1, 5, 15, 30, 60])
    max_sequence_length: int = 1024
    use_rotary_embeddings: bool = True
    
    # Optimization
    batch_size: int = 256  # Optimized for 24GB VRAM
    gradient_checkpointing: bool = True
    mixed_precision: bool = True
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    feature_dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    
    def validate(self):
        assert self.hidden_dim % self.num_attention_heads == 0, \
            "hidden_dim must be divisible by num_attention_heads"
        assert self.num_experts >= 1, "Must have at least 1 expert"
        assert all(x > 0 for x in self.time_scales), \
            "Time scales must be positive"
        assert self.batch_size > 0, "Batch size must be positive"


class ExpertNet(nn.Module):
    """Expert network with fixed dimensions"""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        
        # Fixed dimensions
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = 2  
        self.sequence_length = 300
        
        # Feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # LSTM processing 
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, self.num_classes)
        )
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Dict with logits and features
        """
        # Move input to device
        x = x.to(self.device)
        
        # Feature extraction
        features = self.feature_net(x)  # [batch, seq, hidden]
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)  # [batch, seq, hidden]
        
        # Take final timestep
        final = lstm_out[:, -1]  # [batch, hidden]
        
        # Output projection
        logits = self.output_proj(final)  # [batch, num_classes]
        
        return {
            'logits': logits,      # [batch, num_classes]
            'features': lstm_out   # [batch, seq, hidden] 
        }
        
    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class TemporalExpert(ExpertNet):
    def __init__(self, config: EnsembleNetConfig):
        super().__init__(config)
        
        # Fixed dimensions for RTX 4090
        self.hidden_dim = 2048  # Base hidden dimension
        self.groups = 8
        self.out_channels = 256 * self.groups  # Must be divisible by groups
        
        # Multi-scale temporal convolutions
        self.temporal_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    self.hidden_dim,
                    self.out_channels,  # 2048 = 256 * 8 groups
                    kernel_size=k,
                    padding='same',
                    groups=self.groups
                ),
                nn.BatchNorm1d(self.out_channels),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for k in [3, 5, 7, 11]  # Multiple kernel sizes
        ])
        
        # Output projection
        self.output_proj = nn.Linear(self.out_channels * len(self.temporal_convs), 
                                   self.hidden_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # [batch, seq, hidden] -> [batch, hidden, seq]
        x = x.transpose(1, 2)
        
        # Apply temporal convolutions
        conv_outputs = []
        for conv in self.temporal_convs:
            conv_outputs.append(conv(x))
            
        # Combine outputs [batch, out_channels*num_convs, seq]
        combined = torch.cat(conv_outputs, dim=1)
        
        # Back to [batch, seq, hidden]
        output = combined.transpose(1, 2)
        output = self.output_proj(output)
        
        return {
            'features': output
        }


class FrequencyExpert(ExpertNet):
    """Expert analyzing frequency domain patterns"""
    def __init__(self, config: EnsembleNetConfig):
        super().__init__(config)
        
        self.freq_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8, 
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        self.freq_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Get base processing
        base_out = super().forward(x)
        h = base_out['features']
        
        # FFT features
        fft = torch.fft.rfft(h, dim=1)
        fft_features = torch.cat([
            fft.real,
            fft.imag
        ], dim=-1)
        
        # Project back to hidden dim
        fft_features = self.freq_proj(fft_features)
        
        # Combine with base features
        h = h + self.freq_attention(fft_features, fft_features, h)[0]
        
        # Output
        logits = self.output_projection(h)
        
        return {
            'logits': logits,
            'features': h
        }


class EnhancedEnsemble(nn.Module):
    def __init__(self):
        """Initialize ensemble model with explicit device handling"""
        super().__init__()
        
        self.optimizer = None
        
        # Debug flag
        self.debug = False
       
        # Explicit device initialization
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.set_num_threads(16)
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            logger.info("Using CPU")
            
        # Fixed dimensions optimized for RTX 4090
        self.input_dim = 3072
        self.hidden_dim = 3072
        self.sequence_length = 300
        self.num_experts = 5
        self.batch_size = 64
        # ДОБАВЛЕНО: явное определение num_classes
        self.num_classes = 2

        # Expert feature extractors
        self.expert_features = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ).to(self.device)
            for _ in range(self.num_experts)
        ])
        
        # Expert LSTMs
        self.expert_lstms = nn.ModuleList([
            nn.LSTM(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim//2,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            ).to(self.device)
            for _ in range(self.num_experts)
        ])
        
        # Expert classifiers
        self.expert_classifiers = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.num_classes).to(self.device)
            for _ in range(self.num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.LayerNorm(self.hidden_dim//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim//2, self.num_experts)
        ).to(self.device)
        
        # Initialize metrics
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)

        # Enable gradient checkpointing
        self.use_checkpointing = True
        
        # Memory management thresholds
        self.memory_threshold = 0.8
        self.cleanup_threshold = 0.7
        
        # Set memory limit for RTX 4090
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.95)
            
        # Initialize previous memory stats
        self.prev_allocated = 0
        self.prev_reserved = 0
        
        # Validate initialization
        self._validate_init()


        


    def _get_default_outputs(self, batch_size: int) -> Dict[str, Any]:
        """Get default outputs for error cases"""
        return {
            'loss': torch.tensor(0.0, device=self.device),
            'metrics': {
                'loss': 0.0,
                'accuracy': 0.0,
                'grad_norm': 0.0
            },
            'outputs': {
                'logits': torch.zeros(batch_size, self.num_classes, device=self.device),
                'expert_outputs': torch.zeros(batch_size, self.num_experts, self.num_classes, device=self.device),
                'expert_weights': torch.ones(batch_size, self.num_experts, device=self.device) / self.num_experts,
                'features': torch.zeros(batch_size, self.hidden_dim, device=self.device),
                'uncertainty': torch.zeros(batch_size, 1, device=self.device)
            }
        }





    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with memory optimization and expert outputs"""
        try:
            batch_size = x.size(0)
            all_expert_outputs = []  # ДОБАВЛЕНО: список для всех выходов
            final_features = []

            # Process experts
            for i in range(self.num_experts):
                with torch.set_grad_enabled(self.training):
                    features = self.expert_features[i](x)
                    lstm_out, _ = self.expert_lstms[i](features)
                    final_state = lstm_out[:, -1].detach()
                    logits = self.expert_classifiers[i](final_state)
                    
                    # Store outputs before cleanup
                    all_expert_outputs.append(logits)
                    final_features.append(final_state)
                    
                    del features
                    del lstm_out
                    
                    if torch.cuda.is_available() and i % 2 == 1:
                        torch.cuda.empty_cache()

            # Stack results
            expert_outputs = torch.stack(all_expert_outputs, dim=1)  # [batch, num_experts, num_classes]
            final_features = torch.stack(final_features, dim=1)

            # Calculate weights
            mean_features = torch.mean(final_features, dim=1)
            expert_weights = F.softmax(self.gate(mean_features), dim=-1)

            # Weighted combination
            weighted_outputs = torch.bmm(
                expert_weights.unsqueeze(1),
                expert_outputs.view(batch_size, self.num_experts, -1)
            ).squeeze(1)

            # ИЗМЕНЕНО: добавляем expert_outputs в результат
            outputs = {
                'logits': weighted_outputs,
                'expert_outputs': expert_outputs.detach(),  # Сохраняем выходы экспертов
                'expert_weights': expert_weights,
                'features': mean_features
            }

            # Cleanup
            del final_features
            del all_expert_outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return outputs

        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    def _validate_outputs(self, outputs: Dict[str, torch.Tensor]) -> None:
        """Validate outputs with memory management"""
        required_outputs = {
            'logits': (self.num_classes,),
            'expert_outputs': (self.num_experts, self.num_classes),
            'expert_weights': (self.num_experts,),
            'features': (self.hidden_dim,)
        }
        
        for key, expected_shape in required_outputs.items():
            if key not in outputs:
                raise ValueError(f"Missing {key} in outputs")
            
            if not isinstance(outputs[key], torch.Tensor):
                raise ValueError(f"{key} is not a tensor")
            
            actual_shape = outputs[key].shape[1:]  # Пропускаем batch dimension
            if actual_shape != expected_shape:
                raise ValueError(
                    f"Wrong shape for {key}: "
                    f"got {actual_shape}, expected {expected_shape}"
                )



    def _calculate_metrics(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate comprehensive metrics"""
        try:
            metrics = {}
            
            # Base metrics
            logits = outputs['logits']
            predictions = logits.argmax(dim=-1)
            probs = F.softmax(logits, dim=-1)
            
            # Get targets
            targets = batch.get('targets')
            
            # Basic metrics
            if 'loss' in outputs:
                metrics['loss'] = outputs['loss'].item()
                
            # Classification metrics if targets exist
            if targets is not None:
                metrics['accuracy'] = (predictions == targets).float().mean().item()
                
                # Calculate F1 score
                tp = ((predictions == 1) & (targets == 1)).float().sum()
                fp = ((predictions == 1) & (targets == 0)).float().sum()
                fn = ((predictions == 0) & (targets == 1)).float().sum()
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                metrics['f1_score'] = (2 * precision * recall / (precision + recall + 1e-8)).item()
    
            # Expert metrics
            expert_weights = outputs['expert_weights']
            metrics.update({
                'expert_diversity': (1 - expert_weights.mean()).item(),
                'expert_max_weight': expert_weights.max().item(),
                'expert_min_weight': expert_weights.min().item(),
                'expert_agreement': outputs['expert_outputs'].std(dim=1).mean().item()
            })
    
            # Feature metrics
            if 'features' in outputs:
                features = outputs['features']
                metrics.update({
                    'feature_mean': features.mean().item(),
                    'feature_std': features.std().item(),
                    'feature_activity': (features.abs() > 0.1).float().mean().item()
                })
    
            # Add required metrics if missing
            if 'accuracy' not in metrics:
                metrics['accuracy'] = 0.0
            if 'f1_score' not in metrics:
                metrics['f1_score'] = 0.0
    
            return metrics
    
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {
                'loss': float('inf'),
                'accuracy': 0.0,
                'f1_score': 0.0
            }


    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Training step with gradient handling"""
        try:
            self.train()
            if self.optimizer is None:
                raise ValueError("Optimizer not set")
    
            features = batch['features'].to(self.device)
            targets = batch.get('targets')
            if targets is not None:
                targets = targets.to(self.device)
    
            # Zero gradients
            self.optimizer.zero_grad()
    
            # Forward pass 
            outputs = self(features)
            logits = outputs['logits']
    
            if targets is not None:
                loss = F.cross_entropy(logits, targets)
                loss.backward()
                
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)
                
                self.optimizer.step()
            else:
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)
    
            # Calculate metrics
            metrics = self._calculate_metrics(outputs, batch)
            metrics['loss'] = loss.item()
    
            return {
                'loss': loss,
                'metrics': metrics,
                'outputs': outputs
            }
    
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
            return self._get_default_outputs(features.size(0))
    
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Validation step with comprehensive metrics"""
        try:
            self.eval()
            with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
    
                outputs = self(batch['features'])
                
                if 'targets' in batch:
                    loss = F.cross_entropy(outputs['logits'], batch['targets'])
                else:
                    loss = torch.tensor(0.0, device=self.device)
    
                # Calculate expanded metrics
                metrics = self._calculate_metrics(outputs, batch)
                
                # Add ensemble-specific validation metrics
                expert_weights = outputs['expert_weights']
                metrics.update({
                    'loss': loss.item(),
                    'num_active_experts': (expert_weights > 0.1).float().mean().item(),
                    'expert_agreement': (F.softmax(outputs['expert_outputs'], dim=-1).std(dim=1)).mean().item(),
                    'val_loss': loss.item()
                })
    
                return {
                    'metrics': metrics,
                    'outputs': outputs
                }
    
        except Exception as e:
            logger.error(f"Error in validate step: {str(e)}")
            if self.debug:
                logger.error(traceback.format_exc())
            return {
                'metrics': {'loss': float('inf'), 'accuracy': 0.0},
                'outputs': {}
            }
    
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()






    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Set optimizer"""
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(f"Expected torch.optim.Optimizer, got {type(optimizer)}")
        self.optimizer = optimizer

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


            
    def to(self, device: Union[str, torch.device]) -> 'EnhancedEnsemble':
        """Override to method with proper device handling
        
        Args:
            device: Target device (string or torch.device)
            
        Returns:
            Self for chaining
        """
        if isinstance(device, str):
            device = torch.device(device)
            
        for module in self.modules():
            if isinstance(module, nn.Module):
                module._apply(lambda t: t.to(device))
                
        self.device = device
        return self


    def _memory_status(self) -> Dict[str, float]:
        """Get current memory status"""
        if not torch.cuda.is_available():
            return {}
            
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,
            'reserved': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3
        }

    def _check_memory(self) -> bool:
        """Check if memory cleanup needed"""
        if not torch.cuda.is_available():
            return False
            
        stats = self._memory_status()
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        if stats['allocated'] > self.memory_threshold * total:
            return True
            
        # Check fragmentation
        if stats['allocated'] > self.prev_allocated * 1.2:
            return True
            
        self.prev_allocated = stats['allocated']
        return False




    def _expert_forward(self, x: torch.Tensor, expert_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for single expert with fixed dtype"""
        def _run_expert(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # Ensure input has requires_grad
            if self.training and not x.requires_grad:
                x = x.detach().requires_grad_(True)
            
            # Convert to float32
            x = x.to(dtype=torch.float32)
            
            features = self.expert_features[expert_idx](x)
            lstm_out, (h, c) = self.expert_lstms[expert_idx](features)
            final_state = lstm_out[:, -1]
            logits = self.expert_classifiers[expert_idx](final_state)
            
            # Ensure outputs are float32
            return logits.to(dtype=torch.float32), final_state.to(dtype=torch.float32)
            
        if self.use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _run_expert, 
                x,
                use_reentrant=False,
                preserve_rng_state=True
            )
        return _run_expert(x)


    def _validate_init(self) -> None:
        """Validate initialization with mixed precision support"""
        try:
            x = torch.randn(2, self.sequence_length, self.input_dim,
                          device=self.device)
            
            if self.training:
                x.requires_grad_(True)
            
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
                outputs = self(x)
                
                # Validate shapes and dtypes
                expected_shapes = {
                    'logits': (2, 2),
                    'expert_outputs': (2, self.num_experts, 2),
                    'expert_weights': (2, self.num_experts),
                    'features': (2, self.hidden_dim)
                }
                
                for key, shape in expected_shapes.items():
                    if key not in outputs:
                        raise ValueError(f"Missing {key} in outputs")
                    
                    tensor = outputs[key]
                    if tensor.shape != shape:
                        raise ValueError(
                            f"Wrong shape for {key}: {tensor.shape}, "
                            f"expected {shape}"
                        )
                    
                    # Разрешаем оба типа данных
                    if tensor.dtype not in [torch.float32, torch.float16]:
                        raise ValueError(
                            f"Wrong dtype for {key}: {tensor.dtype}, "
                            f"expected float32 or float16"
                        )
    
                logger.info("Ensemble initialization validated successfully")
    
        except ValueError as e:
            logger.error(f"Initialization validation failed: {str(e)}")
            raise RuntimeError(f"Initialization validation failed: {str(e)}")






    def _calculate_diversity_loss(self, expert_outputs: torch.Tensor) -> torch.Tensor:
        """Calculate diversity loss between experts"""
        diversity_loss = torch.tensor(0.0, device=self.device)
        expert_preds = torch.argmax(expert_outputs, dim=-1)
        
        num_pairs = 0
        for i in range(expert_preds.size(1)):
            for j in range(i+1, expert_preds.size(1)):
                agreement = (expert_preds[:, i] == expert_preds[:, j]).float().mean()
                diversity_loss += agreement
                num_pairs += 1
                
        return diversity_loss / max(1, num_pairs)


    def _get_mean_features(self, features: torch.Tensor) -> torch.Tensor:
        """Get mean features for gating network"""
        if features.dim() == 3:
            return features.mean(dim=1)
        return features




    def save_model(self, path: str) -> None:
        """Сохранение ансамбля моделей с состояниями экспертов
        
        Args:
            path: путь сохранения
                
        Raises:
            RuntimeError: если возникли ошибки при сохранении
        """
        try:
            # Prepare expert states
            expert_states = []
            for i, (features, lstm, classifier) in enumerate(zip(
                self.expert_features,
                self.expert_lstms,
                self.expert_classifiers
            )):
                expert_states.append({
                    'features': features.state_dict(),
                    'lstm': lstm.state_dict(),
                    'classifier': classifier.state_dict()
                })
            
            save_dict = {
                'state_dict': self.state_dict(),
                'expert_states': expert_states,
                'config': {
                    'input_dim': self.input_dim,
                    'hidden_dim': self.hidden_dim,
                    'sequence_length': self.sequence_length,
                    'num_experts': self.num_experts,
                    'batch_size': self.batch_size
                },
                'metrics': {
                    'train': dict(self.train_metrics),
                    'val': dict(self.val_metrics)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Atomic save
            save_path = Path(path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = save_path.with_suffix('.tmp')
            
            torch.save(save_dict, tmp_path)
            tmp_path.replace(save_path)
            
            logger.info(f"Saved EnhancedEnsemble to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving EnhancedEnsemble: {str(e)}")
            if 'tmp_path' in locals() and tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Failed to save EnhancedEnsemble: {str(e)}")
    
    def load_model(self, path: str) -> None:
        """Загрузка ансамбля с восстановлением состояний экспертов
        
        Args:
            path: путь к сохраненной модели
                
        Raises:
            RuntimeError: если возникли ошибки при загрузке
            ValueError: если конфигурация не соответствует
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            # Validate configuration
            saved_config = checkpoint['config']
            critical_params = ['input_dim', 'hidden_dim', 'sequence_length', 
                             'num_experts', 'batch_size']
            
            for param in critical_params:
                if saved_config[param] != getattr(self, param):
                    raise ValueError(
                        f"Parameter mismatch for {param}: "
                        f"{saved_config[param]} != {getattr(self, param)}"
                    )
            
            # Load main model state
            self.load_state_dict(checkpoint['state_dict'])
            
            # Load expert states
            for i, expert_state in enumerate(checkpoint['expert_states']):
                self.expert_features[i].load_state_dict(expert_state['features'])
                self.expert_lstms[i].load_state_dict(expert_state['lstm'])
                self.expert_classifiers[i].load_state_dict(expert_state['classifier'])
            
            # Restore metrics
            if 'metrics' in checkpoint:
                self.train_metrics = defaultdict(list, checkpoint['metrics']['train'])
                self.val_metrics = defaultdict(list, checkpoint['metrics']['val'])
            
            # Apply RTX 4090 optimizations after loading
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                torch.set_num_threads(16)  # Optimized for i7
                
            logger.info(f"Loaded EnhancedEnsemble from {path}")
            
        except Exception as e:
            logger.error(f"Error loading EnhancedEnsemble: {str(e)}")
            raise RuntimeError(f"Failed to load EnhancedEnsemble: {str(e)}")



       
        
     
    def _create_expert(self) -> nn.Module:
        """Create single expert network"""
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim), 
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(self.hidden_dim, self.num_classes)
        )



    def _validate_dimensions(self, outputs: Dict[str, torch.Tensor], batch_size: int) -> None:
        """Validate output dimensions"""
        expected_shapes = {
            'logits': (batch_size, self.num_classes),
            'expert_weights': (batch_size, self.num_experts),
            'uncertainty': (batch_size, 1), 
            'features': (batch_size, self.sequence_length, self.hidden_dim)
        }
        
        for name, shape in expected_shapes.items():
            if name not in outputs:
                raise KeyError(f"Missing output: {name}")
            if outputs[name].shape != shape:
                raise ValueError(
                    f"Wrong shape for {name}: expected {shape}, got {outputs[name].shape}"
                )

                
    def _validate_device(self, tensor: torch.Tensor) -> None:
        """Validate tensor device"""
        if tensor.device != self.device:
            raise ValueError(f"Tensor on wrong device: {tensor.device} vs {self.device}")
            
            




    def _validate_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """Validate batch data
        
        Args:
            batch: Input batch dictionary
            
        Raises:
            ValueError: If batch data is invalid
        """
        if 'features' not in batch or 'targets' not in batch:
            raise ValueError("Batch must contain 'features' and 'targets'")
            
        features = batch['features']
        targets = batch['targets']
        
        self._validate_dimensions(features)
        
        if targets.size(0) != features.size(0):
            raise ValueError(
                f"Batch size mismatch: features {features.size(0)}, "
                f"targets {targets.size(0)}"
            )

    def get_expert_importances(self) -> Dict[str, float]:
        """Get current expert importance weights"""
        with torch.no_grad():
            weights = F.softmax(
                self.expert_selector[-1].weight,
                dim=-1
            )
            return {
                f"expert_{i}": w.mean().item()
                for i, w in enumerate(weights)
            }

    def profile_memory(self, batch_size: int = 32) -> Dict[str, float]:
        """Profile memory usage
        
        Args:
            batch_size: Batch size for profiling
            
        Returns:
            Dict with memory stats in GB
        """
        if not torch.cuda.is_available():
            return {}
            
        try:
            torch.cuda.reset_peak_memory_stats()
            
            # Create dummy batch
            features = torch.randn(
                batch_size,
                self.config.sequence_length,
                self.config.input_dim,
                device='cuda'
            )
            
            # Forward pass
            with torch.no_grad():
                _ = self(features)
                
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**3,
                'reserved': torch.cuda.memory_reserved() / 1024**3,
                'peak': torch.cuda.max_memory_allocated() / 1024**3
            }
            
        finally:
            torch.cuda.empty_cache()

    def get_complexity_stats(self) -> Dict[str, int]:
        """Get model complexity statistics
        
        Returns:
            Dict with parameter and operation counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        
        # Estimate FLOPs for one forward pass
        seq_len = self.config.sequence_length
        hidden_dim = self.config.hidden_dim
        
        attention_flops = (
            4 * seq_len * hidden_dim * hidden_dim + 
            2 * seq_len * seq_len * hidden_dim
        ) * self.config.num_transformer_layers
        
        ffn_flops = (
            2 * seq_len * hidden_dim * self.config.ffn_dim
        ) * self.config.num_transformer_layers
        
        total_flops = attention_flops + ffn_flops
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'attention_flops': attention_flops,
            'ffn_flops': ffn_flops,
            'total_flops': total_flops
        }

    def prepare_for_training(self) -> None:
        """Prepare model for training"""
        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            self.gradient_checkpointing_enable()
            
        # Move to CUDA if available
        if torch.cuda.is_available():
            self.cuda()
            # Enable TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def prepare_for_inference(self) -> None:
        """Prepare model for inference"""
        self.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Fuse layers if possible
        if hasattr(torch, 'jit'):
            try:
                self.eval()
                self = torch.jit.script(self)
            except Exception as e:
                logger.warning(f"JIT compilation failed: {str(e)}")






def init() -> EnhancedEnsemble:
    """Initialize model with default config"""
    return EnhancedEnsemble()