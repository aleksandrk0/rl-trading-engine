import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Union, Any, Type, Generator

import logging
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import csv
# Загрузчики данных нужны только для обучения; импорт опциональный,
# чтобы архитектура модели импортировалась на чистом клоне без data-пакета.
try:
    from data.enhanced_loader import EnhancedDataLoader, DataConfig
    from data.preprocessor import ForexPreprocessor, PreprocessorConfig
except ImportError:
    EnhancedDataLoader = DataConfig = None
    ForexPreprocessor = PreprocessorConfig = None
#from utils.monitoring import TrainingMonitor
#from utils.checkpoints import CheckpointManager
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext
import traceback




import torch._dynamo
torch._dynamo.config.suppress_errors = True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)

@dataclass
class EnhancedModelConfig:
    """Расширенная конфигурация для всей системы"""
    # Base dimensions (as in original data)
    input_dim: int = 3072
    hidden_dim: int = 3072
    sequence_length: int = 300
    num_classes: int = 3  # Up, Down, Sideways
    
    # Architecture
    num_attention_heads: int = 32 
    attention_dropout: float = 0.1
    feature_dropout: float = 0.1
    
    # Training
    learning_rate: float = 2e-4
    batch_size: int = 256  # For RTX 4090
    gradient_clip: float = 1.0
    max_epochs: int = 100
    
    # Feature groups (based on input structure)
    feature_groups: Dict[str, List[int]] = None
    
    def __post_init__(self):
        if self.feature_groups is None:
            # Map to input data columns
            self.feature_groups = {
                'price': list(range(0, 23)),  # price_* columns
                'volume': list(range(23, 33)), # volume_* 
                'momentum': list(range(33, 53)), # momentum_*
                'composite': list(range(53, 71)), # composite_*
                'volatility': list(range(71, 87)), # volatility_*
                'pattern': list(range(87, 95)),  # pattern_*
                'advanced': list(range(95, 133)) # advanced_*
            }


@dataclass
class TradingSystemConfig:
    """Configuration for complete trading system"""
    # System-wide parameters
    batch_size: int = 256
    learning_rate: float = 2e-4
    gradient_clip: float = 1.0
    max_epochs: int = 100
    
    # Feature groups structure
    feature_groups: Dict[str, List[int]] = None
    
    def __post_init__(self):
        if self.feature_groups is None:
            self.feature_groups = {
                'price': list(range(0, 23)),
                'volume': list(range(23, 33)),
                'momentum': list(range(33, 53)),
                'composite': list(range(53, 71)),
                'volatility': list(range(71, 87)),
                'pattern': list(range(87, 95)),
                'advanced': list(range(95, 133))
            }

@dataclass
class RegimeNetConfig:
    """Configuration for regime detection model"""
    input_dim: int = 133
    hidden_dim: int = 3072  # Optimized for RTX 4090
    sequence_length: int = 300
    num_regimes: int = 5
    batch_size: int = 256
    
    # Architecture
    lstm_layers: int = 2
    dropout_rate: float = 0.1
    use_layer_norm: bool = True
    
    # Training
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    def validate(self):
        """Validate configuration parameters"""
        assert self.input_dim > 0, "input_dim must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.sequence_length > 0, "sequence_length must be positive"
        assert self.num_regimes > 1, "num_regimes must be > 1"
        assert 0 <= self.dropout_rate < 1, "dropout must be in [0,1)"



class TransposeModule(nn.Module):
    """Module for tensor transposition"""
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(self.dim0, self.dim1)



class CustomAttention(nn.Module):
    """Custom attention module optimized for RTX 4090"""
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # [B, L, C]
        out, _ = self.attention(x, x, x)
        return out.transpose(1, 2)  # [B, C, L]




@dataclass
class FeatureGroups:
    """Feature group structure for 133 input features"""
    
    PRICE: Dict[str, slice] = field(default_factory=lambda: {
        'OHLC': slice(0, 4),             # price_open,high,low,close
        'EMA': slice(4, 14),             # price_ema_3,5,8,13,21,34,55,89,144
        'SMA': slice(14, 23)             # price_sma_3,5,8,13,21,34,55,89,144
    })
    
    VOLUME: Dict[str, slice] = field(default_factory=lambda: {
        'SMA': slice(23, 29),            # volume_sma_5,8,13,21,34
        'EMA': slice(29, 35),            # volume_ema_5,8,13,21,34
        'EXTRA': slice(35, 37)           # volume_intensity,roi
    })
    
    MOMENTUM: Dict[str, slice] = field(default_factory=lambda: {
        'RSI': slice(37, 43),            # momentum_rsi_3,5,8,13,21,34
        'MACD': slice(43, 52),           # momentum_macd_12_26,5_34,8_21 & signals
        'STOCH': slice(52, 60)           # momentum_stoch_k/d_5,14,21,34
    })
    
    COMPOSITE: Dict[str, slice] = field(default_factory=lambda: {
        'TREND': slice(60, 64),          # composite_trend_strength,volume_trend,efficiency_5,13
        'CONSENSUS': slice(64, 73)        # composite_signal_consensus & trend/volume/momentum
    })
    
    VOLATILITY: Dict[str, slice] = field(default_factory=lambda: {
        'BB': slice(73, 85),             # volatility_bb_upper/middle/lower/width_20,30,40
        'ATR': slice(85, 93)             # volatility_atr_7,14,21,28 & ratios
    })
    
    PATTERN: Dict[str, slice] = field(default_factory=lambda: {
        'CANDLE': slice(93, 101),        # pattern_cdl* patterns
        'PIVOT': slice(101, 106)         # pattern_pivot,r1,s1,r2,s2
    })
    
    ADVANCED: Dict[str, slice] = field(default_factory=lambda: {
        'CAM': slice(106, 114),          # advanced_cam_r*,s*
        'TRIX': slice(114, 120),         # advanced_trix_*
        'TECHNICAL': slice(120, 133)     # advanced_* remaining indicators
    })

def validate_input_features(features: torch.Tensor) -> None:
    """Validate input features dimensions and content

    Args:
        features: Input tensor [batch_size, sequence_length, input_dim]
        
    Raises:
        ValueError: If dimensions or content invalid
    """
    # Check dimensions
    if features.dim() != 3:
        raise ValueError(f"Expected 3D tensor, got {features.dim()}D")
        
    batch_size, seq_len, feat_dim = features.shape
    
    # Required dimensions
    if seq_len != 60: # Fixed sequence length
        raise ValueError(f"Expected sequence length 60, got {seq_len}")
        
    if feat_dim != 133: # Total number of features
        raise ValueError(f"Expected 133 features, got {feat_dim}")
        
    # Check value ranges for key features
    with torch.no_grad():
        # OHLC prices should be positive
        if (features[..., :4] <= 0).any():
            raise ValueError("Found non-positive OHLC prices")
            
        # RSI values between 0-100
        rsi_values = features[..., 37:43]
        if ((rsi_values < 0) | (rsi_values > 100)).any():
            raise ValueError("RSI values out of range [0, 100]")
            
        # Stochastic values between 0-100  
        stoch_values = features[..., 52:60]
        if ((stoch_values < 0) | (stoch_values > 100)).any():
            raise ValueError("Stochastic values out of range [0, 100]")
            
        # Volume should be positive
        if (features[..., 23:37] < 0).any():
            raise ValueError("Found negative volume values")




class RegimeDetector(nn.Module):
    """Detector for market regimes"""
    
    def __init__(self, 
                 input_dim: int = 3072,
                 num_regimes: int = 7,
                 hidden_dim: int = 3072,
                 num_heads: int = 32,
                 num_layers: int = 4,
                 sequence_length: int = 300,
                 dropout: float = 0.1,
                 *args, 
                 **kwargs):
       """Initialize regime detector
        
        Args:
            input_dim: Input dimension (default: 3072 from feature extractor)
            num_regimes: Number of regime classes (default: 7)
            hidden_dim: Hidden layer dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
       """
       super().__init__()
      
       self.debug = kwargs.get('debug', False)
        
        # Сохраняем размерности
       self.input_dim = input_dim
       self.hidden_dim = hidden_dim
       self.num_regimes = num_regimes
       self.num_heads = num_heads
       self.sequence_length = sequence_length
        
        # Основные слои
       self.input_projection = nn.Linear(input_dim, hidden_dim)
       self.position_embedding = nn.Parameter(torch.randn(1, 300, hidden_dim))
       self.num_regimes = 7

       # Memory optimization params
       self.use_checkpointing = True
       self.accumulation_steps = 4
       self.memory_fraction = 0.95

       # Device setup
       self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
       
       # Input projection
       self.input_projection = nn.Sequential(
           nn.Linear(self.input_dim, self.hidden_dim),
           nn.LayerNorm(self.hidden_dim),
           nn.GELU(),
           nn.Dropout(0.1)
       )
       
       # Multi-scale temporal convolutions
       self.temporal_convs = nn.ModuleList([
           nn.Sequential(
               nn.Conv1d(
                   self.hidden_dim,
                   self.hidden_dim // 4,
                   kernel_size=k,
                   padding=k//2,
                   groups=16
               ),
               nn.BatchNorm1d(self.hidden_dim // 4),
               nn.GELU(),
               nn.Dropout(0.1)
           ) for k in [3, 5, 7, 11]  # Multiple scales
       ])
       
       # Attention
       self.attention = nn.MultiheadAttention(
           embed_dim=self.hidden_dim,
           num_heads=16,
           dropout=0.1,
           batch_first=True
       )
       
       # LSTM
       self.lstm = nn.LSTM(
           input_size=self.hidden_dim,
           hidden_size=self.hidden_dim//2,
           num_layers=2,
           dropout=0.1,
           bidirectional=True,
           batch_first=True
       )
       
       # Regime prediction
       self.regime_detector = nn.Sequential(
           nn.Linear(self.hidden_dim, self.hidden_dim//2),
           nn.LayerNorm(self.hidden_dim//2),
           nn.GELU(),
           nn.Dropout(0.1),
           nn.Linear(self.hidden_dim//2, self.num_regimes)
       )
       
       # Market metrics
       self.volatility_predictor = nn.Sequential(
           nn.Linear(self.hidden_dim, self.hidden_dim//4),
           nn.GELU(),
           nn.Linear(self.hidden_dim//4, 1),
           nn.Softplus()
       )
       
       self.liquidity_analyzer = nn.Sequential(
           nn.Linear(self.hidden_dim, self.hidden_dim//4),
           nn.GELU(),
           nn.Linear(self.hidden_dim//4, 1),
           nn.Sigmoid()
       )
        
       self.volume = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//4),
            nn.GELU(),
            nn.Linear(self.hidden_dim//4, 3)
        )

        # Initialize weights and move to device
       self._init_weights()
       self.to(self.device)

        # Set memory limit
       if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction)

       logger.info(
            f"RegimeDetector initialized:\n"
            f"- Input dim: {self.input_dim}\n"
            f"- Hidden dim: {self.hidden_dim}\n"
            f"- Sequence length: {self.sequence_length}\n"
            f"- Num regimes: {self.num_regimes}\n"
            f"- Device: {self.device}"
        )



    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass с валидацией размерностей"""
        try:
            batch_size = x.size(0)
            
            # Input validation
            if x.dim() != 3:
                raise ValueError(f"Expected 3D input, got {x.dim()}D")
                
            if x.size(-1) != self.hidden_dim:
                raise ValueError(
                    f"Expected hidden_dim {self.hidden_dim}, got {x.size(-1)}"
                )
                
            # Input projection
            x = self.input_projection(x)  
            x = x.transpose(1, 2)  # [batch, hidden, seq]
            
            # Process temporal convolutions
            conv_outputs = []
            for conv in self.temporal_convs:
                out = conv(x)
                conv_outputs.append(out)
                
            # Combine scales
            multi_scale = torch.cat(conv_outputs, dim=1)
            features = multi_scale.transpose(1, 2)  # [batch, seq, hidden]
            
            # LSTM processing
            lstm_out, (hidden, cell) = self.lstm(features)
            final_state = lstm_out[:, -1]  # [batch, hidden]
            
            # Predictions
            regime_logits = self.regime_detector(final_state)
            regime_probs = F.softmax(regime_logits, dim=-1)
            
            # Market metrics
            volatility = self.volatility_predictor(final_state)
            liquidity = self.liquidity_analyzer(final_state)
            volume = self.volume(final_state)

            outputs = {
                'logits': regime_logits,
                'regime_probs': regime_probs,
                'features': final_state,
                'market_metrics': {
                    'volatility': volatility,
                    'liquidity': liquidity,
                    'volume': volume
                }
            }

            return outputs
                
        except Exception as e:
            logger.error(f"Forward pass error: {str(e)}")
            if self.debug:
                logger.error(traceback.format_exc())
            raise


    def _calculate_metrics(self, outputs: Dict[str, torch.Tensor], 
                         batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate comprehensive metrics including F1 score
        
        Args:
            outputs: Model outputs
            batch: Input batch
            
        Returns:
            Dict with calculated metrics
        """
        try:
            metrics = {}
            
            # Get predictions and targets
            logits = outputs['logits']  # [batch_size, num_regimes]
            predictions = logits.argmax(dim=-1)  # [batch_size]
            targets = batch.get('targets')
            
            # Basic metrics
            metrics['loss'] = outputs.get('loss', 0.0).item()
            
            if targets is not None:
                # Calculate confusion matrix metrics
                tp = ((predictions == 1) & (targets == 1)).float().sum()
                fp = ((predictions == 1) & (targets == 0)).float().sum()
                tn = ((predictions == 0) & (targets == 0)).float().sum()
                fn = ((predictions == 0) & (targets == 1)).float().sum()
                
                # Calculate derived metrics
                total = tp + tn + fp + fn + 1e-8
                
                # Basic metrics
                metrics.update({
                    'accuracy': (predictions == targets).float().mean().item(),
                    'true_positives': tp.item(),
                    'false_positives': fp.item(),
                    'true_negatives': tn.item(),
                    'false_negatives': fn.item()
                })
                
                # Precision, recall, F1
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
                
                metrics.update({
                    'precision': precision.item(),
                    'recall': recall.item(),
                    'f1_score': f1.item()
                })
            
            # Confidence metrics
            probs = F.softmax(logits, dim=-1)
            metrics['confidence'] = probs.max(dim=-1)[0].mean().item()
            
            # Distribution metrics
            metrics['entropy'] = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean().item()
            
            # Regime-specific metrics
            regime_probs = outputs.get('regime_probs')
            if regime_probs is not None:
                metrics['regime_entropy'] = -(regime_probs * 
                    torch.log(regime_probs + 1e-8)).sum(dim=-1).mean().item()
                    
            # Market metrics
            market_metrics = outputs.get('market_metrics', {})
            for k, v in market_metrics.items():
                if isinstance(v, torch.Tensor):
                    metrics[f'market_{k}'] = v.mean().item()
                    
            # Memory metrics
            if torch.cuda.is_available():
                metrics.update({
                    'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,
                    'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,
                    'gpu_utilization': (torch.cuda.memory_allocated() / 
                        torch.cuda.get_device_properties(0).total_memory)
                })
                
            metrics['epoch'] = getattr(self, 'current_epoch', 0)
            metrics['num_batches'] = getattr(self, 'num_batches', 0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            if self.debug:
                logger.error(traceback.format_exc())
            return {
                'loss': float('inf'),
                'accuracy': 0.0,
                'f1_score': 0.0,
                'confidence': 0.0
            }

    def _validate_outputs(self, outputs: Dict[str, torch.Tensor], 
                         expected_shapes: Dict[str, Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]]) -> None:
        """Validate output dimensions with nested shape checking
        
        Args:
            outputs: Dictionary of model outputs
            expected_shapes: Dictionary of expected shapes
            
        Raises:
            ValueError: If dimensions don't match
        """
        for name, shape in expected_shapes.items():
            if name not in outputs:
                raise ValueError(f"Missing {name} in outputs")
                
            if isinstance(shape, dict):
                # Nested shape validation for market_metrics
                if not isinstance(outputs[name], dict):
                    raise ValueError(f"Expected dict for {name}, got {type(outputs[name])}")
                    
                for metric_name, metric_shape in shape.items():
                    if metric_name not in outputs[name]:
                        raise ValueError(f"Missing {name}.{metric_name} in outputs")
                        
                    if outputs[name][metric_name].shape != metric_shape:
                        raise ValueError(
                            f"Wrong shape for {name}.{metric_name}: "
                            f"got {outputs[name][metric_name].shape}, "
                            f"expected {metric_shape}"
                        )
            else:
                # Direct shape validation
                if outputs[name].shape != shape:
                    raise ValueError(
                        f"Wrong shape for {name}: got {outputs[name].shape}, "
                        f"expected {shape}"
                    )
                    
        # Validate tensor types and devices
        for name, tensor in outputs.items():
            if isinstance(tensor, dict):
                for subname, subtensor in tensor.items():
                    if not isinstance(subtensor, torch.Tensor):
                        raise ValueError(f"{name}.{subname} is not a tensor")
                    if not torch.isfinite(subtensor).all():
                        raise ValueError(f"Non-finite values in {name}.{subname}")
            else:
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(f"{name} is not a tensor")
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Non-finite values in {name}")



    def _process_features_checkpoint(self, x: torch.Tensor) -> torch.Tensor:
        """Process features with dtype control"""
        def _run_features(x: torch.Tensor) -> torch.Tensor:
            x = x.to(dtype=torch.float32)
            features = self.input_projection(x)
            return features.to(dtype=torch.float32)
            
        if self.use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _run_features,
                x,
                use_reentrant=False,
                preserve_rng_state=True
            ).to(dtype=torch.float32)
            
        return _run_features(x)

    def _process_conv_checkpoint(self, x: torch.Tensor, conv: nn.Module) -> torch.Tensor:
        """Convolutional processing with dtype control"""
        def _run_conv(x: torch.Tensor) -> torch.Tensor:
            x = x.to(dtype=torch.float32)
            out = conv(x)
            return out.to(dtype=torch.float32)
            
        if self.use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _run_conv,
                x,
                use_reentrant=False,
                preserve_rng_state=True
            ).to(dtype=torch.float32)
            
        return _run_conv(x)




    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Training step with proper gradient handling"""
        try:
            # Move data to device
            features = batch['features'].to(self.device, non_blocking=True)
            targets = batch.get('targets')
            if targets is not None:
                targets = targets.to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = self(features)
                
                # Убедимся что у нас правильные размерности
                batch_size = features.shape[0]
                expected_shapes = {
                    'logits': (batch_size, self.num_regimes),          # [batch, 7]
                    'regime_probs': (batch_size, self.num_regimes),    # [batch, 7]
                    'features': (batch_size, self.hidden_dim),         # [batch, 3072]
                    'market_metrics': {
                        'volatility': (batch_size, 1),                 # [batch, 1]
                        'liquidity': (batch_size, 1)                   # [batch, 1]
                    }
                }
                
                # Validate outputs
                self._validate_outputs(outputs, expected_shapes)
                
                if targets is not None:
                    loss = F.cross_entropy(outputs['logits'], targets)
                else:
                    loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            # Calculate metrics
            with torch.no_grad():
                predictions = outputs['logits'].argmax(dim=-1)
                probs = F.softmax(outputs['logits'], dim=-1)
                
                metrics = {
                    'loss': loss.item(),
                    'volatility': outputs['market_metrics']['volatility'].mean().item(),
                    'liquidity': outputs['market_metrics']['liquidity'].mean().item()
                }

                # Classification metrics if targets exist
                if targets is not None:
                    tp = ((predictions == 1) & (targets == 1)).float().sum()
                    fp = ((predictions == 1) & (targets == 0)).float().sum()
                    tn = ((predictions == 0) & (targets == 0)).float().sum()
                    fn = ((predictions == 0) & (targets == 1)).float().sum()
                    
                    total = tp + tn + fp + fn + 1e-8
                    
                    metrics.update({
                        'accuracy': (predictions == targets).float().mean().item(),
                        'precision': (tp / (tp + fp + 1e-8)).item(),
                        'recall': (tp / (tp + fn + 1e-8)).item(),
                        'f1_score': (2 * tp / (2 * tp + fp + fn + 1e-8)).item(),
                        'mean_confidence': probs.max(dim=-1)[0].mean().item()
                    })

            # Memory metrics
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                'loss': loss,  # Важно: возвращаем тензор loss с градиентами!
                'metrics': metrics,
                'outputs': {
                    'logits': outputs['logits'].detach(),
                    'predictions': predictions,
                    'probabilities': probs,
                    'features': outputs['features'],
                    'regime_probs': outputs['regime_probs']
                }
            }

        except Exception as e:
            logger.error(f"Error in training step: {str(e)}")
            if self.debug:
                logger.error(traceback.format_exc())
            raise

 


    def _init_weights(self) -> None:
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _validate_input_dimensions(self, x: torch.Tensor) -> None:
        """Validate input tensor dimensions"""
        batch_size, seq_len, feat_dim = x.shape
        
        if seq_len != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, got {seq_len}"
            )
            
        if feat_dim != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, got {feat_dim}"
            )
            
        # Check device with explicit cuda:0
        if x.device != self.device:
            raise ValueError(
                f"Input tensor on wrong device: {x.device} vs {self.device}"
            )

        
    def _validate_initialization(self) -> None:
        """Validate weight initialization"""
        for name, param in self.named_parameters():
            # Check for NaN/Inf values
            if torch.isnan(param).any():
                raise ValueError(f"NaN values in {name}")
            if torch.isinf(param).any():
                raise ValueError(f"Inf values in {name}")
                
            # Check parameter magnitudes
            if param.abs().max() > 10:
                logger.warning(f"Large values in {name}: {param.abs().max()}")



    def _validate_init(self) -> None:
        """Validate model initialization with test input"""
        try:
            # Create test batch
            batch_size = 2
            x = torch.randn(batch_size, self.sequence_length, self.input_dim, device=self.device)
            
            # Test forward pass
            with torch.no_grad():
                outputs = self(x)
                
                # Validate output shapes
                expected_shapes = {
                    'logits': (batch_size, self.num_regimes),
                    'features': (batch_size, self.hidden_dim),
                    'lstm_out': (batch_size, self.sequence_length, self.hidden_dim)
                }
                
                for name, shape in expected_shapes.items():
                    if outputs[name].shape != shape:
                        raise ValueError(
                            f"Wrong shape for {name}: got {outputs[name].shape}, "
                            f"expected {shape}"
                        )
                        
            logger.info("Model initialization validated successfully")
            
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            raise




    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Validation step with metrics"""
        try:
            self.eval()
            with torch.no_grad():
                features = batch['features'].to(self.device)
                targets = batch.get('targets')
                if targets is not None:
                    targets = targets.to(self.device)
    
                # Forward pass
                outputs = self(features)
                
                # Calculate loss if targets exist
                if targets is not None:
                    loss = F.cross_entropy(outputs['logits'], targets)
                else:
                    loss = torch.tensor(0.0, device=self.device)
    
                outputs['loss'] = loss
                metrics = self._calculate_metrics(outputs, batch)
                metrics['loss'] = loss.item()
    
                return {
                    'metrics': metrics,
                    'outputs': {
                        k: v.detach() if isinstance(v, torch.Tensor) else v 
                        for k, v in outputs.items()
                    }
                }
    
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return {
                'metrics': {'loss': float('inf')},
                'outputs': {}
            }


            
    def _validate_batch(self, batch: Dict[str, torch.Tensor]) -> None:
        """Validate batch dimensions and contents
        
        Args:
            batch: Input batch dictionary
            
        Raises:
            ValueError: If dimensions or contents invalid
        """
        if 'features' not in batch:
            raise ValueError("Batch missing features")
            
        features = batch['features']
        if not isinstance(features, torch.Tensor):
            raise ValueError(f"Features must be tensor, got {type(features)}")
            
        if features.dim() != 3:
            raise ValueError(f"Features must be 3D, got {features.dim()}D")
            
        batch_size, seq_len, feat_dim = features.shape
        if seq_len != self.sequence_length:
            raise ValueError(f"Wrong sequence length: {seq_len} vs {self.sequence_length}")
            
        if feat_dim != self.input_dim:
            raise ValueError(f"Wrong feature dimension: {feat_dim} vs {self.input_dim}")
            
        if 'targets' in batch:
            targets = batch['targets']
            if not isinstance(targets, torch.Tensor):
                raise ValueError(f"Targets must be tensor, got {type(targets)}")
                
            if targets.dim() != 1:
                raise ValueError(f"Targets must be 1D, got {targets.dim()}D")
                
            if targets.size(0) != batch_size:
                raise ValueError(
                    f"Targets size mismatch: {targets.size(0)} vs {batch_size}"
                )

    def _log_validation_metrics(self, metrics: Dict[str, float]) -> None:
        """Log validation metrics
        
        Args:
            metrics: Dictionary of metric values
        """
        log_str = []
        for name, value in metrics.items():
            if isinstance(value, float):
                log_str.append(f"{name}: {value:.4f}")
                
        logger.info("Validation metrics: " + " ".join(log_str))

    def _get_regime_labels(self, features: torch.Tensor) -> torch.Tensor:
        """Calculate regime labels from volatility
        
        Args:
            features: Input features [batch_size, sequence_length, input_dim]
            
        Returns:
            Regime labels [batch_size] in range [0, num_regimes-1]
        """
        with torch.no_grad():
            # Get close prices (assumed to be at index 3)
            closes = features[:, :, 3]
            
            # Calculate returns
            returns = torch.diff(closes, dim=1) / closes[:, :-1]
            
            # Calculate rolling volatility (annualized)
            vol = returns.std(dim=1) * np.sqrt(252) * 100
            
            # Thresholds for regime classification
            thresholds = [10, 20, 30, 40]  # Volatility thresholds in %
            
            # Convert thresholds to tensor
            thresholds = torch.tensor(
                thresholds,
                device=vol.device,
                dtype=vol.dtype
            )
            
            # Get regimes based on thresholds
            regimes = torch.zeros_like(vol, dtype=torch.long)
            for i, threshold in enumerate(thresholds):
                regimes = torch.where(vol > threshold, i + 1, regimes)
                
            # Ensure valid range [0, num_regimes-1]
            regimes = regimes.clamp(0, self.num_regimes-1)
            
            return regimes

    def save_model(self, path: str) -> None:
        """Save model state and metrics
        
        Args:
            path: Path to save model
        """
        save_dict = {
            'state_dict': self.state_dict(),
            'train_metrics': dict(self.train_metrics),
            'val_metrics': dict(self.val_metrics),
            'trades': self.trades,
            'config': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'sequence_length': self.sequence_length,
                'num_regimes': self.num_regimes
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Create parent directories if needed
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with temp file
        tmp_path = save_path.with_suffix('.tmp')
        torch.save(save_dict, tmp_path)
        tmp_path.replace(save_path)
        
        logger.info(f"Saved model to {save_path}")

    def load_model(self, path: str) -> None:
        """Load model state and metrics
        
        Args:
            path: Path to model checkpoint
            
        Raises:
            RuntimeError: If loading fails
        """
        try:
            # Load checkpoint
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load state dict
            self.load_state_dict(checkpoint['state_dict'])
            
            # Restore metrics
            self.train_metrics = defaultdict(list, checkpoint['train_metrics'])
            self.val_metrics = defaultdict(list, checkpoint['val_metrics'])
            
            # Restore trades
            self.trades = checkpoint.get('trades', [])
            
            # Validate config
            saved_config = checkpoint['config']
            for key in ['input_dim', 'hidden_dim', 'sequence_length', 'num_regimes']:
                if saved_config[key] != getattr(self, key):
                    raise ValueError(
                        f"Config mismatch for {key}: "
                        f"saved={saved_config[key]}, "
                        f"current={getattr(self, key)}"
                    )
            
            logger.info(f"Loaded model from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def get_metrics(self) -> Dict[str, List[float]]:
        """Get training and validation metrics
        
        Returns:
            Dict containing metrics history
        """
        return {
            'train': dict(self.train_metrics),
            'val': dict(self.val_metrics) 
        }

    def reset_metrics(self) -> None:
        """Reset metrics tracking"""
        self.train_metrics.clear()
        self.val_metrics.clear()
    def _build_network(self) -> None:
        """Build neural network architecture"""
        # Feature extraction optimized for RTX 4090
        self.feature_net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Bi-directional LSTM
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim//2,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.LayerNorm(self.hidden_dim//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim//2, self.num_regimes)
        )

    def _init_trade_tracking(self) -> None:
        """Initialize trade tracking"""
        self.trade_headers = [
            'timestamp', 'direction', 'entry_price', 'exit_price',
            'position_size', 'pnl', 'regime', 'confidence',
            'duration', 'status'
        ]
        
        trades_file = self.trades_dir / f'trades_{datetime.now():%Y%m%d_%H%M%S}.csv'
        with open(trades_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.trade_headers)
            writer.writeheader()


    def _create_trades_file(self, trades_file: Path) -> None:
        """Initialize trades file with proper headers"""
        trades_dir = trades_file.parent
        trades_dir.mkdir(parents=True, exist_ok=True)
        
        with open(trades_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.trade_headers)
            writer.writeheader()

        
    def _get_volatility_regime(self, features: torch.Tensor) -> torch.Tensor:
        """Calculate regime labels from volatility
        
        Args:
            features: [batch_size, sequence_length, input_dim]
            
        Returns:
            regime labels [batch_size] in range [0, num_regimes-1]
        """
        with torch.no_grad():
            # Get close prices (assumed to be at index 3)
            closes = features[:, :, 3]
            
            # Calculate returns
            returns = torch.diff(closes, dim=1) / closes[:, :-1]
            
            # Calculate rolling volatility (annualized)
            vol = returns.std(dim=1) * np.sqrt(252) * 100  # Convert to annual %
            
            # Get regimes based on thresholds
            regimes = torch.zeros_like(vol, dtype=torch.long)
            
            for i, threshold in enumerate(self.vol_thresholds):
                regimes = torch.where(vol > threshold, i + 1, regimes)
                
            # Ensure valid range [0, num_regimes-1]
            regimes = regimes.clamp(0, self.num_regimes-1)
            
            return regimes



    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode sequence"""
        # Project input
        x = self.encoder_projection(x)
        x = x.transpose(1, 2)
        
        # Convolutional encoder
        x = self.encoder_conv1(x)
        x = self.encoder_conv2(x)
        
        # Apply attention
        x = self.attention(x)
        
        # Global pooling and project to latent
        x = x.mean(dim=2)
        latent = self.latent_projection(x)
        
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode from latent"""
        # Project to decoder dims
        x = self.decoder_projection(latent)
        x = x.unsqueeze(2).repeat(1, 1, self.sequence_length)
        
        # Transposed convolutions
        x = self.decoder_conv1(x)
        x = self.decoder_conv2(x)
        
        # Project to input dim
        x = x.transpose(1, 2)
        x = self.output_projection(x)
        
        return x

       
    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate input dimensions"""
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got {x.dim()}D")
            
        batch_size, seq_len, feat_dim = x.shape
        if seq_len != self.sequence_length:
            raise ValueError(f"Expected sequence length {self.sequence_length}, got {seq_len}")
        if feat_dim != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {feat_dim}")



    def _validate_dimensions(self, x: torch.Tensor) -> None:
        """Validate input dimensions"""
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got {x.dim()}D")
            
        batch_size, seq_len, in_dim = x.shape
        if seq_len != self.sequence_length:
            raise ValueError(f"Expected sequence length {self.sequence_length}, got {seq_len}")
        if in_dim != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {in_dim}")


    
    def _setup_optimizations(self):
        """Setup optimizations for RTX 4090"""
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_num_threads(16)  # i7-14700

   
        
    def predict(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate predictions
        
        Args:
            features: Input features [batch_size, seq_len, input_dim]
            
        Returns:
            Dict with predictions and uncertainty
        """
        self.eval()
        with torch.no_grad():
            outputs = self(features)
            predictions = outputs['logits'].argmax(dim=-1)
            probs = F.softmax(outputs['logits'], dim=-1)
            
        return {
            'predictions': predictions,
            'probabilities': probs,
            'uncertainty': outputs['uncertainty']
        }
        
    def save(self, path: Path) -> None:
        """Save model
        
        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        logger.info(f"Saved model to {path}")
        
    def load(self, path: Path) -> None:
        """Load model
        
        Args:
            path: Path to model weights
        """
        state_dict = torch.load(path, map_location=self.device)
        self.load_state_dict(state_dict)
        logger.info(f"Loaded model from {path}")



    def get_regime_probs(self, features: torch.Tensor) -> torch.Tensor:
        """
        Get regime probabilities for given features
        
        Args:
            features: Input features [batch_size, sequence_length, input_dim]
            
        Returns:
            Regime probabilities [batch_size, num_regimes]
        """
        self.eval()
        with torch.no_grad():
            outputs = self(features)
            return outputs['probabilities']


    def get_current_metrics(self, mode: str = 'train') -> Dict[str, float]:
        """
        Get most recent metrics
        
        Args:
            mode: 'train' or 'val'
            
        Returns:
            Dict with current metric values
        """
        metrics = self.get_metrics(mode)
        return {
            k: v[-1] if v else 0.0
            for k, v in metrics.items()
        }

    def save_checkpoint(self, path: str) -> None:
        """
        Save model checkpoint
        
        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'state_dict': self.state_dict(),
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """
        Load model checkpoint
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['state_dict'])
        self.train_metrics = defaultdict(list, checkpoint['train_metrics'])
        self.val_metrics = defaultdict(list, checkpoint['val_metrics'])
        logger.info(f"Loaded checkpoint from {path}")

class EnhancedFeatureProcessor(nn.Module):
    """Enhanced feature processor integrated with existing preprocessor"""
    
    def __init__(self, config: EnhancedModelConfig):
        super().__init__()
        self.config = config
        
        # Create processors for each feature group
        self.group_processors = nn.ModuleDict()
        for name, indices in config.feature_groups.items():
            self.group_processors[name] = nn.Sequential(
                nn.Linear(len(indices), config.hidden_dim // len(config.feature_groups)),
                nn.LayerNorm(config.hidden_dim // len(config.feature_groups)),
                nn.GELU(),
                nn.Dropout(config.feature_dropout)
            )
            
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.feature_dropout)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Process each feature group
        group_features = {}
        for name, indices in self.config.feature_groups.items():
            features = x[..., indices]
            group_features[name] = self.group_processors[name](features)
            
        # Combine groups
        combined = torch.cat(list(group_features.values()), dim=-1)
        fused = self.fusion(combined)
        
        return {
            'fused_features': fused,
            'group_features': group_features
        }

class MarketRegimeAnalyzer(nn.Module):
    """Integrated market regime analyzer"""
    
    def __init__(self, config: EnhancedModelConfig):
        super().__init__()
        self.config = config
        
        # Main regime detection
        self.regime_detector = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.feature_dropout),
            nn.Linear(config.hidden_dim, 5)  # 5 market regimes
        )
        
        # Volatility analysis
        self.volatility_analyzer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(), 
            nn.Dropout(config.feature_dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        # Trend strength
        self.trend_analyzer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.feature_dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Get regime probabilities
        regime_logits = self.regime_detector(features)
        regime_probs = F.softmax(regime_logits, dim=-1)
        
        # Analyze volatility
        volatility = self.volatility_analyzer(features)
        
        # Analyze trend strength  
        trend_strength = self.trend_analyzer(features)
        
        return {
            'regime_probs': regime_probs,
            'regime_logits': regime_logits,
            'volatility': volatility,
            'trend_strength': trend_strength
        }

class DirectionalPredictor(nn.Module):
    """Enhanced directional predictor with regime awareness"""
    
    def __init__(self, config: EnhancedModelConfig):
        super().__init__()
        self.config = config
        
        # Price direction prediction
        self.direction_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.feature_dropout),
            nn.Linear(config.hidden_dim, config.num_classes)
        )
        
        # Uncertainty estimation
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.feature_dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor, regime_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Combine base features with regime information
        combined = torch.cat([features, regime_features], dim=-1)
        
        # Get predictions
        direction_logits = self.direction_predictor(combined)
        direction_probs = F.softmax(direction_logits, dim=-1)
        
        # Estimate uncertainty
        uncertainty = self.uncertainty_estimator(combined)
        
        return {
            'direction_logits': direction_logits,
            'direction_probs': direction_probs,
            'uncertainty': uncertainty
        }

class RiskManager(nn.Module):
    """Enhanced risk management"""
    
    def __init__(self, config: EnhancedModelConfig):
        super().__init__()
        self.config = config
        
        # Risk parameter estimation
        self.risk_estimator = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.feature_dropout),
            nn.Linear(config.hidden_dim, 3)  # position_size, sl, tp
        )
        
        # Additional risk metrics
        self.risk_metrics = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.feature_dropout),
            nn.Linear(config.hidden_dim // 2, 3) # kelly_fraction, max_drawdown, vol_target
        )
        
    def forward(self, 
               features: torch.Tensor,
               regime_features: torch.Tensor,
               direction_features: torch.Tensor) -> Dict[str, torch.Tensor]:
                   
        # Combine all information
        combined = torch.cat([features, regime_features, direction_features], dim=-1)
        
        # Get risk parameters
        risk_params = self.risk_estimator(combined)
        position_size = torch.sigmoid(risk_params[..., 0])
        stop_loss = F.softplus(risk_params[..., 1])
        take_profit = F.softplus(risk_params[..., 2])
        
        # Get additional metrics
        risk_metrics = self.risk_metrics(combined)
        kelly = torch.sigmoid(risk_metrics[..., 0])
        max_dd = F.softplus(risk_metrics[..., 1])
        vol_target = F.softplus(risk_metrics[..., 2])
        
        return {
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'kelly_fraction': kelly,
            'max_drawdown': max_dd,
            'volatility_target': vol_target
        }

class EnhancedTradingSystem(nn.Module):
    """Complete trading system with integration"""
    
    def __init__(self, config: EnhancedModelConfig):
        super().__init__()
        self.config = config
        
        # Initialize components
        self.feature_processor = EnhancedFeatureProcessor(config)
        self.regime_analyzer = MarketRegimeAnalyzer(config)
        self.direction_predictor = DirectionalPredictor(config)
        self.risk_manager = RiskManager(config)
        
        # Checkpointing
#        self.checkpoint_manager = CheckpointManager('trading_system')
        
        # Initialize metrics
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        
        # Model to correct device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Process features
        feature_outputs = self.feature_processor(x)
        features = feature_outputs['fused_features']
        
        # Analyze regime
        regime_outputs = self.regime_analyzer(features)
        
        # Predict direction
        direction_outputs = self.direction_predictor(
            features,
            regime_outputs['regime_probs']
        )
        
        # Manage risk
        risk_outputs = self.risk_manager(
            features,
            regime_outputs['regime_probs'],
            direction_outputs['direction_probs']
        )
        
        return {
            'features': feature_outputs,
            'regime': regime_outputs,
            'direction': direction_outputs,
            'risk': risk_outputs
        }
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Training step with multiple objectives"""
        outputs = self(batch['features'])
        
        # Direction prediction loss
        direction_loss = F.cross_entropy(
            outputs['direction']['direction_logits'],
            batch['direction_targets']
        )
        
        # Regime prediction loss
        regime_loss = F.cross_entropy(
            outputs['regime']['regime_logits'],
            batch['regime_targets']
        )
        
        # Risk estimation loss
        risk_loss = F.mse_loss(
            outputs['risk']['position_size'],
            batch['position_targets']
        )
        
        # Uncertainty calibration
        uncertainty_loss = F.binary_cross_entropy(
            outputs['direction']['uncertainty'].squeeze(),
            (outputs['direction']['direction_probs'].argmax(dim=-1) != 
             batch['direction_targets']).float()
        )
        
        # Total loss with weights
        total_loss = (
            direction_loss + 
            0.3 * regime_loss +
            0.2 * risk_loss + 
            0.1 * uncertainty_loss
        )
        
        # Calculate metrics
        with torch.no_grad():
            metrics = self._calculate_metrics(outputs, batch)
        
        return {
            'loss': total_loss,
            'direction_loss': direction_loss,
            'regime_loss': regime_loss,
            'risk_loss': risk_loss,
            'uncertainty_loss': uncertainty_loss,
            'metrics': metrics,
            'outputs': outputs
        }
        
    def _calculate_metrics(self, 
                         outputs: Dict[str, torch.Tensor],
                         batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate training metrics"""
        
        # Direction accuracy
        predictions = outputs['direction']['direction_probs'].argmax(dim=-1)
        accuracy = (predictions == batch['direction_targets']).float().mean()
        
        # Regime accuracy  
        regime_predictions = outputs['regime']['regime_probs'].argmax(dim=-1)
        regime_accuracy = (regime_predictions == batch['regime_targets']).float().mean()
        
        # Risk metrics
        position_error = F.mse_loss(
            outputs['risk']['position_size'],
            batch['position_targets']
        )
        
        # Uncertainty calibration
        uncertainty = outputs['direction']['uncertainty'].squeeze()
        mistakes = (predictions != batch['direction_targets']).float()
        calibration_error = F.mse_loss(uncertainty, mistakes)
        
        return {
            'accuracy': accuracy.item(),
            'regime_accuracy': regime_accuracy.item(),
            'position_error': position_error.item(),
            'calibration_error': calibration_error.item()
        }
        
    def get_trading_decision(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get complete trading decision"""
        with torch.no_grad():
            outputs = self(features)
            
            # Get directional prediction
            direction_probs = outputs['direction']['direction_probs']
            predicted_direction = direction_probs.argmax(dim=-1)
            
            # Get position size scaled by uncertainty
            position_size = outputs['risk']['position_size']
            position_size = position_size * (1 - outputs['direction']['uncertainty'])
            
            # Scale by regime volatility
            position_size = position_size / (outputs['regime']['volatility'] + 1e-6)
            
            return {
                'direction': predicted_direction,
                'direction_probability': direction_probs.max(dim=-1)[0],
                'position_size': position_size,
                'stop_loss': outputs['risk']['stop_loss'],
                'take_profit': outputs['risk']['take_profit'],
                'regime': outputs['regime']['regime_probs'].argmax(dim=-1),
                'regime_volatility': outputs['regime']['volatility'],
                'uncertainty': outputs['direction']['uncertainty'],
                'kelly_fraction': outputs['risk']['kelly_fraction'],
                'max_drawdown': outputs['risk']['max_drawdown'],
                'volatility_target': outputs['risk']['volatility_target']
            }
            

    def save_checkpoint(self, path: str):
        """Save model checkpoint with metrics"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config.__dict__,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.train_metrics = checkpoint.get('train_metrics', defaultdict(list))
        self.val_metrics = checkpoint.get('val_metrics', defaultdict(list))
        logger.info(f"Loaded checkpoint from {path}")

    def train_model(self, 
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   num_epochs: int = 100) -> Dict[str, List[float]]:
        """Complete training loop optimized for RTX 4090"""
        
        # Initialize optimizers
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.learning_rate,
            epochs=num_epochs,
            steps_per_epoch=len(train_loader)
        )
        
        # Gradient scaler for mixed precision
        scaler = torch.cuda.amp.GradScaler()
        
        # Training loop
        best_val_loss = float('inf')
        patience = 0
        max_patience = 10
        
        for epoch in range(num_epochs):
            # Training phase
            self.train()
            train_losses = []
            train_metrics = defaultdict(list)
            
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
                for batch in pbar:
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass with mixed precision
                    with torch.amp.autocast('cuda'):
                        outputs = self.train_step(batch)
                        loss = outputs['loss']
                    
                    # Backward pass with gradient scaling
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.gradient_clip)
                    
                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # Update LR
                    scheduler.step()
                    
                    # Log metrics
                    train_losses.append(loss.item())
                    for k, v in outputs['metrics'].items():
                        train_metrics[k].append(v)
                        
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': np.mean(train_losses[-100:]),
                        'acc': np.mean(train_metrics['accuracy'][-100:])
                    })
            
            # Validation phase
            val_metrics = self.validate(val_loader)
            
            # Log epoch metrics
            self.train_metrics['loss'].append(np.mean(train_losses))
            for k, v in train_metrics.items():
                self.train_metrics[k].append(np.mean(v))
                
            self.val_metrics['loss'].append(val_metrics['loss'])
            for k, v in val_metrics.items():
                self.val_metrics[k].append(v)
                
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                self.save_checkpoint('best_model.pt')
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    logger.info("Early stopping triggered")
                    break
                    
            # Log epoch summary
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train loss: {np.mean(train_losses):.4f}, "
                f"Val loss: {val_metrics['loss']:.4f}, "
                f"Train acc: {np.mean(train_metrics['accuracy']):.4f}, "
                f"Val acc: {val_metrics['accuracy']:.4f}"
            )
            
        return {
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation loop"""
        self.eval()
        val_losses = []
        val_metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.train_step(batch)
                
                val_losses.append(outputs['loss'].item())
                for k, v in outputs['metrics'].items():
                    val_metrics[k].append(v)
                    
        return {
            'loss': np.mean(val_losses),
            **{k: np.mean(v) for k, v in val_metrics.items()}
        }

class DataPipeline:
    """Enhanced data pipeline optimized for M5 data"""
    
    def __init__(self, config: EnhancedModelConfig):
        self.config = config
        
        # Initialize data loaders
        self.preprocessor = ForexPreprocessor(
            sequence_length=config.sequence_length
        )
        
        self.data_loader = EnhancedDataLoader(
            batch_size=config.batch_size,
            sequence_length=config.sequence_length,
            num_workers=4,  # Optimize for i7-14700
            pin_memory=True,
            prefetch_factor=2
        )
        
    def prepare_data(self, data_path: str) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders"""
        # Load and preprocess data
        features, targets = self.preprocessor.fit_transform(data_path)
        
        # Create datasets
        train_size = int(0.8 * len(features))
        
        train_dataset = EnhancedDataset(
            features=features[:train_size],
            targets=targets[:train_size],
            config=self.config,
            mode='train'
        )
        
        val_dataset = EnhancedDataset(
            features=features[train_size:],
            targets=targets[train_size:],
            config=self.config,
            mode='val'
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2
        )
        
        return train_loader, val_loader

def train_system(data_path: str, 
                output_dir: str,
                config: Optional[EnhancedModelConfig] = None) -> EnhancedTradingSystem:
    """Train complete trading system"""
    
    # Setup config
    if config is None:
        config = EnhancedModelConfig()
        
    # Initialize components
    data_pipeline = DataPipeline(config)
    train_loader, val_loader = data_pipeline.prepare_data(data_path)
    
    # Create model
    model = EnhancedTradingSystem(config)
    
    # Train model
    train_metrics = model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.max_epochs
    )
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.save_checkpoint(output_path / 'final_model.pt')
    
    # Save metrics
    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(train_metrics, f, indent=2)
        
    return model

# Использование:
#if __name__ == '__train_system__':
    
