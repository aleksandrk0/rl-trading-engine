import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union, Any, Type
import numpy as np
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import json
import pandas as pd
import numpy as np
from datetime import datetime, date
from pandas.errors import SpecificationError
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext
from models.trade_recorder import TradeRecorder, Trade
import matplotlib.pyplot as plt
import csv
import traceback


import torch._dynamo
torch._dynamo.config.suppress_errors = True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)



@dataclass
class DirectionalNetConfig:
    """Configuration for DirectionalPredictor"""
    # Base dimensions
    input_dim: int = 3072
    hidden_dim: int = 3072
    sequence_length: int = 300
    num_classes: int = 2
    
    # Architecture params
    num_lstm_layers: int = 2
    num_attention_heads: int = 8
    ffn_dim: int = 2048
    temporal_scales: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    
    # Regularization
    dropout: float = 0.15
    attention_dropout: float = 0.1
    feature_dropout: float = 0.1
    
    # Training specifics
    batch_size: int = 512
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    
    # Advanced features
    use_context: bool = True  # Use market context
    use_feature_groups: bool = True  # Use feature group processing
    use_uncertainty: bool = True  # Enable uncertainty estimation
    use_size_estimation: bool = True  # Enable position size estimation

    def __post_init__(self):
        self.validate()
    
    def validate(self):
        """Validate configuration parameters"""
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
            
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
            
        if self.hidden_dim % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_dim must be divisible by num_attention_heads, "
                f"got {self.hidden_dim} and {self.num_attention_heads}"
            )
            
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0,1), got {self.dropout}")
            
        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {self.sequence_length}")
            
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'DirectionalNetConfig':
        """Create config from dictionary"""
        return cls(**config)


# Market Microstructure Components
class MarketMicrostructure(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Limit Order Book Analysis
        self.lob_analysis = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim//2, 3, padding=1, groups=8),
            nn.BatchNorm1d(hidden_dim//2),
            nn.GELU(),
            nn.Conv1d(hidden_dim//2, hidden_dim, 3, padding=1, groups=8)
        )

        # Volume Profile
        self.volume_profile = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Process limit order book
        x_conv = x.transpose(1, 2)
        lob_features = self.lob_analysis(x_conv).transpose(1, 2)
        
        # Process volume profile
        vol_features = self.volume_profile(x)
        
        return {
            'lob_features': lob_features,
            'volume_features': vol_features
        }

# Time Series Analysis Components 
class TimeSeriesAnalysis(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Trend Analysis
        self.trend_net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 15, padding=7, groups=16),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )

        # Seasonality Detection
        self.season_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, hidden_dim)
        )

        # Stationarity Analysis
        self.stationarity_net = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, 3)  # stationary/non-stationary/quasi-stationary
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Trend extraction
        x_conv = x.transpose(1, 2)
        trend = self.trend_net(x_conv).transpose(1, 2)
        
        # Seasonality
        seasonal = self.season_net(x)
        
        # Stationarity
        combined = torch.cat([trend, seasonal], dim=-1)
        stationarity = self.stationarity_net(combined)
        
        return {
            'trend': trend,
            'seasonality': seasonal,
            'stationarity': stationarity
        }

# Risk Management Components
class RiskManager(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        # Risk Assessment
        self.risk_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )

        # Kelly Criterion
        self.kelly_calculator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )

        # Dynamic Stop Levels
        self.stop_calculator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, 2)  # [stop_loss, take_profit]
        )

    def forward(self, x: torch.Tensor, volatility: torch.Tensor) -> Dict[str, torch.Tensor]:
        risk_score = self.risk_analyzer(x)
        kelly_fraction = self.kelly_calculator(x)
        
        # Scale stops by volatility
        raw_stops = self.stop_calculator(x)
        stops = raw_stops * (1 + volatility)

        return {
            'risk_score': risk_score,
            'kelly_fraction': kelly_fraction,
            'stop_loss': stops[..., 0:1],
            'take_profit': stops[..., 1:2]
        }

# Market State Analysis
class MarketStateAnalyzer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()

        # Regime Detection
        self.regime_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, 5)  # 5 market regimes
        )

        # Volatility Forecasting
        self.volatility_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Softplus()
        )

        # Liquidity Analysis
        self.liquidity_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.LayerNorm(hidden_dim//2),
            nn.GELU(),
            nn.Linear(hidden_dim//2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        regime_logits = self.regime_detector(x)
        regime_probs = F.softmax(regime_logits, dim=-1)
        
        volatility = self.volatility_predictor(x)
        liquidity = self.liquidity_analyzer(x)

        return {
            'regime_logits': regime_logits,
            'regime_probs': regime_probs,
            'volatility': volatility,
            'liquidity': liquidity
        }




class DirectionalPredictor(nn.Module):
    def __init__(self, input_dim: int = 3072, hidden_dim: int = 3072, 
                 sequence_length: int = 300, num_heads: int = 32):
       super().__init__()
        
        # Основные параметры
       self.input_dim = input_dim
       self.hidden_dim = hidden_dim
       self.sequence_length = sequence_length
       self.num_heads = num_heads
       self.debug = False
       self.optimizer = None
       
        # Определяем устройство
       self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Обновленная инициализация GradScaler
       if torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda')
            # Оптимизации для RTX 4090
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')
       else:
            self.scaler = None
            self.autocast = nullcontext
       
        # Base dimensions optimized for RTX 4090
       self.num_classes = 2
       self.dropout_rate = 0.1

    
        # Add uncertainty estimation
       self.uncertainty_net = nn.Sequential(
           nn.Linear(self.hidden_dim, self.hidden_dim//4),
           nn.LayerNorm(self.hidden_dim//4),
           nn.GELU(),
           nn.Linear(self.hidden_dim//4, 1),
           nn.Sigmoid()
        )
    
        # Integrated Components
       self.market_microstructure = MarketMicrostructure(self.hidden_dim)
       self.time_series = TimeSeriesAnalysis(self.hidden_dim)
       self.risk_manager = RiskManager(self.hidden_dim)
       self.market_analyzer = MarketStateAnalyzer(self.hidden_dim)
    
        # Adaptive Weighting
       self.feature_weights = nn.Sequential(
            nn.Linear(self.hidden_dim*4, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 4),
            nn.Softmax(dim=-1)
        )
       
       # Market microstructure analysis
       self.lob_analyzer = nn.Sequential(
           nn.Conv1d(self.hidden_dim, self.hidden_dim//2, kernel_size=3, padding=1, groups=8),
           nn.BatchNorm1d(self.hidden_dim//2),
           nn.GELU(),
           nn.Conv1d(self.hidden_dim//2, self.hidden_dim, kernel_size=3, padding=1, groups=8)
       )
       
       # Volume profile analysis
       self.volume_attn = nn.MultiheadAttention(
           embed_dim=self.hidden_dim,
           num_heads=self.num_heads,
           dropout=0.1,
           batch_first=True
       )
       
       # Order flow imbalance 
       self.flow_analyzer = nn.Sequential(
           nn.Conv1d(self.hidden_dim, self.hidden_dim//2, kernel_size=3, padding=1),
           nn.BatchNorm1d(self.hidden_dim//2),
           nn.GELU(),
           nn.Conv1d(self.hidden_dim//2, self.hidden_dim, kernel_size=3, padding=1)
       )

       # Time series decomposition
       self.trend_extractor = nn.Sequential(
           nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=15, padding=7, groups=16),
           nn.AvgPool1d(kernel_size=5, stride=1, padding=2)
       )

       # Seasonality analysis
       self.seasonality_net = nn.Sequential(
           nn.Linear(self.hidden_dim, self.hidden_dim//2),
           nn.LayerNorm(self.hidden_dim//2),
           nn.GELU(),
           nn.Linear(self.hidden_dim//2, self.hidden_dim),
           nn.Sigmoid()
       )

       # Regime detection
       self.regime_detector = nn.Sequential(
           nn.Linear(self.hidden_dim, self.hidden_dim//2),
           nn.LayerNorm(self.hidden_dim//2),
           nn.GELU(),
           nn.Linear(self.hidden_dim//2, 5)  # 5 market regimes
       )
       
       # Volatility forecasting with high precision
       self.volatility_predictor = nn.Sequential(
           nn.Linear(self.hidden_dim, self.hidden_dim//2),
           nn.LayerNorm(self.hidden_dim//2),
           nn.GELU(),
           nn.Linear(self.hidden_dim//2, 1),
           nn.Softplus()
       )

       # Liquidity estimation
       self.liquidity_estimator = nn.Sequential(
           nn.Linear(self.hidden_dim, self.hidden_dim//2),
           nn.LayerNorm(self.hidden_dim//2),
           nn.GELU(),
           nn.Linear(self.hidden_dim//2, 1),
           nn.Sigmoid()
       )
       
       # Base feature network with optimized capacity
       self.feature_net = nn.Sequential(
           nn.Linear(self.input_dim, self.hidden_dim),
           nn.LayerNorm(self.hidden_dim),
           nn.GELU(),
           nn.Dropout(self.dropout_rate)
       )
       
       # Enhanced LSTM with increased capacity
       self.lstm = nn.LSTM(
           input_size=self.hidden_dim,
           hidden_size=self.hidden_dim//2,
           num_layers=3,
           dropout=self.dropout_rate,
           bidirectional=True,
           batch_first=True
       )

       # Adaptive learning rate by data quality
       self.data_quality = nn.Sequential(
           nn.Linear(self.hidden_dim, self.hidden_dim//2),
           nn.LayerNorm(self.hidden_dim//2),
           nn.GELU(),
           nn.Linear(self.hidden_dim//2, 1),
           nn.Sigmoid()
       )

       # Adaptive component weighting
       self.component_weights = nn.Sequential(
           nn.Linear(self.hidden_dim, self.hidden_dim//2),
           nn.LayerNorm(self.hidden_dim//2),
           nn.GELU(),
           nn.Linear(self.hidden_dim//2, 5),
           nn.Softmax(dim=-1)
       )

       # Risk management
       self.risk_appetite = nn.Sequential(
           nn.Linear(self.hidden_dim, self.hidden_dim//2),
           nn.LayerNorm(self.hidden_dim//2),
           nn.GELU(),
           nn.Linear(self.hidden_dim//2, 1),
           nn.Sigmoid()
       )

       # Kelly criterion estimation
       self.kelly_estimator = nn.Sequential(
           nn.Linear(self.hidden_dim, self.hidden_dim//2),
           nn.LayerNorm(self.hidden_dim//2),
           nn.GELU(),
           nn.Linear(self.hidden_dim//2, 1),
           nn.Sigmoid()
       )

       # Dynamic stop-loss levels
       self.stop_calculator = nn.Sequential(
           nn.Linear(self.hidden_dim, self.hidden_dim//2),
           nn.LayerNorm(self.hidden_dim//2),
           nn.GELU(),
           nn.Linear(self.hidden_dim//2, 2)
       )
       
       # Uncertainty estimation
       self.aleatoric_uncertainty = nn.Sequential(
           nn.Linear(self.hidden_dim, self.hidden_dim//4),
           nn.LayerNorm(self.hidden_dim//4),
           nn.GELU(),
           nn.Linear(self.hidden_dim//4, 1),
           nn.Softplus()
       )
       
       self.epistemic_uncertainty = nn.Sequential(
           nn.Linear(self.hidden_dim, self.hidden_dim//4),
           nn.LayerNorm(self.hidden_dim//4),
           nn.GELU(),
           nn.Linear(self.hidden_dim//4, 1),
           nn.Sigmoid()
       )
       

        # Enhanced classifier с правильными размерностями
       self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2),  # 3072 -> 1536
            nn.LayerNorm(self.hidden_dim//2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim//2, self.num_classes)  # 1536 -> 2
        )

        # Position sizing с правильными размерностями
       self.position_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2),  # 3072 -> 1536
            nn.LayerNorm(self.hidden_dim//2),
            nn.GELU(),
            nn.Linear(self.hidden_dim//2, 1),  # 1536 -> 1
            nn.Sigmoid()
        )

       # Final signal combination
       self.signal_combiner = nn.Sequential(
           nn.Linear(self.hidden_dim * 5, self.hidden_dim),
           nn.LayerNorm(self.hidden_dim),
           nn.GELU(),
           nn.Dropout(self.dropout_rate),
           nn.Linear(self.hidden_dim, self.num_classes)
       )

       # Trading constants
       self.TICK_SIZE = 0.00001
       self.PIP_SIZE = 0.0001  
       self.PIP_VALUE = 10.0
       self.LOT_SIZE = 100000
       self.SPREAD_POINTS = 25  # 2.5 pips
       self.COMMISSION_RATE = 0.0001

       # Risk parameters
       self.MIN_POSITION = 0.01
       self.MAX_POSITION = 0.05
       self.RISK_PER_TRADE = 0.02
       self.BASE_STOP_LOSS = 20  # pips
       self.RR_RATIO = 1.5

       # Initialize metrics tracking
       self.train_metrics = defaultdict(list)
       self.val_metrics = defaultdict(list)
       

       
       # Move model to device
       self.to(self.device)
       self._init_weights()

        # Enable gradient checkpointing
       self.use_checkpointing = True
        
        # Initialize memory tracking
       self.peak_memory = 0
       self.last_memory = 0

        # Add state management
       self.checkpoint_states = {}
       self.use_checkpointing = True

       logger.info(
           f"Initialized DirectionalPredictor with market microstructure:\n"
           f"- Device: {self.device}\n"
           f"- Input dim: {self.input_dim}\n"
           f"- Hidden dim: {self.hidden_dim}\n"
           f"- Sequence length: {self.sequence_length}\n"
           f"- Number of attention heads: {self.num_heads}"
       )

    def _log_initialization(self) -> None:
        """Log initialization details without changing model"""
        try:
            logger.debug(
                f"DirectionalPredictor initialization:\n"
                f"Device: {next(self.parameters()).device}\n" 
                f"Input dim: {self.input_dim}\n"
                f"Hidden dim: {self.hidden_dim}\n"
                f"Feature groups: {len(self.feature_groups)}"
            )
            
            if torch.cuda.is_available():
                logger.debug(
                    f"GPU Memory:\n"
                    f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB\n"
                    f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB"
                )
                
        except Exception as e:
            logger.error(f"Error logging initialization: {str(e)}")






    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Training step with memory optimization"""
        try:
            self.train()  # Устанавливаем режим обучения
            
            if self.optimizer is None:
                raise ValueError("Optimizer not set")

            # Move data to device efficiently
            features = batch['features'].to(self.device, non_blocking=True)
            targets = batch.get('targets')
            if targets is not None:
                targets = targets.to(self.device, non_blocking=True)

            # Clear gradients efficiently
            self.optimizer.zero_grad(set_to_none=True)  # set_to_none=True для лучшей очистки памяти

            # Forward pass with mixed precision
            with torch.amp.autocast(device_type='cuda'):
                outputs = self(features)
                
                if targets is not None:
                    loss = F.cross_entropy(outputs['logits'], targets)
                else:
                    loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            # Calculate metrics before cleanup
            with torch.no_grad():
                predictions = outputs['logits'].argmax(dim=-1)
                metrics = {
                    'loss': loss.item(),
                    'uncertainty': outputs['uncertainty'].mean().item(),
                    'position_size': outputs['position_size'].mean().item(),
                    'volatility': outputs['volatility'].mean().item(),
                    'market_volatility': outputs['volatility'].mean().item(),
                    'market_regime': outputs['market_regime'].argmax(dim=-1).float().mean().item(),
                    'risk_score': outputs['risk_score'].mean().item()
                }

                if targets is not None:
                    tp = ((predictions == 1) & (targets == 1)).float().sum()
                    fp = ((predictions == 1) & (targets == 0)).float().sum()
                    tn = ((predictions == 0) & (targets == 0)).float().sum()
                    fn = ((predictions == 0) & (targets == 1)).float().sum()
                    
                    total = tp + tn + fp + fn + 1e-8
                    total_trades = tp + fp + fn
                    
                    metrics.update({
                        'accuracy': (predictions == targets).float().mean().item(),
                        'true_positives': tp.item(),
                        'false_positives': fp.item(),
                        'true_negatives': tn.item(),
                        'false_negatives': fn.item(),
                        'precision': (tp / (tp + fp + 1e-8)).item(),
                        'recall': (tp / (tp + fn + 1e-8)).item(),
                        'f1_score': (2 * tp / (2 * tp + fp + fn + 1e-8)).item(),
                        'mean_confidence': outputs['logits'].softmax(dim=-1).max(dim=-1)[0].mean().item()
                    })

            # ВАЖНО: Очищаем неиспользуемые тензоры
            del features
            del outputs['features']  # Удаляем большие тензоры
            del outputs['market_regime']
            
            # Return minimal required outputs with detached tensors
            return {
                'loss': loss,
                'metrics': metrics,
                'outputs': {
                    'logits': outputs['logits'].detach(),
                    'predictions': predictions,
                    'uncertainty': outputs['uncertainty'].detach(),
                    'position_size': outputs['position_size'].detach()
                }
            }

        except Exception as e:
            logger.error(f"Error in directional train_step: {str(e)}")
            if self.debug:
                logger.error(traceback.format_exc())
            # Ensure memory cleanup on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise



    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with proper dimensions
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Dict with model outputs
        """
        try:
            batch_size, seq_len, feat_dim = x.shape
            
            if feat_dim != self.input_dim:
                raise ValueError(
                    f"Expected input dimension {self.input_dim}, got {feat_dim}"
                )

            # Memory management
            if torch.cuda.is_available():
                if torch.cuda.memory_allocated() > 0.8 * torch.cuda.get_device_properties(0).total_memory:
                    torch.cuda.empty_cache()

            # Process features
            features = self.feature_net(x)
            
            # LSTM processing
            lstm_out, (hidden, cell) = self.lstm(features)
            final_state = lstm_out[:, -1]

            # Get predictions
            logits = self.classifier(final_state)
            
            # Market analysis
            market_state = self.market_analyzer(final_state)
            
            # Risk assessment
            risk_outputs = self.risk_manager(final_state, market_state['volatility'])

            return {
                'logits': logits,
                'uncertainty': self.uncertainty_net(final_state),
                'position_size': self.position_net(final_state),
                'features': final_state,
                'volatility': market_state['volatility'],
                'market_regime': market_state['regime_probs'],
                'risk_score': risk_outputs['risk_score']
            }

        except Exception as e:
            logger.error(f"Forward pass error: {str(e)}")
            raise




    def _validate_dimensions(self, x: torch.Tensor) -> None:
        """Validate tensor dimensions
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Raises:
            ValueError: If dimensions don't match
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {x.dim()}D")
            
        batch_size, seq_len, feat_dim = x.shape
        
        if feat_dim != self.input_dim:
            raise ValueError(
                f"Wrong feature dimension: got {feat_dim}, "
                f"expected {self.input_dim}"
            )
            
        if seq_len != self.sequence_length:
            raise ValueError(
                f"Wrong sequence length: got {seq_len}, "
                f"expected {self.sequence_length}"
            )




    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Validation step with proper metrics calculation"""
        try:
            self.eval()
            with torch.no_grad():
                # Get predictions
                features = batch['features'].to(self.device)
                targets = batch.get('targets')
                if targets is not None:
                    targets = targets.to(self.device)

                # Forward pass with proper autocast
                with torch.amp.autocast('cuda') if torch.cuda.is_available() else nullcontext():
                    outputs = self(features)
                    
                    # Calculate loss
                    if targets is not None:
                        loss = F.cross_entropy(outputs['logits'], targets)
                    else:
                        loss = torch.tensor(0.0, device=self.device)
                        
                    # Add loss to outputs
                    outputs['loss'] = loss

                # Calculate metrics
                metrics = self._calculate_metrics(outputs=outputs, batch=batch)
                
                # Ensure loss is in metrics
                metrics['loss'] = loss.item()

                return {
                    'metrics': metrics,
                    'outputs': {k: v.detach() if isinstance(v, torch.Tensor) else v
                               for k, v in outputs.items()}
                }

        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            if self.debug:
                logger.error(traceback.format_exc())
            return {
                'metrics': {'loss': float('inf')},
                'outputs': {}
            }
            
    def _calculate_metrics(self, outputs: Dict[str, torch.Tensor], 
                         batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate metrics with proper error handling
        
        Args:
            outputs: Model outputs containing:
                - logits: [batch_size, num_classes]
                - uncertainty: [batch_size, 1]
                - position_size: [batch_size, 1]
                - features: [batch_size, hidden_dim]
                - loss: scalar tensor
            batch: Input batch containing:
                - features: [batch_size, seq_len, input_dim]
                - targets: [batch_size] (optional)
                
        Returns:
            Dict[str, float]: Calculated metrics
        """
        try:
            metrics = {}
            
            # Get base predictions
            logits = outputs['logits']  # [batch_size, num_classes]
            predictions = logits.argmax(dim=-1)  # [batch_size]
            probs = F.softmax(logits, dim=-1)  # [batch_size, num_classes]
            
            # Base metrics
            metrics['loss'] = outputs.get('loss', 0.0).item()
            
            # Process targets if they exist
            if 'targets' in batch:
                targets = batch['targets']
                
                tp = ((predictions == 1) & (targets == 1)).float().sum()
                fp = ((predictions == 1) & (targets == 0)).float().sum()
                tn = ((predictions == 0) & (targets == 0)).float().sum()
                fn = ((predictions == 0) & (targets == 1)).float().sum()
                
                total = tp + tn + fp + fn + 1e-8
                
                metrics.update({
                    'accuracy': (predictions == targets).float().mean().item(),
                    'precision': (tp / (tp + fp + 1e-8)).item(),
                    'recall': (tp / (tp + fn + 1e-8)).item(),
                    'f1_score': (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
                })
                
            # Add model-specific metrics
            if 'uncertainty' in outputs:
                metrics['uncertainty'] = outputs['uncertainty'].mean().item()
                
            if 'position_size' in outputs:
                metrics['position_size'] = outputs['position_size'].mean().item()
                
            if 'volatility' in outputs:
                metrics['volatility'] = outputs['volatility'].mean().item()
                metrics['market_volatility'] = outputs['volatility'].mean().item()
                
            if 'market_regime' in outputs:
                metrics['market_regime'] = outputs['market_regime'].argmax(dim=-1).float().mean().item()
                
            if 'risk_score' in outputs:
                metrics['risk_score'] = outputs['risk_score'].mean().item()
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {
                'loss': float('inf'),
                'accuracy': 0.0,
                'f1_score': 0.0
            }






    def _get_class_weights(self, labels: torch.Tensor) -> torch.Tensor:
       """Calculate class weights to handle imbalance"""
       counts = torch.bincount(labels.long(), minlength=2)
       total = counts.sum()
       weights = total.float() / (counts.float() + 1e-8) 
       weights = weights / weights.sum()
       return weights


    




    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Set optimizer for model"""
        self.optimizer = optimizer





    def _validate_outputs(self, outputs: Dict[str, torch.Tensor], batch_size: int) -> None:
        """Validate output dimensions"""
        expected_shapes = {
            'logits': (batch_size, 2),
            'uncertainty': (batch_size, 1),
            'position_size': (batch_size, 1),
            'features': (batch_size, self.hidden_dim)
        }

        for name, shape in expected_shapes.items():
            if name not in outputs:
                raise ValueError(f"Missing {name} in outputs")
            
            tensor = outputs[name]
            if tensor.shape != shape:
                raise ValueError(
                    f"Wrong shape for {name}: got {tensor.shape}, "
                    f"expected {shape}"
                )


    def _save_rng_state(self) -> None:
        """Save RNG states"""
        self.checkpoint_states['rng'] = {
            'cpu': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        }

    def _restore_rng_state(self) -> None:
        """Restore RNG states"""
        if 'rng' in self.checkpoint_states:
            torch.set_rng_state(self.checkpoint_states['rng']['cpu'])
            if torch.cuda.is_available() and self.checkpoint_states['rng']['cuda'] is not None:
                torch.cuda.set_rng_state(self.checkpoint_states['rng']['cuda'])

    def _run_features(self, x: torch.Tensor) -> torch.Tensor:
        """Feature processing with state preservation"""
        self._save_rng_state()
        features = self.feature_net(x)
        self._restore_rng_state()
        return features

    def _run_lstm(self, features: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """LSTM processing with state preservation"""
        self._save_rng_state()
        lstm_out, (hidden, cell) = self.lstm(features)
        self._restore_rng_state()
        return lstm_out, (hidden, cell)


    def _validate_output_dimensions(self, outputs: Dict[str, torch.Tensor], batch_size: int) -> None:
        expected_shapes = {
            'logits': (batch_size, self.num_classes),
            'uncertainty': (batch_size, 1),
            'position_size': (batch_size, 1),
            'features': (batch_size, self.hidden_dim)
        }

        for name, shape in expected_shapes.items():
            if outputs[name].shape != shape:
                raise ValueError(
                    f"Wrong shape for {name}: got {outputs[name].shape}, "
                    f"expected {shape}"
                )



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

    def get_metrics(self) -> Dict[str, List[float]]:
        """Get training metrics"""
        return {
            'train': dict(self.train_metrics),
            'val': dict(self.val_metrics)
        }



    def epoch_end(self, epoch: int) -> Dict[str, float]:
        """Calculate and report epoch metrics
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dict with epoch metrics
        """
        try:
            # Get epoch metrics from recorder
            metrics = self.trade_recorder.get_metrics('directional') if self.trade_recorder else {}
            
            # Log metrics
            logger.info(f"\nEpoch {epoch} Metrics:")
            logger.info(f"Total Trades: {metrics.get('total_trades', 0)}")
            logger.info(f"Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
            logger.info(f"Total P&L: ${metrics.get('total_pnl', 0):.2f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating epoch metrics: {str(e)}")
            return {}


    def _process_features(self, x: torch.Tensor) -> torch.Tensor:
        """Process features with checkpointing"""
        def _run_features(x: torch.Tensor) -> torch.Tensor:
            return self.feature_net(x)
            
        if self.use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _run_features,
                x,
                use_reentrant=False,
                preserve_rng_state=True
            )
        return _run_features(x)

    def _process_lstm(self, features: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """LSTM processing with checkpointing"""
        def _run_lstm(features: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            return self.lstm(features)
            
        if self.use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _run_lstm,
                features,
                use_reentrant=False,
                preserve_rng_state=True
            )
        return _run_lstm(features)




    def _validate_trade_params(self, trade_data: Dict[str, Any]) -> bool:
        """Validate trade parameters
        
        Args:
            trade_data: Trade parameters dictionary
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Required fields
            required = {'direction', 'entry_price', 'position_size', 'confidence', 'uncertainty'}
            if not all(k in trade_data for k in required):
                return False
                
            # Value ranges
            if not 0 <= trade_data['confidence'] <= 1:
                return False
                
            if not 0 <= trade_data['uncertainty'] <= 1:
                return False
                
            if trade_data['position_size'] <= 0:
                return False
                
            if trade_data['entry_price'] <= 0:
                return False
                
            if trade_data['direction'] not in [0, 1]:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating trade params: {str(e)}")
            return False


    def record_trade(self, trade_data: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """Record trade with calculated metrics"""
        try:
            # Format trade record
            trade_record = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_name': 'directional',
                'direction': int(trade_data['direction']),
                'entry_price': round(float(trade_data['entry_price']), 5),
                'exit_price': 0.0,
                'position_size': round(float(metrics['position_size']), 2),
                'stop_loss': round(float(metrics['stop_loss']), 5),
                'take_profit': round(float(metrics['take_profit']), 5),
                'spread_cost': round(float(metrics['spread_cost']), 2),
                'commission': round(float(metrics['commission']), 2),
                'net_pnl': -round(float(metrics['total_cost']), 2),
                'confidence': round(float(trade_data['confidence']), 4),
                'uncertainty': round(float(trade_data['uncertainty']), 4),
                'volatility': round(float(trade_data['volatility']), 4),
                'trade_duration': 0,
                'status': 'open'
            }
            
            # Save trade
            self._save_trade(trade_record)
            
            logger.info(
                f"Trade recorded:\n"
                f"Direction: {'Long' if trade_record['direction']==1 else 'Short'}\n"
                f"Size: {trade_record['position_size']:.2f}\n"
                f"Entry: {trade_record['entry_price']:.5f}\n"
                f"SL: {trade_record['stop_loss']:.5f}\n"
                f"TP: {trade_record['take_profit']:.5f}\n"
                f"Cost: ${-trade_record['net_pnl']:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error recording trade: {str(e)}")
            raise

    def calculate_volatility(self, price_sequence: torch.Tensor) -> float:
        """Calculate price volatility
        
        Args:
            price_sequence: Price tensor [sequence_length]
            
        Returns:
            Volatility value
        """
        try:
            # Calculate returns
            returns = (price_sequence[1:] / price_sequence[:-1] - 1)
            volatility = float(returns.std() * np.sqrt(252))
            return max(min(volatility, 1.0), 0.01)
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return 0.02




    def _flush_buffer(self) -> None:
        """Flush trades buffer to CSV file"""
        try:
            if not self.trades_buffer:
                return

            # Write trades to CSV
            with open(self.trades_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.trades_headers)
                writer.writerows(self.trades_buffer)

            # Clear buffer
            n_trades = len(self.trades_buffer)
            self.trades_buffer.clear()
            logger.debug(f"Flushed {n_trades} trades to {self.trades_file}")

        except Exception as e:
            logger.error(f"Error flushing trades buffer: {str(e)}")
            raise

    def cleanup(self) -> None:
        """Cleanup before exit"""
        try:
            # Flush remaining trades
            if self.trades_buffer:
                self._flush_buffer()
            logger.info(f"Trading session completed. Trades saved to {self.trades_file}")
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}")




    def save_model(self, path: str) -> None:
        """Save model checkpoint
        
        Args:
            path: Path to save checkpoint
        """
        try:
            checkpoint = {
                'state_dict': self.state_dict(),
                'train_metrics': dict(self.train_metrics),
                'val_metrics': dict(self.val_metrics),
                'trade_history': self.trade_history,
                'batch_count': self._batch_count,
                'timestamp': datetime.now().isoformat()
            }
            
            torch.save(checkpoint, path)
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path: str) -> None:
        """Load model checkpoint
        
        Args:
            path: Path to checkpoint
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.load_state_dict(checkpoint['state_dict'])
            self.train_metrics = defaultdict(list, checkpoint.get('train_metrics', {}))
            self.val_metrics = defaultdict(list, checkpoint.get('val_metrics', {}))
            self.trade_history = checkpoint.get('trade_history', [])
            self._batch_count = checkpoint.get('batch_count', 0)
            
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    



    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save model checkpoint
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'state_dict': self.state_dict(),
            'train_metrics': dict(self.train_metrics),
            'val_metrics': dict(self.val_metrics),
            'metadata': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'num_classes': self.num_classes,
                'sequence_length': self.sequence_length
            },
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            torch.save(checkpoint, path)
            logger.info(f"Saved checkpoint to {path}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            raise
    
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load model checkpoint
        
        Args:
            path: Path to checkpoint
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint['state_dict'])
            
            # Restore metrics
            self.train_metrics = defaultdict(list, checkpoint['train_metrics'])
            self.val_metrics = defaultdict(list, checkpoint['val_metrics'])
            
            # Validate metadata
            metadata = checkpoint['metadata']
            assert metadata['input_dim'] == self.input_dim
            assert metadata['hidden_dim'] == self.hidden_dim
            assert metadata['num_classes'] == self.num_classes
            assert metadata['sequence_length'] == self.sequence_length
            
            logger.info(f"Loaded checkpoint from {path}")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            raise
    


             