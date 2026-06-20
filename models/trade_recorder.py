# C_01_en/models/trade_recorder.py
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Any, Tuple
import torch
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import json
import csv
from concurrent.futures import ThreadPoolExecutor
import threading
import psutil
import sys
from collections import defaultdict
import json
import uuid
import shutil
import traceback
from datetime import datetime, timedelta, date




# Настройка логгера
logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.DEBUG)

# Оптимизация для I7
NUM_THREADS = 16  # Оптимально для i7-14700
torch.set_num_threads(NUM_THREADS)


# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')





class EnhancedTradeRecorder:
    """Enhanced trade recorder for multiple models"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.trades_dir = self.output_dir / 'trades'
        self.trades_dir.mkdir(parents=True, exist_ok=True)
        
        # Headers for CSV
        self.headers = [
            'timestamp', 'model_name', 'trade_id',
            'direction', 'entry_price', 'exit_price',
            'position_size', 'pnl', 'commission',
            'spread_cost', 'net_pnl', 'duration',
            'regime', 'confidence', 'status'
        ]
        
        # Create files for each model
        self.trade_files = {}
        self.trade_buffers = defaultdict(list)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for model_name in ['feature', 'directional', 'regime', 'qnd', 'ensemble']:
            model_dir = self.trades_dir / model_name
            model_dir.mkdir(exist_ok=True)
            
            trade_file = model_dir / f'trades_{timestamp}.csv'
            self.trade_files[model_name] = trade_file
            
            # Initialize CSV
            with open(trade_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()
                
        # Metrics tracking
        self.metrics = defaultdict(lambda: defaultdict(list))
        
    def add_trade(self, model_name: str, trade_data: Dict[str, Any]) -> None:
        """Add trade with validation"""
        try:
            # Validate trade data
            required_fields = {
                'timestamp', 'direction', 'entry_price',
                'position_size', 'status'
            }
            
            missing = required_fields - set(trade_data.keys())
            if missing:
                raise ValueError(f"Missing fields: {missing}")
                
            # Format trade
            formatted_trade = {
                'timestamp': str(trade_data['timestamp']),
                'model_name': model_name,
                'trade_id': str(uuid.uuid4()),
                'direction': str(trade_data['direction']),
                'entry_price': f"{float(trade_data['entry_price']):.5f}",
                'exit_price': f"{float(trade_data.get('exit_price', 0.0)):.5f}",
                'position_size': f"{float(trade_data['position_size']):.4f}",
                'pnl': f"{float(trade_data.get('pnl', 0.0)):.2f}",
                'commission': f"{float(trade_data.get('commission', 0.0)):.2f}",
                'spread_cost': f"{float(trade_data.get('spread_cost', 0.0)):.2f}",
                'net_pnl': f"{float(trade_data.get('net_pnl', 0.0)):.2f}",
                'duration': str(trade_data.get('duration', 0)),
                'regime': str(trade_data.get('regime', 'unknown')),
                'confidence': f"{float(trade_data.get('confidence', 0.0)):.4f}",
                'status': str(trade_data['status'])
            }
            
            # Add to buffer
            self.trade_buffers[model_name].append(formatted_trade)
            
            # Update metrics
            if trade_data['status'] == 'closed':
                self.metrics[model_name]['total_trades'].append(1)
                self.metrics[model_name]['pnl'].append(float(formatted_trade['net_pnl']))
                
            # Write buffer if full
            if len(self.trade_buffers[model_name]) >= 100:
                self._flush_buffer(model_name)
                
        except Exception as e:
            logger.error(f"Error adding trade: {str(e)}")
            raise

    def _flush_buffer(self, model_name: str) -> None:
        """Flush trade buffer to file"""
        if not self.trade_buffers[model_name]:
            return
            
        with open(self.trade_files[model_name], 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.headers)
            writer.writerows(self.trade_buffers[model_name])
            
        self.trade_buffers[model_name].clear()

    def get_metrics(self, model_name: str) -> Dict[str, float]:
        """Get current metrics for model"""
        metrics = {}
        
        if model_name in self.metrics:
            model_metrics = self.metrics[model_name]
            
            metrics = {
                'total_trades': sum(model_metrics['total_trades']),
                'winning_trades': len([p for p in model_metrics['pnl'] if p > 0]),
                'total_pnl': sum(model_metrics['pnl']),
                'avg_trade': np.mean(model_metrics['pnl']) if model_metrics['pnl'] else 0,
                'win_rate': len([p for p in model_metrics['pnl'] if p > 0]) / len(model_metrics['pnl']) if model_metrics['pnl'] else 0
            }
            
        return metrics

    def save_metrics(self) -> None:
        """Save all metrics to file"""
        metrics_file = self.output_dir / 'trading_metrics.json'
        
        metrics = {
            model_name: self.get_metrics(model_name)
            for model_name in self.trade_files.keys()
        }
        
        metrics['timestamp'] = datetime.now().isoformat()
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)


@dataclass
class Trade:
    """Trade data structure"""
    timestamp: datetime
    direction: int  # 0: short, 1: long
    entry_price: float
    position_size: float
    probability: float
    uncertainty: float
    status: str
    exit_price: Optional[float] = None
    pnl: Optional[float] = None




@dataclass
class TradeStats:
    """Статистика торговли по модели"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    balance: float = 10000.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    active_trades: List[Dict] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list)
    metrics: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))





class TradeRecorder:
    """Централизованный менеджер торговых операций"""

    def __init__(self, output_dir: Union[str, Path]) -> None:
        """Initialize trade recorder
        
        Args:
            output_dir: Output directory path
            
        Attributes:
            input_dim: Input features dimension (133)
            sequence_length: Sequence length for models (60)
            PIP_SIZE: Size of one pip (0.0001)
            PIP_VALUE: Value of one pip in USD (10.0)
            LOT_SIZE: Standard lot size (100000)
            SPREAD_POINTS: Spread in points (25) 
            COMMISSION_RATE: Commission rate (0.0001)
            SLIPPAGE_POINTS: Slippage in points (5)
            TICK_SIZE: Tick size (0.00001)
            LEVERAGE: Account leverage (100)
            MARGIN_RATE: Margin rate (1/LEVERAGE)
            
            trades: Dict[str, List[Dict]]: Trade history by model
            trade_buffers: Dict[str, List[Dict]]: Trade buffers by model
            trade_files: Dict[str, Path]: Trade CSV files by model
            metrics: Dict[str, Dict]: Metrics by model
            model_stats: Dict[str, Dict]: Trade statistics by model
            
            file_handles: Dict[str, TextIO]: Open file handles
            csv_writers: Dict[str, csv.DictWriter]: CSV writers
            
            current_epoch: Current training epoch
            current_batch: Current batch number
            session_id: Unique session identifier
            session_start: Session start time
            buffer_size: Buffer size for trades
            debug: Debug mode flag
        """
        try:
            # Base directories
            self.output_dir = Path(output_dir)
            self.trades_dir = self.output_dir / 'trades'
            self.metrics_dir = self.output_dir / 'metrics'
            self.logs_dir = self.output_dir / 'logs'

            for directory in [self.trades_dir, self.metrics_dir, self.logs_dir]:
                directory.mkdir(parents=True, exist_ok=True)

            # Model dimensions
            self.input_dim = 133
            self.sequence_length = 60

            # Trading constants  
            self.PIP_SIZE = 0.0001       # 1 pip
            self.PIP_VALUE = 10.0        # $10 per pip
            self.LOT_SIZE = 100000       # Standard lot
            self.SPREAD_POINTS = 25      # 2.5 pips spread
            self.COMMISSION_RATE = 0.0001 # 0.01% commission
            self.SLIPPAGE_POINTS = 5     # 0.5 pips slippage
            self.TICK_SIZE = 0.00001     # 0.1 pip
            self.LEVERAGE = 100          # 100:1 leverage
            self.MARGIN_RATE = 1/self.LEVERAGE

            # CSV Headers
            self.trades_headers = [
                'timestamp',      # Trade timestamp 
                'model_name',     # Name of model
                'epoch',          # Training epoch
                'batch',          # Batch number
                'direction',      # 1: long, 0: short
                'entry_price',    # Entry price
                'exit_price',     # Exit price 
                'position_size',  # Position size in lots
                'pips',          # Profit/loss in pips
                'profit_loss',   # Raw P&L
                'spread_cost',   # Spread cost
                'commission',    # Commission cost
                'net_pnl',      # Net P&L after costs
                'confidence',    # Model confidence
                'uncertainty',   # Model uncertainty
                'volatility',    # Market volatility
                'trade_duration',# Duration in minutes
                'status'         # 'open' or 'closed'
            ]

            # Initialize containers
            self.model_names = ['feature', 'directional', 'regime', 'qnd']
            self.trades = defaultdict(list)  # Trade history by model
            self.trade_buffers = defaultdict(list)  # Trade buffers
            self.trade_files = {}  # CSV files
            self.metrics = defaultdict(lambda: {
                'total_trades': 0,
                'winning_trades': 0,
                'total_pnl': 0.0,
                'total_pips': 0.0,
                'gross_profit': 0.0,
                'gross_loss': 0.0,
                'max_profit': float('-inf'),
                'max_loss': float('inf'),
                'current_drawdown': 0.0,
                'max_drawdown': 0.0
            })
            
            # File handling
            self.file_handles = {}  # Open file handles
            self.csv_writers = {}   # CSV writers

            # Initialize files for each model
            self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.session_start = datetime.now()

            for model_name in self.model_names:
                model_dir = self.trades_dir / model_name
                model_dir.mkdir(exist_ok=True)
                
                trade_file = model_dir / f'trades_{self.session_id}.csv'
                f = open(trade_file, 'w', newline='', encoding='utf-8')
                writer = csv.DictWriter(f, fieldnames=self.trades_headers)
                writer.writeheader()
                
                self.file_handles[model_name] = f
                self.csv_writers[model_name] = writer
                self.trade_files[model_name] = trade_file

            # Session tracking
            self.current_epoch = 0
            self.current_batch = 0
            self.buffer_size = 1000
            self.debug = False

            # Model stats
            self.model_stats = {
                model_name: {
                    'balance': 10000.0,  # Initial balance
                    'equity': 10000.0,   # Current equity
                    'margin': 0.0,       # Used margin
                    'free_margin': 10000.0, # Free margin
                    'margin_level': 100.0,  # Margin level %
                } for model_name in self.model_names
            }

            # Setup logging
            self._setup_logging()

            logger.info(
                f"Initialized TradeRecorder:\n"
                f"- Output dir: {self.output_dir}\n"
                f"- Models: {len(self.model_names)}\n"
                f"- Start time: {self.session_start}"
            )

        except Exception as e:
            logger.error(f"Error initializing TradeRecorder: {str(e)}\n{traceback.format_exc()}")
            self.cleanup()
            raise RuntimeError(f"Failed to initialize TradeRecorder: {str(e)}")

    def record_trade(self, model_name: str, trade_data: Dict[str, Any]) -> None:
        """Record trade with validation and metrics updating
        
        Args:
            model_name: Name of model making trade
            trade_data: Trade information dictionary with:
                - direction: int (1: long, 0: short)  
                - entry_price: float
                - exit_price: float (optional)
                - position_size: float
                - confidence: float
                - uncertainty: float (optional)
                - volatility: float (optional)
                - status: str ('open' or 'closed')
                
        Raises:
            ValueError: If trade data is invalid
            RuntimeError: If writing to file fails
        """
        try:
            # Validate model exists
            if model_name not in self.model_names:
                raise ValueError(f"Unknown model: {model_name}")
    
            # Validate required fields
            required_fields = {
                'direction', 'entry_price', 'position_size', 'status'
            }
            missing = required_fields - set(trade_data.keys())
            if missing:
                raise ValueError(f"Missing required fields: {missing}")
    
            # Validate types
            if not isinstance(trade_data['direction'], (int, np.integer)):
                raise ValueError(
                    f"Direction must be int, got {type(trade_data['direction'])}"
                )
            if trade_data['direction'] not in [0, 1]:
                raise ValueError(f"Invalid direction: {trade_data['direction']}")
    
            # Calculate trade metrics
            position_value = (float(trade_data['position_size']) * 
                             self.LOT_SIZE * 
                             float(trade_data['entry_price']))
            spread_cost = self.SPREAD_POINTS * self.TICK_SIZE * position_value
            commission = position_value * self.COMMISSION_RATE
            net_pnl = -spread_cost - commission
    
            if trade_data['status'] == 'closed' and 'exit_price' in trade_data:
                # Calculate P&L for closed trade
                entry_price = float(trade_data['entry_price'])
                exit_price = float(trade_data['exit_price'])
                direction = int(trade_data['direction'])
                position_size = float(trade_data['position_size'])
    
                pips = ((exit_price - entry_price) * 10000 
                       if direction == 1 else 
                       (entry_price - exit_price) * 10000)
                profit_loss = pips * self.PIP_VALUE * position_size
                net_pnl = profit_loss - spread_cost - commission
    
            # Format trade record
            formatted_trade = {
                'timestamp': datetime.now().isoformat(),
                'model_name': model_name,
                'epoch': self.current_epoch,
                'batch': self.current_batch,
                'direction': str(trade_data['direction']),
                'entry_price': f"{float(trade_data['entry_price']):.5f}",
                'exit_price': f"{float(trade_data.get('exit_price', 0.0)):.5f}",
                'position_size': f"{float(trade_data['position_size']):.4f}",
                'pips': f"{float(trade_data.get('pips', 0.0)):.1f}",
                'profit_loss': f"{float(trade_data.get('profit_loss', 0.0)):.2f}",
                'spread_cost': f"{float(spread_cost):.2f}",
                'commission': f"{float(commission):.2f}",
                'net_pnl': f"{float(net_pnl):.2f}",
                'confidence': f"{float(trade_data.get('confidence', 0.0)):.4f}",
                'uncertainty': f"{float(trade_data.get('uncertainty', 0.0)):.4f}",
                'volatility': f"{float(trade_data.get('volatility', 0.0)):.4f}",
                'trade_duration': str(trade_data.get('trade_duration', 0)),
                'status': str(trade_data['status'])
            }
    
            # Add to buffer
            self.trade_buffers[model_name].append(formatted_trade)
    
            # Update metrics for closed trades
            if trade_data['status'] == 'closed':
                metrics = self.metrics[model_name]
                metrics['total_trades'] += 1
                if net_pnl > 0:
                    metrics['winning_trades'] += 1
                    metrics['gross_profit'] += net_pnl
                    metrics['max_profit'] = max(metrics['max_profit'], net_pnl)
                else:
                    metrics['gross_loss'] += abs(net_pnl)
                    metrics['max_loss'] = min(metrics['max_loss'], net_pnl)
                    
                metrics['total_pnl'] += net_pnl
                metrics['total_pips'] += abs(float(formatted_trade['pips']))
    
                # Update drawdown
                balance = self.model_stats[model_name]['balance']
                equity = balance + net_pnl
                drawdown = (balance - equity) / balance if balance > 0 else 0
                metrics['current_drawdown'] = drawdown
                metrics['max_drawdown'] = max(metrics['max_drawdown'], drawdown)
    
            # Flush buffer if full
            if len(self.trade_buffers[model_name]) >= self.buffer_size:
                self._flush_buffer(model_name)
    
            # Log trade
            if self.debug:
                logger.debug(
                    f"Recorded trade for {model_name}:\n"
                    f"Direction: {'Long' if trade_data['direction']==1 else 'Short'}\n"
                    f"Size: {trade_data['position_size']:.2f}\n"
                    f"Entry: {trade_data['entry_price']:.5f}\n"
                    f"Status: {trade_data['status']}"
                )
    
        except Exception as e:
            logger.error(
                f"Error recording trade for {model_name}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            raise
    
    def _flush_buffer(self, model_name: str) -> None:
        """Flush trade buffer to CSV file
        
        Args:
            model_name: Name of model
            
        Raises:
            RuntimeError: If writing to file fails
        """
        try:
            if not self.trade_buffers[model_name]:
                return
                
            if model_name not in self.csv_writers:
                raise ValueError(f"No CSV writer for model: {model_name}")
                
            writer = self.csv_writers[model_name]
            
            # Write all trades
            for trade in self.trade_buffers[model_name]:
                writer.writerow(trade)
                
            # Clear buffer
            n_trades = len(self.trade_buffers[model_name])
            self.trade_buffers[model_name].clear()
            
            # Flush file
            self.file_handles[model_name].flush()
            
            if self.debug:
                logger.debug(f"Flushed {n_trades} trades for {model_name}")
                
        except Exception as e:
            logger.error(
                f"Error flushing buffer for {model_name}: {str(e)}\n"
                f"{traceback.format_exc()}"
            )
            raise RuntimeError(f"Failed to flush trade buffer: {str(e)}")
    
    def get_metrics(self, model_name: str) -> Dict[str, float]:
        """Get current metrics for model
        
        Args:
            model_name: Name of model
            
        Returns:
            Dict with metrics:
                - total_trades: Total trades
                - winning_trades: Number of winning trades
                - win_rate: Win rate percentage
                - total_pnl: Total profit/loss
                - gross_profit: Gross profit
                - gross_loss: Gross loss
                - total_pips: Total pips
                - max_profit: Maximum profit trade
                - max_loss: Maximum loss trade
                - max_drawdown: Maximum drawdown percentage
        """
        try:
            if model_name not in self.metrics:
                return {}
                
            metrics = self.metrics[model_name]
            total_trades = metrics['total_trades']
            
            if total_trades == 0:
                return {}
                
            return {
                'total_trades': total_trades,
                'winning_trades': metrics['winning_trades'],
                'win_rate': metrics['winning_trades'] / total_trades * 100,
                'total_pnl': metrics['total_pnl'],
                'gross_profit': metrics['gross_profit'],
                'gross_loss': metrics['gross_loss'],
                'total_pips': metrics['total_pips'],
                'avg_trade': metrics['total_pnl'] / total_trades,
                'avg_win': (metrics['gross_profit'] / metrics['winning_trades'] 
                           if metrics['winning_trades'] > 0 else 0),
                'avg_loss': (metrics['gross_loss'] / 
                            (total_trades - metrics['winning_trades'])
                            if total_trades > metrics['winning_trades'] else 0),
                'profit_factor': (metrics['gross_profit'] / metrics['gross_loss']
                                if metrics['gross_loss'] > 0 else float('inf')),
                'max_profit': metrics['max_profit'],
                'max_loss': metrics['max_loss'],
                'max_drawdown': metrics['max_drawdown'] * 100
            }
        
        except Exception as e:
            logger.error(f"Error getting metrics for {model_name}: {str(e)}")
            return {}
    
    def print_metrics(self, model_name: str, epoch: int) -> None:
        """Print metrics for model to screen
        
        Args:
            model_name: Name of model
            epoch: Current epoch number
        """
        try:
            metrics = self.get_metrics(model_name)
            if not metrics:
                return
                
            print(f"\n=== {model_name.upper()} Trading Metrics - Epoch {epoch} ===")
            print(f"Total Trades: {metrics['total_trades']}")
            print(f"Win Rate: {metrics['win_rate']:.1f}%")
            print(f"Total P&L: ${metrics['total_pnl']:.2f}")
            print(f"Gross Profit: ${metrics['gross_profit']:.2f}")
            print(f"Gross Loss: ${metrics['gross_loss']:.2f}")
            print(f"Total Pips: {metrics['total_pips']:.1f}")
            print(f"Average Trade: ${metrics['avg_trade']:.2f}")
            print(f"Average Win: ${metrics['avg_win']:.2f}")
            print(f"Average Loss: ${metrics['avg_loss']:.2f}")
            print(f"Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"Max Profit Trade: ${metrics['max_profit']:.2f}")
            print(f"Max Loss Trade: ${metrics['max_loss']:.2f}")
            print(f"Max Drawdown: {metrics['max_drawdown']:.1f}%")
            
        except Exception as e:
            logger.error(f"Error printing metrics: {str(e)}")
    
    def cleanup(self) -> None:
        """Cleanup resources and save final trades
        
        This method:
        1. Flushes all trade buffers
        2. Closes file handles
        3. Saves final metrics
        """
        try:
            # Flush all buffers
            for model_name in self.trade_buffers:
                self._flush_buffer(model_name)
                
            # Close file handles
            for f in self.file_handles.values():
                f.close()
                
            # Clear all containers
            self.trade_buffers.clear()
            self.file_handles.clear()
            self.csv_writers.clear()
            
            # Save final metrics
            self._save_final_metrics()
            
            logger.info("Trade recorder cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error in cleanup: {str(e)}")
    
    def _save_final_metrics(self) -> None:
        """Save final metrics for all models"""
        try:
            final_metrics = {
                model_name: self.get_metrics(model_name)
                for model_name in self.model_names
            }
            
            metrics_file = (self.metrics_dir / 
                           f'final_metrics_{self.session_id}.json')
            
            with open(metrics_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'session_id': self.session_id,
                    'duration': str(datetime.now() - self.session_start),
                    'metrics': final_metrics
                }, f, indent=2)
                
            logger.info(f"Saved final metrics to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Error saving final metrics: {str(e)}")


    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        try:
            log_file = self.logs_dir / f'trades_{self.session_id}.log'
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            
            logger.info(f"Logging initialized: {log_file}")
            
        except Exception as e:
            logger.error(f"Error setting up logging: {str(e)}")


    def _validate_models(self) -> None:
        """Validate model configuration"""
        if not self.model_names:
            raise ValueError("No models configured")
            
        for model in self.model_names:
            model_dir = self.trades_dir / model
            model_dir.mkdir(exist_ok=True)
            
        logger.info(f"Validated models: {self.model_names}")


    def _validate_trade_data(self, trade_data: Dict[str, Any]) -> None:
        """Validate trade data completeness and types"""
        required_fields = {
            'timestamp', 'direction', 'entry_price', 
            'position_size', 'status'
        }
        
        missing = required_fields - set(trade_data.keys())
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
            
        # Validate types
        validations = {
            'entry_price': (float, int),
            'position_size': (float, int),
            'direction': (int,),
            'status': (str,)
        }
        
        for field, valid_types in validations.items():
            if not isinstance(trade_data[field], valid_types):
                raise TypeError(
                    f"Invalid type for {field}: "
                    f"got {type(trade_data[field])}, "
                    f"expected one of {valid_types}"
                )

    def _save_trade_record(self, model_name: str, trade_data: Dict[str, Any]) -> None:
        """Save trade record with validation
        
        Args:
            model_name: Name of model
            trade_data: Trade information
        """
        try:
            # Validate model exists
            if model_name not in self.trade_files:
                raise ValueError(f"Unknown model: {model_name}")
                
            # Format trade record
            trade_record = {
                'timestamp': str(datetime.now()),
                'model_name': model_name,
                'epoch': self.current_epoch,
                'batch': self.current_batch,
                'direction': int(trade_data['direction']),
                'entry_price': float(trade_data['entry_price']),
                'exit_price': float(trade_data.get('exit_price', 0.0)),
                'position_size': float(trade_data['position_size']),
                'pips': float(trade_data.get('pips', 0.0)),
                'profit_loss': float(trade_data.get('profit_loss', 0.0)), 
                'spread_cost': float(trade_data.get('spread_cost', 0.0)),
                'commission': float(trade_data.get('commission', 0.0)),
                'net_pnl': float(trade_data.get('net_pnl', 0.0)),
                'confidence': float(trade_data.get('confidence', 0.0)),
                'uncertainty': float(trade_data.get('uncertainty', 0.0)),
                'volatility': float(trade_data.get('volatility', 0.0)),
                'trade_duration': int(trade_data.get('trade_duration', 0)),
                'status': str(trade_data['status'])
            }

            # Add to buffer
            self.trade_buffers[model_name].append(trade_record)
            
            # Write buffer if full 
            if len(self.trade_buffers[model_name]) >= 100:
                self._flush_buffer(model_name)
                
            logger.debug(f"Saved trade record for {model_name}")
            
        except Exception as e:
            logger.error(f"Error saving trade record: {str(e)}")
            raise


    def _calculate_trade_metrics(self, trade: Dict[str, Any]) -> Dict[str, float]:
        """Calculate complete trade metrics
        
        Args:
            trade: Trade record
            
        Returns:
            Dict with calculated metrics
        """
        try:
            # Extract values
            entry_price = float(trade['entry_price'])
            exit_price = float(trade.get('exit_price', entry_price))
            position_size = float(trade['position_size'])
            direction = int(trade['direction'])

            # Calculate pips
            pips = (exit_price - entry_price) * 10000 if direction == 1 else \
                   (entry_price - exit_price) * 10000

            # Calculate costs
            position_value = position_size * 100000 * entry_price
            spread_cost = (25 * 0.0001) * position_value  # 2.5 pips spread
            commission = position_value * 0.0001  # 0.01% commission

            # Calculate P&L
            profit_loss = pips * 10 * position_size  # $10 per pip
            net_pnl = profit_loss - spread_cost - commission

            return {
                'exit_price': exit_price,
                'pips': float(pips),
                'profit_loss': float(profit_loss),
                'spread_cost': float(spread_cost),
                'commission': float(commission),
                'net_pnl': float(net_pnl)
            }

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise



    def _prepare_trade_record(self, model_name: str, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare trade record with calculations
        
        Args:
            model_name: Model name
            trade_data: Raw trade data
            
        Returns:
            Processed trade record
        """
        # Validate required fields
        required = {'direction', 'entry_price', 'position_size'}
        if not all(k in trade_data for k in required):
            raise ValueError(f"Missing required fields: {required - set(trade_data.keys())}")

        # Calculate position value
        position_size = float(trade_data['position_size'])
        entry_price = float(trade_data['entry_price'])
        position_value = position_size * 100000 * entry_price

        # Calculate costs
        spread_cost = (self.SPREAD_POINTS * self.TICK_SIZE) * position_value
        commission = position_value * self.COMMISSION_RATE

        # Calculate stop levels
        volatility = float(trade_data.get('volatility', 0.02))
        sl_pips = 200 * (1 + volatility)
        tp_pips = sl_pips * self.RR_RATIO

        if int(trade_data['direction']) == 1:  # Long
            stop_loss = entry_price * (1 - sl_pips * self.PIP_SIZE)
            take_profit = entry_price * (1 + tp_pips * self.PIP_SIZE)
        else:  # Short
            stop_loss = entry_price * (1 + sl_pips * self.PIP_SIZE)
            take_profit = entry_price * (1 - tp_pips * self.PIP_SIZE)

        return {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'epoch': trade_data.get('epoch', 0),
            'batch': trade_data.get('batch', 0),
            'direction': int(trade_data['direction']),
            'entry_price': float(entry_price),
            'exit_price': float(trade_data.get('exit_price', 0.0)),
            'position_size': float(position_size),
            'stop_loss': float(stop_loss),
            'take_profit': float(take_profit),
            'pips': float(trade_data.get('pips', 0.0)),
            'profit_loss': float(trade_data.get('profit_loss', 0.0)),
            'spread_cost': float(spread_cost),
            'commission': float(commission),
            'net_pnl': float(-spread_cost - commission),
            'confidence': float(trade_data.get('confidence', 0.0)),
            'uncertainty': float(trade_data.get('uncertainty', 0.0)),
            'volatility': float(volatility),
            'trade_duration': int(trade_data.get('trade_duration', 0)),
            'status': 'open'
        }

    def _update_metrics(self, model_name: str, trade: Dict[str, Any]) -> None:
        """Update metrics for model
        
        Args:
            model_name: Model name
            trade: Trade record
        """
        metrics = self.metrics[model_name]
        metrics['total_trades'] += 1
        
        if trade['status'] == 'closed':
            metrics['total_pnl'] += trade['net_pnl']
            if trade['net_pnl'] > 0:
                metrics['winning_trades'] += 1
                
        if metrics['total_trades'] > 0:
            metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']


    def __del__(self):
        """Destructor to ensure files are closed"""
        self.cleanup()



    def update_metrics(self, model_name: str, trade: Dict[str, Any]) -> None:
        """Update trading metrics
        
        Args:
            model_name: Name of model
            trade: Trade data
        """
        try:
            metrics = self.metrics[model_name]
            metrics['total_trades'] += 1

            if trade['status'] == 'closed':
                metrics['total_pnl'] += trade['net_pnl']
                if trade['net_pnl'] > 0:
                    metrics['winning_trades'] += 1

            # Calculate averages
            if metrics['total_trades'] > 0:
                metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']
                metrics['avg_trade'] = metrics['total_pnl'] / metrics['total_trades']

            # Save current metrics
            metrics_file = self.trades_dir / model_name / 'metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")


    def _init_trades_file(self, file_path: Path) -> None:
        """Initialize trade CSV file
        
        Args:
            file_path: Path to CSV file
        """
        try:
            with open(file_path, 'w', newline='') as f:
                # ИСПРАВЛЕНО: Используем trades_headers
                writer = csv.DictWriter(f, fieldnames=self.trades_headers)
                writer.writeheader()
            logger.debug(f"Initialized trades file: {file_path}")
        except Exception as e:
            logger.error(f"Error initializing trades file: {str(e)}")
            raise



    def process_model_prediction(self, 
                               model_name: str,
                               features: torch.Tensor,
                               predictions: torch.Tensor,
                               outputs: Dict[str, torch.Tensor],
                               epoch: int,
                               batch: int) -> None:
        """Обработка предсказаний модели
        
        Args:
            model_name: Название модели
            features: Входные данные [batch_size, seq_len, input_dim]
            predictions: Предсказания [batch_size]
            outputs: Выходы модели
            epoch: Текущая эпоха
            batch: Текущий батч
        """
        try:
            batch_size = features.size(0)
            
            for i in range(batch_size):
                direction = predictions[i].item()
                
                if direction in [0, 1]:  # Торговый сигнал
                    # Создаем данные сделки
                    trade_data = {
                        'timestamp': datetime.now().isoformat(),
                        'model_name': model_name,
                        'trade_id': str(uuid.uuid4()),
                        'epoch': epoch,
                        'batch': batch,
                        'direction': direction,
                        'entry_price': float(features[i, -1, 3]),  # Close price
                        'position_size': float(outputs['position_size'][i]) 
                                       if 'position_size' in outputs else 0.1,
                        'confidence': float(torch.softmax(outputs['logits'][i], dim=0).max()),
                        'uncertainty': float(outputs['uncertainty'][i])
                                      if 'uncertainty' in outputs else 0.0,
                        'volatility': self._calculate_volatility(features[i, :, 3]),
                        'status': 'new'
                    }
                    
                    # Проверяем возможность открытия
                    if self._validate_new_trade(model_name, trade_data):
                        # Рассчитываем уровни
                        levels = self._calculate_trade_levels(
                            trade_data['entry_price'],
                            trade_data['direction'],
                            trade_data['volatility']
                        )
                        trade_data.update(levels)
                        
                        # Рассчитываем затраты
                        costs = self._calculate_trade_costs(
                            trade_data['position_size'],
                            trade_data['entry_price']
                        )
                        trade_data.update(costs)
                        
                        # Сохраняем сделку
                        self._save_trade(trade_data)
                        
                        # Добавляем в активные
                        self.active_trades[model_name].append(trade_data)
                        
                        logger.info(
                            f"New trade ({model_name}):\n"
                            f"Direction: {'Long' if direction == 1 else 'Short'}\n"
                            f"Size: {trade_data['position_size']:.2f}\n"
                            f"Entry: {trade_data['entry_price']:.5f}\n"
                            f"SL: {trade_data['stop_loss']:.5f}\n"
                            f"TP: {trade_data['take_profit']:.5f}"
                        )

        except Exception as e:
            logger.error(f"Error processing predictions: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _validate_new_trade(self, model_name: str, trade_data: Dict[str, Any]) -> bool:
        """Проверка возможности открытия сделки"""
        try:
            stats = self.model_stats[model_name]
            
            # Проверяем количество активных сделок
            if len(self.active_trades[model_name]) >= self.max_trades_per_model:
                return False
                
            # Проверяем общий риск
            position_value = float(trade_data['position_size']) * 100000
            potential_loss = position_value * self.max_risk_per_trade
            
            if potential_loss > stats.balance * self.max_risk_per_trade:
                return False
                
            # Проверяем маржу
            required_margin = position_value / self.leverage
            if required_margin > stats.balance * 0.2:
                return False
                
            # Проверяем конфликты
            if self._has_conflicting_trades(trade_data):
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating trade: {str(e)}")
            return False

    def _calculate_volatility(self, prices: torch.Tensor) -> float:
        """Расчет волатильности"""
        try:
            returns = (prices[1:] / prices[:-1] - 1)
            return float(returns.std() * np.sqrt(252))
        except:
            return 0.0

    def _calculate_trade_levels(self, 
                              entry_price: float,
                              direction: int,
                              volatility: float) -> Dict[str, float]:
        """Расчет уровней входа/выхода"""
        try:
            # Базовые уровни в пунктах
            base_sl = 200
            base_tp = 300
            
            # Корректировка на волатильность
            sl_pips = base_sl * (1 + volatility)
            tp_pips = base_tp * (1 + volatility)
            
            # Конвертация в цену
            if direction == 1:  # Long
                stop_loss = entry_price * (1 - sl_pips/10000)
                take_profit = entry_price * (1 + tp_pips/10000)
            else:  # Short
                stop_loss = entry_price * (1 + sl_pips/10000)
                take_profit = entry_price * (1 - tp_pips/10000)
                
            return {
                'stop_loss': float(stop_loss),
                'take_profit': float(take_profit),
                'sl_pips': float(sl_pips),
                'tp_pips': float(tp_pips)
            }
            
        except Exception as e:
            logger.error(f"Error calculating levels: {str(e)}")
            return {
                'stop_loss': entry_price,
                'take_profit': entry_price,
                'sl_pips': 0.0,
                'tp_pips': 0.0
            }

    def _calculate_trade_costs(self,
                             position_size: float,
                             price: float) -> Dict[str, float]:
        """Расчет затрат на сделку"""
        try:
            position_value = position_size * 100000 * price
            
            spread_cost = (self.spread_pips / 10000) * position_value
            commission = position_value * self.commission_rate
            
            return {
                'spread_cost': float(spread_cost),
                'commission': float(commission),
                'total_cost': float(spread_cost + commission)
            }
            
        except Exception as e:
            logger.error(f"Error calculating costs: {str(e)}")
            return {
                'spread_cost': 0.0,
                'commission': 0.0,
                'total_cost': 0.0
            }

    def _has_conflicting_trades(self, new_trade: Dict[str, Any]) -> bool:
        """Проверка конфликтующих сделок"""
        try:
            new_direction = new_trade['direction']
            
            for model_trades in self.active_trades.values():
                for trade in model_trades:
                    if trade['direction'] != new_direction:
                        return True
            return False
            
        except Exception as e:
            logger.error(f"Error checking conflicts: {str(e)}")
            return True

    def _save_trade(self, trade_data: Dict[str, Any]) -> None:
        """Сохранение сделки в CSV"""
        try:
            model_name = trade_data['model_name']
            trade_file = self.trade_files[model_name]
            
            with open(trade_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.trades_headers)
                writer.writerow(trade_data)
                
        except Exception as e:
            logger.error(f"Error saving trade: {str(e)}")

    def get_model_summary(self, model_name: str) -> Dict[str, Any]:
        """Получение статистики по модели"""
        try:
            stats = self.model_stats[model_name]
            
            return {
                'total_trades': stats.total_trades,
                'winning_trades': stats.winning_trades,
                'losing_trades': stats.losing_trades,
                'total_pnl': stats.total_pnl,
                'current_balance': stats.balance,
                'max_drawdown': stats.max_drawdown,
                'active_trades': len(self.active_trades[model_name]),
                'win_rate': stats.winning_trades / stats.total_trades 
                           if stats.total_trades > 0 else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error getting summary: {str(e)}")
            return {}

    def print_model_metrics(self, model_name: str) -> None:
        """Вывод метрик модели на экран"""
        try:
            summary = self.get_model_summary(model_name)
            
            print(f"\n=== {model_name.upper()} Trading Metrics ===")
            print(f"Total Trades: {summary['total_trades']}")
            print(f"Win Rate: {summary['win_rate']*100:.1f}%")
            print(f"Total P&L: ${summary['total_pnl']:.2f}")
            print(f"Current Balance: ${summary['current_balance']:.2f}")
            print(f"Max Drawdown: {summary['max_drawdown']*100:.1f}%")
            print(f"Active Trades: {summary['active_trades']}")
            
        except Exception as e:
            logger.error(f"Error printing metrics: {str(e)}")




    def validate_trade(self, trade_data: Dict[str, Any], model_name: str) -> bool:
        """Проверка возможности открытия сделки"""
        try:
            # Проверяем баланс модели
            model_balance = self.model_stats[model_name]['balance']
            position_value = float(trade_data['position_size']) * 100000
            
            # Проверяем риск
            max_risk = model_balance * 0.02  # 2% риск на сделку
            potential_loss = position_value * float(trade_data['stop_loss'])
            
            if potential_loss > max_risk:
                logger.warning(f"Trade exceeds risk limit for {model_name}")
                return False
                
            # Проверяем маржу
            required_margin = position_value * 0.01  # 1% маржа
            if required_margin > model_balance * 0.2:  # Не более 20% баланса на маржу
                logger.warning(f"Insufficient margin for {model_name}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating trade: {str(e)}")
            return False

    def calculate_pnl(self, trade_data: Dict[str, Any]) -> Dict[str, float]:
        """Расчет прибыли/убытка по сделке"""
        try:
            # Базовые параметры
            position_size = float(trade_data['position_size'])
            entry_price = float(trade_data['entry_price'])
            exit_price = float(trade_data['exit_price'])
            direction = int(trade_data['direction'])
            
            # Расчет пунктов
            pips = (exit_price - entry_price) * 10000 if direction == 1 else \
                   (entry_price - exit_price) * 10000
                   
            # Стоимость пункта
            pip_value = 10.0
            
            # Расчет P&L
            raw_pnl = pips * pip_value * position_size
            
            # Комиссии
            spread_cost = position_size * 100000 * 0.0002  # 2 пункта спред
            commission = position_size * 100000 * 0.00002  # 0.002% комиссия
            
            # Итоговый P&L
            net_pnl = raw_pnl - spread_cost - commission
            
            return {
                'pips': float(pips),
                'raw_pnl': float(raw_pnl),
                'spread_cost': float(spread_cost), 
                'commission': float(commission),
                'net_pnl': float(net_pnl)
            }
            
        except Exception as e:
            logger.error(f"Error calculating PnL: {str(e)}")
            return {
                'pips': 0.0,
                'raw_pnl': 0.0,
                'spread_cost': 0.0,
                'commission': 0.0,
                'net_pnl': 0.0
            }

    def update_model_stats(self, model_name: str, pnl: float) -> None:
        """Обновление статистики модели"""
        try:
            stats = self.model_stats[model_name]
            
            # Обновляем баланс
            stats['balance'] += pnl
            
            # Обновляем статистику
            stats['total_trades'] += 1
            if pnl > 0:
                stats['winning_trades'] += 1
            else:
                stats['losing_trades'] += 1
            stats['total_pnl'] += pnl
            
            # Обновляем просадку
            peak = max(self.balance_history, key=lambda x: x[1])[1]
            current_drawdown = (peak - stats['balance']) / peak
            stats['current_drawdown'] = current_drawdown
            stats['max_drawdown'] = max(stats['max_drawdown'], current_drawdown)
            
            # Сохраняем историю баланса
            self.balance_history.append((datetime.now(), stats['balance']))
            
        except Exception as e:
            logger.error(f"Error updating model stats: {str(e)}")




    def add_trade(self, model_name: str, trade_data: Dict[str, Any]) -> None:
        """Add trade with validation and saving
        
        Args:
            model_name: Name of model making trade 
            trade_data: Trade information
        """
        try:
            # Validate required fields
            required_fields = {
                'timestamp', 'direction', 'entry_price', 
                'position_size', 'status'
            }
            missing = required_fields - set(trade_data.keys())
            if missing:
                raise ValueError(f"Missing required fields: {missing}")

            # Format trade record
            trade = {
                'timestamp': str(trade_data['timestamp']),
                'model_name': model_name,
                'trade_id': str(uuid.uuid4()),
                'direction': str(trade_data['direction']),
                'entry_price': f"{float(trade_data['entry_price']):.5f}",
                'position_size': f"{float(trade_data['position_size']):.4f}",
                'spread_cost': f"{float(trade_data['spread_cost']):.2f}",
                'commission': f"{float(trade_data['commission']):.2f}",
                'stop_loss': f"{float(trade_data['stop_loss']):.5f}",
                'take_profit': f"{float(trade_data['take_profit']):.5f}",
                'confidence': f"{float(trade_data['confidence']):.4f}",
                'uncertainty': f"{float(trade_data['uncertainty']):.4f}",
                'potential_pips': str(trade_data['potential_pips']),
                'status': str(trade_data['status'])
            }

            # Save directly to CSV
            trades_file = self.trades_dir / model_name / 'trades.csv'
            trades_file.parent.mkdir(exist_ok=True)
            
            # Write header if new file
            write_header = not trades_file.exists()
            
            with open(trades_file, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=trade.keys())
                if write_header:
                    writer.writeheader()
                writer.writerow(trade)
                
            logger.debug(f"Added trade for {model_name}: {trade}")

        except Exception as e:
            logger.error(f"Error adding trade: {str(e)}")
            raise




    def save_metrics(self) -> None:
        """Save metrics to file"""
        try:
            metrics_file = self.trades_dir / f'metrics_{datetime.now():%Y%m%d_%H%M%S}.json'
            
            metrics = {
                model_name: self.get_metrics(model_name)
                for model_name in self.trade_files.keys()
            }
            metrics['timestamp'] = datetime.now().isoformat()
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
                
            logger.info(f"Saved metrics to {metrics_file}")
            
        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            raise





    def calculate_net_pnl(self, trade: Dict[str, Any]) -> float:
        """Calculate net P&L for trade
        
        Args:
            trade: Trade dictionary
            
        Returns:
            float: Net P&L
        """
        profit_loss = float(trade['profit_loss'])
        commission = float(trade.get('commission', 0))
        spread_cost = float(trade.get('spread_cost', 0))
        
        return profit_loss - commission - spread_cost


    def get_trades_summary(self) -> Dict[str, Any]:
        """Get summary of trading performance
        
        Returns:
            Dict with trading metrics
        """
        if not self.trades:
            return {}
            
        df = pd.DataFrame(self.trades)
        
        # Convert numeric columns
        numeric_cols = ['profit_loss', 'net_pnl', 'commission', 'spread_cost']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])

        closed_trades = df[df['status'] == 'closed']
        winning_trades = closed_trades[closed_trades['net_pnl'] > 0]
        
        return {
            'total_trades': len(df),
            'closed_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(closed_trades) - len(winning_trades),
            'win_rate': len(winning_trades) / len(closed_trades) if len(closed_trades) > 0 else 0,
            'total_profit_loss': float(closed_trades['profit_loss'].sum()),
            'total_commission': float(closed_trades['commission'].sum()),
            'total_spread_cost': float(closed_trades['spread_cost'].sum()),
            'total_net_pnl': float(closed_trades['net_pnl'].sum()),
            'avg_trade': float(closed_trades['net_pnl'].mean()) if len(closed_trades) > 0 else 0,
            'max_win': float(winning_trades['net_pnl'].max()) if len(winning_trades) > 0 else 0,
            'max_loss': float(closed_trades[closed_trades['net_pnl'] < 0]['net_pnl'].min()) if len(closed_trades) > 0 else 0
        }

    def _create_trades_file(self) -> None:
        """Create new trades CSV file with headers"""
        try:
            with open(self.trades_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()
                
            logger.debug(f"Created trades file: {self.trades_file}")
            
        except Exception as e:
            logger.error(f"Error creating trades file: {str(e)}")
            raise

    def start_epoch(self, epoch: int) -> None:
        """Start new training epoch"""
        self.current_epoch = epoch
        logger.info(f"Started epoch {epoch}")



    def _create_csv(self, file_path: Path) -> None:
        """Create CSV file with headers"""
        try:
            with open(file_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            logger.debug(f"Created CSV file: {file_path}")
        except Exception as e:
            logger.error(f"Error creating CSV {file_path}: {str(e)}")
            raise


    def end_epoch(self) -> None:
        """End current epoch"""
        # Save summary
        summary = {
            'epoch': self.current_epoch,
            'timestamp': datetime.now().isoformat(),
            'trades_file': str(self.csv_file),
            'num_trades': len(self.trades)
        }
        
        summary_file = self.trades_dir / f'epoch_{self.current_epoch}_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Completed epoch {self.current_epoch}")


    def _write_trade(self, file_path: Path, trade_data: Dict[str, Any]) -> None:
        """Write trade to CSV
        
        Args:
            file_path: Path to CSV file
            trade_data: Trade data to write
        """
        try:
            with open(file_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(trade_data)
                
        except Exception as e:
            logger.error(f"Error writing trade to {file_path}: {str(e)}")
            raise



    def get_trade_stats(self) -> dict:
        """Get trading statistics
        
        Returns:
            Dictionary with trading statistics:
            - total_trades: Total number of trades
            - winning_trades: Number of winning trades
            - losing_trades: Number of losing trades
            - win_rate: Win rate percentage
            - total_pnl: Total profit/loss
            - avg_win: Average winning trade
            - avg_loss: Average losing trade
        """
        try:
            if len(self.df) == 0:
                return {}

            closed_trades = self.df[self.df['status'] == 'closed']
            if len(closed_trades) == 0:
                return {'total_trades': 0}

            winning_trades = closed_trades[closed_trades['pnl'].astype(float) > 0]
            losing_trades = closed_trades[closed_trades['pnl'].astype(float) < 0]

            stats = {
                'total_trades': len(self.df),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(closed_trades) * 100,
                'total_pnl': closed_trades['pnl'].astype(float).sum(),
                'avg_win': winning_trades['pnl'].astype(float).mean() if len(winning_trades) > 0 else 0,
                'avg_loss': losing_trades['pnl'].astype(float).mean() if len(losing_trades) > 0 else 0
            }

            return stats

        except Exception as e:
            logger.error(f"Error calculating trade statistics: {str(e)}")
            raise



if __name__ == '__main__':
    # Example usage
    try:
        recorder = TradeRecorder('results')
        
        # Example trade
        trade = {
            'timestamp': datetime.now(),
            'direction': 1,
            'entry_price': 100.0001,
            'exit_price': 100.0002,
            'position_size': 1.0,
            'probability': 0.9999,
            'uncertainty': 0.0001,
            'pnl': 0.0001,
            'status': 'closed'
        }
        
        recorder.add_trade(trade)
        
        stats = recorder.get_trade_stats()
        print("Trade statistics:", stats)
        
        recorder.cleanup()
        
    except Exception as e:
        print(f"Error in example: {e}")
        sys.exit(1)

