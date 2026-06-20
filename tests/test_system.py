# tests/test_system.py

import unittest
import torch 
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import asyncio
from collections import deque
from typing import Dict, Tuple, Optional, List, Union, Any, Type

from models.feature_net import FeatureExtractor
from models.directional_net import DirectionalPredictor
from models.regime_net import RegimeDetector
from models.qnd_net import QNDAgent
from models.ensemble_net import EnhancedEnsemble
from trader import Trader
from data.trading_preprocessor import TradingPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# tests/test_system.py

class MockTick:
    """Mock tick data for testing"""
    def __init__(self):
        # Текущая цена
        self.ask = 1.10000
        self.bid = 1.09995
        
        # Рассчитываем high/low на основе ask/bid
        self.high = max(self.ask, self.bid)
        self.low = min(self.ask, self.bid)
        
        self.last = (self.ask + self.bid) / 2
        self.volume_real = 1000.0
        self.time = int(datetime.now().timestamp())

class TestTradingSystem(unittest.TestCase):
    @classmethod 
    def setUpClass(cls):
        """Setup class-level test environment"""
        # Constants
        cls.SEQUENCE_LENGTH = 60
        cls.INPUT_DIM = 133
        cls.HIDDEN_DIM = 2048
        
        # Device setup
        cls.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
            torch.set_num_threads(16)
            
        logger.info(f"Using device: {cls.device}")

    def setUp(self):
        """Setup test instance"""
        self.device = self.__class__.device

        # Initialize event loop
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Initialize components
        self.trader = Trader(
            symbol="EURUSD",
            timeframe="M1",
            models_dir="models",
            risk_per_trade=0.02,
            use_gpu=torch.cuda.is_available(),
            debug=True
        )

        self.preprocessor = TradingPreprocessor()
        
        # Initialize buffer with proper shape
        self.buffer = np.zeros((self.SEQUENCE_LENGTH, 5))
        
        # Generate test data
        self.base_price = 1.10000
        self.test_ticks = self._generate_test_ticks()
        self._init_buffer()
        
        # Validate setup
        self._validate_setup()

# tests/test_system.py

    def _validate_setup(self) -> None:
        """Validate test environment initialization
        
        Raises:
            AttributeError: If required attributes are missing
            ValueError: If data dimensions or values are invalid
        """
        # Required attributes
        required_attrs = {
            'device': (torch.device, "CUDA/CPU device"),
            'trader': (Trader, "Trading system"),
            'preprocessor': (TradingPreprocessor, "Data preprocessor"),
            'buffer': (np.ndarray, "Data buffer"),
            'test_ticks': (list, "Test tick data"),
            'loop': (asyncio.AbstractEventLoop, "Event loop"),
            'base_price': ((int, float), "Base price")
        }
        
        # Check attributes existence and types
        for attr_name, (expected_type, description) in required_attrs.items():
            if not hasattr(self, attr_name):
                raise AttributeError(f"Missing {description} attribute: {attr_name}")
                
            attr_value = getattr(self, attr_name)
            if not isinstance(attr_value, expected_type):
                raise TypeError(
                    f"Invalid {description} type: {type(attr_value)}, "
                    f"expected {expected_type}"
                )
    
        # Validate buffer dimensions
        expected_buffer_shape = (self.SEQUENCE_LENGTH, 5)
        if self.buffer.shape != expected_buffer_shape:
            raise ValueError(
                f"Invalid buffer shape: {self.buffer.shape}, "
                f"expected {expected_buffer_shape}"
            )
            
        # Validate test ticks
        if len(self.test_ticks) != self.SEQUENCE_LENGTH:
            raise ValueError(
                f"Invalid test ticks count: {len(self.test_ticks)}, "
                f"expected {self.SEQUENCE_LENGTH}"
            )
            
        # Validate tick data attributes
        required_tick_attrs = {'ask', 'bid', 'high', 'low', 'last', 'volume_real', 'time'}
        for i, tick in enumerate(self.test_ticks):
            missing_attrs = required_tick_attrs - set(dir(tick))
            if missing_attrs:
                raise AttributeError(
                    f"Missing attributes in tick {i}: {missing_attrs}"
                )
    
        # Validate trader models
        required_models = {'feature', 'directional', 'regime', 'qnd', 'ensemble'}
        if not hasattr(self.trader, 'models'):
            raise AttributeError("Trader models not initialized")
            
        missing_models = required_models - set(self.trader.models.keys())
        if missing_models:
            raise ValueError(f"Missing models: {missing_models}")
    
        # Validate model devices
        if torch.cuda.is_available():
            for name, model in self.trader.models.items():
                try:
                    model_device = next(model.parameters()).device
                    if model_device != self.device:
                        raise ValueError(
                            f"Model {name} on wrong device: {model_device}, "
                            f"expected {self.device}"
                        )
                except Exception as e:
                    raise ValueError(f"Error validating model {name}: {str(e)}")
    
        # Validate numeric data
        if not np.isfinite(self.buffer).all():
            raise ValueError("Buffer contains invalid values (inf/nan)")
    
        # Validate price ranges
        min_price = 0.5  # Минимальная разумная цена
        max_price = 2.0  # Максимальная разумная цена
        if not ((self.buffer[:, 0:4] >= min_price) & 
                (self.buffer[:, 0:4] <= max_price)).all():
            raise ValueError("Price values out of reasonable range")
    
        # Validate volumes
        if not (self.buffer[:, 4] > 0).all():
            raise ValueError("Invalid volume values")
    
        logger.info("Environment validation successful")

    def _generate_test_ticks(self) -> List[MockTick]:
        """Generate test tick data"""
        ticks = []
        for i in range(self.SEQUENCE_LENGTH):
            tick = MockTick()
            # Генерируем базовую цену
            base = self.base_price * (1 + np.random.normal(0, 0.0001))
            
            # Устанавливаем цены
            tick.ask = base + 0.00005
            tick.bid = base - 0.00005
            tick.high = max(tick.ask, base + 0.0001)
            tick.low = min(tick.bid, base - 0.0001)
            tick.last = (tick.ask + tick.bid) / 2
            tick.volume_real = 1000 * (1 + abs(np.random.normal(0, 0.1)))
            tick.time = int(datetime.now().timestamp()) + i
            
            ticks.append(tick)
        return ticks

    def _init_buffer(self) -> None:
        """Initialize buffer with OHLCV data"""
        for i in range(self.SEQUENCE_LENGTH):
            tick = self.test_ticks[i]
            self.buffer[i] = np.array([
                tick.ask,
                tick.high,
                tick.low, 
                tick.bid,
                tick.volume_real
            ])

    async def _process_tick_async(self, data: np.ndarray) -> Optional[np.ndarray]:
        """Process tick data asynchronously"""
        # Add to buffer
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = data
        
        # Process through preprocessor
        try:
            return await self.preprocessor.process_tick(self.buffer)
        except Exception as e:
            logger.error(f"Error processing tick: {str(e)}")
            return None

    def process_tick_sync(self, data: np.ndarray) -> Optional[np.ndarray]:
        """Process tick data synchronously"""
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._process_tick_async(data),
                self.loop
            )
            return future.result(timeout=5)
        except Exception as e:
            logger.error(f"Sync processing error: {str(e)}")
            return None

    def test_preprocessor(self):
        """Test preprocessor functionality"""
        try:
            # Convert ticks to OHLCV
            ohlcv = np.array([
                [tick.ask, tick.high, tick.low, tick.bid, tick.volume_real]
                for tick in self.test_ticks
            ])
            
            # Validate shape
            self.assertEqual(ohlcv.shape, (self.SEQUENCE_LENGTH, 5))
            
            # Process features
            features = self.process_tick_sync(ohlcv[-1])
            self.assertIsNotNone(features)
            self.assertEqual(features.shape, (self.SEQUENCE_LENGTH, self.INPUT_DIM))
            
        except Exception as e:
            self.fail(f"Preprocessor test failed: {str(e)}")

    def test_tick_data_format(self):
        """Test tick data format"""
        try:
            tick = self.test_ticks[0]
            data = np.array([
                tick.ask,
                tick.high,
                tick.low,
                tick.bid, 
                tick.volume_real
            ])
            
            features = self.process_tick_sync(data)
            self.assertIsNotNone(features)
            self.assertEqual(features.shape, (self.SEQUENCE_LENGTH, self.INPUT_DIM))
            
        except Exception as e:
            self.fail(f"Tick data test failed: {str(e)}")

    def tearDown(self):
        """Cleanup resources"""
        # Close event loop
        if hasattr(self, 'loop'):
            self.loop.close()
            
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

           
    def _validate_model_shapes(self, model_name: str, input_tensor: torch.Tensor, 
                             output: Dict[str, torch.Tensor]) -> None:
        """Validate model input/output shapes"""
        expected_shapes = {
            'feature': {
                'input': (1, self.SEQUENCE_LENGTH, self.INPUT_DIM),
                'features': (1, self.SEQUENCE_LENGTH, self.HIDDEN_DIM),
                'reconstructed': (1, self.SEQUENCE_LENGTH, self.INPUT_DIM)
            },
            'directional': {
                'input': (1, self.SEQUENCE_LENGTH, self.INPUT_DIM),
                'logits': (1, 2)
            },
            'regime': {
                'input': (1, self.SEQUENCE_LENGTH, self.INPUT_DIM),
                'logits': (1, 5)
            },
            'qnd': {
                'input': (1, self.SEQUENCE_LENGTH, self.INPUT_DIM),
                'logits': (1, 3)
            },
            'ensemble': {
                'input': {'directional': (1, 2), 'regime': (1, 5), 'qnd': (1, 3)},
                'logits': (1, 2)
            }
        }
        
        # Validate input shape
        expected_input = expected_shapes[model_name]['input']
        if isinstance(expected_input, dict):
            for key, shape in expected_input.items():
                self.assertEqual(input_tensor[key].shape, shape)
        else:
            self.assertEqual(input_tensor.shape, expected_input)
            
        # Validate output shapes
        for key, shape in expected_shapes[model_name].items():
            if key != 'input' and key in output:
                self.assertEqual(output[key].shape, shape)




if __name__ == '__main__':
    unittest.main()