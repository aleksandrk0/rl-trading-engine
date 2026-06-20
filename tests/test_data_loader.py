# C_01_en/tests/test_data_loader.py

import sys
import os
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import asyncio
import time
import psutil
import torch

from main import TradingSystem
from data.preprocessor import ForexPreprocessor
from data.trading_preprocessor import TradingPreprocessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestDataLoader:
    @pytest.fixture
    def test_dir(self) -> Path:
        """Create test directory"""
        test_dir = Path(__file__).parent / "test_data"
        test_dir.mkdir(exist_ok=True)
        return test_dir
        
# tests/test_preprocessor.py

class TestForexPreprocessor:
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance"""
        preprocessor = ForexPreprocessor()
        preprocessor.device = torch.device('cpu')  # Force CPU
        return preprocessor
        
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data"""
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='5min')
        np.random.seed(42)
        
        data = []
        base_price = 1.1000
        
        for i in range(1000):
            price_change = np.random.normal(0, 0.0002)
            base_price += price_change
            
            open_price = base_price + np.random.normal(0, 0.0001)
            high_price = base_price + abs(np.random.normal(0, 0.0002))
            low_price = base_price - abs(np.random.normal(0, 0.0002))
            close_price = base_price
            volume = abs(int(np.random.lognormal(3, 1)))
            
            data.append({
                'timestamp': dates[i],
                'open': float(open_price),
                'high': float(high_price),
                'low': float(low_price),
                'close': float(close_price),
                'volume': float(volume)
            })
            
        return pd.DataFrame(data)

    def test_initialization(self, preprocessor):
        """Test CPU initialization"""
        assert preprocessor.device == torch.device('cpu')
        assert preprocessor.input_dim == 133
        assert preprocessor.sequence_length == 60
        assert not preprocessor.is_fitted

    @pytest.mark.asyncio
    async def test_real_time_processing(self, preprocessor, sample_data):
        """Test real-time processing"""
        # Convert to numpy with proper dtypes
        data = sample_data.astype({
            'open': 'float32',
            'high': 'float32', 
            'low': 'float32',
            'close': 'float32',
            'volume': 'float32'
        })
        
        await preprocessor.fit(data.iloc[:100])
        assert preprocessor.is_fitted

    def test_sma_calculation(self, preprocessor):
        """Test Simple Moving Average"""
        x = torch.tensor([1.0, 2.0, 3.0], device='cpu')
        sma = preprocessor._calculate_sma(x, window=2)
        expected = torch.tensor([1.0, 1.5, 2.5], device='cpu')
        assert torch.allclose(sma, expected)

    def test_bollinger_bands(self, preprocessor):
        """Test Bollinger Bands"""
        prices = torch.tensor([1.0, 2.0, 3.0, 2.0, 1.0], device='cpu')
        upper, middle, lower = preprocessor._calculate_bollinger_bands(prices)
        assert torch.all(upper >= middle)
        assert torch.all(middle >= lower)

    
    @pytest.fixture
    def sample_mt5_data(self, test_dir) -> Path:
        """Create sample MT5 CSV data"""
        test_file = test_dir / "test_mt5_data.csv"
        
        # Generate sample data with explicit types
        data = []
        base_time = datetime(2024, 1, 1)
        base_price = 1.1000
        
        for i in range(1000):
            time = base_time + timedelta(minutes=i)
            open_price = base_price * (1 + np.random.normal(0, 0.0002))
            high_price = open_price * (1 + abs(np.random.normal(0, 0.0003)))
            low_price = open_price * (1 - abs(np.random.normal(0, 0.0003)))
            close_price = np.random.uniform(low_price, high_price)
            
            # Generate volume as float
            volume = float(abs(np.random.lognormal(3, 1)))
            
            data.append({
                'DATE': time.strftime("%Y.%m.%d"),
                'TIME': time.strftime("%H:%M:%S"),
                'OPEN': f"{open_price:.5f}",
                'HIGH': f"{high_price:.5f}",
                'LOW': f"{low_price:.5f}",
                'CLOSE': f"{close_price:.5f}",
                'TICKVOL': f"{volume:.1f}",
                'VOL': "0",
                'SPREAD': "5"
            })
            
            base_price = close_price
            
        # Write to CSV
        df = pd.DataFrame(data)
        df.to_csv(test_file, index=False)
        
        return test_file

    @pytest.mark.asyncio
    async def test_data_loading(self, sample_mt5_data):
        """Test complete data loading pipeline"""
        # Create system
        system = TradingSystem()
        
        # Initialize
        await system.initialize(sample_mt5_data)
        
        # Validate state
        assert system.is_running
        
        # Get preprocessor state
        assert system.preprocessor.is_fitted
        
        # Check data quality
        data = await system.preprocessor.get_data()  # Assuming this method exists
        assert isinstance(data, pd.DataFrame)
        
        # Check structure
        expected_columns = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        assert set(data.columns) == expected_columns
        
        # Check types
        assert data['timestamp'].dtype == 'datetime64[ns]'
        assert all(data[col].dtype == np.float64 for col in ['open', 'high', 'low', 'close'])
        assert data['volume'].dtype == np.int64
        
        # Check relationships
        assert all(data['high'] >= data['low'])
        assert all(data['volume'] >= 0)
        
        # Performance test (100 ticks)
        start_time = time.perf_counter()
        for i in range(100):
            tick = {
                'timestamp': datetime.now(),
                'open': 1.1000,
                'high': 1.1001,
                'low': 1.0999,
                'close': 1.1000,
                'volume': 1000
            }
            await system.process_tick(tick)
        
        elapsed = time.perf_counter() - start_time
        ticks_per_second = 100 / elapsed
        
        # Requirements
        assert ticks_per_second > 50, f"Processing too slow: {ticks_per_second:.1f} ticks/s"
        
        memory = psutil.Process().memory_info().rss / 1024**3
        assert memory < 4, f"Memory usage too high: {memory:.1f}GB"

if __name__ == '__main__':
    asyncio.run(pytest.main(['-v', __file__]))