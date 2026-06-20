# tests/test_preprocessor.py

import unittest
import asyncio
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

sys.path.append(str(Path(__file__).parent.parent))

from backtest import TradingSystem

logger = logging.getLogger(__name__)

class TestTradingSystem(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        """Setup test environment"""
        self.initial_balance = 10000.0
        self.system = TradingSystem(initial_balance=self.initial_balance)
        
        # Test data
        self.test_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'open': np.random.uniform(1.0, 1.1, 100),
            'high': np.random.uniform(1.05, 1.15, 100),
            'low': np.random.uniform(0.95, 1.05, 100),
            'close': np.random.uniform(1.0, 1.1, 100),
            'volume': np.random.randint(1, 100, 100)
        })

    async def test_initialization(self):
        """Test system initialization"""
        try:
            # Check balance
            self.assertEqual(self.system.initial_balance, self.initial_balance)
            self.assertEqual(self.system.current_balance, self.initial_balance)
            
            # Check dimensions
            self.assertEqual(self.system.input_dim, 133)
            self.assertEqual(self.system.sequence_length, 60)
            
            # Check device
            self.assertTrue(isinstance(self.system.device, torch.device))
            
            # Check components
            self.assertIsNotNone(self.system.forex_preprocessor)
            self.assertIsNotNone(self.system.trading_preprocessor)
            
            # Check models
            required_models = ['feature_net', 'regime_net', 'directional_net', 
                             'qnd_net', 'ensemble']
            for model_name in required_models:
                self.assertIn(model_name, self.system.models)
                model = self.system.models[model_name]
                self.assertTrue(isinstance(model, torch.nn.Module))
                self.assertEqual(model.device, self.system.device)

            logger.info("Initialization test passed")
            
        except Exception as e:
            self.fail(f"Test failed: {str(e)}")

    async def test_process_features(self):
        """Test feature processing"""
        try:
            # Create test features
            features = torch.randn(
                self.system.sequence_length,
                self.system.input_dim,
                device=self.system.device
            )
            
            # Process features
            decision = await self.system.process_features(features)
            
            # Validate output
            self.assertIsNotNone(decision)
            self.assertIsInstance(decision, dict)
            
            # Check decision fields
            required_fields = ['direction', 'size', 'stop_loss', 'take_profit']
            for field in required_fields:
                self.assertIn(field, decision)
                self.assertIsInstance(decision[field], torch.Tensor)
                
            logger.info("Feature processing test passed")
            
        except Exception as e:
            self.fail(f"Test failed: {str(e)}")

    async def test_error_handling(self):
        """Test error handling"""
        try:
            # Test invalid features
            invalid_features = torch.randn(10, 50)  # Wrong dimensions
            result = await self.system.process_features(invalid_features)
            self.assertIsNone(result)
            
            # Test None input
            result = await self.system.process_features(None)
            self.assertIsNone(result)
            
            # Test nan/inf input
            invalid_features = torch.tensor([[np.nan, np.inf]])
            result = await self.system.process_features(invalid_features)
            self.assertIsNone(result)
            
            logger.info("Error handling test passed")
            
        except Exception as e:
            self.fail(f"Test failed: {str(e)}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main(verbosity=2)
