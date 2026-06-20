# tests/test_sync.py

import os
import sys
import inspect
import ast
import pytest
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from trader import Trader
from data.trading_preprocessor import TradingPreprocessor

logger = logging.getLogger(__name__)

class TestSync:
    """Test synchronous execution"""
    
    def setup_method(self):
        """Setup test environment"""
        self.trader = Trader(
            symbol="EURUSD",
            timeframe="M1",
            sequence_length=300,
            input_dim=133,
            hidden_dim=3072,
            risk_per_trade=0.02,
            use_gpu=True,
            debug=True,
            models_dir="models/pt",
            output_dir="results"
        )
        
    def test_no_async_await(self):
        """Test that no async/await keywords are present in code"""
        trader_file = project_root / 'trader.py'
        preprocessor_file = project_root / 'data' / 'trading_preprocessor.py'
        
        files_to_check = [trader_file, preprocessor_file]
        
        for file_path in files_to_check:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
                
            # Parse code into AST
            tree = ast.parse(code)
            
            # Find async/await nodes
            async_nodes = []
            await_nodes = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.AsyncFunctionDef):
                    async_nodes.append(node.name)
                elif isinstance(node, ast.Await):
                    # Get line number
                    await_nodes.append(node.lineno)
                    
            # Log findings
            if async_nodes or await_nodes:
                logger.error(
                    f"\nFound async/await in {file_path.name}:"
                    f"\nAsync functions: {async_nodes}"
                    f"\nAwait lines: {await_nodes}"
                )
                
            # Assert no async/await
            assert not async_nodes, f"Found async functions: {async_nodes}"
            assert not await_nodes, f"Found await statements: {await_nodes}"
            
    def test_trader_methods(self):
        """Test that all trader methods are synchronous"""
        # Get all methods
        methods = inspect.getmembers(self.trader, predicate=inspect.ismethod)
        
        for name, method in methods:
            # Skip magic methods
            if name.startswith('__'):
                continue
                
            # Get source code
            try:
                source = inspect.getsource(method)
                
                # Check for async/await
                assert 'async ' not in source, f"Found async in {name}"
                assert 'await ' not in source, f"Found await in {name}"
                
            except Exception as e:
                logger.error(f"Error checking method {name}: {str(e)}")
                continue
                
    def test_preprocessor_methods(self):
        """Test that all preprocessor methods are synchronous"""
        preprocessor = self.trader.preprocessor
        
        methods = inspect.getmembers(preprocessor, predicate=inspect.ismethod)
        
        for name, method in methods:
            if name.startswith('__'):
                continue
                
            try:
                source = inspect.getsource(method)
                
                assert 'async ' not in source, f"Found async in {name}"
                assert 'await ' not in source, f"Found await in {name}"
                
            except Exception as e:
                logger.error(f"Error checking method {name}: {str(e)}")
                continue
                
    def test_full_sync_run(self):
        """Test full synchronous execution"""
        try:
            # Initialize
            success = self.trader.run()
            assert success is None
            
            # Check trading state
            assert self.trader.is_running == True
            assert hasattr(self.trader, 'positions_info')
            assert hasattr(self.trader, 'stats')
            
        except Exception as e:
            logger.error(f"Error in sync run test: {str(e)}")
            raise

def main():
    """Run tests"""
    pytest.main([__file__, '-v'])
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
