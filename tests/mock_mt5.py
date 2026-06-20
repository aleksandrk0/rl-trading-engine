# tests/mock_mt5.py

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional, List, Union, Any, Type
import asyncio
import time
import traceback
from dataclasses import dataclass, field
from collections import deque



from data.data_converter import TickBuffer

logger = logging.getLogger(__name__)




@dataclass
class MockTick:
    """Mock MT5 tick data structure"""
    bid: float
    ask: float
    last: float
    volume: float
    time: float
    flags: int = 0
    time_msc: int = 0
    _high: float = 0.0
    _low: float = 0.0
    
    def __post_init__(self):
        """Calculate derived fields"""
        self.time_msc = int(self.time * 1000)
        self._high = max(self.ask, self.last)
        self._low = min(self.bid, self.last)
        self.spread = self.ask - self.bid
    
    def validate(self) -> bool:
        """Validate tick data"""
        try:
            if self.ask <= self.bid:
                return False
            if self.volume <= 0:
                return False
            if self.time <= 0:
                return False
            return True
        except Exception:
            return False
        
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'bid': self.bid,
            'ask': self.ask,
            'last': self.last,
            'volume': self.volume,
            'time': self.time,
            'time_msc': self.time_msc,
            'flags': self.flags,
            'high': self.high,
            'low': self.low
        }
        
    @property
    def high(self) -> float:
        """Get high price"""
        return self._high
        
    @property
    def low(self) -> float:
        """Get low price"""
        return self._low




        
    def to_array(self) -> np.ndarray:
        """Convert to numpy array
        
        Returns:
            np.ndarray: Tick data array [5]
        """
        return np.array([
            self.bid,
            self.ask,
            self.last,
            self.volume,
            self.time
        ], dtype=np.float32)
        

        
    def __getitem__(self, key: str) -> float:
        """Allow dictionary-style access"""
        return getattr(self, key)



@dataclass
class MockPosition:
    """Mock MT5 position with dict-like access"""
    ticket: int
    symbol: str
    type: int  # 0=BUY, 1=SELL
    volume: float
    price_open: float
    sl: float
    tp: float
    price_current: float
    swap: float = 0.0
    profit: float = 0.0
    comment: str = ""
    magic: int = 0
    
    def __post_init__(self):
        """Initialize additional attributes"""
        self._data = {
            'ticket': self.ticket,
            'symbol': self.symbol,
            'type': 'buy' if self.type == 0 else 'sell',
            'volume': self.volume,
            'price': self.price_open,
            'sl': self.sl,
            'tp': self.tp,
            'price_current': self.price_current,
            'swap': self.swap,
            'profit': self.profit,
            'comment': self.comment,
            'magic': self.magic
        }

    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access"""
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Support dictionary-style assignment"""
        self._data[key] = value
        # Sync with dataclass attributes
        if hasattr(self, key):
            setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style get method"""
        return self._data.get(key, default)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return self._data.copy()

    def update(self, current_price: float) -> None:
        """Update position P&L"""
        self.price_current = current_price
        self._data['price_current'] = current_price
        
        # Calculate profit
        multiplier = 1 if self.type == 0 else -1
        self.profit = multiplier * (current_price - self.price_open) * self.volume * 100000
        self._data['profit'] = self.profit


        
class MockMT5:
    def __init__(self, ticks_file: str):
        """Initialize mock MT5"""
        try:
            # Base parameters
            self.ticks_file = self._resolve_file_path(ticks_file)
            self.sequence_length = 60
            self.feature_dim = 5
            self.debug = True
            self.batch_size = 100

            # Initialize statistics
            self.stats = {
                'total_ticks': 0,
                'valid_ticks': 0,
                'processed_ticks': 0,
                'errors': 0,
                'start_time': datetime.now(),
                'last_update': None
            }

            # Initialize buffers using contiguous arrays
            self.tick_cache = deque(maxlen=1000)
            self.current_tick_idx = 0
            self.buffer = np.zeros((1, self.sequence_length, self.feature_dim), 
                                 dtype=np.float32, order='C')
            self.buffer_filled = False
            self.current_pos = 0

            # Load tick data
            self.ticks = self._load_ticks()
            self._validate_ticks()

            logger.info(
                f"MockMT5 initialized:\n"
                f"- Buffer shape: {self.buffer.shape}\n"
                f"- Buffer stride: {self.buffer.strides}\n"
                f"- Memory layout: {'C-contiguous' if self.buffer.flags['C_CONTIGUOUS'] else 'Not C-contiguous'}"
            )

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

    async def symbol_info_tick(self, symbol: str) -> Optional[np.ndarray]:
        """Get tick data with shape (1, sequence_length, feature_dim)"""
        try:
            if self.current_tick_idx >= len(self.ticks):
                return None

            if len(self.tick_cache) == 0:
                batch_end = min(self.current_tick_idx + self.batch_size, len(self.ticks))
                batch = self.ticks.iloc[self.current_tick_idx:batch_end]

                for _, tick in batch.iterrows():
                    if self._validate_tick_data(tick):
                        # Create contiguous array for single tick
                        tick_array = np.ascontiguousarray([
                            tick['ask'],
                            max(tick['ask'], tick['last']),
                            min(tick['bid'], tick['last']),
                            tick['bid'],
                            tick['volume']
                        ], dtype=np.float32)
                        self.tick_cache.append(tick_array)

            if self.tick_cache:
                tick_array = self.tick_cache.popleft()
                
                # Update buffer ensuring contiguous memory
                if self.buffer_filled:
                    # Roll buffer and ensure contiguous memory
                    self.buffer = np.roll(self.buffer, -1, axis=1)
                    self.buffer[0, -1] = tick_array
                else:
                    self.buffer[0, self.current_pos] = tick_array
                    self.current_pos += 1
                    
                    if self.current_pos >= self.sequence_length:
                        self.buffer_filled = True
                        logger.info("Buffer filled successfully")

                self.current_tick_idx += 1
                
                if self.buffer_filled:
                    # Return contiguous copy
                    return np.ascontiguousarray(self.buffer)

            return None

        except Exception as e:
            logger.error(f"Error getting tick: {str(e)}")
            return None

    def initialize(self) -> bool:
        """Initialize MT5 state"""
        try:
            self.current_tick_idx = 0
            self.buffer_filled = False
            self.current_pos = 0
            self.tick_cache.clear()
            
            # Reset statistics
            self.stats.update({
                'processed_ticks': 0,
                'errors': 0,
                'start_time': datetime.now(),
                'last_update': None,
                'cache_hits': 0,
                'cache_misses': 0
            })
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            return False

    def shutdown(self) -> None:
        """Cleanup resources"""
        try:
            # Cleanup buffers
            self.tick_cache.clear()
            self.buffer = np.zeros((self.sequence_length, self.feature_dim), 
                                 dtype=np.float32)
            self.ticks = pd.DataFrame()

            # Log final statistics
            end_time = datetime.now()
            duration = (end_time - self.stats['start_time']).total_seconds()
            
            if duration > 0:
                logger.info(
                    f"Shutdown completed:\n"
                    f"- Processed ticks: {self.stats['processed_ticks']}\n"
                    f"- Duration: {duration:.1f}s\n"
                    f"- Ticks/sec: {self.stats['processed_ticks']/duration:.1f}\n"
                    f"- Errors: {self.stats['errors']}\n"
                    f"- Cache hits: {self.stats['cache_hits']}\n"
                    f"- Cache misses: {self.stats['cache_misses']}"
                )

        except Exception as e:
            logger.error(f"Shutdown error: {str(e)}")
            if self.debug:
                logger.error(traceback.format_exc())

    def _load_ticks(self) -> pd.DataFrame:
        """Load and prepare tick data"""
        try:
            # Read CSV with proper types
            ticks = pd.read_csv(
                self.ticks_file,
                dtype={
                    'timestamp': str,
                    'bid': np.float32,
                    'ask': np.float32,
                    'last': np.float32,
                    'volume': np.float32,
                    'time_msc': np.int64,
                    'flags': np.int32
                }
            )
            
            # Convert timestamp
            ticks['timestamp'] = pd.to_datetime(ticks['timestamp'])

            # Sort by timestamp
            ticks = ticks.sort_values('timestamp').reset_index(drop=True)

            return ticks

        except Exception as e:
            logger.error(f"Error loading ticks: {str(e)}")
            raise

    def _validate_ticks(self) -> None:
        """Validate loaded tick data"""
        initial_size = len(self.ticks)

        # Remove invalid prices
        self.ticks = self.ticks[
            (self.ticks['ask'] > self.ticks['bid']) & 
            (self.ticks['bid'] > 0)
        ].copy()

        # Remove invalid volumes
        self.ticks = self.ticks[self.ticks['volume'] >= 0]

        # Reset index
        self.ticks.reset_index(drop=True, inplace=True)

        logger.info(
            f"Tick validation:\n"
            f"- Initial ticks: {initial_size}\n"
            f"- Valid ticks: {len(self.ticks)}\n"
            f"- Removed: {initial_size - len(self.ticks)}"
        )





    def _validate_tick_data(self, tick: pd.Series) -> bool:
        """Validate tick data with detailed checks"""
        try:
            # Verify fields
            required_fields = ['bid', 'ask', 'last', 'volume', 'timestamp']
            if not all(field in tick.index for field in required_fields):
                logger.debug(f"Missing fields in tick data: {tick.index.tolist()}")
                return False

            # Validate numeric values
            if not all(isinstance(tick[f], (int, float)) for f in ['bid', 'ask', 'last', 'volume']):
                logger.debug("Invalid numeric types in tick data")
                return False

            # Check price relationships
            if not (tick['ask'] > tick['bid'] > 0):
                logger.debug(f"Invalid prices: ask={tick['ask']}, bid={tick['bid']}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating tick: {str(e)}")
            return False





    def _resolve_file_path(self, file_path: str) -> Path:
        """Resolve data file path with multiple locations
        
        Args:
            file_path: Initial file path
            
        Returns:
            Path: Resolved path
            
        Raises:
            FileNotFoundError: If file not found
        """
        # Check multiple possible locations
        search_paths = [
            Path(file_path),
            Path('data') / file_path,
            Path('data/historical') / file_path,
            Path('tests/data') / file_path,
            Path('.') / file_path
        ]
        
        for path in search_paths:
            if path.exists():
                return path
                
        raise FileNotFoundError(f"Data file not found: {file_path}")

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        self.log_file = Path('logs') / f'mock_mt5_{datetime.now():%Y%m%d_%H%M%S}.log'
        self.log_file.parent.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logging.getLogger().addHandler(file_handler)

    def _validate_initialization(self) -> None:
        """Validate initialization state"""
        try:
            if not self.ticks_file.exists():
                raise FileNotFoundError(f"Data file not found: {self.ticks_file}")
                
            if self.ticks.empty:
                raise ValueError("No data loaded")
                
            required_cols = {'bid', 'ask', 'timestamp'}
            missing_cols = required_cols - set(self.ticks.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise




    def create_position(self, order: dict) -> dict:
        """Create new position with proper conversion"""
        try:
            ticket = self.next_ticket
            self.next_ticket += 1
            
            # Convert order type
            order_type = 0 if order['type'].lower() == 'buy' else 1
            
            position = MockPosition(
                ticket=ticket,
                symbol=order['symbol'],
                type=order_type,
                volume=float(order['volume']),
                price_open=float(order['price']),
                sl=float(order.get('sl', 0.0)),
                tp=float(order.get('tp', 0.0)),
                price_current=float(order['price']),
                magic=int(order.get('magic', 0)),
                comment=str(order.get('comment', ''))
            )
            
            self.positions[ticket] = position
            
            logger.info(
                f"Created position:\n"
                f"Ticket: {ticket}\n"
                f"Type: {position['type']}\n"
                f"Price: {position['price']:.5f}\n"
                f"Volume: {position['volume']:.2f}"
            )
            
            return {
                'retcode': self.TRADE_RETCODE_DONE,
                'order': ticket,
                'volume': position.volume,
                'price': position.price_open,
                'comment': position.comment
            }
            
        except Exception as e:
            logger.error(f"Error creating position: {str(e)}")
            return {'retcode': self.TRADE_RETCODE_ERROR}

    def _validate_position(self, position: MockPosition) -> bool:
        """Validate position data"""
        try:
            required = {'ticket', 'symbol', 'type', 'volume', 'price'}
            return all(k in position._data for k in required)
        except Exception as e:
            logger.error(f"Position validation error: {str(e)}")
            return False

    def get_position(self, symbol: str) -> Optional[MockPosition]:
        """Get position by symbol"""
        try:
            # Find position for symbol
            for pos in self.positions.values():
                if pos.symbol == symbol:
                    # Update position with current price
                    if self.current_tick_idx < len(self.ticks):
                        current_price = float(self.ticks.iloc[self.current_tick_idx]['bid'])
                        pos.update(current_price)
                    return pos
            return None
            
        except Exception as e:
            logger.error(f"Error getting position: {str(e)}")
            return None


    def close_position(self, ticket: int) -> dict:
        """Close existing position"""
        try:
            if ticket not in self.positions:
                return {'retcode': self.TRADE_RETCODE_ERROR}
                
            position = self.positions[ticket]
            
            # Update balance
            self.balance += position.profit
            self.equity = self.balance
            
            # Remove position
            del self.positions[ticket]
            
            return {
                'retcode': self.TRADE_RETCODE_DONE,
                'profit': position.profit
            }
            
        except Exception as e:
            logger.error(f"Error closing position: {str(e)}")
            return {'retcode': self.TRADE_RETCODE_ERROR}

    def positions_total(self) -> int:
        """Get total number of positions"""
        return len(self.positions)

    def positions_get(self, symbol: Optional[str] = None) -> List[MockPosition]:
        """Get all positions or positions for symbol"""
        try:
            if symbol:
                return [p for p in self.positions.values() if p.symbol == symbol]
            return list(self.positions.values())
            
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return []
    

    def _load_and_process_data(self) -> None:
        """Load and prepare tick data"""
        try:
            # Чтение файла с явным указанием типов
            dtype_dict = {
                'timestamp': str,
                'bid': np.float32,
                'ask': np.float32,
                'last': np.float32,
                'volume': np.float32,
                'time_msc': np.int64,
                'flags': np.int32
            }
            
            self.ticks = pd.read_csv(
                self.ticks_file,
                dtype=dtype_dict,
                parse_dates=['timestamp']
            )

            # Валидация данных
            self._validate_tick_data_structure()
            
            # Очистка данных
            self._clean_tick_data()
            
            logger.info(
                f"Data loaded successfully:\n"
                f"- Total rows: {len(self.ticks)}\n"
                f"- Memory usage: {self.ticks.memory_usage().sum()/1024**2:.1f}MB\n"
                f"- Date range: {self.ticks['timestamp'].min()} - {self.ticks['timestamp'].max()}"
            )

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            if self.debug:
                logger.error(traceback.format_exc())
            raise

    def _validate_tick_data_structure(self) -> None:
        """Validate tick data structure"""
        required_columns = {
            'timestamp', 'bid', 'ask', 'last', 
            'volume', 'time_msc', 'flags'
        }
        
        # Проверка наличия всех колонок
        missing_columns = required_columns - set(self.ticks.columns)
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")

        # Проверка типов данных
        expected_types = {
            'timestamp': np.dtype('datetime64[ns]'),
            'bid': np.dtype('float32'),
            'ask': np.dtype('float32'),
            'last': np.dtype('float32'),
            'volume': np.dtype('float32'),
            'time_msc': np.dtype('int64'),
            'flags': np.dtype('int32')
        }

        for col, expected_type in expected_types.items():
            if self.ticks[col].dtype != expected_type:
                logger.warning(
                    f"Column {col} has type {self.ticks[col].dtype}, "
                    f"converting to {expected_type}"
                )
                self.ticks[col] = self.ticks[col].astype(expected_type)

    def _clean_tick_data(self) -> None:
        """Clean tick data"""
        initial_rows = len(self.ticks)

        # Удаление строк с невалидными ценами
        self.ticks = self.ticks[
            (self.ticks['ask'] > self.ticks['bid']) & 
            (self.ticks['bid'] > 0) &
            (self.ticks['ask'] > 0)
        ].copy()

        # Удаление дубликатов по времени
        self.ticks = self.ticks.drop_duplicates(subset=['time_msc'])

        # Сортировка по времени
        self.ticks = self.ticks.sort_values('time_msc')

        # Сброс индекса
        self.ticks = self.ticks.reset_index(drop=True)

        rows_removed = initial_rows - len(self.ticks)
        logger.info(
            f"Data cleaning completed:\n"
            f"- Initial rows: {initial_rows}\n"
            f"- Rows removed: {rows_removed}\n"
            f"- Final rows: {len(self.ticks)}"
        )


    def _generate_tick_volumes(self) -> None:
        """Generate synthetic tick volumes based on price movement"""
        try:
            # Group by minute for base volumes
            self.ticks['minute'] = self.ticks['timestamp'].dt.floor('1min')
            ticks_per_minute = self.ticks.groupby('minute').size()
            
            # Calculate price changes
            self.ticks['price_change'] = np.abs(self.ticks['ask'].diff())
            
            # Generate base volume from price changes
            self.ticks['tick_volume'] = self.ticks['price_change'].fillna(0)
            self.ticks['tick_volume'] += 0.1  # Minimum volume
            
            # Scale by tick density
            self.ticks['tick_volume'] *= self.ticks['minute'].map(
                ticks_per_minute / ticks_per_minute.mean()
            )
            
            # Normalize volumes
            mean_volume = self.ticks['tick_volume'].mean()
            self.ticks['tick_volume'] = self.ticks['tick_volume'] / mean_volume
            
            # Clean up
            self.ticks.drop(['minute', 'price_change'], axis=1, inplace=True)
            
            logger.info(
                f"Generated tick volumes:\n"
                f"- Mean volume: {self.ticks['tick_volume'].mean():.2f}\n"
                f"- Min volume: {self.ticks['tick_volume'].min():.2f}\n"
                f"- Max volume: {self.ticks['tick_volume'].max():.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error generating volumes: {str(e)}")
            raise


    def _generate_tick_volume(self) -> None:
        """Generate synthetic tick volume based on price movement"""
        try:
            # Calculate price changes
            self.ticks['price_change'] = (
                self.ticks['ask'].diff().abs() + 
                self.ticks['bid'].diff().abs()
            ) / 2
            
            # Base volume from price changes
            self.ticks['tick_volume'] = (
                self.ticks['price_change'] / 
                self.ticks['price_change'].mean()
            ) * 100
            
            # Add minimum volume
            self.ticks['tick_volume'] = self.ticks['tick_volume'].fillna(1.0) + 1.0
            
            # Convert to float32
            self.ticks['tick_volume'] = self.ticks['tick_volume'].astype(np.float32)
            
            logger.info(
                f"Generated tick volumes:\n"
                f"Mean: {self.ticks['tick_volume'].mean():.2f}\n"
                f"Min: {self.ticks['tick_volume'].min():.2f}\n"
                f"Max: {self.ticks['tick_volume'].max():.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error generating tick volumes: {str(e)}")
            raise

    def _clean_data(self) -> None:
        """Clean tick data with proper volume generation"""
        try:
            initial_size = len(self.ticks)
            
            # Required columns mapping
            column_mapping = {
                'timestamp': ['timestamp', 'time', 'datetime'],
                'ask': ['ask', 'Ask', 'ASK'],
                'bid': ['bid', 'Bid', 'BID'],
                'volume': ['volume', 'Volume', 'VOLUME']
            }
            
            # Standardize column names
            for target, possible_names in column_mapping.items():
                found = False
                for name in possible_names:
                    if name in self.ticks.columns:
                        self.ticks[target] = self.ticks[name]
                        found = True
                        break
                if not found and target != 'volume':
                    raise ValueError(f"Required column not found: {target}")
                    
            # Convert timestamp
            if 'timestamp' in self.ticks.columns:
                self.ticks['timestamp'] = pd.to_datetime(self.ticks['timestamp'])
            
            # Remove NaN
            self.ticks.dropna(subset=['bid', 'ask'], inplace=True)
            
            # Basic validation
            self.ticks = self.ticks[
                (self.ticks['ask'] > 0) & 
                (self.ticks['bid'] > 0) &
                (self.ticks['ask'] > self.ticks['bid'])
            ].copy()
            
            # Generate tick volume if needed
            if 'tick_volume' not in self.ticks.columns:
                self._generate_tick_volume()
                
            # Add required columns
            self.ticks['last'] = self.ticks['bid']
            self.ticks['flags'] = 0
            self.ticks['time_msc'] = (
                self.ticks['timestamp'].astype(np.int64) // 10**6
            )
            
            # Convert types
            self.ticks['bid'] = self.ticks['bid'].astype(np.float32)
            self.ticks['ask'] = self.ticks['ask'].astype(np.float32)
            self.ticks['last'] = self.ticks['last'].astype(np.float32)
            self.ticks['tick_volume'] = self.ticks['tick_volume'].astype(np.float32)
            
            # Reset index
            self.ticks.reset_index(drop=True, inplace=True)
            
            logger.info(
                f"Data cleaning completed:\n"
                f"Initial rows: {initial_size}\n"
                f"Final rows: {len(self.ticks)}\n"
                f"Memory usage: {self.ticks.memory_usage().sum()/1024**2:.1f}MB"
            )
            
        except Exception as e:
            logger.error(f"Error cleaning data: {str(e)}")
            raise


    def _process_batch(self, batch: pd.DataFrame) -> List[MockTick]:
        """Process batch of ticks
        
        Args:
            batch: DataFrame with tick data
            
        Returns:
            List[MockTick]: List of processed ticks
        """
        ticks = []
        for _, row in batch.iterrows():
            try:
                tick = MockTick(
                    bid=float(row['bid']),
                    ask=float(row['ask']),
                    last=float(row.get('last', row['bid'])),
                    volume=float(row['tick_volume']),
                    time=row['timestamp'].timestamp(),
                    flags=int(row.get('flags', 0)),
                    time_msc=int(row.get('time_msc', 0))
                )
                if tick.validate():
                    ticks.append(tick)
            except Exception as e:
                logger.error(f"Error processing tick: {str(e)}")
                continue
        return ticks

    def _preload_data(self) -> None:
        """Preload and prepare data"""
        try:
            # Читаем чанками для больших файлов
            chunks = []
            for chunk in pd.read_csv(self.ticks_file, chunksize=50000):
                chunks.append(chunk)
            self.ticks = pd.concat(chunks)
            
            # Конвертируем типы
            self.ticks['timestamp'] = pd.to_datetime(self.ticks['timestamp'])
            self.ticks['bid'] = self.ticks['bid'].astype(np.float32)
            self.ticks['ask'] = self.ticks['ask'].astype(np.float32)
            
            # Предварительная валидация
            self.ticks = self.ticks[
                (self.ticks['ask'] > self.ticks['bid']) & 
                (self.ticks['bid'] > 0)
            ]
            
            logger.info(
                f"Preloaded data:\n"
                f"Total ticks: {len(self.ticks)}\n"
                f"Memory usage: {self.ticks.memory_usage().sum()/1024**2:.1f}MB"
            )
            
        except Exception as e:
            logger.error(f"Error preloading data: {str(e)}")
            raise
        

    def _validate_tick(self, tick: Dict[str, float]) -> bool:
        """Validate tick data
        
        Args:
            tick: Tick data dictionary
            
        Returns:
            bool: True if valid
        """
        try:
            # Check required fields
            required = {'time', 'bid', 'ask', 'last', 'volume'}
            if not all(k in tick for k in required):
                return False
                
            # Check types
            if not all(isinstance(tick[k], float) for k in required):
                return False
                
            # Validate values
            if any(tick[k] <= 0 for k in ['bid', 'ask', 'volume']):
                return False
                
            # Check prices
            if tick['ask'] <= tick['bid']:
                return False
                
            # Check time
            if tick['time'] <= 0:
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False

    def _format_tick(self, row: pd.Series) -> Dict[str, float]:
        """Format tick data with validation
        
        Args:
            row: DataFrame row
            
        Returns:
            Dict[str, float]: MT5 format tick
        """
        try:
            # Extract base values
            bid = float(row['bid'])
            ask = float(row['ask'])
            volume = float(row['tick_volume'])
            timestamp = float(row['timestamp'].timestamp())
            
            # MT5 style formatting
            tick = {
                'time': timestamp,
                'bid': bid,
                'ask': ask,
                'last': bid,  # Use bid as last if no trades
                'volume': volume,
                'time_msc': int(timestamp * 1000),
                'flags': 0,
                'volume_real': volume,
                'high': ask,  # Use ask as high
                'low': bid    # Use bid as low
            }
            
            return tick
            
        except Exception as e:
            logger.error(f"Error formatting tick: {str(e)}")
            raise



    def get_buffered_ticks(self) -> np.ndarray:
        """Get buffered ticks as numpy array
        
        Returns:
            np.ndarray: Buffered ticks [N, 5]
        """
        if len(self.tick_buffer) == 0:
            return np.array([])
            
        return np.array([
            [t.bid, t.ask, t.last, t.volume, t.time]
            for t in self.tick_buffer
        ])



    def _calculate_tick_volumes(self) -> None:
        """Calculate additional volume metrics"""
        try:
            # Ensure tick_volume exists
            if 'tick_volume' not in self.ticks.columns:
                raise ValueError("tick_volume column missing")
            
            # Group by minute for volume aggregation
            self.ticks['minute'] = self.ticks['timestamp'].dt.floor('1min')
            minute_volumes = self.ticks.groupby('minute')['tick_volume'].sum()
            
            # Calculate relative volumes
            self.ticks['relative_volume'] = self.ticks['tick_volume'] / self.ticks['tick_volume'].mean()
            
            # Calculate cumulative volume
            self.ticks['real_volume'] = self.ticks['tick_volume'].cumsum()
            
            # Cleanup
            self.ticks.drop('minute', axis=1, inplace=True)
            
            logger.info(
                f"Volume calculations completed:\n"
                f"- Total volume: {self.ticks['tick_volume'].sum():.2f}\n"
                f"- Average minute volume: {minute_volumes.mean():.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error calculating volumes: {str(e)}")
            raise

    def _validate_data(self) -> bool:
        """Validate cleaned tick data"""
        try:
            # Check data exists
            if len(self.ticks) == 0:
                raise ValueError("No valid data after cleaning")
                
            # Check required columns
            required_cols = {'timestamp', 'bid', 'ask', 'last', 'tick_volume', 'real_volume'}
            missing_cols = required_cols - set(self.ticks.columns)
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
                
            # Validate data types
            if not pd.api.types.is_datetime64_any_dtype(self.ticks['timestamp']):
                raise ValueError("Invalid timestamp type")
                
            # Check time sequence
            if not self.ticks['timestamp'].is_monotonic_increasing:
                raise ValueError("Timestamps not monotonically increasing")
                
            # Additional statistics
            stats = {
                'avg_spread': (self.ticks['ask'] - self.ticks['bid']).mean(),
                'min_spread': (self.ticks['ask'] - self.ticks['bid']).min(),
                'max_spread': (self.ticks['ask'] - self.ticks['bid']).max(),
                'avg_volume': self.ticks['tick_volume'].mean()
            }
            
            logger.info(
                f"Data validation passed:\n"
                f"- Rows: {len(self.ticks)}\n"
                f"- Average spread: {stats['avg_spread']:.5f}\n"
                f"- Spread range: {stats['min_spread']:.5f} - {stats['max_spread']:.5f}\n"
                f"- Average volume: {stats['avg_volume']:.2f}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise ValueError(f"Data validation failed: {str(e)}")



       

    def copy_rates_from_pos(self, symbol: str, timeframe: int, 
                           start_pos: int, count: int) -> np.ndarray:
        """Get historical rates array (for comparison)
        
        Args:
            symbol: Symbol name
            timeframe: Timeframe
            start_pos: Start position
            count: Number of bars
            
        Returns:
            np.ndarray: OHLCV data array
        """
        start_idx = self.current_idx + start_pos
        end_idx = start_idx + count
        
        if end_idx > len(self.ticks):
            return None
            
        data = self.ticks.iloc[start_idx:end_idx]
        
        # Convert to numpy array [time, open, high, low, close, tick_volume, spread, real_volume]
        return data[['timestamp', 'bid', 'high', 'low', 'ask', 
                    'tick_volume', 'spread', 'real_volume']].values
        

            
    def _validate_row(self, row: pd.Series) -> bool:
        """Validate single tick row
        
        Args:
            row: Tick data row
            
        Returns:
            bool: True if valid
        """
        try:
            # Basic validation
            if row.isna().any():
                logger.debug(f"NaN values in row")
                return False
                
            # Price validation
            if not (0 < row['bid'] < row['ask']):
                logger.debug(f"Invalid prices: bid={row['bid']}, ask={row['ask']}")
                return False
                
            # Spread validation
            spread = row['ask'] - row['bid']
            if not (0.00001 <= spread <= 0.001):
                logger.debug(f"Invalid spread: {spread}")
                return False
                
            # Volume validation
            if row['volume'] <= 0:
                logger.debug(f"Invalid volume: {row['volume']}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Row validation error: {str(e)}")
            return False
            
       
