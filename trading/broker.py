import asyncio
import logging
import json
import websockets
import aiohttp
import hmac
import hashlib
import time
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from decimal import Decimal

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    
class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class OrderRequest:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
@dataclass
class OrderResponse:
    order_id: str
    status: str
    filled_quantity: float
    average_price: float
    commission: float

class RoboforexBroker:
    """Асинхронный брокер для Roboforex с надежным подключением и обработкой ошибок"""
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str = "EURUSD",
        environment: str = "live",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        heartbeat_interval: float = 30.0
    ):
        """
        Инициализация брокера
        
        Args:
            api_key: API ключ Roboforex
            api_secret: API секрет
            symbol: Торговая пара
            environment: 'live' или 'demo'
            max_retries: Максимальное число попыток переподключения
            retry_delay: Задержка между попытками в секундах
            heartbeat_interval: Интервал проверки соединения
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.environment = environment
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.heartbeat_interval = heartbeat_interval
        
        # Base URLs
        self.base_url = "https://api.roboforex.com/live" if environment == "live" else "https://api.roboforex.com/demo"
        self.ws_url = "wss://ws.roboforex.com/live" if environment == "live" else "wss://ws.roboforex.com/demo"
        
        # Session and connection objects
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.reconnect_task: Optional[asyncio.Task] = None
        
        # State management
        self.is_connected = False
        self.last_heartbeat = 0
        self.subscriptions = set()
        
        # Price data
        self.current_bid = 0.0
        self.current_ask = 0.0
        
        # Order tracking
        self.open_orders: Dict[str, OrderRequest] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized RoboforexBroker for {symbol} in {environment} mode")

    async def connect(self) -> None:
        """Установка соединения с брокером"""
        try:
            if self.session is None:
                self.session = aiohttp.ClientSession()
            
            if self.ws is None:
                self.ws = await websockets.connect(
                    self.ws_url,
                    heartbeat=self.heartbeat_interval,
                    ping_timeout=self.heartbeat_interval * 2
                )
                
            # Authenticate
            auth_payload = self._generate_auth_payload()
            await self.ws.send(json.dumps(auth_payload))
            response = await self.ws.recv()
            
            if not self._validate_auth_response(response):
                raise ConnectionError("Authentication failed")
                
            self.is_connected = True
            logger.info("Successfully connected to Roboforex")
            
            # Start heartbeat and message handling
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self.reconnect_task = asyncio.create_task(self._handle_reconnection())
            
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            await self._handle_connection_error(e)

    async def disconnect(self) -> None:
        """Безопасное закрытие соединения"""
        try:
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
            if self.reconnect_task:
                self.reconnect_task.cancel()
                
            if self.ws:
                await self.ws.close()
            if self.session:
                await self.session.close()
                
            self.is_connected = False
            logger.info("Disconnected from Roboforex")
            
        except Exception as e:
            logger.error(f"Error during disconnection: {str(e)}")

    async def place_order(self, order: OrderRequest) -> OrderResponse:
        """
        Размещение ордера с защитой от проскальзывания
        
        Args:
            order: Параметры ордера
            
        Returns:
            OrderResponse с деталями исполнения
            
        Raises:
            ConnectionError: При проблемах с соединением
            ValueError: При неверных параметрах ордера
        """
        try:
            # Validate order parameters
            self._validate_order(order)
            
            # Get current price for slippage check
            current_price = self.current_ask if order.side == OrderSide.BUY else self.current_bid
            
            # Prepare order payload
            payload = {
                "symbol": order.symbol,
                "side": order.side.value,
                "type": order.order_type.value,
                "quantity": str(order.quantity),
                "timestamp": int(time.time() * 1000)
            }
            
            if order.price:
                payload["price"] = str(order.price)
            if order.stop_loss:
                payload["stopLoss"] = str(order.stop_loss)
            if order.take_profit:
                payload["takeProfit"] = str(order.take_profit)
                
            # Sign payload
            payload["signature"] = self._generate_signature(payload)
            
            # Send order
            async with self.session.post(f"{self.base_url}/order", json=payload) as response:
                if response.status != 200:
                    error_msg = await response.text()
                    raise ValueError(f"Order placement failed: {error_msg}")
                    
                result = await response.json()
                
            # Check for excessive slippage
            if order.order_type == OrderType.MARKET:
                filled_price = float(result["averagePrice"])
                slippage = abs(filled_price - current_price) / current_price
                
                if slippage > 0.0003:  # 3 pips maximum slippage
                    await self.cancel_order(result["orderId"])
                    raise ValueError(f"Excessive slippage detected: {slippage:.5f}")
            
            # Track order
            self.open_orders[result["orderId"]] = order
            
            return OrderResponse(
                order_id=result["orderId"],
                status=result["status"],
                filled_quantity=float(result["filledQuantity"]),
                average_price=float(result["averagePrice"]),
                commission=float(result["commission"])
            )
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """
        Отмена ордера
        
        Args:
            order_id: ID ордера для отмены
            
        Returns:
            bool: True если отмена успешна
        """
        try:
            payload = {
                "orderId": order_id,
                "timestamp": int(time.time() * 1000)
            }
            payload["signature"] = self._generate_signature(payload)
            
            async with self.session.delete(f"{self.base_url}/order/{order_id}", json=payload) as response:
                if response.status != 200:
                    error_msg = await response.text()
                    raise ValueError(f"Order cancellation failed: {error_msg}")
                    
                if order_id in self.open_orders:
                    del self.open_orders[order_id]
                    
                return True
                
        except Exception as e:
            logger.error(f"Error cancelling order {order_id}: {str(e)}")
            return False

    async def modify_order(
        self,
        order_id: str,
        new_quantity: Optional[float] = None,
        new_price: Optional[float] = None,
        new_stop_loss: Optional[float] = None,
        new_take_profit: Optional[float] = None
    ) -> bool:
        """
        Модификация существующего ордера
        
        Args:
            order_id: ID ордера
            new_quantity: Новый объем
            new_price: Новая цена
            new_stop_loss: Новый стоп-лосс
            new_take_profit: Новый тейк-профит
            
        Returns:
            bool: True если модификация успешна
        """
        try:
            payload = {
                "orderId": order_id,
                "timestamp": int(time.time() * 1000)
            }
            
            if new_quantity:
                payload["quantity"] = str(new_quantity)
            if new_price:
                payload["price"] = str(new_price)
            if new_stop_loss:
                payload["stopLoss"] = str(new_stop_loss)
            if new_take_profit:
                payload["takeProfit"] = str(new_take_profit)
                
            payload["signature"] = self._generate_signature(payload)
            
            async with self.session.put(f"{self.base_url}/order/{order_id}", json=payload) as response:
                if response.status != 200:
                    error_msg = await response.text()
                    raise ValueError(f"Order modification failed: {error_msg}")
                    
                if order_id in self.open_orders:
                    order = self.open_orders[order_id]
                    if new_quantity:
                        order.quantity = new_quantity
                    if new_price:
                        order.price = new_price
                    if new_stop_loss:
                        order.stop_loss = new_stop_loss
                    if new_take_profit:
                        order.take_profit = new_take_profit
                        
                return True
                
        except Exception as e:
            logger.error(f"Error modifying order {order_id}: {str(e)}")
            return False

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Получение открытых позиций"""
        try:
            payload = {
                "timestamp": int(time.time() * 1000)
            }
            payload["signature"] = self._generate_signature(payload)
            
            async with self.session.get(f"{self.base_url}/positions", json=payload) as response:
                if response.status != 200:
                    error_msg = await response.text()
                    raise ValueError(f"Failed to get positions: {error_msg}")
                    
                positions = await response.json()
                self.positions = {p["positionId"]: p for p in positions}
                return positions
                
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return []

    async def close_position(self, position_id: str) -> bool:
        """
        Закрытие позиции
        
        Args:
            position_id: ID позиции
            
        Returns:
            bool: True если закрытие успешно
        """
        try:
            payload = {
                "positionId": position_id,
                "timestamp": int(time.time() * 1000)
            }
            payload["signature"] = self._generate_signature(payload)
            
            async with self.session.delete(f"{self.base_url}/position/{position_id}", json=payload) as response:
                if response.status != 200:
                    error_msg = await response.text()
                    raise ValueError(f"Position closure failed: {error_msg}")
                    
                if position_id in self.positions:
                    del self.positions[position_id]
                    
                return True
                
        except Exception as e:
            logger.error(f"Error closing position {position_id}: {str(e)}")
            return False

    async def _handle_reconnection(self) -> None:
        """Обработка переподключения при обрыве связи"""
        retries = 0
        
        while True:
            try:
                if not self.is_connected:
                    logger.warning("Connection lost, attempting to reconnect...")
                    
                    for attempt in range(self.max_retries):
                        try:
                            await self.connect()
                            if self.is_connected:
                                logger.info("Successfully reconnected")
                                retries = 0
                                break
                        except Exception as e:
                            logger.error(f"Reconnection attempt {attempt + 1} failed: {str(e)}")
                            await asyncio.sleep(self.retry_delay * (attempt + 1))
                            
                    if not self.is_connected:
                        logger.error("Failed to reconnect after maximum retries")
                        
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in reconnection handler: {str(e)}")
                await asyncio.sleep(self.retry_delay)

    async def _heartbeat_loop(self) -> None:
        """Периодическая проверка соединения"""
        while True:
            try:
                if self.ws and self.is_connected:
                    await self.ws.ping()
                    response = await self.ws.recv()
                    if response == "pong":
                        self.last_heartbeat = time.time()
                    else:
                        logger.warning("Invalid heartbeat response")
                    
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {str(e)}")
                await asyncio.sleep(self.retry_delay)

    async def _message_handler(self) -> None:
        """Обработка входящих WebSocket сообщений"""
        while True:
            try:
                if not self.ws or not self.is_connected:
                    await asyncio.sleep(0.1)
                    continue
                    
                message = await self.ws.recv()
                data = json.loads(message)
                
                if "type" not in data:
                    logger.warning(f"Received message without type: {message}")
                    continue
                    
                await self._process_message(data)
                
            except websockets.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                self.is_connected = False
                
            except json.JSONDecodeError:
                logger.error(f"Failed to decode message: {message}")
                
            except Exception as e:
                logger.error(f"Error in message handler: {str(e)}")
                await asyncio.sleep(0.1)

    async def _process_message(self, data: Dict[str, Any]) -> None:
        """
        Обработка различных типов сообщений
        
        Args:
            data: Декодированное сообщение
        """
        msg_type = data["type"]
        
        try:
            if msg_type == "price":
                await self._handle_price_update(data)
                
            elif msg_type == "order":
                await self._handle_order_update(data)
                
            elif msg_type == "position":
                await self._handle_position_update(data)
                
            elif msg_type == "error":
                await self._handle_error_message(data)
                
            else:
                logger.warning(f"Unknown message type: {msg_type}")
                
        except Exception as e:
            logger.error(f"Error processing {msg_type} message: {str(e)}")

    async def _handle_price_update(self, data: Dict[str, Any]) -> None:
        """
        Обработка обновлений цен
        
        Args:
            data: Данные о ценах
        """
        try:
            symbol = data["symbol"]
            if symbol != self.symbol:
                return
                
            self.current_bid = float(data["bid"])
            self.current_ask = float(data["ask"])
            
            # Check for significant price gaps
            if self._detect_price_gap(self.current_bid, self.current_ask):
                logger.warning(f"Detected price gap: bid={self.current_bid}, ask={self.current_ask}")
                
        except Exception as e:
            logger.error(f"Error handling price update: {str(e)}")

    async def _handle_order_update(self, data: Dict[str, Any]) -> None:
        """
        Обработка обновлений статуса ордеров
        
        Args:
            data: Данные об ордере
        """
        try:
            order_id = data["orderId"]
            new_status = data["status"]
            
            if order_id in self.open_orders:
                if new_status == "FILLED":
                    filled_price = float(data["averagePrice"])
                    filled_quantity = float(data["filledQuantity"])
                    
                    # Check for unusual fills
                    original_order = self.open_orders[order_id]
                    if original_order.order_type == OrderType.MARKET:
                        expected_price = self.current_ask if original_order.side == OrderSide.BUY else self.current_bid
                        price_diff = abs(filled_price - expected_price)
                        
                        if price_diff / expected_price > 0.001:  # More than 0.1% difference
                            logger.warning(
                                f"Unusual fill price for order {order_id}: "
                                f"expected={expected_price}, filled={filled_price}"
                            )
                    
                    del self.open_orders[order_id]
                    logger.info(f"Order {order_id} filled at {filled_price} x {filled_quantity}")
                    
                elif new_status == "REJECTED":
                    reason = data.get("rejectReason", "Unknown reason")
                    logger.error(f"Order {order_id} rejected: {reason}")
                    del self.open_orders[order_id]
                    
        except Exception as e:
            logger.error(f"Error handling order update: {str(e)}")

    async def _handle_position_update(self, data: Dict[str, Any]) -> None:
        """
        Обработка обновлений позиций
        
        Args:
            data: Данные о позиции
        """
        try:
            position_id = data["positionId"]
            
            if "action" in data and data["action"] == "CLOSE":
                if position_id in self.positions:
                    del self.positions[position_id]
                    logger.info(f"Position {position_id} closed")
            else:
                self.positions[position_id] = data
                
                # Monitor position risk
                await self._check_position_risk(position_id)
                
        except Exception as e:
            logger.error(f"Error handling position update: {str(e)}")

    async def _handle_error_message(self, data: Dict[str, Any]) -> None:
        """
        Обработка сообщений об ошибках
        
        Args:
            data: Данные об ошибке
        """
        error_code = data.get("code", "UNKNOWN")
        error_msg = data.get("message", "Unknown error")
        
        logger.error(f"Received error message: [{error_code}] {error_msg}")
        
        if error_code in ["AUTH_FAILED", "SESSION_EXPIRED"]:
            self.is_connected = False
            await self.connect()  # Attempt to reconnect with fresh authentication

    async def _check_position_risk(self, position_id: str) -> None:
        """
        Проверка риска по позиции
        
        Args:
            position_id: ID позиции
        """
        try:
            if position_id not in self.positions:
                return
                
            position = self.positions[position_id]
            entry_price = float(position["entryPrice"])
            current_price = self.current_bid if position["side"] == "SELL" else self.current_ask
            position_size = float(position["quantity"])
            
            # Calculate unrealized P&L
            if position["side"] == "BUY":
                unrealized_pnl = (current_price - entry_price) * position_size
            else:
                unrealized_pnl = (entry_price - current_price) * position_size
                
            # Check risk thresholds
            account_balance = await self._get_account_balance()
            risk_percentage = abs(unrealized_pnl) / account_balance
            
            if risk_percentage > 0.02:  # More than 2% risk
                logger.warning(
                    f"High risk detected for position {position_id}: "
                    f"Risk={risk_percentage:.2%}, P&L=${unrealized_pnl:.2f}"
                )
                
        except Exception as e:
            logger.error(f"Error checking position risk: {str(e)}")

    def _detect_price_gap(self, bid: float, ask: float) -> bool:
        """
        Определение ценовых разрывов
        
        Args:
            bid: Цена покупки
            ask: Цена продажи
            
        Returns:
            bool: True если обнаружен разрыв
        """
        spread = (ask - bid) / bid
        return spread > 0.001  # More than 0.1% spread

    async def _get_account_balance(self) -> float:
        """
        Получение баланса счета
        
        Returns:
            float: Баланс счета
        """
        try:
            payload = {
                "timestamp": int(time.time() * 1000)
            }
            payload["signature"] = self._generate_signature(payload)
            
            async with self.session.get(f"{self.base_url}/account", json=payload) as response:
                if response.status != 200:
                    raise ValueError(f"Failed to get account info: {await response.text()}")
                    
                data = await response.json()
                return float(data["balance"])
                
        except Exception as e:
            logger.error(f"Error getting account balance: {str(e)}")
            return 0.0

    def _generate_signature(self, payload: Dict[str, Any]) -> str:
        """
        Генерация подписи для API запросов
        
        Args:
            payload: Данные для подписи
            
        Returns:
            str: HMAC подпись
        """
        # Sort payload by key
        sorted_payload = dict(sorted(payload.items()))
        
        # Create payload string
        payload_str = "&".join([f"{k}={v}" for k, v in sorted_payload.items()])
        
        # Generate HMAC signature
        signature = hmac.new(
            self.api_secret.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature

    def _validate_order(self, order: OrderRequest) -> None:
        """
        Проверка параметров ордера
        
        Args:
            order: Параметры ордера
            
        Raises:
            ValueError: Если параметры неверны
        """
        if order.quantity <= 0:
            raise ValueError("Order quantity must be positive")
            
        if order.order_type != OrderType.MARKET and not order.price:
            raise ValueError("Limit and Stop orders require a price")
            
        if order.stop_loss:
            if order.side == OrderSide.BUY and order.stop_loss >= self.current_bid:
                raise ValueError("Stop loss must be below current price for long positions")
            if order.side == OrderSide.SELL and order.stop_loss <= self.current_ask:
                raise ValueError("Stop loss must be above current price for short positions")
                
        if order.take_profit:
            if order.side == OrderSide.BUY and order.take_profit <= self.current_bid:
                raise ValueError("Take profit must be above current price for long positions")
            if order.side == OrderSide.SELL and order.take_profit >= self.current_ask:
                raise ValueError("Take profit must be below current price for short positions")