import asyncio
import logging
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from decimal import Decimal

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RiskLimits:
    """Лимиты риска для торговли"""
    max_position_size: float = 0.5  # Максимальный размер позиции в лотах
    max_daily_loss: float = 1000.0  # Максимальный дневной убыток в USD
    max_position_risk: float = 0.02  # Максимальный риск на позицию (2%)
    max_total_risk: float = 0.05    # Максимальный общий риск (5%)
    min_risk_reward: float = 1.5    # Минимальное соотношение риск/прибыль
    max_spread: float = 0.0003      # Максимальный спред (3 пипса)
    max_slippage: float = 0.0002    # Максимальное проскальзывание (2 пипса)

@dataclass
class PositionSizing:
    """Параметры размера позиции"""
    account_balance: float
    stop_loss_pips: float
    risk_per_trade: float
    pip_value: float = 10.0
    min_lot: float = 0.01
    max_lot: float = 5.0

class RiskManager:
    """Управление рисками для торговой системы"""
    
    def __init__(
        self,
        initial_balance: float,
        limits: Optional[RiskLimits] = None,
        check_interval: float = 1.0  # Интервал проверки в секундах
    ):
        """
        Инициализация риск-менеджера
        
        Args:
            initial_balance: Начальный баланс счета
            limits: Лимиты риска (если None, используются значения по умолчанию)
            check_interval: Интервал проверки рисков в секундах
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.limits = limits or RiskLimits()
        self.check_interval = check_interval
        
        # Tracking
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.daily_pnl = 0.0
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0
        
        # Risk stats
        self.risk_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'max_consecutive_losses': 0,
            'current_consecutive_losses': 0
        }
        
        # Historical data
        self.trade_history: List[Dict[str, Any]] = []
        self.daily_stats: Dict[str, Dict[str, float]] = {}
        
        # Monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False
        
        logger.info(
            f"Initialized RiskManager with balance ${initial_balance:.2f} "
            f"and max daily loss ${limits.max_daily_loss:.2f}"
        )

    async def start_monitoring(self) -> None:
        """Запуск мониторинга рисков"""
        if self.monitoring_task is None or self.monitoring_task.done():
            self.is_monitoring = True
            self.monitoring_task = asyncio.create_task(self._risk_monitoring_loop())
            logger.info("Started risk monitoring")

    async def stop_monitoring(self) -> None:
        """Остановка мониторинга рисков"""
        if self.monitoring_task and not self.monitoring_task.done():
            self.is_monitoring = False
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped risk monitoring")

    async def validate_trade(self, 
                           symbol: str,
                           direction: str,
                           entry_price: float,
                           stop_loss: float,
                           take_profit: float,
                           volume: float) -> Tuple[bool, str]:
        """
        Проверка торговой сделки на соответствие рискам
        
        Args:
            symbol: Торговый инструмент
            direction: Направление (buy/sell)
            entry_price: Цена входа
            stop_loss: Уровень стоп-лосса
            take_profit: Уровень тейк-профита
            volume: Объем в лотах
            
        Returns:
            Tuple[bool, str]: (разрешение на сделку, причина отказа)
        """
        try:
            # Check daily loss limit
            if self.daily_pnl <= -self.limits.max_daily_loss:
                return False, "Daily loss limit reached"
                
            # Calculate trade risk
            position_value = volume * 100000  # Standard lot
            if direction == "buy":
                risk_amount = (entry_price - stop_loss) * position_value
                reward_amount = (take_profit - entry_price) * position_value
            else:
                risk_amount = (stop_loss - entry_price) * position_value
                reward_amount = (entry_price - take_profit) * position_value
                
            risk_percent = risk_amount / self.current_balance
            
            # Validate position size
            if volume > self.limits.max_position_size:
                return False, f"Position size {volume} exceeds limit {self.limits.max_position_size}"
                
            # Validate risk percent
            if risk_percent > self.limits.max_position_risk:
                return False, f"Position risk {risk_percent:.2%} exceeds limit {self.limits.max_position_risk:.2%}"
                
            # Validate risk/reward
            risk_reward = reward_amount / risk_amount
            if risk_reward < self.limits.min_risk_reward:
                return False, f"Risk/Reward {risk_reward:.2f} below minimum {self.limits.min_risk_reward}"
                
            # Check total risk
            total_risk = self._calculate_total_risk()
            if total_risk + risk_percent > self.limits.max_total_risk:
                return False, f"Total risk {total_risk+risk_percent:.2%} would exceed limit {self.limits.max_total_risk:.2%}"
                
            # Check consecutive losses
            if self.risk_metrics['current_consecutive_losses'] >= 3:
                reduced_size = volume * 0.5
                if reduced_size < 0.01:
                    return False, "Position size too small after reduction due to consecutive losses"
                logger.warning(
                    f"Reducing position size from {volume} to {reduced_size} "
                    f"due to {self.risk_metrics['current_consecutive_losses']} consecutive losses"
                )
                volume = reduced_size
                
            return True, "Trade validated"
            
        except Exception as e:
            logger.error(f"Error validating trade: {str(e)}")
            return False, f"Validation error: {str(e)}"

    async def calculate_position_size(self,
                                   symbol: str,
                                   stop_loss_pips: float,
                                   risk_per_trade: Optional[float] = None) -> float:
        """
        Расчет оптимального размера позиции
        
        Args:
            symbol: Торговый инструмент
            stop_loss_pips: Размер стоп-лосса в пипсах
            risk_per_trade: Риск на сделку (если None, используется max_position_risk)
            
        Returns:
            float: Размер позиции в лотах
        """
        try:
            # Use default risk if not specified
            risk_per_trade = risk_per_trade or self.limits.max_position_risk
            
            # Calculate pip value for symbol
            pip_value = 10.0  # Стандартное значение для EURUSD
            
            # Calculate risk amount
            risk_amount = self.current_balance * risk_per_trade
            
            # Calculate position size
            position_size = risk_amount / (stop_loss_pips * pip_value)
            
            # Apply limits
            position_size = min(position_size, self.limits.max_position_size)
            position_size = max(position_size, 0.01)  # Minimum 0.01 lot
            
            # Reduce size based on market conditions
            if self.daily_pnl < 0:
                position_size *= 0.75  # Reduce size by 25% on losing day
                
            # Round to 2 decimal places
            position_size = round(position_size, 2)
            
            logger.debug(
                f"Calculated position size: {position_size} lots "
                f"(Risk: ${risk_amount:.2f}, Stop: {stop_loss_pips} pips)"
            )
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return 0.01  # Return minimum size on error

    def update_balance(self, new_balance: float) -> None:
        """
        Обновление баланса счета
        
        Args:
            new_balance: Новый баланс
        """
        self.current_balance = new_balance
        
        # Update peak balance and drawdown
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
        else:
            drawdown = (self.peak_balance - new_balance) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, drawdown)
            
        # Update daily P&L
        today = datetime.now().date()
        if today not in self.daily_stats:
            self.daily_stats[today] = {
                'starting_balance': self.current_balance,
                'pnl': 0.0,
                'trades': 0
            }
        self.daily_pnl = new_balance - self.daily_stats[today]['starting_balance']

    async def record_trade(self,
                         trade_id: str,
                         symbol: str,
                         direction: str,
                         entry_price: float,
                         exit_price: float,
                         volume: float,
                         pnl: float,
                         duration: timedelta) -> None:
        """
        Запись информации о завершенной сделке
        
        Args:
            trade_id: ID сделки
            symbol: Торговый инструмент
            direction: Направление сделки
            entry_price: Цена входа
            exit_price: Цена выхода
            volume: Объем в лотах
            pnl: Прибыль/убыток
            duration: Длительность сделки
        """
        try:
            trade_info = {
                'trade_id': trade_id,
                'symbol': symbol,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'volume': volume,
                'pnl': pnl,
                'duration': duration.total_seconds() / 60,  # Convert to minutes
                'timestamp': datetime.now()
            }
            
            # Update metrics
            self.risk_metrics['total_trades'] += 1
            if pnl > 0:
                self.risk_metrics['winning_trades'] += 1
                self.risk_metrics['total_profit'] += pnl
                self.risk_metrics['current_consecutive_losses'] = 0
            else:
                self.risk_metrics['losing_trades'] += 1
                self.risk_metrics['total_loss'] += abs(pnl)
                self.risk_metrics['current_consecutive_losses'] += 1
                self.risk_metrics['max_consecutive_losses'] = max(
                    self.risk_metrics['max_consecutive_losses'],
                    self.risk_metrics['current_consecutive_losses']
                )
                
            # Update daily stats
            today = datetime.now().date()
            if today in self.daily_stats:
                self.daily_stats[today]['pnl'] += pnl
                self.daily_stats[today]['trades'] += 1
                
            # Store trade
            self.trade_history.append(trade_info)
            
            # Log trade
            logger.info(
                f"Recorded trade {trade_id}: {direction} {volume} lots of {symbol} "
                f"P&L: ${pnl:.2f} Duration: {duration}"
            )
            
        except Exception as e:
            logger.error(f"Error recording trade: {str(e)}")

    async def get_risk_report(self) -> Dict[str, Any]:
        """
        Получение отчета по рискам
        
        Returns:
            Dict с метриками рисков
        """
        try:
            total_trades = self.risk_metrics['total_trades']
            win_rate = self.risk_metrics['winning_trades'] / total_trades if total_trades > 0 else 0
            
            return {
                'current_balance': self.current_balance,
                'daily_pnl': self.daily_pnl,
                'max_drawdown': self.max_drawdown,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': (
                    self.risk_metrics['total_profit'] / abs(self.risk_metrics['total_loss'])
                    if self.risk_metrics['total_loss'] != 0 else float('inf')
                ),
                'avg_win': (
                    self.risk_metrics['total_profit'] / self.risk_metrics['winning_trades']
                    if self.risk_metrics['winning_trades'] > 0 else 0
                ),
                'avg_loss': (
                    self.risk_metrics['total_loss'] / self.risk_metrics['losing_trades']
                    if self.risk_metrics['losing_trades'] > 0 else 0
                ),
                'consecutive_losses': self.risk_metrics['current_consecutive_losses'],
                'max_consecutive_losses': self.risk_metrics['max_consecutive_losses']
            }
            
        except Exception as e:
            logger.error(f"Error generating risk report: {str(e)}")
            return {}

    async def _risk_monitoring_loop(self) -> None:
        """Непрерывный мониторинг рисков"""
        while self.is_monitoring:
            try:
                # Check all positions
                for position_id, position in self.positions.items():
                    await self._check_position_risk(position_id, position)
                    
                # Check daily limits
                await self._check_daily_limits()
                
                # Update metrics
                await self._update_risk_metrics()
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {str(e)}")
                await asyncio.sleep(self.check_interval)

    async def _check_position_risk(self, position_id: str, position: Dict[str, Any]) -> None:
        """
        Проверка риска по позиции
        
        Args:
            position_id: ID позиции
            position: Данные позиции
        """
        try:
            # Calculate unrealized P&L
            entry_price = float(position['entry_price'])
            current_price = float(position['current_price'])
            volume = float(position['volume'])
            direction = position['direction']
            
            if direction == 'buy':
                unrealized_pnl = (current_price - entry_price) * volume * 100000
            else:
                unrealized_pnl = (entry_price - current_price) * volume * 100000
                
            # Calculate risk metrics
            position_risk = abs(unrealized_pnl) / self.current_balance
            
            # Check risk thresholds
            if position_risk > self.limits.max_position_risk:
                logger.warning(
                    f"Position {position_id} exceeds risk limit: "
                    f"{position_risk:.2%} > {self.limits.max_position_risk:.2%}"
                )
                
            # Check drawdown
            if unrealized_pnl < 0:
                drawdown = abs(unrealized_pnl) / self.peak_balance
                if drawdown > self.limits.max_total_risk:
                    logger.warning(
                        f"Position {position_id} exceeds drawdown limit: "
                        f"{drawdown:.2%} > {self.limits.max_total_risk:.2%}"
                    )
                    
            # Update position metrics
            position['unrealized_pnl'] = unrealized_pnl
            position['risk_percent'] = position_risk
            
        except Exception as e:
            logger.error(f"Error checking position risk: {str(e)}")

    async def _check_daily_limits(self) -> None:
        """Проверка дневных лимитов"""
        try:
            today = datetime.now().date()
            if today not in self.daily_stats:
                self.daily_stats[today] = {
                    'starting_balance': self.current_balance,
                    'pnl': 0.0,
                    'trades': 0
                }
            
            # Calculate daily P&L
            total_pnl = self.current_balance - self.daily_stats[today]['starting_balance']
            
            # Check daily loss limit
            if total_pnl <= -self.limits.max_daily_loss:
                logger.warning(
                    f"Daily loss limit reached: ${total_pnl:.2f} <= ${-self.limits.max_daily_loss:.2f}"
                )
                
            # Update daily stats
            self.daily_stats[today]['pnl'] = total_pnl
            
        except Exception as e:
            logger.error(f"Error checking daily limits: {str(e)}")

    async def _update_risk_metrics(self) -> None:
        """Обновление метрик риска"""
        try:
            # Calculate win rate
            if self.risk_metrics['total_trades'] > 0:
                win_rate = (
                    self.risk_metrics['winning_trades'] / 
                    self.risk_metrics['total_trades']
                )
            else:
                win_rate = 0.0
                
            # Calculate profit factor
            if self.risk_metrics['total_loss'] != 0:
                profit_factor = (
                    self.risk_metrics['total_profit'] /
                    abs(self.risk_metrics['total_loss'])
                )
            else:
                profit_factor = float('inf')
                
            # Calculate drawdown
            if self.current_balance < self.peak_balance:
                current_drawdown = (
                    (self.peak_balance - self.current_balance) /
                    self.peak_balance
                )
                self.max_drawdown = max(self.max_drawdown, current_drawdown)
                
            # Log metrics if significant changes
            if (win_rate < 0.4 or profit_factor < 1.0 or
                self.max_drawdown > self.limits.max_total_risk):
                logger.warning(
                    f"Risk metrics warning:\n"
                    f"Win Rate: {win_rate:.2%}\n"
                    f"Profit Factor: {profit_factor:.2f}\n"
                    f"Max Drawdown: {self.max_drawdown:.2%}"
                )
                
        except Exception as e:
            logger.error(f"Error updating risk metrics: {str(e)}")

    def _calculate_total_risk(self) -> float:
        """
        Расчет общего риска по всем позициям
        
        Returns:
            float: Общий риск в процентах
        """
        try:
            total_risk = 0.0
            
            for position in self.positions.values():
                # Get position risk
                entry_price = float(position['entry_price'])
                stop_loss = float(position['stop_loss'])
                volume = float(position['volume'])
                direction = position['direction']
                
                # Calculate risk amount
                if direction == 'buy':
                    risk_amount = (entry_price - stop_loss) * volume * 100000
                else:
                    risk_amount = (stop_loss - entry_price) * volume * 100000
                    
                # Add to total risk
                total_risk += abs(risk_amount) / self.current_balance
                
            return total_risk
            
        except Exception as e:
            logger.error(f"Error calculating total risk: {str(e)}")
            return 0.0

    def get_position_metrics(self, position_id: str) -> Dict[str, Any]:
        """
        Получение метрик по позиции
        
        Args:
            position_id: ID позиции
            
        Returns:
            Dict с метриками позиции
        """
        try:
            if position_id not in self.positions:
                return {}
                
            position = self.positions[position_id]
            metrics = {
                'unrealized_pnl': position.get('unrealized_pnl', 0.0),
                'risk_percent': position.get('risk_percent', 0.0),
                'duration': (
                    datetime.now() - position['entry_time']
                ).total_seconds() / 60,  # в минутах
                'pips': position.get('pips', 0.0),
                'volume': position.get('volume', 0.0)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting position metrics: {str(e)}")
            return {}