# C_01_en/models/feature_net.py

import os
import json
import torch
import dataclasses
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
from pathlib import Path
from .qnd_net import RotaryPositionalEmbedding
from models.qnd_net import QNDAgent
from torch.amp import autocast
from contextlib import nullcontext
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass, field
import numpy as np
from torch.cuda.amp import autocast
import traceback
import importlib.util
from collections import deque



def is_triton_available() -> bool:
    """Check if triton is available and properly installed"""
    try:
        spec = importlib.util.find_spec("triton")
        if spec is not None:
            import triton
            if hasattr(triton, '__version__'):
                logger.debug(f"Found triton version {triton.__version__}")
                return True
        return False
    except ImportError:
        return False

import torch._dynamo
torch._dynamo.config.suppress_errors = True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)



class MetricsAccumulator:
    def __init__(self):
        from collections import defaultdict, deque
        
        self.epoch_metrics = defaultdict(float)
        self.max_history = 100  # Уменьшаем размер истории
        self.window_size = 30  # Размер окна для усреднения
        self.current_count = 0
        self.metric_windows = {}  # Скользящее окно для каждой метрики
        self.metrics = defaultdict(list)
        self.running_metrics = defaultdict(float)
        self.counts = defaultdict(int)

        self.buffers = {
            'current': defaultdict(lambda: deque(maxlen=1)),
            'epoch': defaultdict(lambda: deque(maxlen=100)),
            'batch': defaultdict(lambda: deque(maxlen=10))
        }

        
        # Специальные буферы для торговых метрик
        self.trade_buffers = {
            'direction_acc': deque(maxlen=100),
            'position_acc': deque(maxlen=100),
            'price_error': deque(maxlen=100),
            'significant_moves': deque(maxlen=100)
        }
        
        # Статистика трейдинга
        self.trading_stats = {
            'total_trades': 0,
            'successful_trades': 0,
            'long_trades': 0,
            'short_trades': 0,
            'avg_position_size': 0.0
        }

        # ДОБАВИТЬ: CPU буферы
        self.cpu_buffers = {
            'running_metrics': defaultdict(lambda: deque(maxlen=1000)),
            'epoch_metrics': defaultdict(lambda: deque(maxlen=100)),
            'batch_metrics': defaultdict(lambda: deque(maxlen=50))
        }

        self.trading_stats = {
            'directional': deque(maxlen=100),
            'regime': deque(maxlen=100),
            'qnd': deque(maxlen=100),
            'risk_metrics': deque(maxlen=100),
            'position_size': deque(maxlen=100),
            'price_targets': deque(maxlen=100),
            'uncertainty': deque(maxlen=100),
            'pattern_score': deque(maxlen=100),
            'order_flow': deque(maxlen=100),
            'pnl': deque(maxlen=100),
            'win_rate': deque(maxlen=100),
            'sharpe': deque(maxlen=100)
        }
        
        self.stat_metrics = {
            'total_trades': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'skipped_trades': 0,
            'long_trades': {'count': 0, 'win_rate': 0, 'avg_pnl': 0},
            'short_trades': {'count': 0, 'win_rate': 0, 'avg_pnl': 0}
        }

        self.accuracy_buffer = {
            'direction': deque(maxlen=100),
            'position': deque(maxlen=100),
            'price': deque(maxlen=100),
            'total': deque(maxlen=100)
        }

        # Добавляем буфер для статистики
        self.prediction_stats = {
            'correct_predictions': 0,
            'total_predictions': 0,
            'position_correct': 0,
            'position_total': 0,
            'price_correct': 0,
            'price_total': 0
        }


        # 1. Добавляем буферы для расширенных метрик
        self.extended_metrics = {
            # 1.1. Метрики качества признаков
            'feature_quality': {
                'norm': deque(maxlen=100),           # Норма признаков
                'sparsity': deque(maxlen=100),       # Разреженность активаций
                'entropy': deque(maxlen=100),        # Энтропия признаков
                'disentanglement': deque(maxlen=100) # Независимость признаков
            },
            
            # 1.2. Метрики распределения предсказаний
            'prediction_stats': {
                'confidence': deque(maxlen=100),     # Уверенность предсказаний
                'calibration': deque(maxlen=100),    # Калибровка (соответствие уверенности и точности)
                'long_accuracy': deque(maxlen=100),  # Точность для длинных позиций
                'short_accuracy': deque(maxlen=100), # Точность для коротких позиций
                'auc_score': deque(maxlen=100)       # Area Under Curve ROC
            },
            
            # 1.3. Метрики торговой эффективности
            'trading_performance': {
                'win_rate': deque(maxlen=100),       # Процент успешных сделок
                'profit_factor': deque(maxlen=100),  # Отношение прибыли к убыткам
                'sharpe_ratio': deque(maxlen=100),   # Отношение доходности к волатильности
                'max_drawdown': deque(maxlen=100),   # Максимальная просадка
                'kelly_fraction': deque(maxlen=100)  # Оптимальный размер позиции
            },
            
            # 1.4. Метрики рыночных условий
            'market_conditions': {
                'volatility': deque(maxlen=100),     # Текущая волатильность рынка
                'trend_strength': deque(maxlen=100), # Сила тренда
                'liquidity': deque(maxlen=100),      # Ликвидность рынка
                'ofi': deque(maxlen=100),            # Order Flow Imbalance
                'pattern_score': deque(maxlen=100)   # Оценка ценовых паттернов
            },
            
            # 1.5. Метрики стабильности обучения
            'training_stability': {
                'gradient_norm': deque(maxlen=100),  # Норма градиентов
                'weight_norm': deque(maxlen=100),    # Норма весов
                'update_ratio': deque(maxlen=100),   # Отношение обновления к весам
                'loss': deque(maxlen=100)            # Loss
            }
        }
        
        # 1.6. Расширенные аккумуляторы для метаданных
        self.global_stats = {
            'epoch_count': 0,                # Счетчик эпох
            'batch_count': 0,                # Счетчик батчей
            'training_time': 0.0,            # Общее время обучения
            'validation_time': 0.0,          # Общее время валидации
            'best_metrics': {},              # Лучшие метрики
            'improvement_history': {}        # История улучшения метрик
        }








    def update_all_metrics(
            self, 
            features: torch.Tensor,
            outputs: Dict[str, torch.Tensor],
            batch: Dict[str, torch.Tensor],
            loss: float,
            model: torch.nn.Module = None
        ) -> Dict[str, Dict[str, float]]:
        """
        Обновление всех метрик на основе входных данных, выходов модели и батча
        
        Args:
            features: Исходные или обработанные признаки
            outputs: Выходы модели
            batch: Входной батч данных
            loss: Значение функции потерь
            model: Модель для расчета метрик стабильности (опционально)
            
        Returns:
            Dict[str, Dict[str, float]]: Обновленные метрики
        """
        try:
            # Импортируем функционал PyTorch
            import torch
            import torch.nn.functional as F
            import numpy as np
            
            metrics_result = {}
            
            # 1. Базовая метрика loss
            self.extended_metrics['training_stability']['loss'].append(loss)
            
            # 2. Метрики качества признаков
            if 'features' in outputs:
                feature_metrics = self.calculate_feature_quality(outputs['features'])
                for name, value in feature_metrics.items():
                    self.extended_metrics['feature_quality'][name].append(value)
                metrics_result['feature_quality'] = feature_metrics
            
            # 3. Метрики предсказаний
            if 'logits' in outputs and 'target' in batch:
                uncertainty = outputs.get('uncertainty', None)
                prediction_metrics = self.calculate_prediction_stats(
                    outputs['logits'], 
                    batch['target'], 
                    uncertainty
                )
                for name, value in prediction_metrics.items():
                    if name in self.extended_metrics['prediction_stats']:
                        self.extended_metrics['prediction_stats'][name].append(value)
                metrics_result['prediction_stats'] = prediction_metrics
            
            # 4. Торговые метрики
            if 'logits' in outputs and 'target' in batch:
                # Получаем прогнозы из logits
                probs = F.softmax(outputs['logits'], dim=1) if outputs['logits'].size(1) > 1 else torch.sigmoid(outputs['logits'])
                predictions = torch.argmax(probs, dim=1) if outputs['logits'].size(1) > 1 else (probs > 0.5).long()
                
                # Подготавливаем target
                targets = batch['target']
                if targets.dim() > 1 and targets.size(1) == 1:
                    targets = targets.squeeze(1)
                targets = targets.long()
                
                # Риск-метрики из batch или выходов модели
                risk_metrics = None
                if 'risk_metrics' in batch:
                    risk_metrics = batch['risk_metrics']
                elif 'risk_metrics' in outputs:
                    risk_metrics = outputs['risk_metrics']
                
                trading_metrics = self.calculate_trading_performance(
                    predictions, 
                    targets,
                    features,
                    risk_metrics
                )
                
                for name, value in trading_metrics.items():
                    if name in self.extended_metrics['trading_performance']:
                        self.extended_metrics['trading_performance'][name].append(value)
                        
                metrics_result['trading_performance'] = trading_metrics
            
            # 5. Рыночные метрики
            market_metrics = self.calculate_market_conditions(features)
            for name, value in market_metrics.items():
                if name in self.extended_metrics['market_conditions']:
                    self.extended_metrics['market_conditions'][name].append(value)
            metrics_result['market_conditions'] = market_metrics
            
            # 6. Метрики стабильности обучения
            if model is not None:
                # Норма градиентов
                grad_norm = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5
                
                # Норма весов
                weight_norm = 0.0
                for param in model.parameters():
                    weight_norm += param.norm().item() ** 2
                weight_norm = weight_norm ** 0.5
                
                # Отношение обновления
                update_ratio = grad_norm / (weight_norm + 1e-8)
                
                stability_metrics = {
                    'gradient_norm': grad_norm,
                    'weight_norm': weight_norm,
                    'update_ratio': update_ratio
                }
                
                for name, value in stability_metrics.items():
                    self.extended_metrics['training_stability'][name].append(value)
                metrics_result['training_stability'] = stability_metrics
            
            # 7. Увеличиваем счетчик батчей
            self.global_stats['batch_count'] += 1
            
            return metrics_result
            
        except Exception as e:
            import traceback
            print(f"\033[31mОшибка при обновлении метрик: {str(e)}\033[0m")
            traceback.print_exc()
            return {}
    
    def calculate_feature_quality(self, features: torch.Tensor) -> Dict[str, float]:
        """Расчет метрик качества признаков
        
        Args:
            features: Тензор признаков [batch_size, seq_len, feature_dim]
            
        Returns:
            Dict[str, float]: Метрики качества признаков
        """
        try:
            import torch
            import torch.nn.functional as F
            
            metrics = {}
            
            # 1. Норма признаков
            feature_norm = torch.norm(features, dim=-1).mean().item()
            metrics['norm'] = feature_norm
            
            # 2. Разреженность (процент близких к нулю элементов)
            sparsity = (torch.abs(features) < 0.01).float().mean().item()
            metrics['sparsity'] = sparsity
            
            # 3. Энтропия (распределение активаций)
            # Нормализуем признаки в диапазон [0, 1] для расчета энтропии
            normalized_features = torch.sigmoid(features)
            entropy = -(normalized_features * torch.log(normalized_features + 1e-10) + 
                       (1 - normalized_features) * torch.log(1 - normalized_features + 1e-10))
            metrics['entropy'] = entropy.mean().item()
            
            # 4. Независимость признаков
            # Проверяем размерность для расчета корреляций
            if features.size(0) > 1 and features.size(-1) > 1:
                flat_features = features.reshape(-1, features.size(-1))
                
                # Используем простую аппроксимацию корреляции, если тензор слишком большой
                if flat_features.size(0) > 1000:
                    sampled_indices = torch.randperm(flat_features.size(0))[:1000]
                    flat_features = flat_features[sampled_indices]
                
                # Вычисляем матрицу корреляции между признаками
                try:
                    # Вычисляем корреляцию на CPU, чтобы избежать ошибок с большими тензорами
                    features_cpu = flat_features.detach().cpu()
                    corr_matrix = torch.corrcoef(features_cpu.t())
                    
                    # Убираем диагональные элементы (корреляция с самим собой)
                    mask = ~torch.eye(corr_matrix.size(0), dtype=torch.bool)
                    
                    # Средняя абсолютная корреляция между разными признаками
                    avg_correlation = torch.abs(corr_matrix[mask]).mean().item()
                    
                    # Преобразуем в метрику независимости (1 - полная независимость, 0 - полная зависимость)
                    metrics['disentanglement'] = 1.0 - avg_correlation
                except Exception as corr_error:
                    print(f"Ошибка при расчете корреляции: {str(corr_error)}")
                    metrics['disentanglement'] = 0.5  # Значение по умолчанию
            else:
                metrics['disentanglement'] = 0.5  # Значение по умолчанию
            
            return metrics
            
        except Exception as e:
            print(f"\033[31mОшибка при расчете метрик качества признаков: {str(e)}\033[0m")
            return {'norm': 0.0, 'sparsity': 0.0, 'entropy': 0.0, 'disentanglement': 0.5}

    def calculate_prediction_stats(
            self, 
            logits: torch.Tensor, 
            targets: torch.Tensor, 
            uncertainty: torch.Tensor = None
        ) -> Dict[str, float]:
        """
        Расчет метрик качества предсказаний
        
        Args:
            logits: Выходные логиты модели [batch_size, n_classes]
            targets: Целевые значения [batch_size]
            uncertainty: Оценка неопределенности [batch_size, 1]
            
        Returns:
            Dict[str, float]: Метрики качества предсказаний
        """
        try:
            import torch
            import torch.nn.functional as F
            import numpy as np
            
            metrics = {}
            
            # Проверяем размерность targets и приводим к нужному формату
            if targets.dim() > 1 and targets.size(1) == 1:
                targets = targets.squeeze(1)
            
            # Преобразуем в long для индексации
            targets = targets.long()
            
            # Получаем вероятности и предсказания
            if logits.size(1) > 1:  # Многоклассовая задача
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)
            else:  # Бинарная задача
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
            
            # Общая точность
            accuracy = (preds == targets).float().mean().item()
            metrics['accuracy'] = accuracy
            
            # Уверенность предсказаний
            if logits.size(1) > 1:
                confidence = probs.max(dim=1)[0].mean().item()
            else:
                confidence = torch.mean(torch.max(probs, 1-probs)).item()
            metrics['confidence'] = confidence
            
            # Калибровка (насколько уверенность соответствует точности)
            if uncertainty is not None:
                errors = (preds != targets).float()
                calibration_error = F.mse_loss(uncertainty.squeeze(), errors).item()
                metrics['calibration'] = 1.0 - calibration_error  # Выше лучше
            else:
                # Если uncertainty не передан, используем разницу между уверенностью и точностью
                calibration_error = abs(confidence - accuracy)
                metrics['calibration'] = 1.0 - calibration_error
            
            # Точность для разных классов
            if logits.size(1) <= 2:  # Для бинарной классификации
                # Преобразуем в binary classification, если нужно
                if logits.size(1) == 2:
                    binary_preds = (preds == 1).long()
                    binary_targets = (targets == 1).long()
                else:
                    binary_preds = preds
                    binary_targets = targets
                
                # Маски для разных классов
                long_mask = (binary_targets == 1)
                short_mask = (binary_targets == 0)
                
                if long_mask.sum() > 0:
                    long_accuracy = (binary_preds[long_mask] == binary_targets[long_mask]).float().mean().item()
                    metrics['long_accuracy'] = long_accuracy
                else:
                    metrics['long_accuracy'] = 0.0
                    
                if short_mask.sum() > 0:
                    short_accuracy = (binary_preds[short_mask] == binary_targets[short_mask]).float().mean().item()
                    metrics['short_accuracy'] = short_accuracy
                else:
                    metrics['short_accuracy'] = 0.0
                
                # Приближенное вычисление AUC-ROC для бинарной классификации
                try:
                    if logits.size(1) == 2:
                        # Вероятности положительного класса
                        positive_probs = probs[:, 1].detach().cpu().numpy()
                    else:
                        # Вероятности положительного класса для сигмоиды
                        positive_probs = probs.detach().cpu().numpy()
                    
                    binary_targets_np = binary_targets.detach().cpu().numpy()
                    
                    # Сортируем по вероятностям
                    sorted_indices = np.argsort(positive_probs)[::-1]
                    sorted_targets = binary_targets_np[sorted_indices]
                    
                    # Кумулятивная сумма истинно положительных
                    tp_cumsum = np.cumsum(sorted_targets)
                    
                    # Кумулятивная сумма ложно положительных
                    fp_cumsum = np.cumsum(1 - sorted_targets)
                    
                    # Нормализуем для получения TPR и FPR
                    n_pos = np.sum(binary_targets_np)
                    n_neg = len(binary_targets_np) - n_pos
                    
                    if n_pos > 0 and n_neg > 0:
                        tpr = tp_cumsum / n_pos
                        fpr = fp_cumsum / n_neg
                        
                        # AUC как интеграл TPR по FPR
                        auc = np.trapz(tpr, fpr)
                        metrics['auc_score'] = float(auc)
                    else:
                        metrics['auc_score'] = 0.5  # Значение по умолчанию
                        
                except Exception as auc_error:
                    print(f"Ошибка при расчете AUC: {str(auc_error)}")
                    metrics['auc_score'] = 0.5  # Значение по умолчанию
            
            else:
                # Для многоклассовой классификации метрики по классам не рассчитываем
                metrics['long_accuracy'] = accuracy
                metrics['short_accuracy'] = accuracy
                metrics['auc_score'] = 0.5  # Не применимо напрямую
            
            return metrics
            
        except Exception as e:
            print(f"\033[31mОшибка при расчете метрик предсказаний: {str(e)}\033[0m")
            return {'accuracy': 0.0, 'confidence': 0.0, 'calibration': 0.0, 
                    'long_accuracy': 0.0, 'short_accuracy': 0.0, 'auc_score': 0.5}

    def calculate_trading_performance(
            self, 
            predictions: torch.Tensor,
            targets: torch.Tensor,
            price_data: torch.Tensor,
            risk_metrics: torch.Tensor = None
        ) -> Dict[str, float]:
        """
        Расчет метрик торговой эффективности
        
        Args:
            predictions: Предсказанные направления [batch_size]
            targets: Фактические направления [batch_size]
            price_data: Данные о ценах [batch_size, seq_len, features]
            risk_metrics: Тензор метрик риска [batch_size, 3] (kelly, drawdown, sharpe)
            
        Returns:
            Dict[str, float]: Метрики торговой эффективности
        """
        try:
            import torch
            import numpy as np
            
            metrics = {}
            
            # Win Rate (процент успешных сделок)
            win_rate = (predictions == targets).float().mean().item()
            metrics['win_rate'] = win_rate
            
            # Эмуляция торговли на исторических данных
            batch_size = predictions.size(0)
            
            # Берем цены закрытия из последних временных шагов
            try:
                close_idx = 4  # Индекс цены закрытия
                if price_data.size(-1) > close_idx:
                    close_prices = price_data[:, -1, close_idx]  # Последние цены закрытия
                else:
                    # Если структура данных другая, используем первое измерение
                    close_prices = price_data[:, -1, 0]
            except:
                # Если не получилось получить цены, используем dummy значения
                close_prices = torch.ones(batch_size, device=predictions.device)
            
            # Инициализируем списки для прибылей и убытков
            profits = []
            losses = []
            
            # Предполагаем, что торговля происходит с фиксированным TP/SL
            tp_ratio = 0.01  # 1% Take Profit
            sl_ratio = 0.005  # 0.5% Stop Loss
            
            for i in range(batch_size):
                # Начальная цена
                entry_price = close_prices[i].item()
                
                # Направление сделки (предсказанное)
                is_long = (predictions[i] == 1)
                
                # Результат (фактический)
                is_correct = (predictions[i] == targets[i])
                
                # Расчет PnL
                if is_correct:
                    # Успешная сделка
                    if is_long:
                        pnl = entry_price * tp_ratio  # Прибыль для длинной позиции
                    else:
                        pnl = entry_price * tp_ratio  # Прибыль для короткой позиции
                    profits.append(pnl)
                else:
                    # Убыточная сделка
                    if is_long:
                        pnl = -entry_price * sl_ratio  # Убыток для длинной позиции
                    else:
                        pnl = -entry_price * sl_ratio  # Убыток для короткой позиции
                    losses.append(pnl)
            
            # Profit Factor (отношение прибыли к убыткам)
            total_profit = sum(profits) if profits else 0
            total_loss = abs(sum(losses)) if losses else 1  # Избегаем деления на ноль
            
            profit_factor = total_profit / total_loss
            metrics['profit_factor'] = profit_factor
            
            # Если предоставлены метрики риска, используем их
            if risk_metrics is not None:
                # Убеждаемся, что размерность правильная
                if risk_metrics.size(0) == batch_size and risk_metrics.size(1) >= 3:
                    # Средние значения
                    metrics['kelly_fraction'] = risk_metrics[:, 0].mean().item()
                    metrics['max_drawdown'] = risk_metrics[:, 1].mean().item()
                    metrics['sharpe_ratio'] = risk_metrics[:, 2].mean().item()
                else:
                    # Значения по умолчанию
                    metrics['kelly_fraction'] = win_rate * 2 - 1  # Приблизительная оценка
                    metrics['max_drawdown'] = 0.1  # Значение по умолчанию
                    metrics['sharpe_ratio'] = 1.0  # Значение по умолчанию
            else:
                # Иначе вычисляем приближенно
                pnl_series = profits + losses
                
                # Sharpe Ratio (примерно)
                if len(pnl_series) > 1:
                    sharpe = np.mean(pnl_series) / (np.std(pnl_series) + 1e-8) * np.sqrt(252)  # Годовой
                    metrics['sharpe_ratio'] = float(sharpe)
                else:
                    metrics['sharpe_ratio'] = 0.0
                    
                # Max Drawdown (упрощенно)
                cumulative_pnl = np.cumsum(pnl_series)
                if len(cumulative_pnl) > 0:
                    peak = np.maximum.accumulate(cumulative_pnl)
                    drawdown = (peak - cumulative_pnl) / (peak + 1e-8)
                    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
                    metrics['max_drawdown'] = float(max_drawdown)
                else:
                    metrics['max_drawdown'] = 0.0
                
                # Kelly Fraction (упрощенно)
                win_prob = len(profits) / (len(profits) + len(losses) + 1e-8)
                avg_win = np.mean(profits) if profits else 0
                avg_loss = abs(np.mean(losses)) if losses else 1
                kelly = win_prob - (1 - win_prob) / (avg_win / (avg_loss + 1e-8))
                metrics['kelly_fraction'] = float(kelly)
            
            # Проверка и ограничение значений
            metrics['profit_factor'] = min(max(metrics['profit_factor'], 0.1), 10.0)
            metrics['sharpe_ratio'] = min(max(metrics['sharpe_ratio'], -3.0), 5.0)
            metrics['max_drawdown'] = min(max(metrics['max_drawdown'], 0.0), 1.0)
            metrics['kelly_fraction'] = min(max(metrics['kelly_fraction'], -1.0), 1.0)
            
            return metrics

        except Exception as e:
            print(f"\033[31mОшибка при расчете торговых метрик: {str(e)}\033[0m")
            return {'win_rate': 0.0, 'profit_factor': 1.0, 'sharpe_ratio': 0.0, 
                    'max_drawdown': 0.0, 'kelly_fraction': 0.0}





    
    def calculate_market_conditions(self, price_data: torch.Tensor) -> Dict[str, float]:
        """
        Расчет метрик рыночных условий
        
        Args:
            price_data: Данные о ценах [batch_size, seq_len, features]
            
        Returns:
            Dict[str, float]: Метрики рыночных условий
        """
        try:
            metrics = {}
            
            # 2.15. Извлекаем цены OHLCV (предполагаем стандартный порядок колонок)
            open_prices = price_data[:, :, 1]
            high_prices = price_data[:, :, 2]
            low_prices = price_data[:, :, 3]
            close_prices = price_data[:, :, 4]
            volumes = price_data[:, :, 5]
            
            # 2.16. Волатильность (ATR)
            ranges = torch.maximum(
                high_prices[:, 1:] - low_prices[:, 1:],
                torch.abs(high_prices[:, 1:] - close_prices[:, :-1])
            )
            ranges = torch.maximum(
                ranges,
                torch.abs(low_prices[:, 1:] - close_prices[:, :-1])
            )
            atr = ranges.mean(dim=1).mean().item()
            metrics['volatility'] = atr
            
            # 2.17. Сила тренда
            seq_len = close_prices.size(1)
            x = torch.arange(seq_len, device=close_prices.device).float()
            x_mean = x.mean()
            
            # Регрессия для определения наклона (тренда)
            trends = []
            for i in range(close_prices.size(0)):
                y = close_prices[i]
                y_mean = y.mean()
                
                # Коэффициент beta (наклон) для линейной регрессии
                numerator = ((x - x_mean) * (y - y_mean)).sum()
                denominator = ((x - x_mean).pow(2)).sum()
                
                slope = numerator / (denominator + 1e-8)
                # Нормализуем наклон относительно средней цены
                normalized_slope = slope / (y_mean + 1e-8)
                trends.append(normalized_slope.item())
            
            metrics['trend_strength'] = abs(np.mean(trends))
            
            # 2.18. OFI (Order Flow Imbalance)
            up_volume = torch.sum(volumes * (close_prices > open_prices).float())
            down_volume = torch.sum(volumes * (close_prices < open_prices).float())
            total_volume = up_volume + down_volume
            
            ofi = (up_volume - down_volume) / (total_volume + 1e-8)
            metrics['ofi'] = ofi.item()
            
            # 2.19. Ликвидность (приближенно через объем и спред)
            avg_volume = volumes.mean().item()
            # Используем обратное соотношение между спредом и ликвидностью
            # Спред предполагается в price_data[:, :, 9]
            if price_data.size(-1) > 9:
                spreads = price_data[:, :, 9]
                avg_spread = spreads.mean().item()
                # Ликвидность обратно пропорциональна спреду и прямо пропорциональна объему
                liquidity = avg_volume / (avg_spread + 1e-8)
            else:
                # Если спред недоступен, считаем только по объему
                liquidity = avg_volume
                
            # Нормализуем к диапазону [0, 1]
            metrics['liquidity'] = min(1.0, liquidity / 1000)
            
            # 2.20. Pattern Score (упрощенный индикатор силы паттернов)
            pattern_score = metrics['trend_strength'] * 0.5 + metrics['volatility'] * 0.3 + abs(metrics['ofi']) * 0.2
            metrics['pattern_score'] = pattern_score
            
            return metrics
            
        except Exception as e:
            print(f"\033[31mОшибка при расчете рыночных метрик: {str(e)}\033[0m")
            return {'volatility': 0.0, 'trend_strength': 0.0, 'ofi': 0.0, 
                    'liquidity': 0.0, 'pattern_score': 0.0}
    
    
    
    
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Получение сводной информации о всех метриках
        
        Returns:
            Dict: Сводная информация о метриках
        """
        try:
            summary = {}
            
            # 4.1. Обработка всех групп метрик
            for group_name, metrics_group in self.extended_metrics.items():
                summary[group_name] = {}
                
                for metric_name, values in metrics_group.items():
                    if len(values) > 0:
                        values_array = np.array(list(values))
                        
                        summary[group_name][metric_name] = {
                            'current': float(values_array[-1]) if len(values_array) > 0 else 0.0,
                            'mean': float(np.mean(values_array)),
                            'std': float(np.std(values_array)),
                            'min': float(np.min(values_array)),
                            'max': float(np.max(values_array)),
                            'trend': float(values_array[-1] - values_array[0]) if len(values_array) > 1 else 0.0
                        }
            
            # 4.2. Добавляем глобальные статистики
            summary['global_stats'] = self.global_stats
            
            # 4.3. Формируем интегральные оценки качества по группам
            summary['quality_scores'] = {}
            
            # Оценка качества признаков
            if 'feature_quality' in summary:
                feature_score = (
                    summary['feature_quality'].get('disentanglement', {}).get('current', 0.5) * 0.4 +
                    (1.0 - summary['feature_quality'].get('sparsity', {}).get('current', 0.5)) * 0.3 +
                    summary['feature_quality'].get('entropy', {}).get('current', 0.5) * 0.3
                )
                summary['quality_scores']['feature_quality'] = feature_score
            
            # Оценка качества предсказаний
            if 'prediction_stats' in summary:
                prediction_score = (
                    summary['prediction_stats'].get('accuracy', {}).get('current', 0.5) * 0.4 +
                    summary['prediction_stats'].get('calibration', {}).get('current', 0.5) * 0.3 +
                    summary['prediction_stats'].get('auc_score', {}).get('current', 0.5) * 0.3
                )
                summary['quality_scores']['prediction_quality'] = prediction_score
            
            # Оценка торговой эффективности
            if 'trading_performance' in summary:
                trading_score = (
                    summary['trading_performance'].get('win_rate', {}).get('current', 0.5) * 0.3 +
                    min(1.0, summary['trading_performance'].get('profit_factor', {}).get('current', 1.0) / 2.0) * 0.4 +
                    min(1.0, max(0.0, summary['trading_performance'].get('sharpe_ratio', {}).get('current', 0.0) / 2.0)) * 0.3
                )
                summary['quality_scores']['trading_quality'] = trading_score
            
            # Общая оценка качества
            feature_weight = 0.25
            prediction_weight = 0.35
            trading_weight = 0.4
            
            overall_quality = (
                summary['quality_scores'].get('feature_quality', 0.5) * feature_weight +
                summary['quality_scores'].get('prediction_quality', 0.5) * prediction_weight +
                summary['quality_scores'].get('trading_quality', 0.5) * trading_weight
            )
            
            summary['quality_scores']['overall_quality'] = overall_quality
            
            return summary
            
        except Exception as e:
            print(f"\033[31mОшибка при получении сводки метрик: {str(e)}\033[0m")
            if hasattr(self, 'debug') and self.debug:
                import traceback
                print(traceback.format_exc())
            return {}
    
    
    def print_metrics_dashboard(self) -> None:
        """
        Красивый вывод метрик в консоль с цветным форматированием
        """
        try:
            # Получаем сводку метрик
            summary = self.get_metrics_summary()
            
            # 5.1. Заголовок
            print("\n\033[1;36m" + "="*80 + "\033[0m")
            print("\033[1;36m" + " "*30 + "МЕТРИКИ ОБУЧЕНИЯ" + " "*30 + "\033[0m")
            print("\033[1;36m" + "="*80 + "\033[0m")
            
            # 5.2. Общая оценка качества
            overall_quality = summary.get('quality_scores', {}).get('overall_quality', 0.0)
            quality_color = "\033[1;32m" if overall_quality >= 0.7 else "\033[1;33m" if overall_quality >= 0.5 else "\033[1;31m"
            
            print(f"\n{quality_color}Общая оценка качества: {overall_quality:.2f}/1.0\033[0m")
            
            # 5.3. Метрики качества признаков
            print("\n\033[1;34m1. Качество признаков:\033[0m")
            feature_metrics = summary.get('feature_quality', {})
            for metric_name, values in feature_metrics.items():
                current = values.get('current', 0.0)
                mean = values.get('mean', 0.0)
                trend = values.get('trend', 0.0)
                
                trend_arrow = "↑" if trend > 0 else "↓" if trend < 0 else "→"
                trend_color = "\033[32m" if trend > 0 else "\033[31m" if trend < 0 else "\033[33m"
                
                print(f"  - {metric_name.ljust(15)}: {current:.4f} (среднее: {mean:.4f}) {trend_color}{trend_arrow}\033[0m")
                
            # 5.4. Метрики предсказаний
            print("\n\033[1;34m2. Метрики предсказаний:\033[0m")
            prediction_metrics = summary.get('prediction_stats', {})
            for metric_name, values in prediction_metrics.items():
                current = values.get('current', 0.0)
                mean = values.get('mean', 0.0)
                trend = values.get('trend', 0.0)
                
                trend_arrow = "↑" if trend > 0 else "↓" if trend < 0 else "→"
                trend_color = "\033[32m" if (trend > 0 and metric_name != 'uncertainty') or (trend < 0 and metric_name == 'uncertainty') else "\033[31m" if (trend < 0 and metric_name != 'uncertainty') or (trend > 0 and metric_name == 'uncertainty') else "\033[33m"
                
                print(f"  - {metric_name.ljust(15)}: {current:.4f} (среднее: {mean:.4f}) {trend_color}{trend_arrow}\033[0m")
            
            # 5.5. Торговые метрики
            print("\n\033[1;34m3. Торговые метрики:\033[0m")
            trading_metrics = summary.get('trading_performance', {})
            for metric_name, values in trading_metrics.items():
                current = values.get('current', 0.0)
                mean = values.get('mean', 0.0)
                trend = values.get('trend', 0.0)
                
                trend_arrow = "↑" if trend > 0 else "↓" if trend < 0 else "→"
                trend_color = "\033[32m" if (trend > 0 and metric_name != 'max_drawdown') or (trend < 0 and metric_name == 'max_drawdown') else "\033[31m" if (trend < 0 and metric_name != 'max_drawdown') or (trend > 0 and metric_name == 'max_drawdown') else "\033[33m"
                
                print(f"  - {metric_name.ljust(15)}: {current:.4f} (среднее: {mean:.4f}) {trend_color}{trend_arrow}\033[0m")
            
            # 5.6. Рыночные метрики
            print("\n\033[1;34m4. Рыночные метрики:\033[0m")
            market_metrics = summary.get('market_conditions', {})
            for metric_name, values in market_metrics.items():
                current = values.get('current', 0.0)
                mean = values.get('mean', 0.0)
                
                print(f"  - {metric_name.ljust(15)}: {current:.4f} (среднее: {mean:.4f})")
            
            # 5.7. Метрики стабильности обучения
            print("\n\033[1;34m5. Стабильность обучения:\033[0m")
            stability_metrics = summary.get('training_stability', {})
            for metric_name, values in stability_metrics.items():
                current = values.get('current', 0.0)
                mean = values.get('mean', 0.0)
                trend = values.get('trend', 0.0)
                
                trend_arrow = "↑" if trend > 0 else "↓" if trend < 0 else "→"
                # Для gradient_norm и update_ratio лучше, чтобы они были стабильны или уменьшались
                # А для weight_norm - чтобы увеличивались или были стабильны
                if metric_name in ['gradient_norm', 'update_ratio']:
                    trend_color = "\033[32m" if trend < 0 else "\033[31m" if trend > 0 else "\033[33m"
                else:
                    trend_color = "\033[32m" if trend > 0 else "\033[31m" if trend < 0 else "\033[33m"
                
                print(f"  - {metric_name.ljust(15)}: {current:.4f} (среднее: {mean:.4f}) {trend_color}{trend_arrow}\033[0m")
            
            # 5.8. Статистика обучения
            batch_count = summary.get('global_stats', {}).get('batch_count', 0)
            epoch_count = summary.get('global_stats', {}).get('epoch_count', 0)
            training_time = summary.get('global_stats', {}).get('training_time', 0.0)
            
            print("\n\033[1;34m6. Статистика обучения:\033[0m")
            print(f"  - Эпохи: {epoch_count}")
            print(f"  - Обработано батчей: {batch_count}")
            print(f"  - Время обучения: {training_time:.2f} сек ({training_time/60:.2f} мин)")
            
            # 5.9. Заключение
            print("\n\033[1;36m" + "="*80 + "\033[0m")
            
        except Exception as e:
            print(f"\033[31mОшибка при отображении метрик: {str(e)}\033[0m")
            if hasattr(self, 'debug') and self.debug:
                import traceback
                print(traceback.format_exc())















    def get_average_stats(self) -> Dict[str, float]:
        """Получить усредненные статистики за последние 100 батчей"""
        stats = {}
        for key, values in self.cpu_buffers['epoch_metrics'].items():
            if values:
                values_list = list(values)
                stats[key] = {
                    'mean': float(np.mean(values_list)),
                    'std': float(np.std(values_list)),
                    'min': float(np.min(values_list)),
                    'max': float(np.max(values_list))
                }
        return stats
    
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Получение всех метрик в структурированном виде
        """
        try:
            # Получаем скользящие средние
            avg_stats = self.get_average_stats()
            
            metrics_data = {
                # Текущие метрики
                'current': {
                    name: list(values)[-1] if values else 0.0
                    for name, values in self.cpu_buffers['running_metrics'].items()
                },
                
                # Средние значения по эпохе (последние 100 батчей)
                'epoch': {
                    name: avg_stats[name]['mean'] if name in avg_stats else 0.0
                    for name in self.cpu_buffers['epoch_metrics'].keys()
                },
                
                # Стандартные отклонения
                'performance': avg_stats,
                
                # Торговая статистика
                'trading': {
                    'total_trades': self.stat_metrics['total_trades'],
                    'successful_trades': self.stat_metrics['successful_trades'],
                    'win_rate': (self.stat_metrics['successful_trades'] / 
                              (self.stat_metrics['total_trades'] or 1) * 100),
                    'long_trades': self.stat_metrics['long_trades'],
                    'short_trades': self.stat_metrics['short_trades']
                }
            }
    
            print("\nStats collected:")
            print(f"- Epoch metrics window: {len(self.cpu_buffers['epoch_metrics'].get('direction_accuracy', []))}")
            print(f"- Running metrics count: {len(self.cpu_buffers['running_metrics'])}")
            print(f"- Total trades: {self.stat_metrics['total_trades']}")
    
            return metrics_data
    
        except Exception as e:
            print(f"Error in get_metrics: {str(e)}")
            return {
                'current': {},
                'epoch': {},
                'performance': {},
                'trading': {
                    'total_trades': 0,
                    'successful_trades': 0,
                    'win_rate': 0.0,
                    'long_trades': {'count': 0, 'win_rate': 0.0},
                    'short_trades': {'count': 0, 'win_rate': 0.0}
                }
            }


    def update(self, metrics: Dict[str, float]) -> None:
        """Обновление всех метрик"""
        try:
            # Обновляем все буферы метрик
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.cpu_buffers['running_metrics'][name].append(value)
                    self.cpu_buffers['epoch_metrics'][name].append(value)
                    self.cpu_buffers['batch_metrics'][name].append(value)

            # Обновляем торговые метрики
            if 'trade_type' in metrics:
                trade_type = metrics['trade_type']
                if trade_type == 'long':
                    self.stat_metrics['long_trades']['count'] += 1
                else:
                    self.stat_metrics['short_trades']['count'] += 1

                # Обновляем total_trades
                self.stat_metrics['total_trades'] += 1

                # Определяем успешность сделки
                if metrics.get('trade_result', 0) > 0.5:
                    self.stat_metrics['successful_trades'] += 1
                else:
                    self.stat_metrics['failed_trades'] += 1

            # Обновляем метрики эффективности
            for key in self.trading_stats.keys():
                if key in metrics:
                    self.trading_stats[key].append(metrics[key])

            # Логируем каждые 10 сделок
            if self.stat_metrics['total_trades'] % 10 == 0:
                self._print_statistics()

        except Exception as e:
            print(f"Error updating metrics: {str(e)}")
            print(f"Metrics content: {metrics}")

    def _print_statistics(self) -> None:
        """Вывод торговой статистики"""
        try:
            print("\n=== Trading Performance ===")
            print(f"Total Trades: {self.stat_metrics['total_trades']}")
            
            if self.stat_metrics['total_trades'] > 0:
                win_rate = (self.stat_metrics['successful_trades'] / 
                          self.stat_metrics['total_trades'] * 100)
                print(f"Win Rate: {win_rate:.1f}%")
                
                # Long trades stats
                long_trades = self.stat_metrics['long_trades']
                if long_trades['count'] > 0:
                    print(f"Long Trades: {long_trades['count']} "
                          f"(WR: {long_trades['win_rate']*100:.1f}%)")
                
                # Short trades stats
                short_trades = self.stat_metrics['short_trades']
                if short_trades['count'] > 0:
                    print(f"Short Trades: {short_trades['count']} "
                          f"(WR: {short_trades['win_rate']*100:.1f}%)")

        except Exception as e:
            print(f"Error printing statistics: {str(e)}")


    def reset(self) -> None:
        """Сброс всех метрик"""
        for buffer in self.buffers.values():
            buffer.clear()
        for buffer in self.trade_buffers.values():
            buffer.clear()
        self.trading_stats = {k: 0 for k in self.trading_stats}

    def _update_trade_stats(self, stats: Dict, result: float) -> None:
        """Обновление статистики по типу сделки"""
        stats['count'] += 1
        if result > 0:
            stats['win_rate'] = (stats['win_rate'] * (stats['count']-1) + 1) / stats['count']
        stats['avg_pnl'] = (stats['avg_pnl'] * (stats['count']-1) + result) / stats['count']





    def get_accuracy_stats(self) -> Dict[str, float]:
        """Получение текущей статистики точности"""
        try:
            stats = {}
            for key in self.accuracy_buffer.keys():
                if self.accuracy_buffer[key]:
                    values = list(self.accuracy_buffer[key])
                    stats[f"{key}_mean"] = np.mean(values)
                    stats[f"{key}_std"] = np.std(values)
                    stats[f"{key}_min"] = np.min(values)
                    stats[f"{key}_max"] = np.max(values)
            return stats
        except Exception as e:
            print(f"Error getting accuracy stats: {str(e)}")
            return {}






    def get_average(self) -> Dict[str, float]:
        """Get average from CPU buffers"""
        try:
            averages = {}
            for name, values in self.cpu_buffers['running_metrics'].items():
                if values:
                    # Вычисляем среднее на CPU
                    averages[name] = np.mean(values)
            return averages
        except Exception as e:
            logger.error(f"Error calculating averages: {str(e)}")
            return {}

    def reset_batch(self) -> None:
        """Clear CPU buffers"""
        for buffer in self.cpu_buffers.values():
            buffer.clear()
        torch.cuda.empty_cache()


    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics using numpy"""
        stats = {}
        for name, values in self.cpu_buffers['running_metrics'].items():
            if values:
                values_array = np.array(list(values))
                stats[name] = {
                    'mean': float(np.mean(values_array)),
                    'std': float(np.std(values_array)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                    'last': float(values_array[-1])
                }
        return stats

        
            
    def get_last(self) -> Dict[str, float]:
        """Get last values for all metrics"""
        return {
            name: values[-1] if values else 0.0
            for name, values in self.metrics.items()
        }





    def _cleanup_old_metrics(self):
        """Cleanup old metrics"""
        for k in list(self.metric_windows.keys()):
            while len(self.metric_windows[k]) > self.window_size:
                self.metric_windows[k].popleft()
        self.current_count = self.window_size
        


    def get_batch_metrics(self) -> Dict[str, float]:
        """Get current metrics"""
        return {k: v for k, v in self.metrics.items() if self.counts[k] > 0}

    def update_epoch(self) -> None:
        """Update epoch metrics and cleanup"""
        self.epoch_metrics.update(self.get_batch_metrics())
        self.reset_batch()


    def get_epoch_metrics(self) -> Dict[str, float]:
        """Get epoch metrics safely"""
        return dict(self.epoch_metrics)

    def __len__(self) -> int:
        """Get current metrics count"""
        return self.current_count

            
    def get_averages(self) -> Dict[str, float]:
        return {k: v/self.counts[k] for k, v in self.metrics.items()}
        


    def __str__(self) -> str:
        """String representation of current metrics"""
        metrics = self.get_averages()
        return "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])





class SELayer(nn.Module):
    """Squeeze-and-Excitation layer with fixed dimensionality"""
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        
        # Validate dimensions
        if channel % reduction != 0:
            raise ValueError(
                f"Channel {channel} must be divisible by reduction {reduction}"
            )
            
        self.channel = channel
        self.reduction = reduction
        
        # Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Two-layer bottleneck
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.GELU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, channel]
            
        Returns:
            Scaled tensor of same shape
        """
        # Validate input
        batch_size, seq_len, channels = x.shape
        if channels != self.channel:
            raise ValueError(
                f"Expected {self.channel} channels, got {channels}"
            )
            
        # [batch, seq, channel] -> [batch, channel, seq]
        x_perm = x.permute(0, 2, 1).contiguous()
        
        # Global average pooling
        y = self.avg_pool(x_perm)  # [batch, channel, 1]
        
        # Flatten for FC layers
        y = y.view(batch_size, -1)  # [batch, channel]
        
        # Channel reduction and expansion
        y = self.fc(y)  # [batch, channel]
        
        # Reshape for scaling
        y = y.view(batch_size, -1, 1)  # [batch, channel, 1]
        
        # Scale input features
        x_scaled = x_perm * y  # [batch, channel, seq]
        
        # Return to original format
        return x_scaled.permute(0, 2, 1).contiguous()  # [batch, seq, channel]

    def extra_repr(self) -> str:
        return f'channels={self.channel}, reduction={self.reduction}'

class MultiScaleConvBlock(nn.Module):
    """Enhanced multi-scale convolutional block with fixed dimensions"""
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        
        if hidden_dim % 8 != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by 8")
            
        self.hidden_dim = hidden_dim
        
        # Fixed kernel sizes
        kernel_sizes = [3, 5, 7]
        padding_sizes = [1, 2, 3]
        
        # Group size optimization for RTX 4090
        self.groups = 8
        
        # Multi-scale convolutions
        self.convs = nn.ModuleList()
        for k, p in zip(kernel_sizes, padding_sizes):
            self.convs.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim // len(kernel_sizes),
                         kernel_size=k, padding=p, groups=self.groups),
                nn.BatchNorm1d(hidden_dim // len(kernel_sizes)),
                nn.GELU(),
                nn.Dropout(dropout)
            ))
        
        # 1x1 projection for dimension restoration
        self.projection = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            
        Returns:
            Tensor of shape [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, channels = x.shape
        
        # Validate input dimensions
        if channels != self.hidden_dim:
            raise ValueError(
                f"Expected hidden_dim={self.hidden_dim}, "
                f"got channels={channels}"
            )
        
        # [batch, seq, channels] -> [batch, channels, seq]
        x = x.permute(0, 2, 1).contiguous()
        
        # Store input for residual
        identity = x
        
        # Process at each scale
        outputs = []
        for conv in self.convs:
            out = conv(x)
            outputs.append(out)
            
        # Combine scales
        out = torch.cat(outputs, dim=1)
        
        # Project back to original dimensions
        out = self.projection(out)
        
        # Residual connection
        out = out + identity
        
        # [batch, channels, seq] -> [batch, seq, channels]
        out = out.permute(0, 2, 1).contiguous()
        
        # Final normalization
        out = self.norm(out)
        
        return out

class AdaptiveNormalization(nn.Module):
    def __init__(self, feature_dim: int, momentum: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.momentum = momentum
        
        # Register buffers on the same device as model
        self.register_buffer('running_mean', torch.zeros(feature_dim))
        self.register_buffer('running_var', torch.ones(feature_dim))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with proper device handling
        
        Args:
            x: Input tensor [batch_size, sequence_length, feature_dim]
            
        Returns:
            Normalized tensor [batch_size, sequence_length, feature_dim]
        """
        if self.training:
            # Calculate stats on input device
            mean = x.mean(dim=(0, 1))
            var = x.var(dim=(0, 1), unbiased=False)
            
            # Ensure running stats are on same device
            device = x.device
            if self.running_mean.device != device:
                self.running_mean = self.running_mean.to(device)
                self.running_var = self.running_var.to(device)
                self.num_batches_tracked = self.num_batches_tracked.to(device)
            
            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            self.num_batches_tracked += 1
        else:
            mean = self.running_mean.to(x.device)
            var = self.running_var.to(x.device)
            
        # Normalize
        return (x - mean) / (var + 1e-5).sqrt()

class EnhancedLSTM(nn.Module):
    """Enhanced LSTM with properly sized gates"""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 3,
                 dropout: float = 0.1, bidirectional: bool = True) -> None:
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # Calculate actual output size
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Main LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Input gate with matching dimensions
        self.input_gate = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Sigmoid()
        )
        
        # Output gate with correct output dimension
        self.output_gate = nn.Sequential(
            nn.Linear(self.output_size, self.output_size),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, 
                h0: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with dimension validation
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            h0: Optional initial hidden state
            
        Returns:
            Tuple of:
                - Output tensor [batch_size, seq_len, output_size]
                - Tuple of final hidden states
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Validate input dimensions
        if input_dim != self.input_size:
            raise ValueError(f"Expected input size {self.input_size}, got {input_dim}")
        
        # Apply input gate
        gated_input = x * self.input_gate(x)
        
        # LSTM forward pass
        output, (hn, cn) = self.lstm(gated_input, h0)
        
        # Apply output gate with matching dimensions
        gated_output = output * self.output_gate(output)
        
        return gated_output, (hn, cn)

    def __repr__(self) -> str:
        """String representation"""
        return (f"EnhancedLSTM(input_size={self.input_size}, "
                f"hidden_size={self.hidden_size}, "
                f"num_layers={self.num_layers}, "
                f"dropout={self.dropout}, "
                f"bidirectional={self.bidirectional})")


class AdaptiveAttention(nn.Module):
    """Multi-scale adaptive attention module"""
    def __init__(self, hidden_dim: int = 3072, num_heads: int = 48, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections optimized for RTX 4090
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Rotary embeddings
        self.max_seq_len = 300
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Dropouts
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        # Pre-norm
        x = self.layer_norm(x)
        
        # Linear projections with reshape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Apply rotary embeddings
        q = self._rotate_half(q)
        k = self._rotate_half(k)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection and dropout
        out = self.out_proj(out)
        out = self.out_dropout(out)
        
        return out, attn_weights
        
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int = 3072, num_heads: int = 48, dropout: float = 0.1):
        super().__init__()
        
        assert hidden_dim % num_heads == 0, f"Hidden dim {hidden_dim} not divisible by {num_heads}"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sequence_length = 300  # Fixed length
        
        # Projection layers
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dropouts
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        
        # Initialize
        self._init_weights()
        
    def _init_weights(self) -> None:
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        
        if seq_len != self.sequence_length:
            raise ValueError(f"Expected sequence length {self.sequence_length}, got {seq_len}")
            
        # Pre-norm
        x = self.layer_norm(x)
        
        # Project queries, keys, values
        q = self.q_proj(x)  # [batch, seq, hidden]
        k = self.k_proj(x)  # [batch, seq, hidden]
        v = self.v_proj(x)  # [batch, seq, hidden]
        
        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
            
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention
        out = torch.matmul(attn_weights, v)  # [batch, heads, seq, head_dim]
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(out)
        out = self.out_dropout(out)
        
        return out, attn_weights

    def extra_repr(self) -> str:
        """String representation"""
        return (f"hidden_dim={self.hidden_dim}, "
                f"num_heads={self.num_heads}, "
                f"sequence_length={self.sequence_length}")


class FeatureExtractor(nn.Module):
    # Определяем структуру групп в соответствии с данными
    FEATURE_GROUPS = {
        'price': {'start': 0, 'end': 23},
        'volume': {'start': 23, 'end': 35},
        'momentum': {'start': 35, 'end': 58},
        'composite': {'start': 58, 'end': 73},
        'volatility': {'start': 73, 'end': 93},
        'pattern': {'start': 93, 'end': 106},
        'advanced': {'start': 106, 'end': 133}
    }
    
    FEATURE_DIMS = {
        'price': 23,
        'volume': 12,
        'momentum': 23,
        'composite': 15,
        'volatility': 20,
        'pattern': 13,
        'advanced': 27
    }

    def __init__(self, sequence_length: int = 300) -> None:
        """Initialize the FeatureExtractor

        Args:
            sequence_length: Length of input sequences, defaults to 300
        """
        super().__init__()

        # Base dimensions and parameters
        self.sequence_length = sequence_length
        self.input_dim = 133
        self.hidden_dim = 3072  # Optimized for RTX 4090
        self.num_heads = 48     # Multiple of hidden_dim/64
        self.group_hidden = 384 # Divisible by num_heads
        self.dropout = 0.1
        self.sequence_length = sequence_length
        self.debug = False

        # Добавляем константы для расчетов
        self.RISK_LIMITS = {
            'min_sharpe': -10.0,
            'max_sharpe': 10.0,
            'min_kelly': 0.0,
            'max_kelly': 1.0,
            'default_atr': 0.0002  # Базовое значение ATR для EURUSD
        }
        
        self.PRICE_LIMITS = {
            'tp_factor': 0.002,    # 0.2% от цены
            'sl_factor': 0.001,    # 0.1% от цены
            'min_spread': 0.00001,
            'max_spread': 0.0001
        }


        # Добавляем инициализацию ATR
        self.average_atr = 0.0002  # Базовое значение для EURUSD
        self.atr_period = 14
        self.atr_buffer = deque(maxlen=self.atr_period)
        
        # Feature group dimensions 
        self.group_dims = self.FEATURE_DIMS

        # Input normalization
        self.input_norm = AdaptiveNormalization(self.input_dim)


        # Копируем группы в instance attribute
        self.feature_groups = self.FEATURE_GROUPS.copy()
        
        # Validate dimensions
        total_dims = sum(self.FEATURE_DIMS.values())
        if total_dims != self.input_dim:
            raise ValueError(f"Total feature dimensions {total_dims} != input_dim {self.input_dim}")

        # Initialize group dimensions
        self.group_dims = self.FEATURE_DIMS.copy()    



        
        # Feature group processors
        self.group_processors = nn.ModuleDict({
            'price': nn.Sequential(
                AdaptiveNormalization(23),
                nn.Linear(23, self.group_hidden),
                nn.LayerNorm(self.group_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                MultiScaleConvBlock(self.group_hidden),
                SELayer(self.group_hidden)
            ),
            'volume': nn.Sequential(
                AdaptiveNormalization(12),
                nn.Linear(12, self.group_hidden),
                nn.LayerNorm(self.group_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                MultiScaleConvBlock(self.group_hidden),
                SELayer(self.group_hidden)
            ),
            'momentum': nn.Sequential(
                AdaptiveNormalization(23),
                nn.Linear(23, self.group_hidden),
                nn.LayerNorm(self.group_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                MultiScaleConvBlock(self.group_hidden),
                SELayer(self.group_hidden)
            ),
            'composite': nn.Sequential(
                AdaptiveNormalization(15),
                nn.Linear(15, self.group_hidden),
                nn.LayerNorm(self.group_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                MultiScaleConvBlock(self.group_hidden),
                SELayer(self.group_hidden)
            ),
            'volatility': nn.Sequential(
                AdaptiveNormalization(20),
                nn.Linear(20, self.group_hidden),
                nn.LayerNorm(self.group_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                MultiScaleConvBlock(self.group_hidden),
                SELayer(self.group_hidden)
            ),
            'pattern': nn.Sequential(
                AdaptiveNormalization(13),
                nn.Linear(13, self.group_hidden),
                nn.LayerNorm(self.group_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                MultiScaleConvBlock(self.group_hidden),
                SELayer(self.group_hidden)
            ),
            'advanced': nn.Sequential(
                AdaptiveNormalization(27),
                nn.Linear(27, self.group_hidden),
                nn.LayerNorm(self.group_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                MultiScaleConvBlock(self.group_hidden),
                SELayer(self.group_hidden)
            )
        })

        # Create fusion layer with correct dimensions
        total_group_dims = len(self.feature_groups) * self.group_hidden  # 7 * 384 = 2688
        self.features_fusion = nn.Sequential(
            nn.Linear(total_group_dims, self.hidden_dim),  # 2688 -> 3072
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
        # Features projection
        self.features_projection = nn.Sequential(
            nn.Linear(len(self.feature_groups) * self.group_hidden, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
        # Temporal processors
        self.temporal_scales = [1, 5, 15, 30, 60, 120, 180, 240]
        self.temporal_processors = nn.ModuleList([
            TemporalAttention(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=0.1
            ) for _ in self.temporal_scales
        ])
    
        # Feature processing network
        self.feature_net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # LSTM с увеличенной размерностью
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim//2,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True
        )
        
        # Feature processor - добавляем явно
        self.feature_processor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )        
    
        # Attention blocks
        self.attention_blocks = nn.ModuleList([
            AdaptiveAttention(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=0.1
            ) for _ in range(8)
        ])
    
        # Reconstruction heads
        self.reconstruction_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.LayerNorm(self.hidden_dim // 2),
                nn.GELU(),
                nn.Linear(self.hidden_dim // 2, dim)
            ) for name, dim in {
                'price': 23,
                'volume': 12,
                'momentum': 23,
                'composite': 15,
                'volatility': 20,
                'pattern': 13,
                'advanced': 27
            }.items()
        })

        # Output heads
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.LayerNorm(self.hidden_dim//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim//2, 2)  # Binary classification
        )
    
        self.uncertainty = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//4),
            nn.LayerNorm(self.hidden_dim//4),
            nn.GELU(),
            nn.Linear(self.hidden_dim//4, 1),
            nn.Sigmoid()
        )
    
        self.position_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//4),
            nn.LayerNorm(self.hidden_dim//4),
            nn.GELU(),
            nn.Linear(self.hidden_dim//4, 1),
            nn.Sigmoid()
        )
    
        # Volatility prediction
        self.volatility = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.LayerNorm(self.hidden_dim//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim//2, 1),
            nn.Softplus()
        )
    
        # Risk metrics
        self.risk_metrics = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.LayerNorm(self.hidden_dim//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim//2, 3)  # [kelly_fraction, max_drawdown, sharpe]
        )
    
        # Price targets
        self.price_targets = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.LayerNorm(self.hidden_dim//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim//2, 2)  # [take_profit, stop_loss]
        )

        # Добавляем classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.LayerNorm(self.hidden_dim//2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim//2, 2)  # Binary classification
        )

        # Добавляем uncertainty estimation
        self.uncertainty_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim//4),
            nn.LayerNorm(self.hidden_dim//4),
            nn.GELU(),
            nn.Linear(self.hidden_dim//4, 1),
            nn.Sigmoid()
        )

        if torch.cuda.is_available():
            # Enable TF32
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
            # Basic optimizations only
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.cache_size_limit = 512
            
            # No extra options in compile
            try:
                # Simple compilation without extra options
                self.to('cuda').to(memory_format=torch.channels_last)
            except Exception as e:
                logger.warning(f"Device optimization failed: {str(e)}")

        # Enable gradient checkpointing
        self.use_checkpointing = True
        self.memory_threshold = 0.8

        # Set memory limit
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.95)

        # Statistics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
    
        # Initialize weights

        self._init_weights()
    


        # Вместо автоматической компиляции
        #self.compile_model()
    
        # Validate initialization
        self._validate_initialization()
    
        logger.info(
            f"Initialized FeatureExtractor on {self.device}:\n"
            f"- Input dim: {self.input_dim}\n"
            f"- Hidden dim: {self.hidden_dim}\n"
            f"- Sequence length: {self.sequence_length}\n"
            f"- Num heads: {self.num_heads}\n"
            f"- Group hidden: {self.group_hidden}"
        )


    def _check_memory(self) -> bool:
        """Check if memory cleanup needed"""
        if not torch.cuda.is_available():
            return False
            
        allocated = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
        return allocated > self.memory_threshold

    def _feature_processing_step(self, x: torch.Tensor) -> torch.Tensor:
        """Single step of feature processing with checkpointing"""
        def _process(x: torch.Tensor) -> torch.Tensor:
            # Feature extraction
            features = self.feature_net(x)
            return features
            
        if self.use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                _process, 
                x,
                use_reentrant=False,
                preserve_rng_state=True
            )
        return _process(x)



    def _lstm_processing_step(self, features: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """LSTM processing with type control"""
        def _process(features: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            # Ensure input type
            features = features.to(dtype=torch.float32)
            
            # LSTM processing
            lstm_out, (hidden, cell) = self.lstm(features)
            
            # Ensure output types
            lstm_out = lstm_out.to(dtype=torch.float32)
            hidden = hidden.to(dtype=torch.float32)
            cell = cell.to(dtype=torch.float32)
            
            return lstm_out, (hidden, cell)
            
        if self.use_checkpointing and self.training:
            with torch.set_grad_enabled(True):
                # Process with checkpointing
                outputs = torch.utils.checkpoint.checkpoint(
                    _process,
                    features,
                    use_reentrant=False,
                    preserve_rng_state=True
                )
                return outputs
        
        # Direct processing without checkpointing
        return _process(features)



    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None, 
                mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Forward pass with proper state and mask handling"""
        try:
            batch_size, seq_len, feat_dim = x.shape
            device = x.device

            # Validate dimensions
            if seq_len != self.sequence_length or feat_dim != self.input_dim:
                raise ValueError(f"Expected shape ({batch_size}, {self.sequence_length}, {self.input_dim})")

            # Initialize state if not provided
            if state is None:
                state = torch.zeros(batch_size, self.hidden_dim, device=device)

            # Initialize mask if not provided  
            if mask is None:
                mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

            # Process feature groups
            group_outputs = {}
            for name, processor in self.group_processors.items():
                start = self.feature_groups[name]['start']
                end = self.feature_groups[name]['end']
                group_input = x[..., start:end]
                group_outputs[name] = processor(group_input)

            # Combine features
            combined = torch.cat(list(group_outputs.values()), dim=-1)
            features = self.features_fusion(combined)

            # Process through LSTM
            with torch.amp.autocast('cuda'):
                lstm_out, (hn, cn) = self.lstm(features)
                final_state = lstm_out[:, -1]

                # Model outputs with exact shape control
                logits = self.classifier(final_state)  # [batch_size, 2]
                probs = F.softmax(logits, dim=-1)     # [batch_size, 2]
                
                # Uncertainty and position size
                uncertainty = self.uncertainty(final_state).view(batch_size, 1)
                position_size = self.position_net(final_state).unsqueeze(-1)

                # Calculate metrics
                close_prices = x[..., self.feature_groups['price']['start'] + 3]
                metrics = self._calculate_metrics(features, close_prices)

                return {
                    'logits': logits,
                    'probabilities': probs,
                    'uncertainty': uncertainty,
                    'position_size': position_size,
                    'metrics': metrics,
                    'features': features,
                    'state': hn,
                    'mask': mask
                }

        except Exception as e:
            logger.error(f"Forward pass error: {str(e)}")
            if self.debug:
                logger.error(traceback.format_exc())
            raise

    def _validate_initialization(self) -> None:
        """Validate model initialization"""
        try:
            x = torch.randn(2, self.sequence_length, self.input_dim, device=self.device)
            
            with torch.no_grad():
                outputs = self.forward(x)
                
                expected_shapes = {
                    'features': (2, self.sequence_length, self.hidden_dim),
                    'logits': (2, 2),
                    'uncertainty': (2, 1)
                }
                
                for name, shape in expected_shapes.items():
                    if outputs[name].shape != shape:
                        raise ValueError(f"Wrong shape for {name}: {outputs[name].shape}, expected {shape}")
                    # Verify device
                    if outputs[name].device != self.device:
                        raise ValueError(f"{name} on wrong device: {outputs[name].device}, expected {self.device}")

            logger.debug("Model initialization validated successfully")

        except Exception as e:
            logger.error(f"Initialization validation failed: {str(e)}")
            if self.debug:
                logger.error(traceback.format_exc())
            raise



    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        try:
            # 1. Вывод для отладки
            print("\033[32m[TRAIN_STEP] Начало метода train_step FeatureExtractor\033[0m")
            
            # 2. Перемещение батча на устройство
            features = batch['features'].to(self.device)
            
            # 3. Вывод информации о размерности features
            print(f"\033[32m[TRAIN_STEP] Shape features: {features.shape}\033[0m")
            
            # 4. Получение целевой переменной с проверками
            if 'targets' in batch:
                targets = batch['targets'].to(self.device)
                print(f"\033[32m[TRAIN_STEP] Используем ключ 'targets', shape: {targets.shape}, dtype: {targets.dtype}\033[0m")
            elif 'target' in batch:
                targets = batch['target'].to(self.device)
                print(f"\033[32m[TRAIN_STEP] Используем ключ 'target', shape: {targets.shape}, dtype: {targets.dtype}\033[0m")
            else:
                targets = None
                print("\033[31m[TRAIN_STEP] ВНИМАНИЕ: В батче отсутствуют ключи 'targets' и 'target'\033[0m")
            
            # 5. Forward pass
            outputs = self(features)
            print(f"\033[32m[TRAIN_STEP] Forward pass выполнен, ключи outputs: {list(outputs.keys())}\033[0m")
            
            # 6. Вывод размерности логитов
            if 'logits' in outputs:
                print(f"\033[32m[TRAIN_STEP] Размерность logits: {outputs['logits'].shape}\033[0m")
            
            # 7. Проверка и преобразование размерностей targets
            if targets is not None:
                # Проверка размерности targets и приведение к 1D если необходимо
                if targets.dim() > 1:
                    print(f"\033[33m[TRAIN_STEP] Сжатие размерности targets: {targets.shape} -> ", end="")
                    targets = targets.squeeze()
                    print(f"{targets.shape}\033[0m")
                
                # Проверка типа targets и приведение к long если необходимо
                if targets.dtype != torch.long:
                    print(f"\033[33m[TRAIN_STEP] Приведение targets к типу long: {targets.dtype} -> torch.long\033[0m")
                    targets = targets.long()
                
                # 8. Расчет функции потерь
                try:
                    loss = F.cross_entropy(outputs['logits'], targets)
                    print(f"\033[32m[TRAIN_STEP] Расчет loss выполнен успешно: {loss.item():.6f}\033[0m")
                except Exception as loss_e:
                    print(f"\033[31m[TRAIN_STEP] Ошибка при расчете loss: {str(loss_e)}\033[0m")
                    print(f"\033[31m[TRAIN_STEP] logits shape: {outputs['logits'].shape}, targets shape: {targets.shape}\033[0m")
                    # Возвращаем фиктивное значение loss
                    loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                # 9. Обработка случая отсутствия targets
                print("\033[33m[TRAIN_STEP] Отсутствует targets, используем нулевой loss\033[0m")
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            # 10. Расчет метрик
            with torch.no_grad():
                predictions = outputs['logits'].argmax(dim=-1)
                metrics = {
                    'loss': loss.item(),
                    'accuracy': (predictions == targets).float().mean().item() if targets is not None else 0.0
                }
            
            print(f"\033[32m[TRAIN_STEP] Метод train_step завершен, loss: {metrics['loss']:.6f}, accuracy: {metrics['accuracy']:.4f}\033[0m")
            
            # 11. Возвращаем результат
            return {
                'loss': loss,
                'metrics': metrics,
                'outputs': outputs
            }
                
        except Exception as e:
            print(f"\033[31m[TRAIN_STEP] Ошибка в train_step: {str(e)}\033[0m")
            # Выводим стек вызовов в режиме отладки
            if hasattr(self, 'debug') and self.debug:
                import traceback
                print(traceback.format_exc())
            # Создаем dummy loss для продолжения обучения
            dummy_loss = torch.tensor(1.0, device=self.device, requires_grad=True)
            return {
                'loss': dummy_loss,
                'metrics': {'loss': 1.0, 'error': str(e)},
                'outputs': {'logits': torch.zeros((1, 2), device=self.device)}
            }



    @torch.jit.ignore
    def _create_group_processor(self, input_dim: int) -> nn.Sequential:
        """Create feature group processor with proper compilation settings"""
        return nn.Sequential(
            AdaptiveNormalization(input_dim),
            nn.Linear(input_dim, self.group_hidden),
            nn.LayerNorm(self.group_hidden),
            nn.GELU(),
            nn.Dropout(self.dropout),
            MultiScaleConvBlock(self.group_hidden, self.dropout),
            SELayer(self.group_hidden)
        )


    def _compute_metrics(self, outputs: Dict[str, torch.Tensor], 
                        batch: Dict[str, torch.Tensor],
                        loss: torch.Tensor) -> Dict[str, float]:
        """Compute training metrics including loss
        
        Args:
            outputs: Model outputs
            batch: Input batch
            loss: Calculated loss
            
        Returns:
            Dict of computed metrics
        """
        metrics = {}
        predictions = outputs['logits'].argmax(dim=-1)
        
        # Add loss to metrics
        metrics['loss'] = loss.item()
        
        if 'targets' in batch:
            targets = batch['targets']
            
            # Basic metrics
            accuracy = (predictions == targets).float().mean()
            metrics['accuracy'] = accuracy.item()
            
            # Confusion matrix metrics
            tp = ((predictions == 1) & (targets == 1)).float().sum()
            fp = ((predictions == 1) & (targets == 0)).float().sum()
            tn = ((predictions == 0) & (targets == 0)).float().sum()
            fn = ((predictions == 0) & (targets == 1)).float().sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            metrics.update({
                'precision': precision.item(),
                'recall': recall.item(),
                'f1_score': f1_score.item()
            })

        # Uncertainty metrics
        metrics['uncertainty'] = outputs['uncertainty'].mean().item()
        
        # Feature statistics
        metrics['feature_mean'] = outputs['features'].mean().item()
        metrics['feature_std'] = outputs['features'].std().item()
        
        return metrics





    def _calculate_price_metrics(self, prices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate price based metrics safely"""
        try:
            batch_size = prices.shape[0]
            device = prices.device
            
            # Calculate basic metrics with shape checking
            returns = torch.zeros_like(prices, device=device)
            returns[:, 1:] = (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1]
            
            # Volatility calculation
            volatility = torch.std(returns, dim=1)  # [batch_size]
            
            # Trend calculation 
            ma_fast = torch.mean(prices[:, -20:], dim=1)  # [batch_size]
            ma_slow = torch.mean(prices, dim=1)  # [batch_size]
            trend = (ma_fast - ma_slow) / ma_slow  # [batch_size]
            
            # Momentum calculation
            momentum = prices[:, -1] - prices[:, 0]  # [batch_size]
            momentum = momentum / prices[:, 0]  # [batch_size]
            
            return {
                'volatility': volatility,
                'trend': trend,
                'momentum': momentum
            }
            
        except Exception as e:
            logger.error(f"Price metrics error: {str(e)}")
            return {
                'volatility': torch.zeros(batch_size, device=device),
                'trend': torch.zeros(batch_size, device=device), 
                'momentum': torch.zeros(batch_size, device=device)
            }


    def _format_metrics(self, metrics_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Convert tensor metrics to dict of floats"""
        try:
            formatted = {}
            for name, tensor in metrics_dict.items():
                # Handle different tensor shapes
                if tensor.dim() == 0:
                    formatted[name] = tensor.item()
                elif tensor.dim() == 1:
                    # Take first element for batched metrics
                    formatted[name] = tensor[0].item()
                else:
                    # For higher dimensions, take mean
                    formatted[name] = tensor.mean().item()
            
            return formatted

        except Exception as e:
            logger.error(f"Error formatting metrics: {str(e)}")
            return {
                'volatility': 0.0,
                'trend_strength': 0.0,
                'momentum': 0.0,
                'uncertainty': 0.5,
                'risk': 0.0,
                'reward': 0.0
            }


                
        except Exception as e:
            logger.error(f"Initialization validation failed: {str(e)}")
            raise

    def _validate_dimensions(self, tensor: torch.Tensor, expected_shape: tuple) -> None:
        if tensor.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {tensor.shape}")


    def _calculate_metrics(self, features: torch.Tensor, prices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Calculate metrics preserving batch dimension"""
        try:
            batch_size = features.shape[0]
            device = features.device
            
            # Returns calculation
            returns = torch.zeros_like(prices, device=device)
            returns[:, 1:] = (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1]
            
            # Calculate metrics preserving batch dimension
            volatility = torch.std(returns, dim=1, keepdim=True)  # [batch_size, 1]
            
            ma_fast = torch.mean(prices[:, -20:], dim=1, keepdim=True)  # [batch_size, 1]
            ma_slow = torch.mean(prices, dim=1, keepdim=True)  # [batch_size, 1]
            trend = (ma_fast - ma_slow) / ma_slow  # [batch_size, 1]
            
            # Momentum
            momentum = ((prices[:, -1] - prices[:, 0]) / prices[:, 0]).unsqueeze(-1)  # [batch_size, 1]

            metrics = {
                'volatility': volatility,                                    # [batch_size, 1]
                'trend_strength': trend,                                     # [batch_size, 1] 
                'momentum': momentum,                                        # [batch_size, 1]
                'uncertainty': torch.ones(batch_size, 1, device=device) * 0.5  # [batch_size, 1]
            }

            return metrics

        except Exception as e:
            logger.error(f"Metrics calculation error: {str(e)}")
            return {
                'volatility': torch.zeros(batch_size, 1, device=device),
                'trend_strength': torch.zeros(batch_size, 1, device=device),
                'momentum': torch.zeros(batch_size, 1, device=device),
                'uncertainty': torch.ones(batch_size, 1, device=device) * 0.5
            }


 

    def _calculate_safe_metrics(self, features: torch.Tensor, prices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Safe metrics calculation with dimension checks"""
        try:
            batch_size = features.shape[0]
            device = features.device

            # Calculate returns
            returns = torch.zeros_like(prices, device=device)
            returns[:, 1:] = (prices[:, 1:] - prices[:, :-1]) / prices[:, :-1]

            # Basic metrics
            volatility = torch.std(returns, dim=1)
            trend = self._calculate_trend_strength(prices)
            momentum = self._calculate_momentum(prices)

            metrics = {
                'volatility': volatility,
                'trend_strength': trend,
                'momentum': momentum,
                'returns_mean': returns.mean(dim=1),
                'returns_std': returns.std(dim=1)
            }

            # Validate outputs
            for k, v in metrics.items():
                if v.shape[0] != batch_size:
                    raise ValueError(f"Invalid shape for {k}: {v.shape}")

            return metrics

        except Exception as e:
            logger.error(f"Metrics calculation error: {str(e)}")
            # Return zeros with correct shape
            return {
                'volatility': torch.zeros(batch_size, device=device),
                'trend_strength': torch.zeros(batch_size, device=device),
                'momentum': torch.zeros(batch_size, device=device),
                'returns_mean': torch.zeros(batch_size, device=device),
                'returns_std': torch.zeros(batch_size, device=device)
            }






    def _calculate_trend_strength(self, prices: torch.Tensor) -> torch.Tensor:
        """Calculate trend strength indicator"""
        ma_fast = torch.nn.functional.avg_pool1d(
            prices.unsqueeze(1), kernel_size=20
        ).squeeze(1)
        ma_slow = torch.nn.functional.avg_pool1d(
            prices.unsqueeze(1), kernel_size=50
        ).squeeze(1)
        
        trend_strength = (ma_fast - ma_slow).abs() / ma_slow
        return trend_strength[:, -1]



    def _calculate_risk_metrics(self, features: torch.Tensor, close_prices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate risk metrics with proper validation
        
        Args:
            features: Feature tensor [batch_size, seq_len, feature_dim]
            close_prices: Close prices [batch_size, seq_len]
            
        Returns:
            Dict with risk metrics
        """
        try:
            device = close_prices.device
            batch_size = close_prices.shape[0]
            
            # Calculate returns with padding for first element
            returns = torch.zeros_like(close_prices, device=device)
            returns[:, 1:] = (close_prices[:, 1:] - close_prices[:, :-1]) / close_prices[:, :-1]
            
            # Validate returns
            returns = torch.nan_to_num(returns, 0.0)
            returns = torch.clamp(returns, -0.1, 0.1)  # Ограничиваем экстремальные значения
            
            # Kelly Fraction
            mean_return = returns.mean(dim=1)
            var_return = returns.var(dim=1) + 1e-8
            kelly = mean_return / var_return
            kelly = torch.clamp(kelly, 
                              self.RISK_LIMITS['min_kelly'], 
                              self.RISK_LIMITS['max_kelly'])
            
            # Max Drawdown
            cumulative = torch.cumprod(1 + returns, dim=1)
            running_max, _ = torch.cummax(cumulative, dim=1)
            drawdowns = (cumulative - running_max) / running_max
            max_drawdown = drawdowns.min(dim=1)[0]
            
            # Sharpe Ratio
            rf_daily = torch.full((batch_size,), 0.02/252, device=device)
            excess_returns = returns - rf_daily.unsqueeze(1)
            mean_excess = excess_returns.mean(dim=1)
            std_excess = excess_returns.std(dim=1) + 1e-8
            
            sharpe = torch.sqrt(torch.tensor(252., device=device)) * (mean_excess / std_excess)
            sharpe = torch.clamp(sharpe, 
                               self.RISK_LIMITS['min_sharpe'], 
                               self.RISK_LIMITS['max_sharpe'])
            
            return {
                'kelly_fraction': kelly.detach(),
                'max_drawdown': max_drawdown.detach(),
                'sharpe_ratio': sharpe.detach()
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {
                'kelly_fraction': torch.zeros(batch_size, device=device),
                'max_drawdown': torch.zeros(batch_size, device=device),
                'sharpe_ratio': torch.zeros(batch_size, device=device)
            }

    def _calculate_price_targets(self, 
                               features: torch.Tensor,
                               current_price: torch.Tensor,
                               direction: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate price targets with proper direction handling"""
        try:
            device = features.device
            batch_size = features.shape[0]
            
            # Calculate ATR
            high = features[..., 1]  # High prices
            low = features[..., 2]   # Low prices
            close = features[..., 3]  # Close prices
            
            tr1 = high[..., 1:] - low[..., 1:]
            tr2 = torch.abs(high[..., 1:] - close[..., :-1])
            tr3 = torch.abs(low[..., 1:] - close[..., :-1])
            
            true_range = torch.maximum(tr1, torch.maximum(tr2, tr3))
            atr = torch.mean(true_range, dim=1)  # [batch_size]
            
            # Validate ATR
            atr = torch.where(
                torch.isfinite(atr),
                atr,
                torch.full_like(atr, self.RISK_LIMITS['default_atr'])
            )
            
            # Calculate target distances
            tp_distance = torch.maximum(
                atr * 2.0,
                current_price * self.PRICE_LIMITS['tp_factor']
            )
            
            sl_distance = torch.maximum(
                atr,
                current_price * self.PRICE_LIMITS['sl_factor']
            )
            
            # Get direction sign (-1 or 1)
            dir_sign = torch.where(direction > 0, 1.0, -1.0)
            
            # Calculate targets using sign
            take_profit = current_price + (tp_distance * dir_sign)
            stop_loss = current_price - (sl_distance * dir_sign)
            
            # Validate outputs
            take_profit = torch.where(
                torch.isfinite(take_profit),
                take_profit,
                current_price * (1 + self.PRICE_LIMITS['tp_factor'] * dir_sign)
            )
            
            stop_loss = torch.where(
                torch.isfinite(stop_loss),
                stop_loss,
                current_price * (1 - self.PRICE_LIMITS['sl_factor'] * dir_sign)
            )
            
            return take_profit, stop_loss
            
        except Exception as e:
            logger.error(f"Error calculating price targets: {str(e)}")
            return current_price * 1.002, current_price * 0.998

    def _calculate_momentum(self, prices: torch.Tensor) -> torch.Tensor:
        """Calculate price momentum with proper dimensions"""
        try:
            # Get sequence length from prices
            seq_len = prices.shape[1]
            
            # Create proper length range
            weights = torch.arange(seq_len, device=prices.device, dtype=prices.dtype)
            
            # Calculate returns
            returns = torch.diff(prices, dim=1)  # [batch, seq_len-1]
            
            # Adjust weights to match returns length
            weights = weights[:returns.shape[1]]
            
            # Calculate weighted sum
            momentum = torch.sum(returns * weights, dim=1)
            
            return momentum
            
        except Exception as e:
            logger.error(f"Error calculating momentum: {str(e)}")
            return torch.zeros_like(prices[:, 0])

    


    def compile_model(self) -> None:
        """Compile model with proper backend selection"""
        if torch.cuda.is_available():
            try:
                # Avoid inductor if triton not available
                backend = "cudnn" if not is_triton_available() else "inductor"
                
                self.compile(
                    backend=backend,
                    mode="reduce-overhead",
                    fullgraph=False,
                    dynamic=True
                )
            except Exception as e:
                logger.warning(f"Model compilation failed: {str(e)}")
                
    def is_triton_available() -> bool:
        """Check if triton is available and properly installed"""
        try:
            spec = importlib.util.find_spec("triton")
            if spec is not None:
                import triton
                if hasattr(triton, '__version__'):
                    logger.debug(f"Found triton version {triton.__version__}")
                    return True
            return False
        except ImportError:
            return False


    def _process_features(self, x: torch.Tensor) -> torch.Tensor:
        """Process input features"""
        # Input normalization
        x = self.input_norm(x)
        
        features = []
        for name, group in self.feature_groups.items():
            # Extract and process features
            start, end = group['start'], group['end']
            group_input = x[..., start:end]
            group_features = self.group_processors[name](group_input)
            features.append(group_features)

        # Combine features
        combined = torch.cat(features, dim=-1)
        return self.fusion(combined)



    def _init_weights(self) -> None:
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)



    def _validate_output_dimensions(self, outputs: Dict[str, torch.Tensor]) -> None:
        """Validate output dimensions
        
        Args:
            outputs: Dictionary containing model outputs with tensors
            
        Raises:
            ValueError: If dimensions don't match expected sizes
        """
        try:
            batch_size = 2  # Test batch size
            
            # Validate features
            features = outputs.get('features')
            if features is None:
                raise ValueError("Missing 'features' in outputs")
                
            expected_feature_shape = (
                batch_size,
                self.sequence_length,
                len(self.FEATURE_GROUPS) * self.group_hidden
            )
            if features.shape != expected_feature_shape:
                raise ValueError(
                    f"Invalid features shape: got {features.shape}, "
                    f"expected {expected_feature_shape}"
                )
                
            # Validate group features
            group_features = outputs.get('group_features', {})
            for name, tensor in group_features.items():
                expected_shape = (batch_size, self.sequence_length, self.group_hidden)
                if tensor.shape != expected_shape:
                    raise ValueError(
                        f"Invalid shape for group {name}: "
                        f"got {tensor.shape}, expected {expected_shape}"
                    )
                    
            # Validate group inputs
            group_inputs = outputs.get('group_inputs', {})
            for name, tensor in group_inputs.items():
                expected_dim = self.FEATURE_DIMS[name]
                if tensor.shape[-1] != expected_dim:
                    raise ValueError(
                        f"Invalid input dimension for group {name}: "
                        f"got {tensor.shape[-1]}, expected {expected_dim}"
                    )
                    
            # Validate temporal outputs
            temporal_attention = outputs.get('temporal_attention', [])
            if len(temporal_attention) != len(self.temporal_scales):
                raise ValueError(
                    f"Wrong number of temporal attention outputs: "
                    f"got {len(temporal_attention)}, "
                    f"expected {len(self.temporal_scales)}"
                )
                
            # Memory check for RTX 4090
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1e9
                memory_reserved = torch.cuda.memory_reserved() / 1e9
                if memory_allocated > 0.9 * memory_reserved:
                    logger.warning(
                        f"High GPU memory usage: {memory_allocated:.1f}GB "
                        f"of {memory_reserved:.1f}GB reserved"
                    )
                    
            logger.info("Output dimensions validated successfully")
            
        except Exception as e:
            logger.error(f"Output validation failed: {str(e)}")
            if self.debug:
                logger.error(traceback.format_exc())
            raise



    def _validate_outputs(self, outputs: Dict[str, torch.Tensor], batch_size: int) -> None:
        expected_shapes = {
            'features': (batch_size, self.sequence_length, self.hidden_dim),
            'group_features': dict,
            'temporal_attention': list
        }

        for name, expected in expected_shapes.items():
            if name not in outputs:
                raise ValueError(f"Missing output: {name}")

            if isinstance(expected, tuple):
                if outputs[name].shape != expected:
                    raise ValueError(
                        f"Wrong shape for {name}: got {outputs[name].shape}, "
                        f"expected {expected}"
                    )


       
    def _create_reconstruction_head(self, output_dim: int) -> nn.Sequential:
        """Create reconstruction head for feature group"""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, output_dim)
        )




    def validate_initialization(self) -> None:
        """Validate parameter initialization with adjusted thresholds"""
        try:
            with torch.no_grad():
                weight_norms = []
                bias_norms = []
                
                for name, param in self.named_parameters():
                    if not param.requires_grad:
                        continue
                        
                    if not torch.isfinite(param).all():
                        raise ValueError(f"Invalid values in {name}")
                        
                    norm = param.norm().item()
                    
                    if 'weight' in name:
                        weight_norms.append(norm)
                        # Более мягкие пороги для весов
                        if norm > 100:
                            logger.warning(f"Very large weight norm ({norm:.2f}) for {name}")
                    elif 'bias' in name:
                        bias_norms.append(norm)
                        if norm > 20:
                            logger.warning(f"Large bias norm ({norm:.2f}) for {name}")

                # Log statistics
                logger.info(f"Weight norms - Mean: {np.mean(weight_norms):.2f}, "
                          f"Min: {min(weight_norms):.2f}, Max: {max(weight_norms):.2f}")
                if bias_norms:
                    logger.info(f"Bias norms - Mean: {np.mean(bias_norms):.2f}, "
                              f"Min: {min(bias_norms):.2f}, Max: {max(bias_norms):.2f}")

                logger.info("Parameter initialization validated successfully")

        except Exception as e:
            logger.error(f"Initialization validation failed: {str(e)}")
            if self.debug:
                logger.error(traceback.format_exc())
            raise

    
    
    def _validate_attention_weights(self) -> None:
        """Validate attention layer weights"""
        with torch.no_grad():
            for name, module in self.named_modules():
                if isinstance(module, (AdaptiveAttention, TemporalAttention)):
                    for param_name, param in module.named_parameters():
                        if 'weight' in param_name:
                            norm = param.norm().item()
                            if norm < 0.1 or norm > 10:
                                logger.warning(
                                    f"Unusual attention weight norm ({norm:.2f}) "
                                    f"in {name}.{param_name}"
                                )
    
    def _validate_conv_weights(self) -> None:
        """Validate convolutional layer weights"""
        with torch.no_grad():
            for name, module in self.named_modules():
                if isinstance(module, nn.Conv1d):
                    weight_norm = module.weight.norm().item()
                    if weight_norm < 0.1 or weight_norm > 10:
                        logger.warning(
                            f"Unusual conv weight norm ({weight_norm:.2f}) "
                            f"in {name}"
                        )
    
    def _validate_lstm_weights(self) -> None:
        """Validate LSTM layer weights"""
        with torch.no_grad():
            for name, module in self.named_modules():
                if isinstance(module, (nn.LSTM, EnhancedLSTM)):
                    for param_name, param in module.named_parameters():
                        if 'weight' in param_name:
                            norm = param.norm().item()
                            if norm < 0.1 or norm > 10:
                                logger.warning(
                                    f"Unusual LSTM weight norm ({norm:.2f}) "
                                    f"in {name}.{param_name}"
                                )




    def _calculate_detailed_metrics(self, 
                                  predictions: torch.Tensor,
                                  targets: torch.Tensor,
                                  probs: torch.Tensor) -> Dict[str, float]:
        """Calculate detailed metrics for validation
        
        Args:
            predictions: Model predictions
            targets: Ground truth
            probs: Prediction probabilities
            
        Returns:
            Dict with detailed metrics
        """
        tp = ((predictions == 1) & (targets == 1)).float().sum()
        fp = ((predictions == 1) & (targets == 0)).float().sum()
        tn = ((predictions == 0) & (targets == 0)).float().sum()
        fn = ((predictions == 0) & (targets == 1)).float().sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'precision': precision.item(),
            'recall': recall.item(),
            'f1_score': f1_score.item(),
            'confidence': probs.max(dim=1)[0].mean().item()
        }



    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Validation step with detailed metrics"""
        try:
            print("\033[34m[VALIDATE_STEP] Начало метода validate_step FeatureExtractor\033[0m")
            
            self.eval()
            with torch.no_grad():
                # 1. Обработка батча
                features = batch['features'].to(self.device)
                print(f"\033[34m[VALIDATE_STEP] Shape features: {features.shape}\033[0m")
                
                # 2. Получение целевой переменной с проверками
                if 'targets' in batch:
                    targets = batch['targets'].to(self.device)
                    print(f"\033[34m[VALIDATE_STEP] Используем ключ 'targets', shape: {targets.shape}\033[0m")
                elif 'target' in batch:
                    targets = batch['target'].to(self.device)
                    print(f"\033[34m[VALIDATE_STEP] Используем ключ 'target', shape: {targets.shape}\033[0m")
                else:
                    targets = None
                    print("\033[31m[VALIDATE_STEP] ВНИМАНИЕ: В батче отсутствуют ключи 'targets' и 'target'\033[0m")
                
                # 3. Forward pass
                outputs = self(features)
                logits = outputs['logits']
                uncertainty = outputs['uncertainty']
                print(f"\033[34m[VALIDATE_STEP] Forward pass выполнен, logits shape: {logits.shape}\033[0m")
                
                # 4. Расчет метрик
                metrics = {}
                
                if targets is not None:
                    # 5. Обработка размерности targets
                    if targets.dim() > 1:
                        print(f"\033[33m[VALIDATE_STEP] Сжатие размерности targets: {targets.shape} -> ", end="")
                        targets = targets.squeeze()
                        print(f"{targets.shape}\033[0m")
                    
                    # 6. Приведение к типу long
                    if targets.dtype != torch.long:
                        print(f"\033[33m[VALIDATE_STEP] Приведение targets к типу long: {targets.dtype} -> torch.long\033[0m")
                        targets = targets.long()
                    
                    # 7. Расчет метрик классификации
                    predictions = logits.argmax(dim=-1)
                    correct = (predictions == targets).float()
                    
                    # 8. Расчет loss
                    try:
                        loss = F.cross_entropy(logits, targets)
                        metrics['loss'] = loss.item()
                        print(f"\033[34m[VALIDATE_STEP] Расчет loss выполнен успешно: {loss.item():.6f}\033[0m")
                    except Exception as loss_e:
                        print(f"\033[31m[VALIDATE_STEP] Ошибка при расчете loss: {str(loss_e)}\033[0m")
                        metrics['loss'] = 0.0
                    
                    metrics['accuracy'] = correct.mean().item()
                    
                    # 9. Расчет метрик confusion matrix
                    tp = ((predictions == 1) & (targets == 1)).float().sum()
                    fp = ((predictions == 1) & (targets == 0)).float().sum()
                    tn = ((predictions == 0) & (targets == 0)).float().sum()
                    fn = ((predictions == 0) & (targets == 1)).float().sum()
                    
                    # 10. Дополнительные метрики
                    precision = tp / (tp + fp + 1e-8)
                    recall = tp / (tp + fn + 1e-8)
                    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
                    
                    metrics.update({
                        'precision': precision.item(),
                        'recall': recall.item(),
                        'f1_score': f1_score.item(),
                        'true_positives': tp.item(),
                        'false_positives': fp.item(),
                        'true_negatives': tn.item(),
                        'false_negatives': fn.item()
                    })
                
                # 11. Метрики модели
                metrics.update({
                    'uncertainty_mean': uncertainty.mean().item(),
                    'uncertainty_std': uncertainty.std().item(),
                    'logits_mean': logits.mean().item(),
                    'logits_std': logits.std().item()
                })
                
                # 12. Метрики памяти на GPU
                if torch.cuda.is_available():
                    metrics['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**3
                    metrics['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**3
                    
                    # Очистка кэша при высоком использовании памяти
                    if metrics['gpu_memory_allocated'] > 0.8 * metrics['gpu_memory_reserved']:
                        torch.cuda.empty_cache()
                
                print(f"\033[34m[VALIDATE_STEP] Метод validate_step завершен успешно\033[0m")
                
                return {
                    'metrics': metrics,
                    'outputs': outputs,
                    'predictions': logits.argmax(dim=-1) if targets is not None else None
                }
        
        except Exception as e:
            print(f"\033[31m[VALIDATE_STEP] Ошибка в validate_step: {str(e)}\033[0m")
            if hasattr(self, 'debug') and self.debug:
                import traceback
                print(traceback.format_exc())
            return {
                'metrics': {'loss': float('inf'), 'error': str(e)},
                'outputs': {},
                'predictions': None
            }




    def _log_metrics(self, phase: str, metrics: Dict[str, float], epoch: int) -> None:
        """Log metrics with proper formatting"""
        metric_str = [
            f"Epoch {epoch}:",
            f"Phase: {phase}",
            f"Loss: {metrics.get('loss', float('inf')):.4f}",
            f"Accuracy: {metrics.get('accuracy', 0):.4f}",
            f"F1 Score: {metrics.get('f1_score', 0):.4f}",
            f"Precision: {metrics.get('precision', 0):.4f}",
            f"Recall: {metrics.get('recall', 0):.4f}",
            f"Uncertainty: {metrics.get('uncertainty_mean', 0):.4f}±{metrics.get('uncertainty_std', 0):.4f}"
        ]
    
        if 'gpu_memory_allocated' in metrics:
            metric_str.append(
                f"GPU Memory: {metrics['gpu_memory_allocated']:.1f}GB / {metrics['gpu_memory_reserved']:.1f}GB"
            )
    
        logger.info("\n".join(metric_str))


    def _calculate_r2(self, pred: torch.Tensor, true: torch.Tensor) -> float:
        """Calculate R² score
        
        Args:
            pred: Predicted values
            true: True values
            
        Returns:
            float: R² score
        """
        try:
            ss_res = (true - pred).pow(2).sum()
            ss_tot = (true - true.mean()).pow(2).sum()
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            return r2.item()
        except Exception as e:
            logger.warning(f"Error calculating R2: {str(e)}")
            return 0.0

    def get_memory_stats(self) -> Dict[str, float]:
        """Get GPU memory statistics in GB"""
        try:
            if torch.cuda.is_available():
                stats = {
                    'allocated': torch.cuda.memory_allocated(self.device) / 1e9,
                    'reserved': torch.cuda.memory_reserved(self.device) / 1e9,
                    'max_allocated': torch.cuda.max_memory_allocated(self.device) / 1e9
                }
                
                if stats['allocated'] > 0.8 * stats['reserved']:
                    logger.warning(
                        f"High memory usage: {stats['allocated']:.1f}GB "
                        f"of {stats['reserved']:.1f}GB reserved"
                    )
                    torch.cuda.empty_cache()
                
                return stats
            return {}
        except Exception as e:
            logger.error(f"Error getting memory stats: {str(e)}")
            return {}




    def get_feature_importance(self, x: torch.Tensor) -> Dict[str, float]:
        """Calculate feature importance using attention weights
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            
        Returns:
            Dict mapping feature groups to importance scores
        """
        self.eval()
        
        with torch.no_grad():
            try:
                outputs = self(x)
                attention = torch.stack(outputs['attention_weights']).mean(dim=(0,1,2))  # Average over layers, heads, batch
                
                importance = {}
                feature_groups = {
                    'price': slice(0, 23),
                    'volume': slice(23, 35),
                    'momentum': slice(35, 58),
                    'composite': slice(58, 73),
                    'volatility': slice(73, 93),
                    'pattern': slice(93, 106),
                    'advanced': slice(106, 133)
                }
                
                for name, slice_obj in feature_groups.items():
                    importance[name] = attention[slice_obj].mean().item()
                    
                # Normalize to sum to 1
                total = sum(importance.values())
                importance = {k: v/total for k,v in importance.items()}
                
                return importance
                
            except Exception as e:
                logger.error(f"Error calculating feature importance: {str(e)}")
                return {}



    def get_optimization_groups(self) -> List[Dict[str, Any]]:
        """Get parameter groups for optimizer with different learning rates
        
        Returns:
            List of parameter group dictionaries
        """
        return [
            {
                'params': [p for name, p in self.named_parameters() if 'attention' in name],
                'lr': 2e-4,
                'weight_decay': 1e-5
            },
            {
                'params': [p for name, p in self.named_parameters() if 'lstm' in name],
                'lr': 1e-4,
                'weight_decay': 1e-5
            },
            {
                'params': [p for name, p in self.named_parameters() 
                          if not any(x in name for x in ['attention', 'lstm'])],
                'lr': 3e-4,
                'weight_decay': 1e-5
            }
        ]

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Configure optimizer and scheduler
        
        Returns:
            Tuple of (optimizer, scheduler)
        """
        # Optimizer with parameter groups
        optimizer = torch.optim.AdamW(
            self.get_optimization_groups(),
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine scheduler with warmup
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[g['lr'] for g in self.get_optimization_groups()],
            epochs=100,
            steps_per_epoch=100,
            pct_start=0.1,
            anneal_strategy='cos',
            final_div_factor=1e3
        )
        
        return optimizer, scheduler

    def mixed_precision_context(self) -> Any:
        """Get mixed precision context for RTX 4090
        
        Returns:
            Context manager for mixed precision
        """
        if torch.cuda.is_available():
            return torch.amp.autocast('cuda')
        return nullcontext()

    def get_gradients_info(self) -> Dict[str, float]:
        """Get gradient statistics
        
        Returns:
            Dict with gradient information
        """
        grads = {
            name: p.grad.abs().mean().item()
            for name, p in self.named_parameters()
            if p.grad is not None
        }
        
        return {
            'mean_grad': np.mean(list(grads.values())),
            'max_grad': max(grads.values()),
            'min_grad': min(grads.values()),
            'grad_by_layer': grads
        }

    def _update_stats(self, phase: str, metrics: Dict[str, float]) -> None:
        """Update running statistics
        
        Args:
            phase: 'train' or 'val'
            metrics: Current metrics
        """
        stats = self.train_metrics if phase == 'train' else self.val_metrics
        
        for k, v in metrics.items():
            stats[k].append(v)
            
        # Keep last 100 values only
        for k in stats:
            stats[k] = stats[k][-100:]

    def get_latest_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get latest metrics for train and val
        
        Returns:
            Dict with train and val metrics
        """
        return {
            'train': {k: np.mean(v[-10:]) for k, v in self.train_metrics.items()},
            'val': {k: np.mean(v[-10:]) for k, v in self.val_metrics.items()}
        }

    @property
    def device(self) -> torch.device:
        """Get model device"""
        return next(self.parameters()).device

    @property
    def num_parameters(self) -> int:
        """Get number of model parameters"""
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"FeatureExtractor(\n"
            f"  input_dim: {self.input_dim}\n"
            f"  hidden_dim: {self.hidden_dim}\n"
            f"  sequence_length: {self.sequence_length}\n"
            f"  num_heads: {self.num_heads}\n"
            f"  num_layers: {self.num_layers}\n"
            f"  num_parameters: {self.num_parameters:,}\n"
            f"  device: {self.device}\n"
            f")"
        )