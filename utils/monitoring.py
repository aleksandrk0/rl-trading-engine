# C_01_en/utils/monitoring.py
import time
import logging
import psutil
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class TrainingMonitor:
    """Monitor training progress and metrics"""
    def __init__(self):
        self.metrics_history: List[Dict] = []
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.log_interval = 50

    def log_system_metrics(self) -> Dict[str, float]:
        """Get system metrics including CPU, RAM and GPU
        
        Returns:
            Dict[str, float]: System metrics dictionary
        """
        metrics = {
            'cpu_percent': psutil.cpu_percent(),
            'ram_percent': psutil.virtual_memory().percent
        }
        
        if torch.cuda.is_available():
            metrics.update({
                'gpu_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'gpu_reserved_gb': torch.cuda.memory_reserved() / (1024**3)
            })
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'system',
            'metrics': metrics
        }
        self.metrics_history.append(entry)
            
        return metrics

    def log_batch(self, 
                epoch: int,
                batch_idx: int,
                total_batches: int,
                metrics: Dict[str, float],
                model_name: str) -> None:
        """Log batch metrics
        
        Args:
            epoch: Current epoch
            batch_idx: Current batch index
            total_batches: Total number of batches
            metrics: Dict of metrics to log
            model_name: Name of model being trained
        """
        if batch_idx % self.log_interval != 0:
            return
            
        current_time = time.time()
        elapsed = current_time - self.start_time
        batch_time = current_time - self.last_log_time
        self.last_log_time = current_time
        
        sys_metrics = self.get_system_metrics()
        
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k,v in metrics.items()])
        sys_str = ", ".join([f"{k}: {v:.2f}" for k,v in sys_metrics.items()])
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'batch',
            'epoch': epoch,
            'batch': batch_idx,
            'metrics': metrics,
            'system': sys_metrics,
            'model': model_name,
            'time': {
                'elapsed': elapsed,
                'batch': batch_time
            }
        }
        self.metrics_history.append(entry)
        
        logger.info(
            f"[{model_name}] Epoch: {epoch}, "
            f"Batch: {batch_idx}/{total_batches} "
            f"({100. * batch_idx / total_batches:.1f}%) "
            f"Time: {elapsed:.1f}s ({batch_time:.3f}s/batch) "
            f"| {metrics_str} | {sys_str}"
        )

    def get_system_metrics(self) -> Dict[str, float]:
        """Get system metrics (alias for log_system_metrics)"""
        return self.log_system_metrics()

    def log_epoch(self,
                 epoch: int,
                 train_metrics: Dict[str, float],
                 val_metrics: Dict[str, float],
                 model_name: str) -> None:
        """Log epoch metrics"""
        epoch_time = time.time() - self.last_log_time
        
        train_str = ", ".join([f"train_{k}: {v:.4f}" for k,v in train_metrics.items()])
        val_str = ", ".join([f"val_{k}: {v:.4f}" for k,v in val_metrics.items()])
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'type': 'epoch',
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'model': model_name,
            'time': epoch_time,
            'system': self.get_system_metrics()
        }
        self.metrics_history.append(entry)
        
        logger.info(
            f"[{model_name}] Epoch {epoch} completed in {epoch_time:.1f}s | "
            f"{train_str} | {val_str}" 
        )

    def get_history(self) -> List[Dict]:
        """Get complete metrics history"""
        return self.metrics_history