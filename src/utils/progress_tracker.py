"""
Progress tracking utilities for large dataset processing.

This module provides comprehensive progress tracking with memory monitoring,
ETA calculation, and performance metrics for data processing operations.
"""

import time
import psutil
import logging
from typing import Optional, Dict, Any, Callable
from tqdm import tqdm
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProgressMetrics:
    """Container for progress tracking metrics."""
    total_items: int
    processed_items: int
    start_time: float
    current_time: float
    memory_usage_mb: float
    cpu_percent: float
    items_per_second: float
    eta_seconds: float
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        return (self.processed_items / self.total_items) * 100 if self.total_items > 0 else 0.0
    
    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time in seconds."""
        return self.current_time - self.start_time
    
    @property
    def eta_formatted(self) -> str:
        """Format ETA as human-readable string."""
        if self.eta_seconds < 60:
            return f"{self.eta_seconds:.1f}s"
        elif self.eta_seconds < 3600:
            minutes = int(self.eta_seconds // 60)
            seconds = int(self.eta_seconds % 60)
            return f"{minutes}m {seconds}s"
        else:
            hours = int(self.eta_seconds // 3600)
            minutes = int((self.eta_seconds % 3600) // 60)
            return f"{hours}h {minutes}m"


class ProgressTracker:
    """
    Advanced progress tracker with system monitoring and performance metrics.
    """
    
    def __init__(
        self,
        total_items: int,
        description: str = "Processing",
        update_interval: float = 1.0,
        log_interval: int = 100,
        memory_threshold_mb: float = 1000.0,
        enable_system_monitoring: bool = True
    ):
        """
        Initialize progress tracker.
        
        Args:
            total_items: Total number of items to process
            description: Description for progress display
            update_interval: Update interval in seconds
            log_interval: Log progress every N items
            memory_threshold_mb: Memory usage threshold for warnings
            enable_system_monitoring: Enable system resource monitoring
        """
        self.total_items = total_items
        self.description = description
        self.update_interval = update_interval
        self.log_interval = log_interval
        self.memory_threshold_mb = memory_threshold_mb
        self.enable_system_monitoring = enable_system_monitoring
        
        # Progress tracking
        self.processed_items = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_log_time = self.start_time
        
        # Performance metrics
        self.processing_times = []
        self.memory_usage_history = []
        self.cpu_usage_history = []
        
        # System monitoring
        self.process = psutil.Process() if enable_system_monitoring else None
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        # Progress bar
        self.pbar = tqdm(
            total=total_items,
            desc=description,
            unit="items",
            dynamic_ncols=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )
        
        # Callbacks
        self.progress_callbacks = []
        self.warning_callbacks = []
        
        # Start monitoring
        if self.enable_system_monitoring:
            self._start_monitoring()
    
    def _start_monitoring(self):
        """Start system resource monitoring in background thread."""
        def monitor():
            while not self.stop_monitoring.wait(self.update_interval):
                try:
                    # Get memory usage
                    memory_info = self.process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024
                    self.memory_usage_history.append(memory_mb)
                    
                    # Get CPU usage
                    cpu_percent = self.process.cpu_percent()
                    self.cpu_usage_history.append(cpu_percent)
                    
                    # Check memory threshold
                    if memory_mb > self.memory_threshold_mb:
                        self._trigger_warning(
                            f"High memory usage: {memory_mb:.1f} MB (threshold: {self.memory_threshold_mb} MB)"
                        )
                    
                    # Keep only recent history (last 100 measurements)
                    if len(self.memory_usage_history) > 100:
                        self.memory_usage_history = self.memory_usage_history[-100:]
                    if len(self.cpu_usage_history) > 100:
                        self.cpu_usage_history = self.cpu_usage_history[-100:]
                        
                except Exception as e:
                    logger.warning(f"Error in system monitoring: {str(e)}")
        
        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
    
    def update(self, increment: int = 1, **kwargs):
        """
        Update progress by specified increment.
        
        Args:
            increment: Number of items processed
            **kwargs: Additional information to display
        """
        self.processed_items += increment
        current_time = time.time()
        
        # Update progress bar
        self.pbar.update(increment)
        
        # Update postfix with additional info
        if kwargs:
            self.pbar.set_postfix(**kwargs)
        
        # Log progress at intervals
        if (current_time - self.last_log_time) >= (self.log_interval * self.update_interval):
            self._log_progress()
            self.last_log_time = current_time
        
        # Trigger progress callbacks
        if self.progress_callbacks:
            metrics = self.get_current_metrics()
            for callback in self.progress_callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    logger.warning(f"Error in progress callback: {str(e)}")
    
    def set_description(self, description: str):
        """Update progress description."""
        self.description = description
        self.pbar.set_description(description)
    
    def add_progress_callback(self, callback: Callable[[ProgressMetrics], None]):
        """Add callback function to be called on progress updates."""
        self.progress_callbacks.append(callback)
    
    def add_warning_callback(self, callback: Callable[[str], None]):
        """Add callback function to be called on warnings."""
        self.warning_callbacks.append(callback)
    
    def _trigger_warning(self, message: str):
        """Trigger warning callbacks."""
        logger.warning(message)
        for callback in self.warning_callbacks:
            try:
                callback(message)
            except Exception as e:
                logger.warning(f"Error in warning callback: {str(e)}")
    
    def _log_progress(self):
        """Log current progress with metrics."""
        metrics = self.get_current_metrics()
        logger.info(
            f"{self.description}: {metrics.processed_items}/{metrics.total_items} "
            f"({metrics.progress_percent:.1f}%) - "
            f"{metrics.items_per_second:.1f} items/s - "
            f"ETA: {metrics.eta_formatted} - "
            f"Memory: {metrics.memory_usage_mb:.1f} MB"
        )
    
    def get_current_metrics(self) -> ProgressMetrics:
        """Get current progress metrics."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Calculate processing rate
        items_per_second = self.processed_items / elapsed_time if elapsed_time > 0 else 0
        
        # Calculate ETA
        remaining_items = self.total_items - self.processed_items
        eta_seconds = remaining_items / items_per_second if items_per_second > 0 else 0
        
        # Get system metrics
        memory_usage_mb = 0
        cpu_percent = 0
        if self.enable_system_monitoring and self.process:
            try:
                memory_info = self.process.memory_info()
                memory_usage_mb = memory_info.rss / 1024 / 1024
                cpu_percent = self.process.cpu_percent()
            except Exception:
                pass
        
        return ProgressMetrics(
            total_items=self.total_items,
            processed_items=self.processed_items,
            start_time=self.start_time,
            current_time=current_time,
            memory_usage_mb=memory_usage_mb,
            cpu_percent=cpu_percent,
            items_per_second=items_per_second,
            eta_seconds=eta_seconds
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        metrics = self.get_current_metrics()
        
        summary = {
            'total_items': metrics.total_items,
            'processed_items': metrics.processed_items,
            'progress_percent': metrics.progress_percent,
            'elapsed_time_seconds': metrics.elapsed_time,
            'items_per_second': metrics.items_per_second,
            'eta_seconds': metrics.eta_seconds,
            'eta_formatted': metrics.eta_formatted,
            'current_memory_mb': metrics.memory_usage_mb,
            'current_cpu_percent': metrics.cpu_percent
        }
        
        # Add historical metrics if available
        if self.memory_usage_history:
            summary.update({
                'avg_memory_mb': sum(self.memory_usage_history) / len(self.memory_usage_history),
                'max_memory_mb': max(self.memory_usage_history),
                'min_memory_mb': min(self.memory_usage_history)
            })
        
        if self.cpu_usage_history:
            summary.update({
                'avg_cpu_percent': sum(self.cpu_usage_history) / len(self.cpu_usage_history),
                'max_cpu_percent': max(self.cpu_usage_history),
                'min_cpu_percent': min(self.cpu_usage_history)
            })
        
        return summary
    
    def finish(self):
        """Finish progress tracking and cleanup."""
        # Stop monitoring
        if self.monitoring_thread:
            self.stop_monitoring.set()
            self.monitoring_thread.join(timeout=2.0)
        
        # Close progress bar
        self.pbar.close()
        
        # Log final summary
        summary = self.get_performance_summary()
        logger.info(f"Processing completed: {summary}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()


class BatchProgressTracker:
    """
    Progress tracker specifically designed for batch processing operations.
    """
    
    def __init__(
        self,
        total_batches: int,
        batch_size: int,
        description: str = "Processing batches",
        log_every_n_batches: int = 10
    ):
        """
        Initialize batch progress tracker.
        
        Args:
            total_batches: Total number of batches
            batch_size: Size of each batch
            description: Description for progress display
            log_every_n_batches: Log progress every N batches
        """
        self.total_batches = total_batches
        self.batch_size = batch_size
        self.total_items = total_batches * batch_size
        self.log_every_n_batches = log_every_n_batches
        
        self.tracker = ProgressTracker(
            total_items=total_batches,
            description=description,
            log_interval=log_every_n_batches
        )
        
        self.current_batch = 0
        self.batch_times = []
    
    def update_batch(self, batch_size: Optional[int] = None, **kwargs):
        """
        Update progress for one batch.
        
        Args:
            batch_size: Actual batch size (if different from default)
            **kwargs: Additional information to display
        """
        actual_batch_size = batch_size or self.batch_size
        self.current_batch += 1
        
        # Record batch processing time
        current_time = time.time()
        if hasattr(self, '_last_batch_time'):
            batch_time = current_time - self._last_batch_time
            self.batch_times.append(batch_time)
        self._last_batch_time = current_time
        
        # Calculate additional metrics
        if self.batch_times:
            avg_batch_time = sum(self.batch_times) / len(self.batch_times)
            kwargs['avg_batch_time'] = f"{avg_batch_time:.2f}s"
        
        kwargs['batch'] = f"{self.current_batch}/{self.total_batches}"
        kwargs['batch_size'] = actual_batch_size
        
        self.tracker.update(1, **kwargs)
    
    def get_batch_statistics(self) -> Dict[str, float]:
        """Get batch processing statistics."""
        if not self.batch_times:
            return {}
        
        return {
            'avg_batch_time': sum(self.batch_times) / len(self.batch_times),
            'min_batch_time': min(self.batch_times),
            'max_batch_time': max(self.batch_times),
            'total_batch_time': sum(self.batch_times),
            'batches_per_second': len(self.batch_times) / sum(self.batch_times)
        }
    
    def finish(self):
        """Finish batch progress tracking."""
        self.tracker.finish()
        
        # Log batch statistics
        stats = self.get_batch_statistics()
        if stats:
            logger.info(f"Batch processing statistics: {stats}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()


def create_file_progress_tracker(
    file_path: str,
    description: str = "Processing file"
) -> ProgressTracker:
    """
    Create progress tracker for file processing based on file size.
    
    Args:
        file_path: Path to file being processed
        description: Description for progress display
        
    Returns:
        ProgressTracker instance
    """
    file_size = Path(file_path).stat().st_size
    return ProgressTracker(
        total_items=file_size,
        description=f"{description} ({file_size / 1024 / 1024:.1f} MB)"
    )


if __name__ == "__main__":
    # Example usage
    import random
    
    # Test basic progress tracker
    print("Testing basic progress tracker...")
    with ProgressTracker(1000, "Processing items") as tracker:
        for i in range(1000):
            # Simulate work
            time.sleep(0.001)
            tracker.update(1, current_item=i)
    
    # Test batch progress tracker
    print("\nTesting batch progress tracker...")
    with BatchProgressTracker(50, 32, "Processing batches") as batch_tracker:
        for batch_idx in range(50):
            # Simulate batch processing
            time.sleep(0.01)
            batch_tracker.update_batch(loss=random.uniform(0.1, 1.0))
    
    print("Progress tracker tests completed!")