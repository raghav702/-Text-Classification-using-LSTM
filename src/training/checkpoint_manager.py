"""
Advanced checkpoint management and early stopping utilities.

This module provides enhanced checkpoint management functionality
beyond the basic trainer capabilities.
"""

import os
import glob
import torch
import json
import shutil
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging

from models.lstm_model import LSTMClassifier


class CheckpointManager:
    """
    Advanced checkpoint management with automatic cleanup and versioning.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 5,
        save_best_n: int = 3
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of regular checkpoints to keep
            save_best_n: Number of best models to keep
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.save_best_n = save_best_n
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.best_models_dir = os.path.join(checkpoint_dir, "best_models")
        os.makedirs(self.best_models_dir, exist_ok=True)
        
        # Track best models
        self.best_models = []  # List of (loss, accuracy, filepath) tuples
        self._load_best_models_registry()
    
    def _load_best_models_registry(self):
        """Load registry of best models from disk."""
        registry_path = os.path.join(self.best_models_dir, "registry.json")
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    data = json.load(f)
                    self.best_models = [
                        (item['loss'], item['accuracy'], item['filepath'])
                        for item in data.get('best_models', [])
                    ]
            except Exception as e:
                self.logger.warning(f"Failed to load best models registry: {e}")
                self.best_models = []
    
    def _save_best_models_registry(self):
        """Save registry of best models to disk."""
        registry_path = os.path.join(self.best_models_dir, "registry.json")
        try:
            data = {
                'best_models': [
                    {
                        'loss': loss,
                        'accuracy': accuracy,
                        'filepath': filepath,
                        'timestamp': datetime.now().isoformat()
                    }
                    for loss, accuracy, filepath in self.best_models
                ],
                'last_updated': datetime.now().isoformat()
            }
            with open(registry_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save best models registry: {e}")
    
    def save_checkpoint(
        self,
        model: LSTMClassifier,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_accuracy: float,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        metadata: Dict[str, Any] = None,
        is_best: bool = False
    ) -> str:
        """
        Save a checkpoint with automatic management.
        
        Args:
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            train_loss: Training loss
            val_loss: Validation loss
            val_accuracy: Validation accuracy
            scheduler: Learning rate scheduler (optional)
            metadata: Additional metadata to save
            is_best: Whether this is a best model
            
        Returns:
            Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if is_best:
            filename = f"best_model_epoch_{epoch}_{timestamp}.pth"
            filepath = os.path.join(self.best_models_dir, filename)
        else:
            filename = f"checkpoint_epoch_{epoch}_{timestamp}.pth"
            filepath = os.path.join(self.checkpoint_dir, filename)
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_config': model.get_model_info(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'timestamp': datetime.now().isoformat(),
            'is_best': is_best
        }
        
        # Add scheduler state
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Add metadata
        if metadata is not None:
            checkpoint['metadata'] = metadata
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")
        
        # Manage best models
        if is_best:
            self._add_best_model(val_loss, val_accuracy, filepath)
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        return filepath
    
    def _add_best_model(self, val_loss: float, val_accuracy: float, filepath: str):
        """Add a model to the best models list and manage the list size."""
        self.best_models.append((val_loss, val_accuracy, filepath))
        
        # Sort by validation loss (ascending) then by accuracy (descending)
        self.best_models.sort(key=lambda x: (x[0], -x[1]))
        
        # Keep only the best N models
        if len(self.best_models) > self.save_best_n:
            # Remove the worst model
            _, _, removed_filepath = self.best_models.pop()
            if os.path.exists(removed_filepath):
                os.remove(removed_filepath)
                self.logger.info(f"Removed old best model: {removed_filepath}")
        
        # Save registry
        self._save_best_models_registry()
    
    def _cleanup_checkpoints(self):
        """Remove old regular checkpoints, keeping only the most recent ones."""
        # Get all regular checkpoint files
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pth")
        checkpoint_files = glob.glob(pattern)
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return
        
        # Sort by modification time (newest first)
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        
        # Remove old checkpoints
        for old_checkpoint in checkpoint_files[self.max_checkpoints:]:
            try:
                os.remove(old_checkpoint)
                self.logger.info(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                self.logger.warning(f"Failed to remove checkpoint {old_checkpoint}: {e}")
    
    def load_best_checkpoint(self, criterion: str = 'loss') -> Optional[str]:
        """
        Get path to the best checkpoint based on specified criterion.
        
        Args:
            criterion: 'loss' for best validation loss, 'accuracy' for best accuracy
            
        Returns:
            Path to best checkpoint or None if no checkpoints exist
        """
        if not self.best_models:
            return None
        
        if criterion == 'loss':
            # Best model has lowest validation loss
            best_model = min(self.best_models, key=lambda x: x[0])
        elif criterion == 'accuracy':
            # Best model has highest validation accuracy
            best_model = max(self.best_models, key=lambda x: x[1])
        else:
            raise ValueError("Criterion must be 'loss' or 'accuracy'")
        
        return best_model[2]
    
    def load_latest_checkpoint(self) -> Optional[str]:
        """
        Get path to the most recent checkpoint.
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pth")
        checkpoint_files = glob.glob(pattern)
        
        if not checkpoint_files:
            return None
        
        # Return the most recently modified checkpoint
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        return latest_checkpoint
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Get information about a checkpoint without loading the full model.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary with checkpoint information
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        info = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'train_loss': checkpoint.get('train_loss', 'unknown'),
            'val_loss': checkpoint.get('val_loss', 'unknown'),
            'val_accuracy': checkpoint.get('val_accuracy', 'unknown'),
            'timestamp': checkpoint.get('timestamp', 'unknown'),
            'is_best': checkpoint.get('is_best', False),
            'model_config': checkpoint.get('model_config', {}),
            'file_size': os.path.getsize(checkpoint_path)
        }
        
        return info
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints with their information.
        
        Returns:
            List of checkpoint information dictionaries
        """
        checkpoints = []
        
        # Regular checkpoints
        pattern = os.path.join(self.checkpoint_dir, "checkpoint_epoch_*.pth")
        for filepath in glob.glob(pattern):
            try:
                info = self.get_checkpoint_info(filepath)
                info['filepath'] = filepath
                info['type'] = 'regular'
                checkpoints.append(info)
            except Exception as e:
                self.logger.warning(f"Failed to read checkpoint {filepath}: {e}")
        
        # Best model checkpoints
        pattern = os.path.join(self.best_models_dir, "best_model_epoch_*.pth")
        for filepath in glob.glob(pattern):
            try:
                info = self.get_checkpoint_info(filepath)
                info['filepath'] = filepath
                info['type'] = 'best'
                checkpoints.append(info)
            except Exception as e:
                self.logger.warning(f"Failed to read best model {filepath}: {e}")
        
        # Sort by epoch
        checkpoints.sort(key=lambda x: x.get('epoch', 0))
        
        return checkpoints
    
    def cleanup_all(self):
        """Remove all checkpoints and reset the manager."""
        try:
            shutil.rmtree(self.checkpoint_dir)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            os.makedirs(self.best_models_dir, exist_ok=True)
            self.best_models = []
            self.logger.info("All checkpoints cleaned up")
        except Exception as e:
            self.logger.error(f"Failed to cleanup checkpoints: {e}")


class EarlyStopping:
    """
    Enhanced early stopping with multiple criteria and restoration capabilities.
    """
    
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.001,
        restore_best_weights: bool = True,
        mode: str = 'min',
        baseline: float = None,
        cooldown: int = 0
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
            mode: 'min' for loss, 'max' for accuracy
            baseline: Baseline value that must be reached
            cooldown: Number of epochs to wait after lr reduction
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.baseline = baseline
        self.cooldown = cooldown
        
        # State variables
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.cooldown_counter = 0
        self.stopped_epoch = 0
        
        # Comparison function
        if mode == 'min':
            self.monitor_op = lambda current, best: current < best - self.min_delta
        else:
            self.monitor_op = lambda current, best: current > best + self.min_delta
        
        self.logger = logging.getLogger(__name__)
    
    def __call__(
        self,
        current_score: float,
        model: torch.nn.Module,
        epoch: int
    ) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_score: Current validation score
            model: Model to potentially save weights from
            epoch: Current epoch number
            
        Returns:
            True if training should stop
        """
        # Handle cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False
        
        # Check baseline
        if self.baseline is not None:
            if self.mode == 'min' and current_score > self.baseline:
                return False
            elif self.mode == 'max' and current_score < self.baseline:
                return False
        
        # First epoch
        if self.best_score is None:
            self.best_score = current_score
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            return False
        
        # Check for improvement
        if self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        # Check if should stop
        if self.counter >= self.patience:
            self.stopped_epoch = epoch
            if self.restore_best_weights and self.best_weights is not None:
                self.logger.info("Restoring best model weights")
                model.load_state_dict(self.best_weights)
            return True
        
        return False
    
    def on_lr_reduction(self):
        """Call this when learning rate is reduced to reset cooldown."""
        self.cooldown_counter = self.cooldown
    
    def get_state(self) -> Dict[str, Any]:
        """Get current early stopping state."""
        return {
            'best_score': self.best_score,
            'counter': self.counter,
            'stopped_epoch': self.stopped_epoch,
            'patience': self.patience,
            'mode': self.mode
        }


def create_checkpoint_manager(
    checkpoint_dir: str = "checkpoints",
    max_checkpoints: int = 5,
    save_best_n: int = 3
) -> CheckpointManager:
    """
    Factory function to create a checkpoint manager.
    
    Args:
        checkpoint_dir: Directory for checkpoints
        max_checkpoints: Maximum regular checkpoints to keep
        save_best_n: Number of best models to keep
        
    Returns:
        Configured CheckpointManager instance
    """
    return CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_checkpoints=max_checkpoints,
        save_best_n=save_best_n
    )


def create_early_stopping(
    patience: int = 7,
    min_delta: float = 0.001,
    monitor: str = 'val_loss',
    restore_best_weights: bool = True
) -> EarlyStopping:
    """
    Factory function to create early stopping.
    
    Args:
        patience: Epochs to wait for improvement
        min_delta: Minimum improvement threshold
        monitor: Metric to monitor ('val_loss' or 'val_accuracy')
        restore_best_weights: Whether to restore best weights
        
    Returns:
        Configured EarlyStopping instance
    """
    mode = 'min' if 'loss' in monitor else 'max'
    
    return EarlyStopping(
        patience=patience,
        min_delta=min_delta,
        mode=mode,
        restore_best_weights=restore_best_weights
    )