"""
Model Optimization Module for LSTM Sentiment Classifier

This module provides functionality for optimizing trained models through:
- Model quantization (INT8, FP16)
- Model pruning
- ONNX export for cross-platform deployment
- Performance benchmarking and profiling
"""

import os
import time
import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.nn.utils import prune
import onnx
import onnxruntime as ort
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
from pathlib import Path

from models.lstm_model import LSTMClassifier


class ModelOptimizer:
    """
    Comprehensive model optimization toolkit for LSTM sentiment classifier.
    
    Provides quantization, pruning, ONNX export, and benchmarking capabilities
    to optimize model performance for production deployment.
    """
    
    def __init__(self, model: LSTMClassifier, device: str = None):
        """
        Initialize the model optimizer.
        
        Args:
            model: Trained LSTM model to optimize
            device: Device to run optimization on
        """
        self.original_model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimized_models = {}
        self.benchmark_results = {}
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Ensure model is in eval mode
        self.original_model.eval()
    
    def quantize_model_dynamic(self, 
                              dtype: torch.dtype = torch.qint8,
                              qconfig_spec: Dict = None) -> torch.nn.Module:
        """
        Apply dynamic quantization to the model.
        
        Dynamic quantization quantizes weights ahead of time but quantizes
        activations dynamically during inference.
        
        Args:
            dtype: Quantization data type (torch.qint8 or torch.float16)
            qconfig_spec: Custom quantization configuration
            
        Returns:
            Dynamically quantized model
        """
        self.logger.info(f"Applying dynamic quantization with dtype: {dtype}")
        
        # Create a copy of the model
        model_copy = type(self.original_model)(
            **self.original_model.get_model_info()
        )
        model_copy.load_state_dict(self.original_model.state_dict())
        model_copy.eval()
        
        # Define layers to quantize
        if qconfig_spec is None:
            qconfig_spec = {
                nn.Linear: torch.quantization.default_dynamic_qconfig,
                nn.LSTM: torch.quantization.default_dynamic_qconfig
            }
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model_copy,
            qconfig_spec,
            dtype=dtype
        )
        
        # Store the quantized model
        model_name = f"dynamic_quantized_{str(dtype).split('.')[-1]}"
        self.optimized_models[model_name] = quantized_model
        
        self.logger.info(f"Dynamic quantization completed: {model_name}")
        return quantized_model
    
    def quantize_model_static(self,
                             calibration_data: torch.utils.data.DataLoader,
                             dtype: torch.dtype = torch.qint8) -> torch.nn.Module:
        """
        Apply static quantization to the model.
        
        Static quantization requires calibration data to determine
        optimal quantization parameters for activations.
        
        Args:
            calibration_data: DataLoader with calibration samples
            dtype: Quantization data type
            
        Returns:
            Statically quantized model
        """
        self.logger.info(f"Applying static quantization with dtype: {dtype}")
        
        # Create a copy of the model
        model_copy = type(self.original_model)(
            **self.original_model.get_model_info()
        )
        model_copy.load_state_dict(self.original_model.state_dict())
        
        # Prepare model for quantization
        model_copy.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model_copy, inplace=True)
        
        # Calibrate with sample data
        self.logger.info("Calibrating model with sample data...")
        model_copy.eval()
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(calibration_data):
                if batch_idx >= 100:  # Limit calibration samples
                    break
                model_copy(data.to(self.device))
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_copy, inplace=False)
        
        # Store the quantized model
        model_name = f"static_quantized_{str(dtype).split('.')[-1]}"
        self.optimized_models[model_name] = quantized_model
        
        self.logger.info(f"Static quantization completed: {model_name}")
        return quantized_model
    
    def prune_model(self,
                   pruning_ratio: float = 0.2,
                   structured: bool = False,
                   global_pruning: bool = True) -> torch.nn.Module:
        """
        Apply pruning to remove unnecessary model parameters.
        
        Args:
            pruning_ratio: Fraction of parameters to prune (0.0 to 1.0)
            structured: Whether to use structured pruning
            global_pruning: Whether to apply global pruning across all layers
            
        Returns:
            Pruned model
        """
        self.logger.info(f"Applying pruning with ratio: {pruning_ratio}")
        
        # Create a copy of the model
        model_copy = type(self.original_model)(
            **self.original_model.get_model_info()
        )
        model_copy.load_state_dict(self.original_model.state_dict())
        
        # Identify layers to prune
        parameters_to_prune = []
        for name, module in model_copy.named_modules():
            if isinstance(module, (nn.Linear, nn.LSTM)):
                if hasattr(module, 'weight'):
                    parameters_to_prune.append((module, 'weight'))
        
        if global_pruning:
            # Global magnitude-based pruning
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_ratio,
            )
        else:
            # Layer-wise pruning
            for module, param_name in parameters_to_prune:
                if structured:
                    # Structured pruning (remove entire channels/neurons)
                    prune.ln_structured(
                        module, param_name, amount=pruning_ratio, n=2, dim=0
                    )
                else:
                    # Unstructured magnitude-based pruning
                    prune.l1_unstructured(module, param_name, amount=pruning_ratio)
        
        # Make pruning permanent
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)
        
        # Store the pruned model
        model_name = f"pruned_{int(pruning_ratio*100)}pct"
        self.optimized_models[model_name] = model_copy
        
        self.logger.info(f"Pruning completed: {model_name}")
        return model_copy
    
    def export_to_onnx(self,
                      model: torch.nn.Module,
                      output_path: str,
                      input_shape: Tuple[int, ...] = (1, 500),
                      opset_version: int = 11,
                      dynamic_axes: Dict = None) -> str:
        """
        Export model to ONNX format for cross-platform deployment.
        
        Args:
            model: Model to export
            output_path: Path to save ONNX model
            input_shape: Shape of input tensor (batch_size, sequence_length)
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes specification for variable input sizes
            
        Returns:
            Path to exported ONNX model
        """
        self.logger.info(f"Exporting model to ONNX: {output_path}")
        
        # Create dummy input
        dummy_input = torch.randint(0, 1000, input_shape, dtype=torch.long)
        
        # Default dynamic axes for variable batch size and sequence length
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size'}
            }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Export to ONNX
        model.eval()
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
        
        # Verify ONNX model
        try:
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            self.logger.info(f"ONNX model exported and verified: {output_path}")
        except Exception as e:
            self.logger.error(f"ONNX model verification failed: {e}")
            raise
        
        return output_path
    
    def benchmark_model(self,
                       model: torch.nn.Module,
                       input_shape: Tuple[int, ...] = (1, 500),
                       num_runs: int = 100,
                       warmup_runs: int = 10) -> Dict:
        """
        Benchmark model inference performance.
        
        Args:
            model: Model to benchmark
            input_shape: Shape of input tensor
            num_runs: Number of inference runs for timing
            warmup_runs: Number of warmup runs
            
        Returns:
            Dictionary with benchmark results
        """
        self.logger.info(f"Benchmarking model with {num_runs} runs")
        
        # Create test input
        test_input = torch.randint(0, 1000, input_shape, dtype=torch.long)
        test_input = test_input.to(self.device)
        
        model.eval()
        model.to(self.device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(test_input)
        
        # Synchronize GPU if using CUDA
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()
        
        # Benchmark runs
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.perf_counter()
                output = model(test_input)
                
                if self.device.startswith('cuda'):
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        # Calculate statistics
        times = np.array(times)
        
        # Memory usage
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())
        
        results = {
            'mean_inference_time': float(np.mean(times)),
            'std_inference_time': float(np.std(times)),
            'min_inference_time': float(np.min(times)),
            'max_inference_time': float(np.max(times)),
            'median_inference_time': float(np.median(times)),
            'throughput_samples_per_sec': float(input_shape[0] / np.mean(times)),
            'model_size_bytes': model_size,
            'model_size_mb': model_size / (1024 * 1024),
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'device': str(self.device),
            'input_shape': input_shape
        }
        
        return results
    
    def benchmark_onnx_model(self,
                            onnx_path: str,
                            input_shape: Tuple[int, ...] = (1, 500),
                            num_runs: int = 100,
                            providers: List[str] = None) -> Dict:
        """
        Benchmark ONNX model performance.
        
        Args:
            onnx_path: Path to ONNX model
            input_shape: Shape of input tensor
            num_runs: Number of inference runs
            providers: ONNX Runtime execution providers
            
        Returns:
            Dictionary with benchmark results
        """
        self.logger.info(f"Benchmarking ONNX model: {onnx_path}")
        
        if providers is None:
            providers = ['CPUExecutionProvider']
        
        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Create test input
        test_input = np.random.randint(0, 1000, input_shape, dtype=np.int64)
        
        # Warmup runs
        for _ in range(10):
            _ = session.run(None, {'input': test_input})
        
        # Benchmark runs
        times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            output = session.run(None, {'input': test_input})
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        times = np.array(times)
        
        # Get model size
        model_size = os.path.getsize(onnx_path)
        
        results = {
            'mean_inference_time': float(np.mean(times)),
            'std_inference_time': float(np.std(times)),
            'min_inference_time': float(np.min(times)),
            'max_inference_time': float(np.max(times)),
            'median_inference_time': float(np.median(times)),
            'throughput_samples_per_sec': float(input_shape[0] / np.mean(times)),
            'model_size_bytes': model_size,
            'model_size_mb': model_size / (1024 * 1024),
            'providers': providers,
            'input_shape': input_shape
        }
        
        return results
    
    def compare_models(self,
                      test_input: torch.Tensor,
                      tolerance: float = 1e-3) -> Dict:
        """
        Compare outputs of optimized models with original model.
        
        Args:
            test_input: Test input tensor
            tolerance: Tolerance for output comparison
            
        Returns:
            Dictionary with comparison results
        """
        self.logger.info("Comparing optimized models with original")
        
        # Get original model output
        self.original_model.eval()
        with torch.no_grad():
            original_output = self.original_model(test_input.to(self.device))
        
        comparison_results = {}
        
        for model_name, model in self.optimized_models.items():
            try:
                model.eval()
                with torch.no_grad():
                    optimized_output = model(test_input.to(self.device))
                
                # Calculate differences
                diff = torch.abs(original_output - optimized_output)
                max_diff = torch.max(diff).item()
                mean_diff = torch.mean(diff).item()
                
                # Check if outputs are close
                outputs_close = torch.allclose(
                    original_output, optimized_output, atol=tolerance
                )
                
                comparison_results[model_name] = {
                    'max_difference': max_diff,
                    'mean_difference': mean_diff,
                    'outputs_close': outputs_close,
                    'tolerance': tolerance
                }
                
            except Exception as e:
                comparison_results[model_name] = {
                    'error': str(e)
                }
        
        return comparison_results
    
    def profile_model_layers(self,
                           model: torch.nn.Module,
                           input_tensor: torch.Tensor,
                           num_runs: int = 10) -> Dict:
        """
        Profile individual layer performance in the model.
        
        Args:
            model: Model to profile
            input_tensor: Input tensor for profiling
            num_runs: Number of runs for averaging
            
        Returns:
            Dictionary with layer-wise profiling results
        """
        self.logger.info("Profiling model layers")
        
        model.eval()
        model.to(self.device)
        input_tensor = input_tensor.to(self.device)
        
        # Use PyTorch profiler
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model(input_tensor)
        
        # Extract profiling results
        events = prof.key_averages().table(sort_by="cpu_time_total", row_limit=20)
        
        # Parse profiling data
        layer_stats = {}
        for event in prof.key_averages():
            if event.key not in layer_stats:
                layer_stats[event.key] = {
                    'cpu_time': event.cpu_time_total,
                    'cuda_time': event.cuda_time_total,
                    'cpu_memory': event.cpu_memory_usage,
                    'cuda_memory': event.cuda_memory_usage,
                    'count': event.count
                }
        
        return {
            'layer_statistics': layer_stats,
            'profiler_table': events,
            'total_cpu_time': sum(event.cpu_time_total for event in prof.key_averages()),
            'total_cuda_time': sum(event.cuda_time_total for event in prof.key_averages())
        }
    
    def optimize_all(self,
                    calibration_data: torch.utils.data.DataLoader = None,
                    pruning_ratio: float = 0.2,
                    export_onnx: bool = True,
                    onnx_output_dir: str = "optimized_models") -> Dict:
        """
        Apply all optimization techniques and benchmark results.
        
        Args:
            calibration_data: Data for static quantization calibration
            pruning_ratio: Ratio for model pruning
            export_onnx: Whether to export ONNX models
            onnx_output_dir: Directory to save ONNX models
            
        Returns:
            Dictionary with all optimization results
        """
        self.logger.info("Starting comprehensive model optimization")
        
        results = {
            'original_model': {},
            'optimized_models': {},
            'onnx_models': {},
            'comparisons': {},
            'summary': {}
        }
        
        # Benchmark original model
        original_benchmark = self.benchmark_model(self.original_model)
        results['original_model'] = original_benchmark
        
        # Apply optimizations
        optimizations = [
            ('dynamic_quantization_int8', lambda: self.quantize_model_dynamic(torch.qint8)),
            ('dynamic_quantization_fp16', lambda: self.quantize_model_dynamic(torch.float16)),
            ('pruning', lambda: self.prune_model(pruning_ratio)),
        ]
        
        # Add static quantization if calibration data provided
        if calibration_data is not None:
            optimizations.append(
                ('static_quantization', lambda: self.quantize_model_static(calibration_data))
            )
        
        # Apply each optimization
        for opt_name, opt_func in optimizations:
            try:
                self.logger.info(f"Applying {opt_name}")
                optimized_model = opt_func()
                
                # Benchmark optimized model
                benchmark_results = self.benchmark_model(optimized_model)
                results['optimized_models'][opt_name] = benchmark_results
                
                # Export to ONNX if requested
                if export_onnx:
                    onnx_path = os.path.join(onnx_output_dir, f"{opt_name}.onnx")
                    try:
                        self.export_to_onnx(optimized_model, onnx_path)
                        onnx_benchmark = self.benchmark_onnx_model(onnx_path)
                        results['onnx_models'][opt_name] = onnx_benchmark
                    except Exception as e:
                        self.logger.warning(f"ONNX export failed for {opt_name}: {e}")
                
            except Exception as e:
                self.logger.error(f"Optimization {opt_name} failed: {e}")
                results['optimized_models'][opt_name] = {'error': str(e)}
        
        # Compare model outputs
        test_input = torch.randint(0, 1000, (4, 500), dtype=torch.long)
        comparisons = self.compare_models(test_input)
        results['comparisons'] = comparisons
        
        # Generate summary
        summary = self._generate_optimization_summary(results)
        results['summary'] = summary
        
        self.logger.info("Comprehensive optimization completed")
        return results
    
    def _generate_optimization_summary(self, results: Dict) -> Dict:
        """Generate optimization summary statistics."""
        original_time = results['original_model']['mean_inference_time']
        original_size = results['original_model']['model_size_mb']
        
        summary = {
            'original_inference_time': original_time,
            'original_model_size_mb': original_size,
            'optimizations': {}
        }
        
        for opt_name, opt_results in results['optimized_models'].items():
            if 'error' not in opt_results:
                speedup = original_time / opt_results['mean_inference_time']
                size_reduction = (original_size - opt_results['model_size_mb']) / original_size
                
                summary['optimizations'][opt_name] = {
                    'speedup': speedup,
                    'size_reduction_ratio': size_reduction,
                    'inference_time': opt_results['mean_inference_time'],
                    'model_size_mb': opt_results['model_size_mb']
                }
        
        return summary
    
    def save_optimization_report(self, results: Dict, output_path: str):
        """Save optimization results to JSON file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Optimization report saved to: {output_path}")


def create_model_optimizer(model_path: str, vocab_size: int, device: str = None) -> ModelOptimizer:
    """
    Factory function to create model optimizer from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        vocab_size: Vocabulary size for model initialization
        device: Device to use
        
    Returns:
        Initialized ModelOptimizer
    """
    # Load model
    checkpoint = torch.load(model_path, map_location=device or 'cpu')
    
    # Extract model config
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    else:
        model_config = {
            'vocab_size': vocab_size,
            'embedding_dim': 300,
            'hidden_dim': 128,
            'output_dim': 1,
            'n_layers': 2,
            'dropout': 0.3,
            'bidirectional': True
        }
    
    # Initialize model
    model = LSTMClassifier(**model_config)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return ModelOptimizer(model, device)