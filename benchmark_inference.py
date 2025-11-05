#!/usr/bin/env python3
"""
Inference Performance Benchmarking Script

This script provides comprehensive benchmarking and optimization capabilities
for the LSTM sentiment classifier, including model quantization, pruning,
ONNX export, and performance profiling.
"""

import argparse
import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from optimization.model_optimizer import ModelOptimizer, create_model_optimizer
from data.text_preprocessor import TextPreprocessor
from inference.inference_engine import InferenceEngine
import torch
import torch.utils.data as data


class BenchmarkDataset(data.Dataset):
    """Simple dataset for benchmarking purposes."""
    
    def __init__(self, num_samples: int = 1000, seq_length: int = 500):
        self.num_samples = num_samples
        self.seq_length = seq_length
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random sequences for benchmarking
        sequence = torch.randint(0, 10000, (self.seq_length,), dtype=torch.long)
        label = torch.randint(0, 2, (1,), dtype=torch.float)
        return sequence, label


def setup_logging(log_level: str = 'INFO'):
    """Set up logging configuration."""
    level = getattr(logging, log_level.upper())
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def benchmark_original_model(args) -> Dict:
    """Benchmark the original unoptimized model."""
    print("Benchmarking original model...")
    
    # Create optimizer with original model
    optimizer = create_model_optimizer(
        args.model_path, 
        args.vocab_size, 
        args.device
    )
    
    # Benchmark original model
    results = optimizer.benchmark_model(
        optimizer.original_model,
        input_shape=(args.batch_size, args.sequence_length),
        num_runs=args.num_runs,
        warmup_runs=args.warmup_runs
    )
    
    print(f"Original model inference time: {results['mean_inference_time']:.4f}s")
    print(f"Original model size: {results['model_size_mb']:.2f} MB")
    print(f"Throughput: {results['throughput_samples_per_sec']:.1f} samples/sec")
    
    return results


def run_quantization_benchmark(args) -> Dict:
    """Run quantization benchmarks."""
    print("\nRunning quantization benchmarks...")
    
    optimizer = create_model_optimizer(
        args.model_path, 
        args.vocab_size, 
        args.device
    )
    
    results = {}
    
    # Dynamic quantization INT8
    print("  - Dynamic quantization (INT8)...")
    try:
        quant_model = optimizer.quantize_model_dynamic(torch.qint8)
        quant_results = optimizer.benchmark_model(
            quant_model,
            input_shape=(args.batch_size, args.sequence_length),
            num_runs=args.num_runs
        )
        results['dynamic_int8'] = quant_results
        print(f"    Inference time: {quant_results['mean_inference_time']:.4f}s")
        print(f"    Model size: {quant_results['model_size_mb']:.2f} MB")
    except Exception as e:
        print(f"    Failed: {e}")
        results['dynamic_int8'] = {'error': str(e)}
    
    # Dynamic quantization FP16
    print("  - Dynamic quantization (FP16)...")
    try:
        quant_model = optimizer.quantize_model_dynamic(torch.float16)
        quant_results = optimizer.benchmark_model(
            quant_model,
            input_shape=(args.batch_size, args.sequence_length),
            num_runs=args.num_runs
        )
        results['dynamic_fp16'] = quant_results
        print(f"    Inference time: {quant_results['mean_inference_time']:.4f}s")
        print(f"    Model size: {quant_results['model_size_mb']:.2f} MB")
    except Exception as e:
        print(f"    Failed: {e}")
        results['dynamic_fp16'] = {'error': str(e)}
    
    # Static quantization (if calibration data available)
    if args.enable_static_quantization:
        print("  - Static quantization (INT8)...")
        try:
            # Create calibration dataset
            calib_dataset = BenchmarkDataset(100, args.sequence_length)
            calib_loader = data.DataLoader(calib_dataset, batch_size=16, shuffle=False)
            
            quant_model = optimizer.quantize_model_static(calib_loader, torch.qint8)
            quant_results = optimizer.benchmark_model(
                quant_model,
                input_shape=(args.batch_size, args.sequence_length),
                num_runs=args.num_runs
            )
            results['static_int8'] = quant_results
            print(f"    Inference time: {quant_results['mean_inference_time']:.4f}s")
            print(f"    Model size: {quant_results['model_size_mb']:.2f} MB")
        except Exception as e:
            print(f"    Failed: {e}")
            results['static_int8'] = {'error': str(e)}
    
    return results


def run_pruning_benchmark(args) -> Dict:
    """Run pruning benchmarks."""
    print("\nRunning pruning benchmarks...")
    
    optimizer = create_model_optimizer(
        args.model_path, 
        args.vocab_size, 
        args.device
    )
    
    results = {}
    
    for pruning_ratio in args.pruning_ratios:
        print(f"  - Pruning {int(pruning_ratio*100)}%...")
        try:
            pruned_model = optimizer.prune_model(
                pruning_ratio=pruning_ratio,
                structured=args.structured_pruning,
                global_pruning=True
            )
            
            pruned_results = optimizer.benchmark_model(
                pruned_model,
                input_shape=(args.batch_size, args.sequence_length),
                num_runs=args.num_runs
            )
            
            results[f'pruned_{int(pruning_ratio*100)}pct'] = pruned_results
            print(f"    Inference time: {pruned_results['mean_inference_time']:.4f}s")
            print(f"    Model size: {pruned_results['model_size_mb']:.2f} MB")
            print(f"    Parameters: {pruned_results['num_parameters']:,}")
            
        except Exception as e:
            print(f"    Failed: {e}")
            results[f'pruned_{int(pruning_ratio*100)}pct'] = {'error': str(e)}
    
    return results


def run_onnx_benchmark(args) -> Dict:
    """Run ONNX export and benchmarking."""
    print("\nRunning ONNX benchmarks...")
    
    optimizer = create_model_optimizer(
        args.model_path, 
        args.vocab_size, 
        args.device
    )
    
    results = {}
    
    # Export original model to ONNX
    onnx_dir = Path(args.output_dir) / "onnx_models"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    
    print("  - Exporting original model to ONNX...")
    try:
        onnx_path = str(onnx_dir / "original_model.onnx")
        optimizer.export_to_onnx(
            optimizer.original_model,
            onnx_path,
            input_shape=(args.batch_size, args.sequence_length)
        )
        
        # Benchmark ONNX model
        onnx_results = optimizer.benchmark_onnx_model(
            onnx_path,
            input_shape=(args.batch_size, args.sequence_length),
            num_runs=args.num_runs
        )
        results['original_onnx'] = onnx_results
        print(f"    ONNX inference time: {onnx_results['mean_inference_time']:.4f}s")
        print(f"    ONNX model size: {onnx_results['model_size_mb']:.2f} MB")
        
    except Exception as e:
        print(f"    Failed: {e}")
        results['original_onnx'] = {'error': str(e)}
    
    # Export optimized models to ONNX
    if args.export_optimized_onnx:
        print("  - Exporting optimized models to ONNX...")
        
        # Quantized models
        try:
            quant_model = optimizer.quantize_model_dynamic(torch.qint8)
            onnx_path = str(onnx_dir / "quantized_int8.onnx")
            optimizer.export_to_onnx(quant_model, onnx_path)
            
            onnx_results = optimizer.benchmark_onnx_model(onnx_path)
            results['quantized_int8_onnx'] = onnx_results
            print(f"    Quantized INT8 ONNX time: {onnx_results['mean_inference_time']:.4f}s")
        except Exception as e:
            print(f"    Quantized INT8 ONNX failed: {e}")
        
        # Pruned model
        try:
            pruned_model = optimizer.prune_model(pruning_ratio=0.2)
            onnx_path = str(onnx_dir / "pruned_20pct.onnx")
            optimizer.export_to_onnx(pruned_model, onnx_path)
            
            onnx_results = optimizer.benchmark_onnx_model(onnx_path)
            results['pruned_20pct_onnx'] = onnx_results
            print(f"    Pruned 20% ONNX time: {onnx_results['mean_inference_time']:.4f}s")
        except Exception as e:
            print(f"    Pruned ONNX failed: {e}")
    
    return results


def run_comprehensive_optimization(args) -> Dict:
    """Run comprehensive optimization with all techniques."""
    print("\nRunning comprehensive optimization...")
    
    optimizer = create_model_optimizer(
        args.model_path, 
        args.vocab_size, 
        args.device
    )
    
    # Prepare calibration data if needed
    calibration_data = None
    if args.enable_static_quantization:
        calib_dataset = BenchmarkDataset(100, args.sequence_length)
        calibration_data = data.DataLoader(calib_dataset, batch_size=16, shuffle=False)
    
    # Run comprehensive optimization
    results = optimizer.optimize_all(
        calibration_data=calibration_data,
        pruning_ratio=args.default_pruning_ratio,
        export_onnx=args.export_onnx,
        onnx_output_dir=str(Path(args.output_dir) / "onnx_models")
    )
    
    return results


def run_accuracy_validation(args) -> Dict:
    """Validate that optimized models maintain accuracy."""
    print("\nValidating model accuracy...")
    
    optimizer = create_model_optimizer(
        args.model_path, 
        args.vocab_size, 
        args.device
    )
    
    # Create test input
    test_input = torch.randint(0, args.vocab_size, (10, args.sequence_length), dtype=torch.long)
    
    # Apply optimizations
    optimizer.quantize_model_dynamic(torch.qint8)
    optimizer.prune_model(pruning_ratio=0.2)
    
    # Compare outputs
    comparisons = optimizer.compare_models(test_input, tolerance=1e-2)
    
    print("Model output comparisons:")
    for model_name, comparison in comparisons.items():
        if 'error' not in comparison:
            print(f"  {model_name}:")
            print(f"    Max difference: {comparison['max_difference']:.6f}")
            print(f"    Mean difference: {comparison['mean_difference']:.6f}")
            print(f"    Outputs close: {comparison['outputs_close']}")
        else:
            print(f"  {model_name}: Error - {comparison['error']}")
    
    return comparisons


def profile_model_performance(args) -> Dict:
    """Profile detailed model performance."""
    print("\nProfiling model performance...")
    
    optimizer = create_model_optimizer(
        args.model_path, 
        args.vocab_size, 
        args.device
    )
    
    # Create test input
    test_input = torch.randint(0, args.vocab_size, (args.batch_size, args.sequence_length), dtype=torch.long)
    
    # Profile original model
    profile_results = optimizer.profile_model_layers(
        optimizer.original_model,
        test_input,
        num_runs=10
    )
    
    print("Layer profiling completed")
    print(f"Total CPU time: {profile_results['total_cpu_time']:.4f}ms")
    print(f"Total CUDA time: {profile_results['total_cuda_time']:.4f}ms")
    
    return profile_results


def generate_benchmark_report(all_results: Dict, args) -> str:
    """Generate comprehensive benchmark report."""
    report_path = Path(args.output_dir) / "benchmark_report.json"
    
    # Add metadata
    report_data = {
        'metadata': {
            'model_path': args.model_path,
            'vocab_size': args.vocab_size,
            'batch_size': args.batch_size,
            'sequence_length': args.sequence_length,
            'num_runs': args.num_runs,
            'device': args.device,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'results': all_results
    }
    
    # Save report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\nBenchmark report saved to: {report_path}")
    return str(report_path)


def print_summary(all_results: Dict):
    """Print benchmark summary."""
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    if 'original' in all_results:
        original = all_results['original']
        print(f"Original Model:")
        print(f"  Inference Time: {original['mean_inference_time']:.4f}s")
        print(f"  Model Size: {original['model_size_mb']:.2f} MB")
        print(f"  Throughput: {original['throughput_samples_per_sec']:.1f} samples/sec")
    
    if 'comprehensive' in all_results and 'summary' in all_results['comprehensive']:
        summary = all_results['comprehensive']['summary']
        print(f"\nOptimization Results:")
        
        for opt_name, opt_data in summary['optimizations'].items():
            print(f"  {opt_name}:")
            print(f"    Speedup: {opt_data['speedup']:.2f}x")
            print(f"    Size Reduction: {opt_data['size_reduction_ratio']:.1%}")
            print(f"    Inference Time: {opt_data['inference_time']:.4f}s")
    
    print("="*60)


def main():
    """Main benchmarking function."""
    parser = argparse.ArgumentParser(
        description="LSTM Sentiment Classifier - Performance Benchmarking",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('-m', '--model-path', required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--vocab-size', type=int, default=10000,
                       help='Vocabulary size for model initialization')
    
    # Benchmarking options
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for benchmarking')
    parser.add_argument('--sequence-length', type=int, default=500,
                       help='Sequence length for benchmarking')
    parser.add_argument('--num-runs', type=int, default=100,
                       help='Number of inference runs for timing')
    parser.add_argument('--warmup-runs', type=int, default=10,
                       help='Number of warmup runs')
    
    # Optimization options
    parser.add_argument('--enable-static-quantization', action='store_true',
                       help='Enable static quantization (requires calibration)')
    parser.add_argument('--pruning-ratios', type=float, nargs='+', 
                       default=[0.1, 0.2, 0.3],
                       help='Pruning ratios to test')
    parser.add_argument('--default-pruning-ratio', type=float, default=0.2,
                       help='Default pruning ratio for comprehensive optimization')
    parser.add_argument('--structured-pruning', action='store_true',
                       help='Use structured pruning instead of unstructured')
    
    # ONNX options
    parser.add_argument('--export-onnx', action='store_true',
                       help='Export models to ONNX format')
    parser.add_argument('--export-optimized-onnx', action='store_true',
                       help='Export optimized models to ONNX')
    
    # Benchmark modes
    parser.add_argument('--mode', choices=[
        'original', 'quantization', 'pruning', 'onnx', 'comprehensive', 
        'accuracy', 'profile', 'all'
    ], default='all', help='Benchmarking mode')
    
    # Output options
    parser.add_argument('--output-dir', default='benchmark_results',
                       help='Directory to save benchmark results')
    parser.add_argument('--device',
                       help='Device to use (cpu/cuda, default: auto-detect)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Auto-detect device if not specified
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Running benchmarks on device: {args.device}")
    print(f"Model: {args.model_path}")
    print(f"Batch size: {args.batch_size}, Sequence length: {args.sequence_length}")
    print(f"Number of runs: {args.num_runs}")
    
    # Run benchmarks based on mode
    all_results = {}
    
    try:
        if args.mode in ['original', 'all']:
            all_results['original'] = benchmark_original_model(args)
        
        if args.mode in ['quantization', 'all']:
            all_results['quantization'] = run_quantization_benchmark(args)
        
        if args.mode in ['pruning', 'all']:
            all_results['pruning'] = run_pruning_benchmark(args)
        
        if args.mode in ['onnx', 'all']:
            all_results['onnx'] = run_onnx_benchmark(args)
        
        if args.mode in ['comprehensive', 'all']:
            all_results['comprehensive'] = run_comprehensive_optimization(args)
        
        if args.mode in ['accuracy', 'all']:
            all_results['accuracy'] = run_accuracy_validation(args)
        
        if args.mode in ['profile', 'all']:
            all_results['profile'] = profile_model_performance(args)
        
        # Generate report
        report_path = generate_benchmark_report(all_results, args)
        
        # Print summary
        print_summary(all_results)
        
    except Exception as e:
        print(f"Benchmarking failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()