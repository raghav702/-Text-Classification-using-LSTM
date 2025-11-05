#!/usr/bin/env python3
"""
Utility script to run hyperparameter optimization with configuration files.

This script provides an easy way to run hyperparameter optimization experiments
using YAML configuration files with predefined settings.
"""

import argparse
import os
import sys
import yaml
import logging
from datetime import datetime

# Import the main hyperparameter optimization script
from hyperparameter_optimization import main as hp_main


def load_config(config_path: str, config_section: str = None) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        config_section: Specific section to load (optional)
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if config_section:
        if config_section not in config:
            raise ValueError(f"Configuration section '{config_section}' not found")
        
        # Merge base config with section-specific config
        base_config = {k: v for k, v in config.items() if not isinstance(v, dict) or k == config_section}
        section_config = config[config_section]
        
        # Deep merge configurations
        merged_config = base_config.copy()
        for key, value in section_config.items():
            if key in merged_config and isinstance(merged_config[key], dict) and isinstance(value, dict):
                merged_config[key].update(value)
            else:
                merged_config[key] = value
        
        return merged_config
    
    return config


def config_to_args(config: dict) -> list:
    """
    Convert configuration dictionary to command line arguments.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of command line arguments
    """
    args = []
    
    # Optimization settings
    if 'optimization' in config:
        opt_config = config['optimization']
        if 'method' in opt_config:
            args.extend(['--method', opt_config['method']])
        if 'n_iter' in opt_config:
            args.extend(['--n-iter', str(opt_config['n_iter'])])
        if 'n_calls' in opt_config:
            args.extend(['--n-calls', str(opt_config['n_calls'])])
        if 'n_initial_points' in opt_config:
            args.extend(['--n-initial-points', str(opt_config['n_initial_points'])])
    
    # Cross-validation settings
    if 'cross_validation' in config:
        cv_config = config['cross_validation']
        if 'n_folds' in cv_config:
            args.extend(['--cv-folds', str(cv_config['n_folds'])])
        if cv_config.get('stratified', True):
            args.append('--stratified-cv')
    
    # Data settings
    if 'data' in config:
        data_config = config['data']
        if 'data_dir' in data_config:
            args.extend(['--data-dir', data_config['data_dir']])
        if 'subset_size' in data_config and data_config['subset_size']:
            args.extend(['--subset-size', str(data_config['subset_size'])])
        if 'max_vocab_size' in data_config:
            args.extend(['--max-vocab-size', str(data_config['max_vocab_size'])])
        if 'max_sequence_length' in data_config:
            args.extend(['--max-sequence-length', str(data_config['max_sequence_length'])])
        if 'min_word_freq' in data_config:
            args.extend(['--min-word-freq', str(data_config['min_word_freq'])])
    
    # Hyperparameter space settings
    if 'hyperparameter_space' in config:
        hp_config = config['hyperparameter_space']
        
        if hp_config.get('use_default_space', False):
            args.append('--use-default-space')
        
        # Tuning flags
        tuning_flags = [
            'tune_hidden_dim', 'tune_n_layers', 'tune_dropout', 'tune_embedding_dim',
            'tune_learning_rate', 'tune_weight_decay', 'tune_batch_size',
            'tune_optimizer', 'tune_scheduler', 'tune_gradient_clip'
        ]
        
        for flag in tuning_flags:
            if hp_config.get(flag, False):
                args.append(f'--{flag.replace("_", "-")}')
    
    # Model settings
    if 'model' in config:
        model_config = config['model']
        if 'embedding_dim' in model_config:
            args.extend(['--embedding-dim', str(model_config['embedding_dim'])])
        if 'hidden_dim' in model_config:
            args.extend(['--hidden-dim', str(model_config['hidden_dim'])])
        if 'n_layers' in model_config:
            args.extend(['--n-layers', str(model_config['n_layers'])])
        if 'dropout' in model_config:
            args.extend(['--dropout', str(model_config['dropout'])])
        if model_config.get('bidirectional', True):
            args.append('--bidirectional')
    
    # Training settings
    if 'training' in config:
        train_config = config['training']
        if 'epochs' in train_config:
            args.extend(['--epochs', str(train_config['epochs'])])
        if 'optimizer_type' in train_config:
            args.extend(['--optimizer-type', train_config['optimizer_type']])
        if 'scheduler_type' in train_config:
            args.extend(['--scheduler-type', train_config['scheduler_type']])
        if 'learning_rate' in train_config:
            args.extend(['--learning-rate', str(train_config['learning_rate'])])
        if 'weight_decay' in train_config:
            args.extend(['--weight-decay', str(train_config['weight_decay'])])
        if 'gradient_clip_type' in train_config:
            args.extend(['--gradient-clip-type', train_config['gradient_clip_type']])
        if 'gradient_clip_value' in train_config:
            args.extend(['--gradient-clip-value', str(train_config['gradient_clip_value'])])
        if 'accumulation_steps' in train_config:
            args.extend(['--accumulation-steps', str(train_config['accumulation_steps'])])
        if 'early_stopping_patience' in train_config:
            args.extend(['--early-stopping-patience', str(train_config['early_stopping_patience'])])
    
    # GloVe settings
    if 'glove' in config:
        glove_config = config['glove']
        if glove_config.get('use_glove', True):
            args.append('--use-glove')
        if 'glove_corpus' in glove_config:
            args.extend(['--glove-corpus', glove_config['glove_corpus']])
        if glove_config.get('freeze_embeddings', False):
            args.append('--freeze-embeddings')
        if 'glove_cache_dir' in glove_config:
            args.extend(['--glove-cache-dir', glove_config['glove_cache_dir']])
    
    # System settings
    if 'system' in config:
        sys_config = config['system']
        if 'device' in sys_config:
            args.extend(['--device', sys_config['device']])
        if 'n_jobs' in sys_config:
            args.extend(['--n-jobs', str(sys_config['n_jobs'])])
        if 'seed' in sys_config:
            args.extend(['--seed', str(sys_config['seed'])])
    
    # Output settings
    if 'output' in config:
        output_config = config['output']
        if 'output_dir' in output_config:
            args.extend(['--output-dir', output_config['output_dir']])
    
    # Logging settings
    if 'logging' in config:
        log_config = config['logging']
        if 'log_level' in log_config:
            args.extend(['--log-level', log_config['log_level']])
        if 'log_dir' in log_config:
            args.extend(['--log-dir', log_config['log_dir']])
    
    return args


def main():
    """Main function for configuration-based hyperparameter optimization."""
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization with configuration files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', required=True,
                       help='Path to YAML configuration file')
    parser.add_argument('--config-section', 
                       help='Specific configuration section to use (e.g., fast_search, comprehensive_search)')
    parser.add_argument('--list-sections', action='store_true',
                       help='List available configuration sections and exit')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show command that would be executed without running it')
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config, args.config_section)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # List sections if requested
    if args.list_sections:
        with open(args.config, 'r') as f:
            full_config = yaml.safe_load(f)
        
        print(f"Available configuration sections in {args.config}:")
        for key, value in full_config.items():
            if isinstance(value, dict) and any(isinstance(v, dict) for v in value.values()):
                print(f"  - {key}")
        sys.exit(0)
    
    # Convert configuration to command line arguments
    cmd_args = config_to_args(config)
    
    # Add timestamp to output directory if not specified
    if '--output-dir' not in cmd_args:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"hyperparameter_optimization_{timestamp}"
        if args.config_section:
            output_dir += f"_{args.config_section}"
        cmd_args.extend(['--output-dir', output_dir])
    
    # Show command if dry run
    if args.dry_run:
        print("Command that would be executed:")
        print(f"python hyperparameter_optimization.py {' '.join(cmd_args)}")
        sys.exit(0)
    
    # Print configuration summary
    print("=" * 80)
    print("HYPERPARAMETER OPTIMIZATION CONFIGURATION")
    print("=" * 80)
    print(f"Configuration file: {args.config}")
    if args.config_section:
        print(f"Configuration section: {args.config_section}")
    print(f"Method: {config.get('optimization', {}).get('method', 'random')}")
    print(f"CV Folds: {config.get('cross_validation', {}).get('n_folds', 3)}")
    
    if 'data' in config and 'subset_size' in config['data'] and config['data']['subset_size']:
        print(f"Data subset size: {config['data']['subset_size']}")
    else:
        print("Using full dataset")
    
    print("=" * 80)
    
    # Run hyperparameter optimization
    try:
        # Temporarily modify sys.argv to pass arguments to the main script
        original_argv = sys.argv.copy()
        sys.argv = ['hyperparameter_optimization.py'] + cmd_args
        
        # Run the main hyperparameter optimization function
        hp_main()
        
        # Restore original argv
        sys.argv = original_argv
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nOptimization failed: {e}")
        raise


if __name__ == '__main__':
    main()