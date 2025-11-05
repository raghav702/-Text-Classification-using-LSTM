#!/usr/bin/env python3
"""
Complete workflow example for LSTM Sentiment Classifier.

This script demonstrates the complete workflow from training to evaluation
using the LSTM sentiment classifier.
"""

import os
import sys
import subprocess
import tempfile
import shutil

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def run_command(cmd, description):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, cwd='..')
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ {description} failed: {e}")
        return False


def main():
    """Run complete workflow example."""
    print("LSTM Sentiment Classifier - Complete Workflow Example")
    print("=" * 60)
    print("This example demonstrates the complete workflow:")
    print("1. Setup project structure")
    print("2. Train a quick model (5 epochs)")
    print("3. Make predictions")
    print("4. Evaluate the model")
    print("=" * 60)
    
    # Change to parent directory
    os.chdir('..')
    
    # Step 1: Setup project
    if not run_command(['python', 'lstm_sentiment.py', 'setup'], 
                      "Setting up project structure"):
        return False
    
    # Step 2: Create quick training config
    if not run_command(['python', 'lstm_sentiment.py', 'config', '--create', 'quick', '--output', 'quick_config.yaml'],
                      "Creating quick training configuration"):
        return False
    
    # Step 3: Train model (assuming IMDB data is available)
    print("\nNOTE: This step requires IMDB dataset in data/imdb/")
    print("Download the dataset manually or skip this step if not available.")
    
    user_input = input("\nDo you want to run training? (requires IMDB data) [y/N]: ")
    if user_input.lower() in ['y', 'yes']:
        if not run_command(['python', 'lstm_sentiment.py', 'train', '--config', 'quick_config.yaml'],
                          "Training LSTM model (quick configuration)"):
            print("Training failed. This is expected if IMDB data is not available.")
            print("Download IMDB dataset to data/imdb/ to run training.")
        else:
            # Step 4: Make predictions (if training succeeded)
            model_files = [f for f in os.listdir('models') if f.endswith('.pth') and 'vocab' not in f]
            vocab_files = [f for f in os.listdir('models') if f.endswith('.pth') and 'vocab' in f]
            
            if model_files and vocab_files:
                model_path = os.path.join('models', model_files[0])
                vocab_path = os.path.join('models', vocab_files[0])
                
                # Single prediction
                run_command(['python', 'lstm_sentiment.py', 'predict', 
                           '-m', model_path, '-v', vocab_path, 
                           '-t', 'This movie was absolutely fantastic!'],
                          "Making single prediction")
                
                # Evaluation
                run_command(['python', 'lstm_sentiment.py', 'evaluate',
                           '-m', model_path, '-v', vocab_path, '-d', 'data/imdb'],
                          "Evaluating model performance")
            else:
                print("No trained model found for prediction and evaluation.")
    
    # Step 5: Demonstrate configuration management
    run_command(['python', 'lstm_sentiment.py', 'config', '--show', 'quick_config.yaml'],
              "Showing configuration file")
    
    print("\n" + "=" * 60)
    print("WORKFLOW EXAMPLE COMPLETED")
    print("=" * 60)
    print("Key files created:")
    print("- quick_config.yaml: Training configuration")
    if os.path.exists('models'):
        model_files = os.listdir('models')
        if model_files:
            print("- models/: Trained model files")
            for f in model_files[:3]:  # Show first 3 files
                print(f"  - {f}")
    
    print("\nNext steps:")
    print("1. Download IMDB dataset to data/imdb/")
    print("2. Run full training: python lstm_sentiment.py train --config configs/examples/production_training.yaml")
    print("3. Experiment with different configurations")
    print("4. Compare multiple models using evaluation")
    
    # Cleanup
    if os.path.exists('quick_config.yaml'):
        os.remove('quick_config.yaml')
        print("\nCleaned up temporary configuration file.")


if __name__ == '__main__':
    main()