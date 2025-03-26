import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import sys
import os

from model_params import ModelParams
from hardware_params import HardwareParams
from tokenomics_model import TokenomicsModel
from visualizations import plot_throughput_vs_batch_size, plot_throughput_comparison
from benchmark_utils import load_benchmark_data

def run_simulation(model, hardware, input_tokens=2035, output_tokens=300):
    """Run a simulation without benchmark comparison"""
    # Create tokenomics model
    model_obj = TokenomicsModel(model, hardware)

    # Print basic information
    print(f"Model: {model.name}")
    print(f"Total parameters: {model.total_params / 1e9:.2f} billion")
    print(f"Model size: {model.model_size_bytes / 1e9:.2f} GB")
    print()
    print(f"Hardware: {hardware.name} x {hardware.num_gpus}")
    print(f"Effective peak performance: {hardware.effective_tflops:.2f} TFLOPS")
    print(f"Effective memory bandwidth: {hardware.effective_memory_bandwidth_GBs:.2f} GB/s")
    print(f"Total memory: {hardware.total_memory_GB:.2f} GB")

    # Generate and show the plots
    throughput_plot, bottleneck_plot, data_points = plot_throughput_vs_batch_size(
        model_obj, input_tokens, output_tokens
    )

    # Show the plots
    plt.tight_layout()
    plt.show()
    
    return throughput_plot, bottleneck_plot, data_points

def run_benchmark_comparison(benchmark_file, model_type=None, hardware_type=None):
    """Run a simulation with benchmark comparison"""
    # Load the benchmark data
    results, metadata, benchmark_model_name = load_benchmark_data(benchmark_file)
    
    # Use provided model or create a new one
    if model_type is None:
        # Use the model from the benchmark if available, otherwise use default
        model = ModelParams(name="llama-3-3-70b")
    else:
        model = model_type
    
    # Use default hardware or custom hardware params if provided
    hardware = hardware_type if hardware_type is not None else HardwareParams()
    
    # Create tokenomics model
    tokenomics_model = TokenomicsModel(model, hardware)
    
    # Use the input and output token counts from the benchmark
    batch_size_1_data = results.get("1", {})
    input_tokens = int(batch_size_1_data.get("avg_input_tokens", 2035))
    output_tokens = int(batch_size_1_data.get("avg_output_tokens", 300))
    
    # Get the batch sizes from the metadata if available, otherwise from the results
    if "batch_sizes" in metadata:
        batch_sizes = metadata["batch_sizes"]
    else:
        batch_sizes = sorted([int(k) for k in results.keys()])
    
    # Print basic information
    print(f"Model: {model.name}")
    print(f"Total parameters: {model.total_params / 1e9:.2f} billion")
    print(f"Model size: {model.model_size_bytes / 1e9:.2f} GB")
    print()
    print(f"Hardware: {hardware.name} x {hardware.num_gpus}")
    print(f"Effective peak performance: {hardware.effective_tflops:.2f} TFLOPS")
    print(f"Effective memory bandwidth: {hardware.effective_memory_bandwidth_GBs:.2f} GB/s")
    print(f"Total memory: {hardware.total_memory_GB:.2f} GB")
    print()
    print(f"Benchmark: {benchmark_model_name}")
    print(f"Input tokens: {input_tokens}")
    print(f"Output tokens: {output_tokens}")
    print(f"Batch sizes: {batch_sizes}")
    
    # Generate and show the plots
    throughput_plot, bottleneck_plot, efficiency_plot, data_points = plot_throughput_comparison(
        tokenomics_model, input_tokens, output_tokens, results, batch_sizes
    )
    
    # Show the plots
    plt.tight_layout()
    plt.show()
    
    return throughput_plot, bottleneck_plot, efficiency_plot, data_points

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='LLM Tokenomics Model')
    parser.add_argument('--benchmark', '-b', type=str, help='Path to benchmark JSON file')
    parser.add_argument('--model', '-m', type=str, default='llama-3-3-70b', 
                        choices=['llama-3-1-8b', 'llama-3-3-70b'],
                        help='Model name (default: llama-3-3-70b)')
    parser.add_argument('--input-tokens', '-i', type=int, default=2035,
                        help='Number of input tokens (default: 2035)')
    parser.add_argument('--output-tokens', '-o', type=int, default=300,
                        help='Number of output tokens (default: 300)')
    parser.add_argument('--gpus', '-g', type=int, default=4,
                        help='Number of GPUs (default: 4)')
    
    # Parse arguments or use hardcoded values if run without arguments
    if len(sys.argv) > 1:
        args = parser.parse_args()
        benchmark_file = args.benchmark
        model_name = args.model
        input_tokens = args.input_tokens
        output_tokens = args.output_tokens
        num_gpus = args.gpus
    else:
        # Default values or look for a JSON file in the current directory
        benchmark_file = None
        json_files = [f for f in os.listdir('.') if f.endswith('.json')]
        if json_files:
            benchmark_file = json_files[0]  # Use the first JSON file found
            print(f"Using benchmark file: {benchmark_file}")
        
        model_name = "llama-3-3-70b"
        input_tokens = 2035
        output_tokens = 300
        num_gpus = 4
    
    model = ModelParams(name=model_name)
    hardware = HardwareParams(num_gpus=num_gpus)
    
    # Run simulation with or without benchmark comparison
    if benchmark_file:
        run_benchmark_comparison(benchmark_file, model, hardware)
    else:
        run_simulation(model, hardware, input_tokens, output_tokens)

if __name__ == "__main__":
    main()