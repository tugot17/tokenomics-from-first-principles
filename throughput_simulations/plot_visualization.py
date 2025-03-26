import numpy as np
import matplotlib.pyplot as plt

from model_params import ModelParams
from hardware_params import HardwareParams
from tokenomics_model import TokenomicsModel

def plot_throughput_vs_batch_size(model, input_tokens=1024, output_tokens=128):
    """Plot total and per-request throughput against batch size with bottleneck analysis"""
    # Hardcoded batch sizes as requested
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    total_throughputs = []
    per_request_throughputs = []
    
    # Analyze bottlenecks
    bottleneck_analysis = model.analyze_bottlenecks(input_tokens, output_tokens, batch_sizes)
    compute_times = bottleneck_analysis["compute_times"]
    memory_times = bottleneck_analysis["memory_times"]
    bottlenecks = bottleneck_analysis["bottlenecks"]
    
    for batch_size in batch_sizes:
        total_throughput = model.calculate_total_throughput(input_tokens, output_tokens, batch_size)
        per_request_throughput = model.calculate_per_request_throughput(input_tokens, output_tokens, batch_size)
        
        total_throughputs.append(total_throughput)
        per_request_throughputs.append(per_request_throughput)
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot total throughput
    ax1.plot(batch_sizes, total_throughputs, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Total Throughput (tokens/second)')
    ax1.set_title(f'Theoretical Total Throughput vs Batch Size\n{model.model.name} on {model.hardware.num_gpus}x {model.hardware.name}\nInput Tokens: {input_tokens}, Output Tokens: {output_tokens}')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)  # Use log scale for x-axis
    ax1.set_xticks(batch_sizes)
    ax1.set_xticklabels(batch_sizes)
    
    # Add values above points in total throughput plot
    for i, v in enumerate(total_throughputs):
        ax1.text(batch_sizes[i], v + (max(total_throughputs) * 0.03), f"{v:.1f}", 
                ha='center', va='bottom', fontweight='bold')
    
    # Plot per-request throughput
    ax2.plot(batch_sizes, per_request_throughputs, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Per-Request Throughput (tokens/second)')
    ax2.set_title(f'Theoretical Per-Request Throughput vs Batch Size\n{model.model.name} on {model.hardware.num_gpus}x {model.hardware.name}\nInput Tokens: {input_tokens}, Output Tokens: {output_tokens}')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)  # Use log scale for x-axis
    ax2.set_xticks(batch_sizes)
    ax2.set_xticklabels(batch_sizes)
    
    # Add values above points in per-request throughput plot
    for i, v in enumerate(per_request_throughputs):
        ax2.text(batch_sizes[i], v + (max(per_request_throughputs) * 0.03), f"{v:.1f}", 
                ha='center', va='bottom', fontweight='bold')
    
    # Create a figure for bottleneck analysis
    fig2, ax3 = plt.subplots(figsize=(12, 7))
    
    # Plot compute and memory times
    ax3.plot(batch_sizes, compute_times, 'b-', linewidth=2, label='Compute Time')
    ax3.plot(batch_sizes, memory_times, 'r-', linewidth=2, label='Memory Time')
    
    # Add markers for bottlenecks
    for i, bottleneck in enumerate(bottlenecks):
        marker_color = 'blue' if bottleneck == 'compute' else 'red'
        marker_style = 'o' if bottleneck == 'compute' else 's'
        ax3.plot(batch_sizes[i], max(compute_times[i], memory_times[i]), 
                marker=marker_style, markersize=10, color=marker_color)
    
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title(f'Compute vs Memory Bottleneck Analysis\n{model.model.name} on {model.hardware.num_gpus}x {model.hardware.name}\nInput Tokens: {input_tokens}, Output Tokens: {output_tokens}')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)  # Use log scale for x-axis
    ax3.set_xticks(batch_sizes)
    ax3.set_xticklabels(batch_sizes)
    ax3.legend()
    
    # Find crossover point
    for i in range(len(batch_sizes) - 1):
        if bottlenecks[i] != bottlenecks[i+1]:
            crossover_point = batch_sizes[i+1]
            ax3.axvline(x=crossover_point, color='green', linestyle='--', 
                    label=f'Bottleneck Shift at Batch Size {crossover_point}')
            ax3.legend()
            break
    
    # Create a data table with the values and bottlenecks
    data = []
    for i, bs in enumerate(batch_sizes):
        data.append((
            bs, 
            round(total_throughputs[i], 2),
            round(per_request_throughputs[i], 2),
            bottlenecks[i],
            round(compute_times[i], 6),
            round(memory_times[i], 6)
        ))
    
    print("\nThroughput and Bottleneck Data:")
    print(f"{'Batch Size':<12} {'Total Throughput':<20} {'Per-Request':<15} {'Bottleneck':<15} {'Compute Time':<15} {'Memory Time':<15}")
    print("-" * 95)
    for bs, tt, prt, bot, ct, mt in data:
        print(f"{bs:<12} {tt:<20.2f} {prt:<15.2f} {bot:<15} {ct:<15.6f} {mt:<15.6f}")
    
    plt.tight_layout()
    
    # Save the figures to files
    fig.savefig(f"throughput_plots_{model.model.name.replace(' ', '_')}_{model.hardware.num_gpus}x_{input_tokens}in_{output_tokens}out.png", 
                dpi=300, bbox_inches='tight')
    fig2.savefig(f"bottleneck_analysis_{model.model.name.replace(' ', '_')}_{model.hardware.num_gpus}x_{input_tokens}in_{output_tokens}out.png", 
                dpi=300, bbox_inches='tight')
    
    print(f"\nFigures saved as:")
    print(f"- throughput_plots_{model.model.name.replace(' ', '_')}_{model.hardware.num_gpus}x_{input_tokens}in_{output_tokens}out.png")
    print(f"- bottleneck_analysis_{model.model.name.replace(' ', '_')}_{model.hardware.num_gpus}x_{input_tokens}in_{output_tokens}out.png")
    
    return fig, fig2, data


def main():
    # Create model and hardware objects
    llama = ModelParams()  # Default is LLama 3.3 70B
    h100 = HardwareParams()  # Default is 4x NVIDIA H100

    # Create tokenomics model
    model = TokenomicsModel(llama, h100)

    # Print basic information
    print(f"Model: {llama.name}")
    print(f"Total parameters: {llama.total_params / 1e9:.2f} billion")
    print(f"Model size: {llama.model_size_bytes / 1e9:.2f} GB")
    print()
    print(f"Hardware: {h100.name} x {h100.num_gpus}")
    print(f"Effective peak performance: {h100.effective_tflops:.2f} TFLOPS")
    print(f"Effective memory bandwidth: {h100.effective_memory_bandwidth_GBs:.2f} GB/s")
    print(f"Total memory: {h100.total_memory_GB:.2f} GB")

    # Generate and show the plots
    input_tokens = 2035
    output_tokens = 300

    throughput_plot, bottleneck_plot, data_points = plot_throughput_vs_batch_size(
        model, input_tokens, output_tokens
    )

    # Show the plots
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()