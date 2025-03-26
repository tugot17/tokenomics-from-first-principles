import numpy as np
import matplotlib.pyplot as plt

from model_params import ModelParams
from hardware_params import HardwareParams
from tokenomics_model import TokenomicsModel
from visualizations import plot_throughput_vs_batch_size

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