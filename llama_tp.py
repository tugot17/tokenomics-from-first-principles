import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn

from llama2_model import Transformer, ModelArgs

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel
)

"""
This script demonstrates how to perform Tensor Parallelism (TP) on a Llama2 model
without Data Parallelism (DP). We set up a single device mesh for TP and
execute a single forward pass.
"""

def main():
    tp_size = 2  # Adjust this value based on the number of GPUs you want to use for TP

    # Initialize the distributed process group
    dist.init_process_group(backend='nccl')

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Starting PyTorch TP example on rank {rank}.")

    # Ensure the world size matches the TP size
    assert world_size == tp_size, f"World size {world_size} must be equal to TP size {tp_size}"

    # Create a device mesh with only the tensor parallel dimension
    device_mesh = init_device_mesh("cuda", (tp_size,), mesh_dim_names=("tp",))

    # Create the model and move it to GPU
    simple_llama2_config = ModelArgs(dim=256, n_layers=2, n_heads=16, vocab_size=32000)
    model = Transformer.from_model_args(simple_llama2_config).to("cuda")

    # Initialize model weights
    model.init_weights()

    # Parallelize the first embedding and the last linear output projection
    model = parallelize_module(
        model,
        device_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate()
            ),
        }
    )

    # Apply tensor parallelism to each transformer block
    for layer_id, transformer_block in enumerate(model.layers):
        layer_tp_plan = {
            "attention_norm": SequenceParallel(),
            "attention": PrepareModuleInput(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            "attention.wq": ColwiseParallel(),
            "attention.wk": ColwiseParallel(),
            "attention.wv": ColwiseParallel(),
            "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward": PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": ColwiseParallel(),
            "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
            "feed_forward.w3": ColwiseParallel(),
        }

        # Adjust attention module to use the local number of heads
        attn_layer = transformer_block.attention
        attn_layer.n_heads = attn_layer.n_heads // device_mesh.size()
        attn_layer.n_kv_heads = attn_layer.n_kv_heads // device_mesh.size()

        # Apply the custom parallelization plan to the transformer block
        parallelize_module(
            module=transformer_block,
            device_mesh=device_mesh,
            parallelize_plan=layer_tp_plan
        )

    # Generate input and perform a single forward pass
    torch.manual_seed(0)
    inp = torch.randint(32000, (8, 256), device="cuda")  # Input shape: [sequence_length, batch_size]

    output = model(inp)
    print(f"Rank {rank} forward pass completed. Output shape: {output.shape}")

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
