import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm
from collections import OrderedDict

import torch
from safetensors.torch import load_file, save_file

from kernel import weight_dequant

def main(fp8_path, bf16_path):
    """
    Converts FP8 weights to BF16 and saves the converted weights.

    This function reads FP8 weights from the specified directory, converts them to BF16,
    and saves the converted weights to another specified directory. It also updates the
    model index file to reflect the changes.

    Args:
    fp8_path (str): The path to the directory containing the FP8 weights and model index file.
    bf16_path (str): The path to the directory where the converted BF16 weights will be saved.

    Raises:
    KeyError: If a required scale_inv tensor is missing for a weight.

    Notes:
    - The function assumes that the FP8 weights are stored in safetensor files.
    - The function caches loaded safetensor files to optimize memory usage.
    - The function updates the model index file to remove references to scale_inv tensors.
    """
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]
    
    # Pre-compute scale_inv mappings for faster lookup
    scale_inv_map = {k: v for k, v in weight_map.items() if k.endswith("_scale_inv")}
    
    # LRU cache for loaded safetensor files with size-based eviction
    loaded_files = OrderedDict()
    max_cache_size_gb = 4  # Maximum cache size in GB
    current_cache_size = 0
    fp8_weight_names = []

    def get_file_size_gb(tensor_dict):
        """Calculate approximate memory size of tensor dict in GB"""
        total_bytes = sum(t.numel() * t.element_size() for t in tensor_dict.values())
        return total_bytes / (1024**3)

    def get_tensor(tensor_name, device="cpu"):
        """
        Retrieves a tensor from the cached safetensor files or loads it from disk if not cached.
        Uses LRU caching with size-based eviction.

        Args:
            tensor_name (str): The name of the tensor to retrieve.
            device (str): Device to load tensor to. Default is "cpu".

        Returns:
            torch.Tensor: The retrieved tensor.

        Raises:
            KeyError: If the tensor does not exist in the safetensor file.
        """
        nonlocal current_cache_size
        
        file_name = weight_map[tensor_name]
        
        # Move to end (most recently used) if already in cache
        if file_name in loaded_files:
            loaded_files.move_to_end(file_name)
            return loaded_files[file_name][tensor_name].to(device)
        
        # Load new file
        file_path = os.path.join(fp8_path, file_name)
        new_tensors = load_file(file_path, device="cpu")  # Load to CPU first
        new_size = get_file_size_gb(new_tensors)
        
        # Evict old files if cache would exceed size limit
        while current_cache_size + new_size > max_cache_size_gb and loaded_files:
            oldest_file, oldest_tensors = loaded_files.popitem(last=False)
            current_cache_size -= get_file_size_gb(oldest_tensors)
            del oldest_tensors
        
        # Add new file to cache
        loaded_files[file_name] = new_tensors
        current_cache_size += new_size
        
        return new_tensors[tensor_name].to(device)

    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    safetensor_files.sort()
    
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        
        # Load current file to CPU first for efficient processing
        current_state_dict = load_file(safetensor_file, device="cpu")
        
        # Batch process tensors to improve GPU utilization
        fp8_weights_batch = []
        scale_inv_batch = []
        weight_names_batch = []
        new_state_dict = {}
        
        for weight_name, weight in current_state_dict.items():
            if weight_name.endswith("_scale_inv"):
                continue
            elif weight.element_size() == 1:  # FP8 weight
                scale_inv_name = f"{weight_name}_scale_inv"
                try:
                    # Check if scale_inv exists in weight_map
                    if scale_inv_name in scale_inv_map:
                        scale_inv = get_tensor(scale_inv_name, device="cpu")
                        fp8_weights_batch.append(weight.cuda())
                        scale_inv_batch.append(scale_inv.cuda())
                        weight_names_batch.append(weight_name)
                        fp8_weight_names.append(weight_name)
                    else:
                        print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                        new_state_dict[weight_name] = weight
                except KeyError:
                    print(f"Warning: Missing scale_inv tensor for {weight_name}, skipping conversion")
                    new_state_dict[weight_name] = weight
            else:
                new_state_dict[weight_name] = weight
        
        # Process FP8 weights in batch for better GPU utilization
        for i, (fp8_weight, scale_inv, weight_name) in enumerate(zip(fp8_weights_batch, scale_inv_batch, weight_names_batch)):
            new_state_dict[weight_name] = weight_dequant(fp8_weight, scale_inv)
            # Free GPU memory immediately after processing each tensor
            del fp8_weight, scale_inv
            if i % 10 == 9:  # Periodic cleanup every 10 tensors
                torch.cuda.empty_cache()
        
        # Clear batch arrays to free memory
        del fp8_weights_batch, scale_inv_batch, weight_names_batch
        
        new_safetensor_file = os.path.join(bf16_path, file_name)
        save_file(new_state_dict, new_safetensor_file)
        
        # Explicit cleanup after processing each file
        del current_state_dict, new_state_dict
        torch.cuda.empty_cache()
    
    # Update model index
    new_model_index_file = os.path.join(bf16_path, "model.safetensors.index.json")
    for weight_name in fp8_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map:
            weight_map.pop(scale_inv_name)
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)

    # Explicitly free up memory
    del loaded_files
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-fp8-hf-path", type=str, required=True)
    parser.add_argument("--output-bf16-hf-path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_fp8_hf_path, args.output_bf16_hf_path)