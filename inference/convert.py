import os
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file


mapping = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate": ("gate", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norm", None),
    "lm_head": ("head", 0),
    "scale": ("scale", None),
}


def main(hf_ckpt_path, save_path, n_experts, mp):
    """
    Converts and saves model checkpoint files into a specified format.

    Args:
        hf_ckpt_path (str): Path to the directory containing the input checkpoint files.
        save_path (str): Path to the directory where the converted checkpoint files will be saved.
        n_experts (int): Total number of experts in the model.
        mp (int): Model parallelism factor.
        
    Returns:
        None
    """
    torch.set_num_threads(8)
    n_local_experts = n_experts // mp
    
    # Pre-compile regex patterns for better performance
    model_prefix = "model."
    model_prefix_len = len(model_prefix)
    
    # Pre-compute expert ranges for each partition
    expert_ranges = [(i * n_local_experts, (i + 1) * n_local_experts) for i in range(mp)]
    
    # Create save directory once
    os.makedirs(save_path, exist_ok=True)
    
    # Process each file and write immediately to reduce memory usage
    file_paths = glob(os.path.join(hf_ckpt_path, "*.safetensors"))
    
    for i in range(mp):
        state_dict = {}
        
        for file_path in tqdm(file_paths, desc=f"Processing partition {i+1}/{mp}"):
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for name in f.keys():
                    if "model.layers.61" in name:
                        continue
                    
                    # Process name transformations
                    if name.startswith(model_prefix):
                        processed_name = name[model_prefix_len:]
                    else:
                        processed_name = name
                    
                    processed_name = processed_name.replace("self_attn", "attn")
                    processed_name = processed_name.replace("mlp", "ffn")
                    processed_name = processed_name.replace("weight_scale_inv", "scale")
                    processed_name = processed_name.replace("e_score_correction_bias", "bias")
                    
                    # Extract key and apply mapping
                    key = processed_name.split(".")[-2]
                    assert key in mapping
                    new_key, dim = mapping[key]
                    processed_name = processed_name.replace(key, new_key)
                    
                    # Check if this parameter belongs to current partition
                    is_expert = "experts" in processed_name and "shared_experts" not in processed_name
                    if is_expert:
                        expert_idx = int(processed_name.split(".")[-3])
                        start_idx, end_idx = expert_ranges[i]
                        if expert_idx < start_idx or expert_idx >= end_idx:
                            continue
                    
                    # Load and process tensor
                    param = f.get_tensor(name)
                    
                    # Apply sharding if needed
                    if not is_expert and dim is not None:
                        assert param.size(dim) % mp == 0
                        shard_size = param.size(dim) // mp
                        param = param.narrow(dim, i * shard_size, shard_size).contiguous()
                    
                    state_dict[processed_name] = param
        
        # Save this partition's state dict
        save_file(state_dict, os.path.join(save_path, f"model{i}-mp{mp}.safetensors"))
        
        # Clear the state dict to free memory
        del state_dict

    # Copy tokenizer files
    for file_path in glob(os.path.join(hf_ckpt_path, "*token*")):
        new_file_path = os.path.join(save_path, os.path.basename(file_path))
        shutil.copyfile(file_path, new_file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--model-parallel", type=int, required=True)
    args = parser.parse_args()
    assert args.n_experts % args.model_parallel == 0
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)