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
    
    # Pre-compute expert ranges for each model parallel rank
    expert_ranges = [(i * n_local_experts, (i + 1) * n_local_experts) for i in range(mp)]
    
    # Create save directory early
    os.makedirs(save_path, exist_ok=True)
    
    # Process each file separately to reduce peak memory usage
    safetensor_files = glob(os.path.join(hf_ckpt_path, "*.safetensors"))
    
    for file_path in tqdm(safetensor_files):
        state_dicts = [{} for _ in range(mp)]
        
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                if "model.layers.61" in name:
                    continue
                    
                param = f.get_tensor(name)
                
                # Apply name transformations
                if name.startswith("model."):
                    name = name[6:]  # More efficient than name[len("model."):]
                
                # Chain replace operations to avoid multiple string operations
                name = (name.replace("self_attn", "attn")
                           .replace("mlp", "ffn")
                           .replace("weight_scale_inv", "scale")
                           .replace("e_score_correction_bias", "bias"))
                
                key = name.split(".")[-2]
                assert key in mapping
                new_key, dim = mapping[key]
                name = name.replace(key, new_key)
                
                # Check if this is an expert parameter
                is_expert = "experts" in name and "shared_experts" not in name
                expert_idx = None
                if is_expert:
                    expert_idx = int(name.split(".")[-3])
                
                for i in range(mp):
                    # Skip expert if it doesn't belong to this rank
                    if is_expert:
                        start_idx, end_idx = expert_ranges[i]
                        if expert_idx < start_idx or expert_idx >= end_idx:
                            continue
                        state_dicts[i][name] = param
                    elif dim is not None:
                        # Shard parameter along specified dimension
                        assert param.size(dim) % mp == 0
                        shard_size = param.size(dim) // mp
                        new_param = param.narrow(dim, i * shard_size, shard_size).contiguous()
                        state_dicts[i][name] = new_param
                    else:
                        # Parameter is replicated across all ranks
                        state_dicts[i][name] = param
        
        # Save state dicts for this file and clear memory
        for i in range(mp):
            if state_dicts[i]:  # Only save if there are parameters
                output_file = os.path.join(save_path, f"model{i}-mp{mp}-{os.path.basename(file_path)}")
                save_file(state_dicts[i], output_file)
        
        # Clear memory after processing each file
        del state_dicts

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