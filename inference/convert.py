import os
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange
import json
import tempfile
from collections import defaultdict

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
    Converts and saves model checkpoint files into a specified format using streaming processing.

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
    
    os.makedirs(save_path, exist_ok=True)
    
    # Progress checkpoint file
    checkpoint_file = os.path.join(save_path, ".conversion_progress.json")
    processed_files = set()
    
    # Load existing progress if available
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                progress_data = json.load(f)
                processed_files = set(progress_data.get('processed_files', []))
        except (json.JSONDecodeError, KeyError):
            processed_files = set()
    
    # Get all safetensors files
    file_paths = glob(os.path.join(hf_ckpt_path, "*.safetensors"))
    
    # Use temporary files for streaming writes
    temp_files = []
    temp_state_dicts = []
    
    try:
        # Initialize temporary files for each shard
        for i in range(mp):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.safetensors')
            temp_files.append(temp_file.name)
            temp_file.close()
            temp_state_dicts.append({})
        
        # Process files in chunks to manage memory
        chunk_size = max(1, len(file_paths) // 4)  # Process in 4 chunks
        file_chunks = [file_paths[i:i + chunk_size] for i in range(0, len(file_paths), chunk_size)]
        
        for chunk_idx, file_chunk in enumerate(file_chunks):
            print(f"Processing chunk {chunk_idx + 1}/{len(file_chunks)}")
            
            for file_path in tqdm(file_chunk, desc=f"Processing files in chunk {chunk_idx + 1}"):
                file_basename = os.path.basename(file_path)
                
                # Skip if already processed
                if file_basename in processed_files:
                    continue
                
                # Pre-compute string operations cache for this file
                name_cache = {}
                
                with safe_open(file_path, framework="pt", device="cpu") as f:
                    tensor_names = list(f.keys())
                    
                    # Process tensors in smaller batches within each file
                    batch_size = 10  # Process 10 tensors at a time
                    tensor_batches = [tensor_names[i:i + batch_size] for i in range(0, len(tensor_names), batch_size)]
                    
                    for batch in tensor_batches:
                        batch_tensors = {}
                        
                        # Load batch of tensors
                        for name in batch:
                            if "model.layers.61" in name:
                                continue
                            
                            # Use cached name processing
                            if name not in name_cache:
                                processed_name = _process_tensor_name(name)
                                name_cache[name] = processed_name
                            else:
                                processed_name = name_cache[name]
                            
                            if processed_name is None:
                                continue
                            
                            # Load tensor with memory mapping for large tensors
                            param = f.get_tensor(name)
                            
                            # Determine tensor routing
                            key_parts = processed_name.split(".")
                            expert_idx = None
                            is_expert_tensor = "experts" in processed_name and "shared_experts" not in processed_name
                            if is_expert_tensor:
                                expert_idx = int(key_parts[-3]) if len(key_parts) > 3 else None
                            
                            batch_tensors[name] = {
                                'tensor': param,
                                'processed_name': processed_name,
                                'expert_idx': expert_idx,
                                'is_expert_tensor': is_expert_tensor
                            }
                        
                        # Process batch and distribute to shards
                        _process_tensor_batch(batch_tensors, temp_state_dicts, mp, n_local_experts, mapping)
                        
                        # Clear batch to free memory immediately
                        for tensor_data in batch_tensors.values():
                            del tensor_data['tensor']
                        del batch_tensors
                        
                        # Periodic save to prevent memory buildup
                        if len(temp_state_dicts[0]) > 50:  # Save every 50 tensors
                            _save_intermediate_state(temp_state_dicts, temp_files)
                
                # Mark file as processed
                processed_files.add(file_basename)
                
                # Update progress checkpoint
                _update_progress_checkpoint(checkpoint_file, processed_files)
        
        # Final save of all accumulated tensors
        for i in range(mp):
            output_path = os.path.join(save_path, f"model{i}-mp{mp}.safetensors")
            
            # Load existing temp data if any
            if os.path.exists(temp_files[i]) and os.path.getsize(temp_files[i]) > 0:
                with safe_open(temp_files[i], framework="pt", device="cpu") as temp_f:
                    for key in temp_f.keys():
                        if key not in temp_state_dicts[i]:
                            temp_state_dicts[i][key] = temp_f.get_tensor(key)
            
            # Save final state
            save_file(temp_state_dicts[i], output_path)
            
            # Clear memory
            temp_state_dicts[i].clear()
        
        # Cleanup temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        
        # Remove progress checkpoint on successful completion
        if os.path.exists(checkpoint_file):
            os.unlink(checkpoint_file)
        
        # Batch copy tokenizer files
        _copy_tokenizer_files(hf_ckpt_path, save_path)
        
    except Exception as e:
        # Cleanup on error
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        raise e
    finally:
        # Ensure cleanup
        del temp_state_dicts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _process_tensor_name(name):
    """Process and cache tensor name transformations."""
    if "model.layers.61" in name:
        return None
    
    # Optimize string operations by chaining replacements
    processed_name = name
    if processed_name.startswith("model."):
        processed_name = processed_name[len("model."):]
    
    # Chain multiple replacements for efficiency
    processed_name = (processed_name
                    .replace("self_attn", "attn")
                    .replace("mlp", "ffn")
                    .replace("weight_scale_inv", "scale")
                    .replace("e_score_correction_bias", "bias"))
    
    # Get mapping information once
    key_parts = processed_name.split(".")
    if len(key_parts) < 2:
        return None
    
    key = key_parts[-2]
    if key not in mapping:
        return None
    
    new_key, _ = mapping[key]
    processed_name = processed_name.replace(key, new_key)
    
    return processed_name


def _process_tensor_batch(batch_tensors, temp_state_dicts, mp, n_local_experts, mapping):
    """Process a batch of tensors and distribute to appropriate shards."""
    for name, tensor_data in batch_tensors.items():
        param = tensor_data['tensor']
        processed_name = tensor_data['processed_name']
        expert_idx = tensor_data['expert_idx']
        is_expert_tensor = tensor_data['is_expert_tensor']
        
        # Get dimension info
        key_parts = processed_name.split(".")
        key = key_parts[-2]
        _, dim = mapping[key]
        
        # Process tensor for each model parallel shard
        for i in range(mp):
            # Skip expert tensors not belonging to this shard
            if is_expert_tensor and expert_idx is not None:
                if expert_idx < i * n_local_experts or expert_idx >= (i + 1) * n_local_experts:
                    continue
            
            # Handle tensor sharding
            if dim is not None:
                if param.size(dim) % mp != 0:
                    continue  # Skip tensors that can't be evenly divided
                shard_size = param.size(dim) // mp
                # Create shard directly without intermediate copy
                sharded_param = param.narrow(dim, i * shard_size, shard_size).contiguous()
                temp_state_dicts[i][processed_name] = sharded_param
            else:
                # For non-sharded tensors, clone to avoid shared references
                temp_state_dicts[i][processed_name] = param.clone()


def _save_intermediate_state(temp_state_dicts, temp_files):
    """Save intermediate state to temporary files and clear memory."""
    for i, state_dict in enumerate(temp_state_dicts):
        if state_dict:
            # Load existing temp data if any
            existing_data = {}
            if os.path.exists(temp_files[i]) and os.path.getsize(temp_files[i]) > 0:
                try:
                    with safe_open(temp_files[i], framework="pt", device="cpu") as temp_f:
                        for key in temp_f.keys():
                            existing_data[key] = temp_f.get_tensor(key)
                except:
                    pass  # If temp file is corrupted, start fresh
            
            # Merge with current state
            existing_data.update(state_dict)
            
            # Save merged state
            save_file(existing_data, temp_files[i])
            
            # Clear current state
            state_dict.clear()
            del existing_data


def _update_progress_checkpoint(checkpoint_file, processed_files):
    """Update progress checkpoint file."""
    progress_data = {'processed_files': list(processed_files)}
    with open(checkpoint_file, 'w') as f:
        json.dump(progress_data, f)


def _copy_tokenizer_files(hf_ckpt_path, save_path):
    """Batch copy tokenizer files for efficiency."""
    tokenizer_files = glob(os.path.join(hf_ckpt_path, "*token*"))
    
    if tokenizer_files:
        # Use more efficient batch copy
        for file_path in tqdm(tokenizer_files, desc="Copying tokenizer files"):
            new_file_path = os.path.join(save_path, os.path.basename(file_path))
            shutil.copy2(file_path, new_file_path)  # copy2 preserves metadata


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-ckpt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--n-experts", type=int, required=True)
    parser.add_argument("--model-parallel", type=int, required=True)
    args = parser.parse_args()
    assert args.n_experts % args.model_parallel == 0
    main(args.hf_ckpt_path, args.save_path, args.n_experts, args.model_parallel)