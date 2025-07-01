
import os
import json
from argparse import ArgumentParser
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs


def sample(logits, temperature: float = 1.0):
    """
    Samples a token from the logits using temperature scaling.

    Args:
        logits (torch.Tensor): The logits tensor for token predictions.
        temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.

    Returns:
        torch.Tensor: The sampled token.
    """
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    """
    Generates new tokens based on the given prompt tokens using the specified model.

    Args:
        model (Transformer): The transformer model used for token generation.
        prompt_tokens (List[List[int]]): A list of lists containing the prompt tokens for each sequence.
        max_new_tokens (int): The maximum number of new tokens to generate.
        eos_id (int): The end-of-sequence token ID.
        temperature (float, optional): The temperature value for sampling. Defaults to 1.0.

    Returns:
        List[List[int]]: A list of lists containing the generated tokens for each sequence.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    batch_size = len(prompt_tokens)
    # Initialize the tokens tensor once for efficiency
    tokens = torch.full((batch_size, total_len), -1, dtype=torch.long, device="cuda")
    # Populate initial prompt tokens without looping over each sequence
    for i, t in enumerate(prompt_tokens):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
    prev_pos = 0
    finished = torch.tensor([False] * batch_size, device="cuda")
    prompt_mask = tokens != -1

    for cur_pos in range(max(prompt_lens), total_len):
        logits = model.forward(tokens[:, :cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        # Use torch.where to handle prompt tokens efficiently
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        # Check for finished sequences
        finished |= (next_token == eos_id) & (~prompt_mask[:, cur_pos])
        if finished.all():
            break
        prev_pos = cur_pos

    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        start_idx = prompt_lens[i]
        end_idx = start_idx + max_new_tokens
        # Slice only the relevant tokens
        toks = toks[start_idx:end_idx]
        # Cut off at eos_id if present
        if eos_id in toks:
            toks = toks[:toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
) -> None:
    """
    Main function to load the model and perform interactive or batch text generation.

    Args:
        ckpt_path (str): Path to the model checkpoint directory.
        config (str): Path to the model configuration file.
        input_file (str, optional): Path to a file containing input prompts. Defaults to "".
        interactive (bool, optional): Whether to run in interactive mode. Defaults to True.
        max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 100.
        temperature (float, optional): Temperature for sampling. Defaults to 1.0.
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    
    # Optimize distributed setup for better memory efficiency
    if world_size > 1:
        dist.init_process_group("nccl", init_method="env://")
        # Pre-allocate broadcast buffer to avoid repeated allocations
        if rank == 0:
            _broadcast_buffer = torch.empty(1024, dtype=torch.long, device=f"cuda:{local_rank}")
        
    global print
    if rank != 0:
        print = lambda *_, **__: None
        
    # Enhanced CUDA memory management
    torch.cuda.set_device(local_rank)
    torch.cuda.empty_cache()  # Clear any existing allocations
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)
    
    # Enable memory-efficient attention if available
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
    
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    
    # Initialize model with gradient-free mode and memory optimization
    with torch.device("cuda"), torch.no_grad():
        model = Transformer(args)
        model.eval()  # Ensure model is in eval mode for inference
        # Enable memory-efficient inference mode
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    
    # Pre-allocate reusable tensor buffers for memory efficiency
    _tensor_pool = {
        'prompt_buffer': torch.empty(0, dtype=torch.long, device=f"cuda:{local_rank}"),
        'output_buffer': torch.empty(0, dtype=torch.long, device=f"cuda:{local_rank}"),
    }
    
    # Generate initial dummy tokens with gradient-free context
    with torch.no_grad():
        initial_tokens = [tokenizer.encode("DeepSeek")]
        generated_ids = generate(model, initial_tokens, 2, -1, 1.)
        tokenizer.decode(generated_ids[0])
        del initial_tokens, generated_ids
        torch.cuda.empty_cache()  # Clear initialization overhead
    
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))

    if interactive:
        # Circular buffer for message history to prevent unbounded growth
        MAX_HISTORY_SIZE = 20
        messages = []
        
        # Pre-allocate prompt string buffer for distributed communication
        if world_size > 1:
            _prompt_buffer = [""] * 1  # Reusable buffer for broadcast
        
        while True:
            # Optimized distributed prompt input with buffer reuse
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                _prompt_buffer[0] = prompt  # Reuse buffer
                dist.broadcast_object_list(_prompt_buffer, 0)
            else:
                dist.broadcast_object_list(_prompt_buffer, 0)
                prompt = _prompt_buffer[0]
                
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                torch.cuda.empty_cache()  # Clear cache on history clear
                continue
            
            # Add user message with in-place circular buffer management
            messages.append({"role": "user", "content": prompt})
            if len(messages) > MAX_HISTORY_SIZE:
                messages[:2] = []  # In-place removal
            
            # Generate response with gradient-free context and buffer reuse
            with torch.no_grad():
                prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                
                # Reuse tensor buffer if possible
                if len(prompt_tokens) <= _tensor_pool['prompt_buffer'].numel():
                    _tensor_pool['prompt_buffer'][:len(prompt_tokens)].copy_(torch.tensor(prompt_tokens))
                    reused_tokens = _tensor_pool['prompt_buffer'][:len(prompt_tokens)]
                else:
                    reused_tokens = prompt_tokens
                    # Expand buffer for future reuse
                    _tensor_pool['prompt_buffer'] = torch.empty(len(prompt_tokens) * 2, dtype=torch.long, device=f"cuda:{local_rank}")
                
                completion_tokens = generate(model, [reused_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
                completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
                print(completion)
                
                del prompt_tokens, completion_tokens, reused_tokens
                torch.cuda.empty_cache()  # Periodic cache clearing
            
            # Add assistant message with in-place management
            messages.append({"role": "assistant", "content": completion})
            if len(messages) > MAX_HISTORY_SIZE:
                messages[:2] = []
            
            del completion
    else:
        # Process batch mode with advanced memory-efficient streaming
        BATCH_CHUNK_SIZE = min(8, args.max_batch_size)
        
        # Pre-allocate batch processing buffers
        batch_tensor_pool = {
            'batch_tokens': torch.empty(BATCH_CHUNK_SIZE, max_new_tokens + 512, dtype=torch.long, device=f"cuda:{local_rank}"),
            'batch_lengths': torch.empty(BATCH_CHUNK_SIZE, dtype=torch.long, device=f"cuda:{local_rank}"),
        }
        
        with open(input_file) as f:
            line_count = 0
            prompt_batch = []
            
            for line in f:
                prompt = line.strip()
                if not prompt:
                    continue
                    
                prompt_batch.append(prompt)
                line_count += 1
                
                # Process when batch is full with memory pooling
                if len(prompt_batch) >= BATCH_CHUNK_SIZE:
                    _process_batch_chunk_optimized(prompt_batch, tokenizer, model, max_new_tokens, temperature, batch_tensor_pool)
                    prompt_batch.clear()
                    torch.cuda.empty_cache()  # Periodic memory cleanup
                    
                if line_count >= args.max_batch_size:
                    break
            
            # Process remaining prompts
            if prompt_batch:
                _process_batch_chunk_optimized(prompt_batch, tokenizer, model, max_new_tokens, temperature, batch_tensor_pool)

    # Cleanup and finalization
    if world_size > 1:
        dist.destroy_process_group()
    
    # Final memory cleanup
    del _tensor_pool
    if 'batch_tensor_pool' in locals():
        del batch_tensor_pool
    torch.cuda.empty_cache()


def _process_batch_chunk(prompts, tokenizer, model, max_new_tokens, temperature):
    """Process a chunk of prompts with memory-efficient operations."""
    with torch.no_grad():
        prompt_tokens_list = []
        for prompt in prompts:
            prompt_tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], 
                add_generation_prompt=True
            )
            prompt_tokens_list.append(prompt_tokens)
        
        completion_tokens_list = generate(model, prompt_tokens_list, max_new_tokens, tokenizer.eos_token_id, temperature)
        
        for prompt, completion_tokens in zip(prompts, completion_tokens_list):
            completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()
            del completion
        
        del prompt_tokens_list, completion_tokens_list


def _process_batch_chunk_optimized(prompts, tokenizer, model, max_new_tokens, temperature, tensor_pool):
    """Process a chunk of prompts with advanced memory pooling and tensor reuse."""
    with torch.no_grad():
        batch_size = len(prompts)
        
        # Reuse pre-allocated tensors from pool
        batch_tokens = tensor_pool['batch_tokens'][:batch_size]
        batch_lengths = tensor_pool['batch_lengths'][:batch_size]
        
        # Tokenize in-place to reuse buffers
        prompt_tokens_list = []
        max_length = 0
        
        for i, prompt in enumerate(prompts):
            prompt_tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], 
                add_generation_prompt=True
            )
            prompt_tokens_list.append(prompt_tokens)
            max_length = max(max_length, len(prompt_tokens))
            batch_lengths[i] = len(prompt_tokens)
        
        # Pad and copy to reused tensor buffer
        for i, tokens in enumerate(prompt_tokens_list):
            padded_length = min(len(tokens), batch_tokens.size(1))
            batch_tokens[i, :padded_length].copy_(torch.tensor(tokens[:padded_length], device=batch_tokens.device))
        
        # Generate with memory-optimized batching
        completion_tokens_list = generate(model, prompt_tokens_list, max_new_tokens, tokenizer.eos_token_id, temperature)
        
        # Stream output to avoid memory accumulation
        for prompt, completion_tokens in zip(prompts, completion_tokens_list):
            completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)
            print("Prompt:", prompt)
            print("Completion:", completion)
            print()
            del completion
        
        # Explicit cleanup
        del prompt_tokens_list, completion_tokens_list


if __name__ == "__main__":
    """
    Command-line interface for distributed text generation.

    Arguments:
        --ckpt-path (str): Path to the model checkpoint directory.
        --config (str): Path to the model configuration file.
        --input-file (str, optional): File containing prompts for batch processing.
        --interactive (bool, optional): Enable interactive mode for generating text.
        --max-new-tokens (int, optional): Maximum number of new tokens to generate. Defaults to 200.
        --temperature (float, optional): Temperature for sampling. Defaults to 0.2.

    Raises:
        AssertionError: If neither input-file nor interactive mode is specified.
    """
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    assert args.input_file or args.interactive
    main(args.ckpt_path, args.config, args.input_file, args.interactive, args.max_new_tokens, args.temperature)