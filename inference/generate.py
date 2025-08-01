
import os
import json
from argparse import ArgumentParser
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model
from torch.nn.utils.rnn import pad_sequence # Added for efficient batch padding

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
    # In-place division for memory efficiency
    logits.div_(max(temperature, 1e-5))
    # Use top-k sampling when applicable to avoid full softmax
    if logits.size(-1) > 100:
        top_k = 50
        top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
        probs = torch.softmax(top_k_logits, dim=-1)
        # In-place operations to avoid memory allocation
        probs.div_(torch.empty_like(probs).exponential_(1))
        selected = probs.argmax(dim=-1)
        return torch.gather(top_k_indices, -1, selected.unsqueeze(-1)).squeeze(-1)
    else:
        probs = torch.softmax(logits, dim=-1)
        probs.div_(torch.empty_like(probs).exponential_(1))
        return probs.argmax(dim=-1)


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
    # Calculate prompt lengths once
    prompt_lens = [len(t) for t in prompt_tokens]
    max_prompt_len = max(prompt_lens)
    assert max_prompt_len <= model.max_seq_len
    
    # Optimize total length calculation
    total_len = min(model.max_seq_len, max_new_tokens + max_prompt_len)
    batch_size = len(prompt_tokens)
    
    # Convert prompt tokens to tensors directly on GPU with pre-allocated memory
    # Create a single tensor of the right size to avoid multiple allocations
    tokens = torch.full((batch_size, total_len), -1, dtype=torch.long, device="cuda")
    
    # Fill in the prompt tokens efficiently
    for i, prompt in enumerate(prompt_tokens):
        tokens[i, :len(prompt)] = torch.tensor(prompt, dtype=torch.long, device="cuda")
    
    # Create prompt mask once (True where tokens are part of the prompt)
    prompt_mask = tokens != -1
    
    # Preallocate finished tensor
    finished = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
    
    # Start generation
    prev_pos = 0
    for cur_pos in range(max_prompt_len, total_len):
        # Only process unfinished sequences
        if prev_pos == 0 or not torch.all(finished):
            # Forward pass with the required tokens only
            logits = model.forward(tokens[:, :cur_pos], prev_pos)
            
            # Get next token based on temperature
            if temperature > 0:
                next_token = sample(logits, temperature)
            else:
                next_token = logits.argmax(dim=-1)
            
            # Check if position is part of prompt for any sequence
            any_prompt_at_pos = prompt_mask[:, cur_pos].any()
            if any_prompt_at_pos:
                # Only use where when needed to save compute
                next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            
            # Update tokens with next token
            tokens[:, cur_pos] = next_token
            
            # Update finished state for non-prompt positions
            non_prompt_pos = ~prompt_mask[:, cur_pos]
            if non_prompt_pos.any():
                finished[non_prompt_pos] |= (next_token[non_prompt_pos] == eos_id)
            
            # Early stopping if all sequences are finished
            if finished.all():
                break
            
            prev_pos = cur_pos
    
    # Extract completion tokens efficiently
    completion_tokens = []
    for i in range(batch_size):
        # Get tokens after the prompt
        completion = tokens[i, prompt_lens[i]:min(prompt_lens[i] + max_new_tokens, total_len)].tolist()
        
        # Find the first EOS token
        try:
            eos_idx = completion.index(eos_id)
            completion = completion[:eos_idx]
        except ValueError:
            # No EOS found, keep all tokens
            pass
            
        completion_tokens.append(completion)
        
    return completion_tokens


import os
import json
import torch
import torch.distributed as dist
from transformers import AutoTokenizer

# Note: The following imports are assumed to be available from the project structure.
# from model import Transformer, ModelArgs
# from generation import generate
# from modeling.load import load_model

# --- Helper function defined before it is called ---
def _process_batch_chunk(prompts, tokenizer, model, max_new_tokens, temperature):
    """Processes a chunk of prompts for batch generation."""
    # Pre-allocate list with known size
    batch_size = len(prompts)
    prompt_tokens_list = [None] * batch_size
    
    # Process prompts in parallel using tokenizer batch capabilities
    prompt_objs = [[{"role": "user", "content": prompt}] for prompt in prompts]
    
    # Generate all tokens at once
    for i, prompt_obj in enumerate(prompt_objs):
        prompt_tokens_list[i] = tokenizer.apply_chat_template(prompt_obj, add_generation_prompt=True)
    
    # Generate completions with optimized memory usage
    with torch.cuda.amp.autocast():
        completion_tokens_list = generate(model, prompt_tokens_list, max_new_tokens, tokenizer.eos_token_id, temperature)
    
    # Decode completions all at once
    completions = tokenizer.batch_decode(completion_tokens_list, skip_special_tokens=True)

    # Print results
    for prompt, completion in zip(prompts, completions):
        print("Prompt:", prompt)
        print("Completion:", completion)
        print()

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
    # Setup distributed environment
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    
    # Redirect print to be a no-op on non-main processes
    if rank != 0:
        global print
        print = lambda *_, **__: None

    # Optimize CUDA setup
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(min(os.cpu_count(), 4))  # Limit thread count for better efficiency
    torch.manual_seed(965)
    
    # Load configuration efficiently
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    
    # Create model on CUDA
    model = Transformer(args).cuda()
    
    # Use from_pretrained with caching to avoid reloading
    tokenizer_kwargs = {"local_files_only": True} if os.path.exists(os.path.join(ckpt_path, "tokenizer.json")) else {}
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, **tokenizer_kwargs)
    
    # Load model weights directly
    model_path = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    load_model(model, model_path)
    
    # Run a small warmup for the first compilation
    with torch.no_grad(), torch.cuda.amp.autocast():
        initial_tokens = [tokenizer.encode("DeepSeek", add_special_tokens=False)]
        generate(model, initial_tokens, 2, tokenizer.eos_token_id, 1.0)
        # Clear CUDA cache after warmup
        torch.cuda.empty_cache()

    if interactive:
        # Interactive mode with memory-efficient message handling
        MAX_HISTORY_SIZE = 20
        messages = []
        
        while True:
            # Get user input based on distributed setup
            if world_size > 1:
                if rank == 0:
                    prompt = input(">>> ")
                    dist.broadcast_object_list([prompt], 0)
                else:
                    objects = [None]
                    dist.broadcast_object_list(objects, 0)
                    prompt = objects[0]
            else:
                prompt = input(">>> ")
                
            if prompt == "/exit":
                break
            if prompt == "/clear":
                messages.clear()
                print("History cleared.")
                torch.cuda.empty_cache()  # Clear CUDA cache after clearing history
                continue
            
            # Process user message
            messages.append({"role": "user", "content": prompt})
            
            # Generate completion with proper memory management
            with torch.cuda.amp.autocast():
                prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=False)
                completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
                completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            
            print(completion)
            
            # Update conversation history
            messages.append({"role": "assistant", "content": completion})

            # Trim history to maintain memory usage
            if len(messages) > MAX_HISTORY_SIZE:
                messages = messages[-MAX_HISTORY_SIZE:]
                
            # Periodically clear CUDA cache to prevent memory fragmentation
            if len(messages) % 10 == 0:
                torch.cuda.empty_cache()
    else:
        # Batch processing with streaming to avoid loading entire file
        BATCH_CHUNK_SIZE = min(4, getattr(args, 'max_batch_size', 8))
        
        try:
            # Stream processing of input file
            with open(input_file) as f:
                prompt_batch = []
                for line in f:
                    prompt = line.strip()
                    if prompt:
                        prompt_batch.append(prompt)
                    
                    if len(prompt_batch) >= BATCH_CHUNK_SIZE:
                        _process_batch_chunk(prompt_batch, tokenizer, model, max_new_tokens, temperature)
                        prompt_batch.clear()
                        torch.cuda.empty_cache()  # Clear cache between batches
                
                # Process remaining prompts
                if prompt_batch:
                    _process_batch_chunk(prompt_batch, tokenizer, model, max_new_tokens, temperature)
        except FileNotFoundError:
            print(f"Error: Input file not found at {input_file}")

    # Cleanup
    torch.cuda.empty_cache()
    if world_size > 1:
        dist.destroy_process_group()



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