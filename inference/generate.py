
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
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float = 1.0
) -> List[List[int]]:
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    batch_size = len(prompt_tokens)
    # Convert prompt_tokens to tensors on CUDA, no grad
    prompt_tensors = [torch.tensor(p, dtype=torch.long, device="cuda", requires_grad=False) for p in prompt_tokens]
    # Pad them to max prompt length in batch using -1
    padded_prompt_tensor = pad_sequence(prompt_tensors, batch_first=True, padding_value=-1)

    tokens = torch.full((batch_size, total_len), -1, dtype=torch.long, device="cuda", requires_grad=False)
    tokens[:, :padded_prompt_tensor.shape[1]] = padded_prompt_tensor

    prev_pos = 0
    finished = torch.zeros(batch_size, dtype=torch.bool, device="cuda")
    prompt_mask = tokens != -1

    # Use model once on prompt to cache key-values if possible, then generate one token at a time
    for cur_pos in range(max(prompt_lens), total_len):
        logits = model.forward(tokens[:, :cur_pos], prev_pos)
        if temperature > 0:
            next_token = sample(logits, temperature)
        else:
            next_token = logits.argmax(dim=-1)
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= (next_token == eos_id) & (~prompt_mask[:, cur_pos])
        if finished.all():
            break
        prev_pos = cur_pos

    completion_tokens = [tokens[i, prompt_lens[i]:prompt_lens[i] + max_new_tokens].tolist() for i in range(batch_size)]
    for i in range(batch_size):
        eos_pos_in_slice = (tokens[i, prompt_lens[i]:prompt_lens[i] + max_new_tokens] == eos_id).nonzero(as_tuple=True)[0]
        if eos_pos_in_slice.numel() > 0:
            completion_tokens[i] = completion_tokens[i][:eos_pos_in_slice[0].item()]
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
    prompt_tokens_list = []
    for prompt in prompts:
        prompt_obj = [{"role": "user", "content": prompt}]
        prompt_tokens_list.append(tokenizer.apply_chat_template(prompt_obj, add_generation_prompt=True))
    
    completion_tokens_list = generate(model, prompt_tokens_list, max_new_tokens, tokenizer.eos_token_id, temperature)
    completions = tokenizer.batch_decode(completion_tokens_list, skip_special_tokens=True)

    # Print results for the processed chunk
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
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    
    # Disable printing on non-primary processes
    if rank != 0:
        global print
        print = lambda *args, **kwargs: None

    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)
    
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    
    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    
    # Generate initial dummy tokens to initialize model device
    initial_tokens = [tokenizer.encode("DeepSeek")]
    generated_ids = generate(model, initial_tokens, 2, -1, 1.)
    tokenizer.decode(generated_ids[0])
    
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))

    if interactive:
        MAX_HISTORY_SIZE = 20
        messages = []
        
        while True:
            if world_size > 1 and rank == 0:
                prompt = input(">>> ")
                dist.broadcast_object_list([prompt], 0)
            elif world_size > 1:
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
                continue
            
            messages.append({"role": "user", "content": prompt})
            
            prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id, temperature)
            completion = tokenizer.decode(completion_tokens[0], skip_special_tokens=True)
            print(completion)
            
            messages.append({"role": "assistant", "content": completion})

            while len(messages) > MAX_HISTORY_SIZE:
                del messages[:2]
    else:
        BATCH_CHUNK_SIZE = min(8, args.max_batch_size)
        
        try:
            with open(input_file) as f:
                prompt_batch = []
                for line in f:
                    prompt = line.strip()
                    if prompt:
                        prompt_batch.append(prompt)
                    
                    if len(prompt_batch) >= BATCH_CHUNK_SIZE:
                        _process_batch_chunk(prompt_batch, tokenizer, model, max_new_tokens, temperature)
                        prompt_batch.clear()
                
                if prompt_batch:
                    _process_batch_chunk(prompt_batch, tokenizer, model, max_new_tokens, temperature)
        except FileNotFoundError:
            print(f"Error: Input file not found at {input_file}")

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