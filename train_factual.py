import os 

from datetime import datetime

from utils.utils_all import seed_everything, scheduling

from datasets.counterfact import get_factual

import numpy as np

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F
import torch.optim as optim

import time
import csv

from torch.nn.utils.rnn import pad_sequence

# Get the current time
current_time = datetime.now().strftime("%Y-%m-%d")

import argparse
parser = argparse.ArgumentParser(description="Causal Tracing")
parser.add_argument("--model", type=str, help = 'model type', default='distilgpt2', choices=['distilgpt2', 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', "EleutherAI/gpt-j-6B"])
parser.add_argument("--lambda1", type=float, help = 'Regularization 1' , default=1e-4)
parser.add_argument("--lambda2", type=float, help = 'Regularization 2' ,default=1e-4)
parser.add_argument("--scheduling_method", type=str, help = 'Scheduling method', default="linear")
parser.add_argument("--lr", type=float, help = 'Learning rate' ,default=5e-1)
parser.add_argument("--batch_size", type=int, help = 'Training batch size' ,default=32)
# parser.add_argument("--eval_batch_size", type=int, help = 'Evaluation batch size' ,default=64)
parser.add_argument("--epochs", type=int, help = 'Total epochs' ,default=15)
parser.add_argument("--log_freq", type=int, help = 'Logging frequency' ,default=10)
parser.add_argument("--threshold", type=float, help = 'Truncation threshold' ,default=0.5)
parser.add_argument("--device", type=str, help = 'model type', default='cuda')
parser.add_argument("--trail", type=int, help = 'model type', default=42)
parser.add_argument("--cache_dir", type=str, help = 'model save dir', default=None)
parser.add_argument("--std", type=float, help = 'variance', default=3 * 0.2015)

args = parser.parse_args()


print("**********************")
print(args.model)

l1 = args.lambda1
l2 = args.lambda2   # Weight for the binary-like penalty
method = args.scheduling_method

cache_dir=args.cache_dir

##########################
# Create a new directory with the current time as its name
directory_name = f"results/Factual/Our/{current_time}/{args.model}_l1_{args.lambda1}_l2_{args.lambda2}_scheduling_{method}_lr_{args.lr}"
os.makedirs(directory_name, exist_ok=True)
os.makedirs(f"{directory_name}/output", exist_ok=True)
os.makedirs(f"{directory_name}/ckpt", exist_ok=True)



##########################
# save the args

# Save args to a text file
with open(f"{directory_name}/output/args.txt", 'w') as f:
    for arg, value in vars(args).items():
        f.write(f"{arg}: {value}\n")

##########################
# seed for reproduce
seed_everything(args.trail)



##########################
# load the model
print("loading model and dataset")

model_type = args.model
device = torch.device(args.device)

if args.cache_dir:
    print("&&& using cache_dir &&&")
    # Step 1: Initialize the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model, cache_dir=cache_dir)

    # Step 2: Initialize the model
    model = GPT2LMHeadModel.from_pretrained(args.model, cache_dir=cache_dir, output_hidden_states=True)
else:
    # Step 1: Initialize the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)

    # Step 2: Initialize the model
    model = GPT2LMHeadModel.from_pretrained(args.model, output_hidden_states=True)

# Add pad token
tokenizer.pad_token = tokenizer.eos_token



model = model.to(device)
model.eval()
print("Done load model")

# Get the configuration object
config = model.config

# Number of transformer layers
num_layers = config.n_layer

# Number of attention heads per layer
num_attention_heads = config.n_head

# Hidden size for each layer (used in MLP computations)
hidden_size = config.n_embd

print("Start load dataset")

batch_texts, target_new_list, target_true_list = get_factual()

print("Done load dataset")


batch_all = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
batch_lengths = batch_all["attention_mask"].sum(dim=1)

#generate fixed pertubation in the embedding

# Compute standard deviation from the desired variance
# std = 3 * 0.2015
# std = 0.1 * 0.2015
# std = 0
# std = math.sqrt(variance)

# Generate a list of tensors filled with Gaussian noise
gaussian_tensors = [torch.randn(int(length), hidden_size).to(device) * args.std for length in batch_lengths]



z = torch.full((num_layers, hidden_size), 0.5, requires_grad=True, device=device)
# z = torch.full((num_layers, hidden_size), 1.0, requires_grad=True, device=device)



optimizer = optim.Adam([z], lr=args.lr)




# Multi-step gradient descent
num_steps = args.epochs
losses = []
sparsities = []



def evaluate(z_truncated):
    odds_all = []
    with torch.no_grad():
        for bb in range(0, len(batch_texts), batch_size):
            time1=time.time()
            batch_chunk = batch_texts[bb:bb + batch_size]
            batch_true = target_true_list[bb:bb + batch_size]
            batch_gaussian = gaussian_tensors[bb:bb + batch_size]

            # Tokenize the current batch and move it to the device
            batch_inputs = tokenizer(batch_chunk, return_tensors="pt", padding=True, truncation=True)
            batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}

            # batch_new_output = tokenizer(batch_new, return_tensors="pt", padding=True, truncation=True)

            batch_true_output = tokenizer(batch_true, return_tensors="pt", padding=True, truncation=True)

            batch_lengths = batch_inputs["attention_mask"].sum(dim=1)

            max_input_lengths = max(batch_inputs["attention_mask"].sum(dim=1)).item()

            # Create a zero tensor by converting size_tensor to a tuple
            size_tensor = torch.tensor([max_input_lengths, batch_size, hidden_size, hidden_size])



            time2=time.time()

            processed_contexts = [
                torch.tensor([token for token in context if token != 50256], dtype=torch.long)
                for context in batch_inputs['input_ids'].tolist()
            ]

            candidate_this = [
                torch.tensor([token for token in context if token != 50256], dtype=torch.long)
                for context in batch_true_output['input_ids'].tolist()
            ]

            true_all_mean_probs = []
            # Process each candidate group (each candidate group is a list with one candidate per context)
            combined_sequences = []   # To hold context + candidate (minus one token for prediction)
            candidate_tokens = []     # To hold candidate token tensors (for scoring)
            context_lens = []         # Save lengths of each (cleaned) context
            
            # For each context, build the combined sequence and candidate tensor.
            for context, candidate in zip(processed_contexts, candidate_this):
                # Save the (cleaned) context length.
                context_lens.append(len(context))
                
                # Convert candidate to tensor.
                # cand_tensor = torch.tensor(candidate, dtype=torch.long)
                cand_tensor = candidate.clone().detach()
                candidate_tokens.append(cand_tensor)
                
                # Concatenate context and candidate tokens.
                combined = torch.cat([context, cand_tensor])
                # Remove the last token so that each token's prediction comes from the previous token.
                combined_sequences.append(combined[:-1])
                
            # Pad the combined sequences to the same length.
            # (This is more efficient than padding manually in Python.)
            padded_combined = pad_sequence(combined_sequences, batch_first=True, padding_value=50256)
            # Send input to device.
            input_ids = padded_combined.to(device)
            
            # Run the model in one batch forward pass.
            # Assume the model returns logits with shape (batch_size, seq_len, vocab_size)
            logits = model(input_ids)[0]
            log_probs = F.log_softmax(logits, dim=-1)
            
            mean_probs_for_this_batch = []
            # Now compute each candidate's mean probability.
            # For each example in the batch:
            for idx, (ctx_len, cand_tensor) in enumerate(zip(context_lens, candidate_tokens)):
                # In your original code, you used:
                #   context_end = len(context) - 1
                # so that the first candidate token is predicted by the last token in the context.
                start = ctx_len - 1
                cand_len = len(cand_tensor)
                
                # Construct the positions at which the candidate tokens are predicted.
                # For each candidate token j, its probability is in log_probs[idx, start + j, candidate[j]].
                positions = torch.arange(start, start + cand_len, device=device)
                # Use advanced indexing to select the candidate token log–probs.
                candidate_log_probs = log_probs[idx, positions, cand_tensor.to(device)]
                # Average the log–probabilities and exponentiate to get the mean probability.
                mean_prob = torch.exp(candidate_log_probs.mean())
                mean_probs_for_this_batch.append(mean_prob)
                
            # Collect mean probabilities for this candidate group.
            true_all_mean_probs.append(mean_probs_for_this_batch)

            time3=time.time()
            
            #get corrupted hidden states
            input_ids = batch_inputs['input_ids']


            sequence_length = input_ids.size(1)

            # Build position_ids
            position_ids = torch.arange(
                0, 
                sequence_length, 
                dtype=torch.long, 
                device=input_ids.device
            ).unsqueeze(0).expand_as(input_ids)

            inputs_embeds = model.transformer.wte(input_ids)

            position_embeds = model.transformer.wpe(position_ids)

            hidden_states_0 = inputs_embeds + position_embeds
            hidden_states_0 = model.transformer.drop(hidden_states_0)

            for i, noise in enumerate(batch_gaussian):
                seq_len = noise.size(0)
                # Add noise to the first seq_len time steps for the i-th batch element
                hidden_states_0[i, :seq_len, :] += noise

            hidden_states_corrupt= []
            hidden_states_corrupt.append(hidden_states_0)

            mlp_corrupt = []

            # Perform mixing operation in batches
            for layer_idx in range(1, num_layers+1):
                inputs_for_layer = {'hidden_states': hidden_states_corrupt[layer_idx - 1]}

                # Unpack the hidden states
                hidden_states = inputs_for_layer['hidden_states']

                # Layer Normalization before self-attention
                hidden_states_norm_1 = model.transformer.h[layer_idx - 1].ln_1(hidden_states)

                # Self-Attention mechanism
                attention_outputs = model.transformer.h[layer_idx - 1].attn(hidden_states_norm_1)
                attention_output = attention_outputs[0]  # Take only the output (ignore other outputs if present)

                # Add skip connection (residual connection) after self-attention
                hidden_states_after_attention = hidden_states + attention_output

                # Layer Normalization before feed-forward network
                hidden_states_norm_2 = model.transformer.h[layer_idx - 1].ln_2(hidden_states_after_attention)

                # Feed-Forward Network (MLP)
                mlp_output = model.transformer.h[layer_idx - 1].mlp(hidden_states_norm_2)
                mlp_corrupt.append(mlp_output)
                # Add skip connection (residual connection) after the feed-forward network
                hidden_states_output = hidden_states_after_attention + mlp_output

                if layer_idx == num_layers:
                    # Final layer normalization and logits calculation for the mixed hidden states
                    hidden_states_corrupt.append(model.transformer.ln_f(hidden_states_output))
                else:
                    hidden_states_corrupt.append(hidden_states_output)

            del hidden_states_corrupt

            time4=time.time()
            #Now do mixture forward and get the result
            # including the embedding layer
            hidden_states_mix = [torch.zeros(*size_tensor.tolist()) for _ in range(num_layers+1)]
            mix_all_mean_probs = []

            input_ids = padded_combined.to(device)

            sequence_length = input_ids.size(1)

            # Build position_ids
            position_ids = torch.arange(
                0, 
                sequence_length, 
                dtype=torch.long, 
                device=input_ids.device
            ).unsqueeze(0).expand_as(input_ids)

            inputs_embeds = model.transformer.wte(input_ids)

            position_embeds = model.transformer.wpe(position_ids)

            hidden_states_0 = inputs_embeds + position_embeds
            hidden_states_0 = model.transformer.drop(hidden_states_0)

            hidden_states = hidden_states_0

            mlp_mix = []
            
            # Run the model in one batch forward pass.
            # Assume the model returns logits with shape (batch_size, seq_len, vocab_size)
            # Perform mixing operation in batches
            for layer_idx in range(num_layers):
                block = model.transformer.h[layer_idx]

                # -- LN -> Attention ----------------------------------------------
                hidden_states_norm_1 = block.ln_1(hidden_states)
                # Usually GPT-style attention returns (attn_output, present) or so:
                attn_output, _ = block.attn(hidden_states_norm_1)
                # Residual
                hidden_states = hidden_states + attn_output

                # -- LN -> MLP ---------------------------------------------------
                hidden_states_norm_2 = block.ln_2(hidden_states)
                mlp_output = block.mlp(hidden_states_norm_2)

                z = z_truncated[layer_idx]
                # If it's a scalar:
                # mlp_output = mlp_output*(1 - z) + mlp_corrupt[layer_idx]*z
                #
                # Now enforce the valid lengths only:
                # Build a mask [B, seq_len], True where index < batch_length[i]
                # Suppose mlp_output has shape [B, T, H]
                # batch_lengths.shape[0] is your batch size
                B = batch_lengths.size(0)
                T = mlp_output.shape[1]

                # 1) Create a range [0 .. T-1], shape [T]
                # 2) Unsqueeze to [1, T] so we can compare it to each row's batch_lengths
                # 3) Compare < batch_lengths.unsqueeze(1) to get a boolean mask [B, T]
                # 4) Convert to int/float as desired
                mask = (
                    torch.arange(T, device=batch_lengths.device)
                    .unsqueeze(0)                                # shape [1, T]
                    .expand(B, T)                                # shape [B, T]
                    < batch_lengths.unsqueeze(1)                 # shape [B, T], boolean
                ).long()                                         # or .float(), etc.
                mask = mask.unsqueeze(-1) 
                mask = mask.bool()   
                # print(mask.shape)

                mixed = mlp_output[:,:mlp_corrupt[layer_idx].shape[1],:]*(1 - z) + mlp_corrupt[layer_idx]*(z)


                
                # We'll do the mixing only where mask==True.  A standard trick is:
                #    mlp_output = torch.where(mask, mix_value, original_value)
                # but we can do it in two steps to keep it simple.
                B, T_out, H = mlp_output.shape
                B_in, T_in,  H_in = mixed.shape

                # Check that batch size & hidden dim match (assuming that's required)
                assert B == B_in,    f"Mismatch in batch size: {B} vs {B_in}"
                assert H == H_in,    f"Mismatch in hidden dim: {H} vs {H_in}"

                if T_in < T_out:
                    pad_amount = T_out - T_in
                    # 'F.pad(input, (left, right, top, bottom, ...))'
                    # For a 3D tensor [B, T, H], the 4-tuple (0,0,0,pad_amount)
                    # means "pad dimension -1 by 0/0, and dimension -2 by 0/pad_amount".
                    #
                    mixed = F.pad(mixed, (0, 0, 0, pad_amount), value=0.0)
                    # Now 'mixed' has shape [B, T_out, H]

                mlp_output = torch.where(mask, mixed, mlp_output)

                # -- Residual -----------------------------------------------------
                hidden_states = hidden_states + mlp_output

            # -------------------------------------------------------------------------
            # 3) Final layer normalization + logits
            # -------------------------------------------------------------------------
            hidden_states = model.transformer.ln_f(hidden_states)
            logits_mix = model.lm_head(hidden_states)
            log_probs_mix = F.log_softmax(logits_mix, dim=-1)


            time5=time.time()

            mean_probs_for_this_batch = []
            # Now compute each candidate's mean probability.
            # For each example in the batch:
            for idx, (ctx_len, cand_tensor) in enumerate(zip(context_lens, candidate_tokens)):
                # In your original code, you used:
                #   context_end = len(context) - 1
                # so that the first candidate token is predicted by the last token in the context.
                start = ctx_len - 1
                cand_len = len(cand_tensor)
                
                # Construct the positions at which the candidate tokens are predicted.
                # For each candidate token j, its probability is in log_probs[idx, start + j, candidate[j]].
                positions = torch.arange(start, start + cand_len, device=device)
                # Use advanced indexing to select the candidate token log–probs.
                candidate_log_probs = log_probs_mix[idx, positions, cand_tensor.to(device)]
                # Average the log–probabilities and exponentiate to get the mean probability.
                mean_prob = torch.exp(candidate_log_probs.mean())
                mean_probs_for_this_batch.append(mean_prob)
                
            # Collect mean probabilities for this candidate group.
            mix_all_mean_probs.append(mean_probs_for_this_batch)

            # Calculate odds ratio and loss in batches
            odds_ratios = [f / s for f, s in zip(true_all_mean_probs[0], mix_all_mean_probs[0])]


            # Inverse the values for males, keep the values for females
            # torch.where is used to apply conditions while preserving gradient flow
            modified_odds_ratios = [ odds_ratio - 1 for odds_ratio in odds_ratios]

            odds_all += modified_odds_ratios
            time6=time.time()
            # print(time6-time5,time5-time4,time4-time3,time3-time2,time2-time1)
    odds_average = np.mean([o.cpu() for o in odds_all])

    return odds_all, odds_average

batch_size = args.batch_size





outputfile = f"{directory_name}/output/log.csv"

# os.makedirs(os.path.dirname(outputfile), exist_ok=True)
header = ['Epoch', 'Step', 'Running_odds', 'Running_loss','Sparsity','Non_zeros','violation']

# File path
file_path_add = f"{directory_name}/output/evaluate.csv"

# Initialize the file if it doesn't exist
with open(file_path_add, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)  # Example headers


with open(outputfile, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    start_time = time.time()
    for step in range(num_steps):
        print(f"in step {step+1}")
        running_loss = 0
        running_odds = 0

        # # Assuming batch_texts is a tensor with shape [N, ...]
        N = len(batch_texts)
        shuffled_indices = torch.randperm(N)  # Generate a random permutation of indices

        # Now collect pairs from batch_texts using the shuffled indices
        shuffled_texts = []
        # shuffled_new_list = []
        shuffled_true_list = []
        shuffled_gaussian_tensors = []
        # shuffled_gender = []
        for i in shuffled_indices:
            shuffled_texts.append(batch_texts[i])
            # shuffled_new_list.append(target_new_list[i])
            shuffled_true_list.append(target_true_list[i])
            shuffled_gaussian_tensors.append(gaussian_tensors[i])

        for bb in range(0, len(batch_texts), batch_size):
            print(f"in batch {bb+1}")
            with torch.no_grad():
                batch_chunk = shuffled_texts[bb:bb + batch_size]
                # batch_new = shuffled_new_list[bb:bb + batch_size]
                batch_true = shuffled_true_list[bb:bb + batch_size]
                batch_gaussian = shuffled_gaussian_tensors[bb:bb + batch_size]
                # gender_chuch = shuffled_gender[int(bb/2):int(bb/2) + int(batch_size/2)]

                # Tokenize the current batch and move it to the device
                batch_inputs = tokenizer(batch_chunk, return_tensors="pt", padding=True, truncation=True)
                batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}

                # batch_new_output = tokenizer(batch_new, return_tensors="pt", padding=True, truncation=True)

                batch_true_output = tokenizer(batch_true, return_tensors="pt", padding=True, truncation=True)

                batch_lengths = batch_inputs["attention_mask"].sum(dim=1)

                max_input_lengths = max(batch_inputs["attention_mask"].sum(dim=1)).item()

                # Create a zero tensor by converting size_tensor to a tuple
                size_tensor = torch.tensor([max_input_lengths, batch_size, hidden_size, hidden_size])




                processed_contexts = [
                    torch.tensor([token for token in context if token != 50256], dtype=torch.long)
                    for context in batch_inputs['input_ids'].tolist()
                ]

                candidate_this = [
                    torch.tensor([token for token in context if token != 50256], dtype=torch.long)
                    for context in batch_true_output['input_ids'].tolist()
                ]

                true_all_mean_probs = []
                # Process each candidate group (each candidate group is a list with one candidate per context)
                combined_sequences = []   # To hold context + candidate (minus one token for prediction)
                candidate_tokens = []     # To hold candidate token tensors (for scoring)
                context_lens = []         # Save lengths of each (cleaned) context
                
                # For each context, build the combined sequence and candidate tensor.
                for context, candidate in zip(processed_contexts, candidate_this):
                    # Save the (cleaned) context length.
                    context_lens.append(len(context))
                    
                    # Convert candidate to tensor.
                    # cand_tensor = torch.tensor(candidate, dtype=torch.long)
                    cand_tensor = candidate.clone().detach()
                    candidate_tokens.append(cand_tensor)
                    
                    # Concatenate context and candidate tokens.
                    combined = torch.cat([context, cand_tensor])
                    # Remove the last token so that each token's prediction comes from the previous token.
                    combined_sequences.append(combined[:-1])
                    
                # Pad the combined sequences to the same length.
                # (This is more efficient than padding manually in Python.)
                padded_combined = pad_sequence(combined_sequences, batch_first=True, padding_value=50256)
                # Send input to device.
                input_ids = padded_combined.to(device)
                
                # Run the model in one batch forward pass.
                # Assume the model returns logits with shape (batch_size, seq_len, vocab_size)
                logits = model(input_ids)[0]
                log_probs = F.log_softmax(logits, dim=-1)

                del logits, padded_combined
                
                print('done with part one')

                mean_probs_for_this_batch = []
                # Now compute each candidate's mean probability.
                # For each example in the batch:
                for idx, (ctx_len, cand_tensor) in enumerate(zip(context_lens, candidate_tokens)):
                    # In your original code, you used:
                    #   context_end = len(context) - 1
                    # so that the first candidate token is predicted by the last token in the context.
                    start = ctx_len - 1
                    cand_len = len(cand_tensor)
                    
                    # Construct the positions at which the candidate tokens are predicted.
                    # For each candidate token j, its probability is in log_probs[idx, start + j, candidate[j]].
                    positions = torch.arange(start, start + cand_len, device=device)
                    # Use advanced indexing to select the candidate token log–probs.
                    candidate_log_probs = log_probs[idx, positions, cand_tensor.to(device)]
                    # Average the log–probabilities and exponentiate to get the mean probability.
                    mean_prob = torch.exp(candidate_log_probs.mean())
                    mean_probs_for_this_batch.append(mean_prob)
                    
                # Collect mean probabilities for this candidate group.
                true_all_mean_probs.append(mean_probs_for_this_batch)
                
                #get corrupted hidden states
                input_ids = batch_inputs['input_ids']


                sequence_length = input_ids.size(1)

                # Build position_ids
                position_ids = torch.arange(
                    0, 
                    sequence_length, 
                    dtype=torch.long, 
                    device=input_ids.device
                ).unsqueeze(0).expand_as(input_ids)

                inputs_embeds = model.transformer.wte(input_ids)

                position_embeds = model.transformer.wpe(position_ids)

                hidden_states_0 = inputs_embeds + position_embeds
                hidden_states_0 = model.transformer.drop(hidden_states_0)

                for i, noise in enumerate(batch_gaussian):
                    seq_len = noise.size(0)
                    # Add noise to the first seq_len time steps for the i-th batch element
                    hidden_states_0[i, :seq_len, :] += noise

                hidden_states_corrupt= []
                hidden_states_corrupt.append(hidden_states_0)

                mlp_corrupt = []

                # Perform mixing operation in batches
                for layer_idx in range(1, num_layers+1):
                    inputs_for_layer = {'hidden_states': hidden_states_corrupt[layer_idx - 1]}

                    # Unpack the hidden states
                    hidden_states = inputs_for_layer['hidden_states']

                    # Layer Normalization before self-attention
                    hidden_states_norm_1 = model.transformer.h[layer_idx - 1].ln_1(hidden_states)

                    # Self-Attention mechanism
                    attention_outputs = model.transformer.h[layer_idx - 1].attn(hidden_states_norm_1)
                    attention_output = attention_outputs[0]  # Take only the output (ignore other outputs if present)

                    # Add skip connection (residual connection) after self-attention
                    hidden_states_after_attention = hidden_states + attention_output

                    # Layer Normalization before feed-forward network
                    hidden_states_norm_2 = model.transformer.h[layer_idx - 1].ln_2(hidden_states_after_attention)

                    # Feed-Forward Network (MLP)
                    mlp_output = model.transformer.h[layer_idx - 1].mlp(hidden_states_norm_2)
                    mlp_corrupt.append(mlp_output)


                    # Add skip connection (residual connection) after the feed-forward network
                    hidden_states_output = hidden_states_after_attention + mlp_output

                    if layer_idx == num_layers:
                        # Final layer normalization and logits calculation for the mixed hidden states
                        hidden_states_corrupt.append(model.transformer.ln_f(hidden_states_output))
                    else:
                        hidden_states_corrupt.append(hidden_states_output)

                del hidden_states_corrupt, hidden_states_0

                print('done with part 2')

            #Now do mixture forward and get the result
            # including the embedding layer
            hidden_states_mix = [torch.zeros(*size_tensor.tolist()) for _ in range(num_layers+1)]
            mix_all_mean_probs = []
            # Process each candidate group (each candidate group is a list with one candidate per context)
            combined_sequences = []   # To hold context + candidate (minus one token for prediction)
            candidate_tokens = []     # To hold candidate token tensors (for scoring)
            context_lens = []         # Save lengths of each (cleaned) context
            
            # For each context, build the combined sequence and candidate tensor.
            for context, candidate in zip(processed_contexts, candidate_this):
                # Save the (cleaned) context length.
                context_lens.append(len(context))
                
                # Convert candidate to tensor.
                # cand_tensor = torch.tensor(candidate, dtype=torch.long)
                cand_tensor = candidate.clone().detach()
                candidate_tokens.append(cand_tensor)
                
                # Concatenate context and candidate tokens.
                combined = torch.cat([context, cand_tensor])
                # Remove the last token so that each token's prediction comes from the previous token.
                combined_sequences.append(combined[:-1])
                
            # Pad the combined sequences to the same length.
            # (This is more efficient than padding manually in Python.)
            padded_combined = pad_sequence(combined_sequences, batch_first=True, padding_value=50256)
            # Send input to device.
            input_ids = padded_combined.to(device)

            sequence_length = input_ids.size(1)

            # Build position_ids
            position_ids = torch.arange(
                0, 
                sequence_length, 
                dtype=torch.long, 
                device=input_ids.device
            ).unsqueeze(0).expand_as(input_ids)

            inputs_embeds = model.transformer.wte(input_ids)

            position_embeds = model.transformer.wpe(position_ids)

            hidden_states_0 = inputs_embeds + position_embeds
            hidden_states_0 = model.transformer.drop(hidden_states_0)

            hidden_states_mix[0] = hidden_states_0

            
            hidden_states = hidden_states_0

            mlp_mix = []

            
            for layer_idx in range(num_layers):
                block = model.transformer.h[layer_idx]

                # -- LN -> Attention ----------------------------------------------
                hidden_states_norm_1 = block.ln_1(hidden_states)
                # Usually GPT-style attention returns (attn_output, present) or so:
                attn_output, _ = block.attn(hidden_states_norm_1)
                # Residual
                hidden_states = hidden_states + attn_output

                # -- LN -> MLP ---------------------------------------------------
                hidden_states_norm_2 = block.ln_2(hidden_states)
                mlp_output = block.mlp(hidden_states_norm_2)

                z_ = z[layer_idx]
                # If it's a scalar:
                # mlp_output = mlp_output*(1 - z) + mlp_corrupt[layer_idx]*z
                #
                # Now enforce the valid lengths only:
                # Build a mask [B, seq_len], True where index < batch_length[i]
                # Suppose mlp_output has shape [B, T, H]
                # batch_lengths.shape[0] is your batch size
                B = batch_lengths.size(0)
                T = mlp_output.shape[1]

                # 1) Create a range [0 .. T-1], shape [T]
                # 2) Unsqueeze to [1, T] so we can compare it to each row's batch_lengths
                # 3) Compare < batch_lengths.unsqueeze(1) to get a boolean mask [B, T]
                # 4) Convert to int/float as desired
                mask = (
                    torch.arange(T, device=batch_lengths.device)
                    .unsqueeze(0)                                # shape [1, T]
                    .expand(B, T)                                # shape [B, T]
                    < batch_lengths.unsqueeze(1)                 # shape [B, T], boolean
                ).long()                                         # or .float(), etc.
                mask = mask.unsqueeze(-1) 
                mask = mask.bool()   
                # print(mask.shape)

                mixed = mlp_output[:,:mlp_corrupt[layer_idx].shape[1],:]*(1 - z_) + mlp_corrupt[layer_idx]*(z_)


                
                # We'll do the mixing only where mask==True.  A standard trick is:
                #    mlp_output = torch.where(mask, mix_value, original_value)
                # but we can do it in two steps to keep it simple.
                B, T_out, H = mlp_output.shape
                B_in, T_in,  H_in = mixed.shape

                # Check that batch size & hidden dim match (assuming that's required)
                assert B == B_in,    f"Mismatch in batch size: {B} vs {B_in}"
                assert H == H_in,    f"Mismatch in hidden dim: {H} vs {H_in}"

                if T_in < T_out:
                    pad_amount = T_out - T_in
                    # 'F.pad(input, (left, right, top, bottom, ...))'
                    # For a 3D tensor [B, T, H], the 4-tuple (0,0,0,pad_amount)
                    # means "pad dimension -1 by 0/0, and dimension -2 by 0/pad_amount".
                    #
                    mixed = F.pad(mixed, (0, 0, 0, pad_amount), value=0.0)
                    # Now 'mixed' has shape [B, T_out, H]

                mlp_output = torch.where(mask, mixed, mlp_output)

                # -- Residual -----------------------------------------------------
                hidden_states = hidden_states + mlp_output

            print('done with part 3 fwd')


            # -------------------------------------------------------------------------
            # 3) Final layer normalization + logits
            # -------------------------------------------------------------------------
            hidden_states = model.transformer.ln_f(hidden_states)
            logits_mix = model.lm_head(hidden_states)
            log_probs_mix = F.log_softmax(logits_mix, dim=-1)

            mean_probs_for_this_batch = []
            # Now compute each candidate's mean probability.
            # For each example in the batch:
            for idx, (ctx_len, cand_tensor) in enumerate(zip(context_lens, candidate_tokens)):
                # In your original code, you used:
                #   context_end = len(context) - 1
                # so that the first candidate token is predicted by the last token in the context.
                start = ctx_len - 1
                cand_len = len(cand_tensor)
                
                # Construct the positions at which the candidate tokens are predicted.
                # For each candidate token j, its probability is in log_probs[idx, start + j, candidate[j]].
                positions = torch.arange(start, start + cand_len, device=device)
                # Use advanced indexing to select the candidate token log–probs.
                candidate_log_probs = log_probs_mix[idx, positions, cand_tensor.to(device)]
                # Average the log–probabilities and exponentiate to get the mean probability.
                mean_prob = torch.exp(candidate_log_probs.mean())
                mean_probs_for_this_batch.append(mean_prob)
                
            # Collect mean probabilities for this candidate group.
            mix_all_mean_probs.append(mean_probs_for_this_batch)

            # Calculate odds ratio and loss in batches
            # print("odds")
            # print(true_all_mean_probs[0])
            # print(mix_all_mean_probs[0])
            odds_ratios = [f / s for f, s in zip(true_all_mean_probs[0], mix_all_mean_probs[0])]
            # print("odds ratios")
            # print(odds_ratios)


            # Inverse the values for males, keep the values for females
            # torch.where is used to apply conditions while preserving gradient flow
            modified_odds_ratios = torch.stack([ odds_ratio - 1 for odds_ratio in odds_ratios])
           
              
            l1_norm = torch.sum(torch.abs(z))
            l1_reg = scheduling(l1, step, method=method)

            l2_reg = scheduling(l2, step, method=method)   # Weight for the binary-like penalty

            # Binary-like penalty (pushing values to 0 or 1)
            l2_norm = torch.sum(z * (1 - z))

            # print("norm", l1_norm, l2_norm)

            losses_ = 1 / (1 + modified_odds_ratios) + l1_reg * l1_norm + l2_reg * l2_norm
            # losses_ = - modified_odds_ratios + l1_reg * l1_norm + l2_reg * l2_norm

            print('done with part 4 loss')


            # print("losses_",losses_)

            loss_average = torch.mean(losses_)
            odds_average = torch.mean(modified_odds_ratios)
            loss_average.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss_average.item()
            running_odds += odds_average
            with torch.no_grad():
                z.clamp_(0, 1)
            
            torch.cuda.empty_cache()
            if bb% args.log_freq ==0:
                #save to csv file
                print(f"Step {step + 1}/{num_steps} in {bb}/ {len(batch_texts)}")
                print(f"Odds: {running_odds/(bb/batch_size+1)}")
                print(f"Loss: {running_loss/(bb/batch_size+1)}")
                losses.append(running_loss/(bb/batch_size+1))

                # Sparsity of z
                sparsity = torch.sum(z == 0).item() / z.numel()
                non_zero = z.numel() - torch.sum(z == 0).item()
                violation = torch.sum(z * (1 - z))
                print(f"Sparsity of z: {sparsity * 100:.2f}%")
                print(f"non_zero of z: {non_zero}")
                print(f"Violation of z: {violation}")
                sparsities.append(sparsity * 100)
                writer.writerow([step, bb, running_odds.item()/(bb/batch_size+1), running_loss/(bb/batch_size+1), sparsity * 100, non_zero,violation.item()])

            print('done with part 5:end')

     
        # Save z for the current step
        z_save_path = os.path.join(f"{directory_name}/ckpt/", f"z_step_{step + 1}.pt")
        torch.save(z, z_save_path)

        z_truncated = (z >= args.threshold).float()
        z_truncated.requires_grad = False
        sparsity = torch.sum(z_truncated == 0).item() / z_truncated.numel()
        non_zero = z_truncated.numel() - torch.sum(z_truncated == 0).item()
        print('start evaluating')
        _, odds_average = evaluate(z_truncated)
        print("odds_average",odds_average)
        writer.writerow([step, -1, odds_average, "", sparsity, non_zero, 0.0 ])


        # Write the same information to an additional file
        with open(file_path_add, mode='a', newline='') as additional_file:
            additional_writer = csv.writer(additional_file)
            additional_writer.writerow([step, -1, odds_average, "", sparsity, non_zero, 0.0])
    end_time = time.time()
    execution_time = end_time-start_time
    print("Time: ", execution_time)
    writer.writerow(["Time", execution_time])

    num_batch = int(np.ceil(len(batch_texts)/ batch_size))

    # Create the repeating divisor pattern [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, ...]
    pattern = np.arange(1, num_batch + 1, dtype=float)  # Creates [1, 2, 3, 4, 5]
    divisor = np.tile(pattern, len(losses) // num_batch + 1)[:len(losses)]  # Repeat and trim to length N
    losses_array = np.array(losses, dtype=float) / divisor
    sparsity_array = np.array(sparsities, dtype=float)

    np.save(f'{directory_name}/output/losses.npy', losses_array)
    np.save(f'{directory_name}/output/sparsities.npy', sparsity_array)


    # # Flatten the tensor and get the top 25 values and their indices
    # top_values, top_indices = torch.topk(z.view(-1), 25)

    # # Convert the flattened indices back to the original shape
    # # original_indices = torch.unravel_index(top_indices, z.shape)

    # # Convert the flattened indices back to the original shape
    # # Manual unravel index equivalent for PyTorch
    # indices = []
    # for size in reversed(z.shape):
    #     indices.append(top_indices % size)
    #     top_indices = top_indices // size

    # # The indices are calculated in reverse order, so reverse them back
    # original_indices = tuple(reversed(indices))

    # print("Top-25 elements in z:")
    # for i in range(25):
    #     index = (original_indices[0][i].item(),original_indices[1][i].item())
    #     value = top_values[i].item()
    #     print(f"Index: {index}, Value: {value}")






