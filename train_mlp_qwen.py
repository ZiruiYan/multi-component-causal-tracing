import os 


from datetime import datetime

from datasets.professions import get_template_list, word_list_female, word_list_male
from utils.utils_qwen import evaluate
from utils.utils_all import seed_everything, scheduling

import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import torch.optim as optim

import time
import csv

# Get the current time
current_time = datetime.now().strftime("%Y-%m-%d")

import argparse
parser = argparse.ArgumentParser(description="Causal Tracing")
parser.add_argument("--lambda1", type=float, help = 'Regularization 1' , default=1e-4)
parser.add_argument("--model", type=str, help = 'model type', default='Qwen')
parser.add_argument("--lambda2", type=float, help = 'Regularization 2' ,default=1e-4)
parser.add_argument("--scheduling_method", type=str, help = 'Scheduling method', default="linear")
parser.add_argument("--lr", type=float, help = 'Learning rate' ,default=5e-1)
parser.add_argument("--batch_size", type=int, help = 'Training batch size' ,default=64)
# parser.add_argument("--eval_batch_size", type=int, help = 'Evaluation batch size' ,default=64)
parser.add_argument("--epochs", type=int, help = 'Total epochs' ,default=30)
parser.add_argument("--log_freq", type=int, help = 'Logging frequency' ,default=10)
parser.add_argument("--threshold", type=float, help = 'Truncation threshold' ,default=0.5)
parser.add_argument("--device", type=str, help = 'model type', default='cuda')
parser.add_argument("--trail", type=int, help = 'model type', default=42)
parser.add_argument("--cache_dir", type=str, help = 'model save dir', default=None)
parser.add_argument("--model_name", type=str, help='Hugging Face model id or local model path', default='Qwen/Qwen3-1.7B-Base')

args = parser.parse_args()


print("**********************")
print(args.model)

l1 = args.lambda1
l2 = args.lambda2   # Weight for the binary-like penalty
method = args.scheduling_method

cache_dir=args.cache_dir

##########################
# Create a new directory with the current time as its name
directory_name = f"results/MLP/Our/{current_time}/{args.model}_l1_{args.lambda1}_l2_{args.lambda2}_scheduling_{method}_lr_{args.lr}"
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
# model_name = "Qwen/Qwen2.5-1.5B"
model_name = args.model_name

model_type = args.model
device = torch.device(args.device)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name,  output_hidden_states=True, cache_dir=cache_dir)



model = model.to(device)
model.eval()

# print(model)

# Get the configuration object
config = model.config


num_layers = config.num_hidden_layers
num_attention_heads = config.num_attention_heads
hidden_size = config.hidden_size


templates = get_template_list()



z = torch.full((num_layers, hidden_size), 0.5, requires_grad=True, device=device)



optimizer = optim.Adam([z], lr=args.lr)



# Step 6: Extract hidden states for "he" and "she" tokens and normalize
# Get token indices for "he" and "she"
he_index = tokenizer.encode(" he", add_special_tokens=False)[0]
she_index = tokenizer.encode(" she", add_special_tokens=False)[0]


# Multi-step gradient descent
num_steps = args.epochs
losses = []
sparsities = []

batch_texts = []  # Collect texts for the batch
batch_texts_gender = []
for gender in ['male', 'female']:
    # print(f"in gender {gender}")
    word_list = word_list_male if gender == 'male' else word_list_female

    for template in templates:
        for word in word_list:
            # Collect texts for batch processing
            if gender == 'male':
                batch_texts.append(template.format(word))
                batch_texts.append(template.format("woman"))
                batch_texts_gender.append('male')
            elif gender == 'female':
                batch_texts.append(template.format(word))
                batch_texts.append(template.format("man"))
                batch_texts_gender.append('female')


batch_size = args.batch_size
# eval_batch_size = args.eval_batch_size





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
        shuffled_indices = torch.randperm(int(N/2))  # Generate a random permutation of indices

        # Now collect pairs from batch_texts using the shuffled indices
        shuffled_texts = []
        shuffled_gender = []
        for i in shuffled_indices:
            shuffled_texts.append(batch_texts[i*2])        # Collect the item at index i
            shuffled_texts.append(batch_texts[i*2 + 1])    # Collect the adjacent item at index i + 1
            shuffled_gender.append(batch_texts_gender[i])

        for bb in range(0, len(batch_texts), batch_size):
            batch_chunk = shuffled_texts[bb:bb + batch_size]
            gender_chuch = shuffled_gender[int(bb/2):int(bb/2) + int(batch_size/2)]

            # Tokenize the current batch and move it to the device
            batch_inputs = tokenizer(batch_chunk, return_tensors="pt", padding=True, truncation=True)
            batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}

            batch_lengths = batch_inputs["attention_mask"].sum(dim=1) - 1


            seq_len = batch_inputs["input_ids"].size(1)
            position_ids_full = torch.arange(seq_len, device=device).unsqueeze(0).expand_as(batch_inputs["input_ids"])

            # Get mask for Qwen
            attn_mask_full = batch_inputs["attention_mask"].to(torch.bool)          # shape [B, L]
            attn_mask_1     = attn_mask_full[::2]                    # even indices
            attn_mask_2     = attn_mask_full[1::2] 
            pos_ids_1       = position_ids_full[::2]
            pos_ids_2       = position_ids_full[1::2]


            # Run the batch through the model
            with torch.no_grad():
                mlp_outputs = []
                def hook_fn(module, input, output):
                    mlp_outputs.append(output.detach())

                hook_handles = []

                # For GPT2LMHeadModel, the blocks are in `model.transformer.h`
                for block in model.model.layers:
                    handle = block.mlp.register_forward_hook(hook_fn)
                    hook_handles.append(handle)

                batch_outputs = model(**batch_inputs)
                batch_hidden_states = batch_outputs.hidden_states

                mlp_outputs_1 = tuple(m[::2, :, :] for m in mlp_outputs)  # Take even indices
                mlp_outputs_2 = tuple(m[1::2, :, :] for m in mlp_outputs)  # Take odd indices

                # Process hidden states for pairs in batches
                hidden_states_1 = tuple(h[::2, :, :] for h in batch_hidden_states)  # Take even indices
                hidden_states_2 = tuple(h[1::2, :, :] for h in batch_hidden_states)  # Take odd indices

                # 1) Create a range [0,1,2,...,batch_size-1] to index the batch dimension
                batch_indices = torch.arange(batch_outputs.logits.shape[0], device=device)

                # 2) Use advanced indexing:
                picked = batch_outputs.logits[batch_indices,  batch_lengths, :]

                logits_1 = picked[::2, :]
                logits_2 = picked[1::2, :]

                logits_1 = F.softmax(logits_1, dim=-1)
                logits_2 = F.softmax(logits_2, dim=-1)
                # Calculate logits for gender-specific words
                logits_heshe_1 = torch.cat((logits_1[:, he_index].unsqueeze(1), logits_1[:, she_index].unsqueeze(1)), dim=1)
                logits_heshe_2 = torch.cat((logits_2[:, he_index].unsqueeze(1), logits_2[:, she_index].unsqueeze(1)), dim=1)

                # Initialize the mixed hidden states as a clone of hidden_states_1
                hidden_states_mix = [hidden_state.clone() for hidden_state in hidden_states_1]
                hidden_states_mix[0] = hidden_states_1[0]

                # When you're done collecting the outputs and want to remove the hooks:
                for handle in hook_handles:
                    handle.remove()

            # Perform mixing operation in batches
            for layer_idx in range(1, len(hidden_states_1)):
                inputs_for_layer = {'hidden_states': hidden_states_mix[layer_idx - 1]}

                # Unpack the hidden states
                hidden_states = inputs_for_layer['hidden_states']

                # Layer Normalization before self-attention
                hidden_states_norm_1 = model.model.layers[layer_idx - 1].input_layernorm(hidden_states)

                block = model.model.layers[layer_idx - 1]
                seq_len = hidden_states_norm_1.size(1)
                batch_size_ = hidden_states_norm_1.size(0)

                # Build position_ids and RoPE once
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size_, -1)
                cos, sin = model.model.rotary_emb(
                              hidden_states_norm_1,         # dummy tensor, only dtype/device matter
                              position_ids
                          )
                position_embeddings = (cos, sin)

                # # Causal mask in 4-D format [B, 1, T, T]
                # attn_mask = model.model._prepare_decoder_attention_mask(
                #                 batch_inputs["attention_mask"],
                #                 (batch_size, seq_len),
                #                 dtype=hidden_states_norm_1.dtype,
                #                 device=device,
                # )

                # print("hidden_states_norm_1")
                # print(hidden_states_norm_1.shape)
                # print("position_embeddings")
                # print(position_embeddings[0].shape)
                # print("attn_mask_1")
                # print(attn_mask_1.shape)

                # Self-Attention mechanism
                attention_outputs = model.model.layers[layer_idx - 1].self_attn(
                                                                    hidden_states_norm_1,
                                                                    position_ids=pos_ids_1,
                                                                    position_embeddings = position_embeddings,         # NEW
                                                                    attention_mask=None,
                                                                    padding_mask= attn_mask_1,                   # NEW
                                                                    )
                attention_output = attention_outputs[0]  # Select the main attention output

                # Add skip connection (residual connection) after self-attention
                hidden_states_after_attention = hidden_states + attention_output

                # Layer Normalization before feed-forward network
                hidden_states_norm_2 = model.model.layers[layer_idx - 1].post_attention_layernorm(hidden_states_after_attention)

                # Feed-Forward Network (MLP)
                mlp_output = model.model.layers[layer_idx - 1].mlp(hidden_states_norm_2)
                # print(layer_idx)
                mlp_output = mlp_output * (1 - z[layer_idx - 1]) + mlp_outputs_2[layer_idx - 1] * z[layer_idx - 1]


                # Add skip connection (residual connection) after the feed-forward network
                hidden_states_output = hidden_states_after_attention + mlp_output

                if layer_idx == len(hidden_states_1)-1:
                    # Final layer normalization and logits calculation for the mixed hidden states
                    hidden_states_mix[layer_idx] = model.model.norm(hidden_states_output)
                else:
                    hidden_states_mix[layer_idx] = hidden_states_output

            logits_mix = model.lm_head(hidden_states_mix[-1])


            batch_indices = torch.arange(logits_mix.shape[0], device=device)
            batch_lengths = batch_lengths[::2]

            # 2) Use advanced indexing:
            picked = logits_mix[batch_indices,  batch_lengths, :]


            logits_mix = F.softmax(picked,dim=-1)

            # Compute softmax for mixed logits
            # logits_heshe_mix = F.softmax(torch.cat((logits_mix[:, he_index].unsqueeze(1), logits_mix[:, she_index].unsqueeze(1)), dim=1), dim=-1)
            logits_heshe_mix = torch.cat((logits_mix[:, he_index].unsqueeze(1), logits_mix[:, she_index].unsqueeze(1)), dim=1)

            # Calculate odds ratio and loss in batches
            odds_ratios = (logits_heshe_mix[:, 0] / logits_heshe_mix[:, 1]) / (logits_heshe_1[:, 0] / logits_heshe_1[:, 1])

            # 1 represents 'male' and 0 represents 'female'
            gender_chuch_tensor = torch.tensor([1 if gender == 'male' else 0 for gender in gender_chuch])

            # Create masks for male and female
            male_mask = (gender_chuch_tensor == 1).to(device)
            # female_mask = gender_chuch_tensor == 0

            # Inverse the values for males, keep the values for females
            # torch.where is used to apply conditions while preserving gradient flow
            modified_odds_ratios = torch.where(male_mask, 1.0 / odds_ratios, odds_ratios) - 1
           
              
            l1_norm = torch.sum(torch.abs(z))
            l1_reg = scheduling(l1, step, method=method)

            l2_reg = scheduling(l2, step, method=method)   # Weight for the binary-like penalty

            # Binary-like penalty (pushing values to 0 or 1)
            l2_norm = torch.sum(z * (1 - z))

            losses_ = 1 / (1 + modified_odds_ratios) + l1_reg * l1_norm + l2_reg * l2_norm
            # losses_ = - modified_odds_ratios + l1_reg * l1_norm + l2_reg * l2_norm

            loss_average = torch.mean(losses_)
            odds_average = torch.mean(modified_odds_ratios)
            loss_average.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss += loss_average.item()
            running_odds += odds_average
            with torch.no_grad():
                z.clamp_(0, 1)
            
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

     
        # Save z for the current step
        z_save_path = os.path.join(f"{directory_name}/ckpt/", f"z_step_{step + 1}.pt")
        torch.save(z, z_save_path)

        z_truncated = (z >= args.threshold).float()
        sparsity = torch.sum(z_truncated == 0).item() / z_truncated.numel()
        non_zero = z_truncated.numel() - torch.sum(z_truncated == 0).item()
        _, _, _, _, odds_average = evaluate(model, z_truncated, batch_texts, batch_texts_gender, tokenizer, device)
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







