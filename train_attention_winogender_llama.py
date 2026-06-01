import os 
from datetime import datetime

from utils.utils_attention_new import accelerated_candidate_probs, get_probabilities_for_examples_multitoken_batch, get_probabilities_for_examples_multitoken_batch_grad, MyAttentionWrapper, evaluate

from utils.utils_all import seed_everything, scheduling

import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import torch.optim as optim

import time
import csv

### Define the dataset
from datasets import winogender


torch.autograd.set_detect_anomaly(True)

# Get the current time
current_time = datetime.now().strftime("%Y-%m-%d")

import argparse
parser = argparse.ArgumentParser(description="Causal Tracing")
parser.add_argument("--start", type=str, help = 'start for dataset', default='bergsma', choices=['bergsma', 'bls'])
parser.add_argument("--model", type=str, help = 'model type', default='llama')
parser.add_argument("--lambda1", type=float, help = 'Regularization 1' , default=1e-4)
parser.add_argument("--lambda2", type=float, help = 'Regularization 2' ,default=1e-4)
parser.add_argument("--scheduling_method", type=str, help = 'Scheduling method', default="linear")
parser.add_argument("--lr", type=float, help = 'Learning rate' ,default=1e-1)
parser.add_argument("--batch_size", type=int, help = 'Training batch size' ,default=64)
# parser.add_argument("--eval_batch_size", type=int, help = 'Evaluation batch size' ,default=64)
parser.add_argument("--epochs", type=int, help = 'Total epochs' ,default=15)
parser.add_argument("--log_freq", type=int, help = 'Logging frequency' ,default=2)
parser.add_argument("--threshold", type=float, help = 'Truncation threshold' ,default=0.5)
parser.add_argument("--device", type=str, help = 'model type', default='cuda')
parser.add_argument("--trail", type=int, help = 'model type', default=42)
parser.add_argument("--model_name", type=str, help='Hugging Face model id or local model path', default='unsloth/Llama-3.2-1B')
parser.add_argument("--cache_dir", type=str, help = 'model save dir', default=None)


args = parser.parse_args()


print("**********************")
print(args.model)

l1 = args.lambda1
l2 = args.lambda2   # Weight for the binary-like penalty
method = args.scheduling_method


cache_dir=args.cache_dir

##########################
# Create a new directory with the current time as its name
directory_name = f"results/Attention_winogender/Ours/{current_time}/{args.model}_l1_{args.lambda1}_l2_{args.lambda2}_scheduling_{method}_lr_{args.lr}_trail_{args.trail}"
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

model_name = args.model_name

model_type = args.model
device = torch.device(args.device)

if args.cache_dir:
    print("&&& using cache_dir &&&")
    # Step 1: Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # Step 2: Initialize the model
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, output_hidden_states=True, attn_implementation="eager")
else:
    # Step 1: Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Step 2: Initialize the model
    model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, attn_implementation="eager")



model = model.to(device)
model.eval()

# Get the configuration object
config = model.config

num_layers = config.num_hidden_layers
num_attention_heads = config.num_attention_heads
hidden_size = config.hidden_size


# 1) Store original attention modules.
original_attentions = []
for i, layer in enumerate(model.model.layers):
    original_attentions.append(layer.self_attn)


examples = winogender.load_examples()



z = torch.full((num_layers, num_attention_heads), 0.5, requires_grad=True, device=device)


print("processing data")

batch_texts = [
    example.base_string.format(suffix)
    for example in examples
    for suffix in ["she", "he"]
]
if args.start=='bergsma':
    batch_candidates = [
    " " + candidate
    for example in examples
    for candidate in (
        [example.continuation_occupation, example.continuation_participant]
        if example.bergsma_pct_female > 50
        else [example.continuation_participant, example.continuation_occupation]
    )
    ]
    # pct_female = self.bergsma_pct_female
elif args.start == 'bls':
    batch_candidates = [
    " " + candidate
    for example in examples
    for candidate in (
        [example.continuation_occupation, example.male_continuation_participant]
        if example.bls_pct_female > 50
        else [example.continuation_participant, example.continuation_occupation]
    )
    ]
    # pct_female = self.bls_pct_female
else:
    raise ValueError('Invalid: ' + args.stat)

print("end processing data")

optimizer = optim.Adam([z], lr=args.lr)

# Multi-step gradient descent
num_steps = args.epochs
losses = []
sparsities = []

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
        # print(f"in step {step+1}")
        running_loss = 0
        running_odds = 0

        # # Assuming batch_texts is a tensor with shape [N, ...]
        N = len(batch_texts)
        shuffled_indices = torch.randperm(int(N/2))  # Generate a random permutation of indices

        # Now collect pairs from batch_texts using the shuffled indices
        shuffled_texts = []
        # shuffled_gender = []
        shuffled_candidates = []
        for i in shuffled_indices:
            shuffled_texts.append(batch_texts[i*2])        # Collect the item at index i
            shuffled_texts.append(batch_texts[i*2 + 1])    # Collect the adjacent item at index i + 1
            # shuffled_gender.append(batch_texts_gender[i])
            shuffled_candidates.append(batch_candidates[i*2])
            shuffled_candidates.append(batch_candidates[i*2 + 1])

        for bb in range(0, len(batch_texts), batch_size):
            # print("new bb")

            batch_chunk = shuffled_texts[bb:bb + batch_size]
            # gender_chuch = shuffled_gender[int(bb/2):int(bb/2) + int(batch_size/2)]
            candidate_chuck = shuffled_candidates[bb:bb + batch_size]

            # Tokenize the current batch and move it to the device
            batch_inputs = tokenizer(batch_chunk, return_tensors="pt", padding=True, truncation=True)

            batch_lengths = batch_inputs["attention_mask"].sum(dim=1)
            batch_lengths = batch_lengths[::2]

            batch_inputs = {key: value.to(device) for key, value in batch_inputs.items()}

            # candidates
            batch_candidates_idx = [
                tokenizer.encode(text, add_special_tokens=False)
                for text in candidate_chuck
            ]

            # Group them into pairs
            batch_candidates_idx = [
                batch_candidates_idx[i : i+2] 
                for i in range(0, len(batch_candidates_idx), 2)
            ]

            # Now transpose so that the first elements of each pair form one list,
            # and the second elements form another.
            batch_candidates_idx = list(map(list, zip(*batch_candidates_idx)))

            # print("through model")


            # print("Outside")
            # print(batch_inputs["input_ids"].shape)

            # Run the batch through the model
            with torch.no_grad():
                # 3) Reassign the original attentions when you're done.
                for i, layer in enumerate(model.model.layers):
                    layer.self_attn = original_attentions[i]

                batch_outputs = model(**batch_inputs, output_attentions=True)
                attention_weights = batch_outputs.attentions  # Extract original attention weights

                logits = batch_outputs[0]
                # print(logits.shape)
                # print("Done")


                original_attention_weights = tuple(w[::2, :, :] for w in attention_weights)
                conterfactual_attention_weights = tuple(w[1::2, :, :] for w in attention_weights)

                result_base = get_probabilities_for_examples_multitoken_batch(batch_inputs["input_ids"][::2].tolist(), batch_candidates_idx,model,tokenizer,device)

                first, second = result_base

                # Elementwise division: (first_list[i] / second_list[i])
                odds_base = [s / f for f, s in zip(first, second)]



            # print("through override model")
            # Suppose `mask_shape` is the desired shape of each attention override mask.
            # For each layer, create the mask directly from z with proper broadcasting:
            attention_override_mask = [ z[layer_].view(1, -1, 1, 1).expand(conterfactual_attention_weights[layer_].shape) for layer_ in range(num_layers) ]

            # attention_override_mask = [torch.zeros_like(layer_attention_override) for layer_attention_override in conterfactual_attention_weights]
            # for layer_ in range(12):
            #     for head_ in range(12):
            #         attention_override_mask[layer_][:,head_] = z[layer_][head_]  # Override for head 1 only (example)

            # Replace GPT-2 attention layers with counterfactual attention layers
            for i, layer in enumerate(model.model.layers):
                model.model.layers[i].self_attn = MyAttentionWrapper(
                    orig_attn=original_attentions[i],
                    attn_override=conterfactual_attention_weights[i],
                    attn_override_mask=attention_override_mask[i],
                    override_seq_len = batch_lengths,
                    layer_idx = i
                )
                # model.model.layers[i].self_attn = MyAttentionWrapper(
                #     original_attention=original_attentions[i],
                #     attn_override=conterfactual_attention_weights[i],
                #     attn_override_mask=attention_override_mask[i]
                # )

            results = accelerated_candidate_probs(batch_inputs["input_ids"][::2].tolist(), batch_candidates_idx,model,device)

            # print("***********************")
            # print(results)

            first, second = results
            odds_ratio = [s / f for f, s in zip(first, second)]

            modified_odds_ratios = torch.stack([ (o_r - o_b) / o_b for (o_r,o_b) in zip(odds_ratio, odds_base)])
           
              
            l1_norm = torch.sum(torch.abs(z))
            l1_reg = scheduling(l1, step, method=method)
            l2_reg = scheduling(l2, step, method=method)   # Weight for the binary-like penalty

            # Binary-like penalty (pushing values to 0 or 1)
            l2_norm = torch.sum(z * (1 - z))

            losses_ = 1 / (1 + modified_odds_ratios) + l1_reg * l1_norm + l2_reg * l2_norm


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

        for i, layer in enumerate(model.model.layers):
            layer.self_attn = original_attentions[i]

        # Save z for the current step
        z_save_path = os.path.join(f"{directory_name}/ckpt/", f"z_step_{step + 1}.pt")
        torch.save(z, z_save_path)

        z_truncated = (z >= args.threshold).float()
        odds_all, odds_average = evaluate(model, z_truncated, batch_texts, batch_candidates, tokenizer,original_attentions, device)
        print(f"Step {step+1}, odds_average: {odds_average}")
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






