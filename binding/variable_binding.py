import os
import random
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from baukit import Trace, TraceDict
from transformers import LlamaTokenizer, LlamaForCausalLM
from functools import partial
import importlib 

try:
    from .variable_binding_utils import get_data, predict_answers
    from . import desideratum
except ImportError:
    from variable_binding_utils import get_data, predict_answers
    import desideratum

from collections import OrderedDict
# from desideratum import ValueSwitchDesideratum, TaskSwitchDesideratum

SEED = 1234
os.environ["PYTHONHASHSEED"] = str(SEED)  # Python's hash-based randomness
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Make CUDA/cuDNN pick deterministic algorithms where possible
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Optional: fail/warn on non-deterministic ops (einsum, some reductions).
# Use warn_only=True to avoid hard errors if an op lacks a deterministic path.
torch.use_deterministic_algorithms(True, warn_only=True)

# %%

importlib.reload(desideratum)

# %% Load model 

WEIGHT_PATH = os.environ.get("VBD_MODEL_PATH")
if WEIGHT_PATH is None:
    raise RuntimeError("Set VBD_MODEL_PATH to a local or Hugging Face Llama model path before running VBD.")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
print("Using device:", DEVICE)

# don't reload model if already loaded
if "model" not in locals():

    print("Loading model...")
    # configure model
    tokenizer = LlamaTokenizer.from_pretrained(f"{WEIGHT_PATH}")
    model = LlamaForCausalLM.from_pretrained(f"{WEIGHT_PATH}")

    tokenizer.pad_token_id = tokenizer.eos_token_id

    # set to float16 TODO undo
    model.half()
    model.to(DEVICE)
    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)
NUM_HEADS = model.config.num_attention_heads
HEAD_SIZE = model.config.hidden_size // NUM_HEADS

# %% Get data. Throughout this file, ValueSwitch means Value Dependence and Task Switch means Operation Invariance.

train_value_switch = desideratum.ValueSwitchDesideratum(tokenizer, operations = ["+", "-"], num_samples = 50, var_values = range(10, 100), device = DEVICE)
train_task_switch = desideratum.TaskSwitchDesideratum(tokenizer, operations = ["+", "-"], num_samples = 50, var_values = range(10, 100), device = DEVICE)

desids = [train_value_switch, train_task_switch]

# store the logits for the "original" sequences, for use as a comparison to patched model outputs later

with torch.no_grad():
    for desid in desids:
        desid.set_target_data(model(**desid.tokenized_to_samples)["logits"])

# %% Evaluate baseline model accuracies on the two desiderata

FIRST_DIGIT = True
generated_answers = predict_answers(model, tokenizer, train_value_switch.from_samples, first_digit_only=FIRST_DIGIT)
_correct_answers = [int(str(right.item())[0]) for right in train_value_switch.from_answers] if FIRST_DIGIT else train_value_switch.to_answers
print(f"Value Switch Accuracy (first digit): {np.mean(np.array(generated_answers) == np.array(_correct_answers))}")

generated_answers = predict_answers(model, tokenizer, train_task_switch.to_samples, first_digit_only=FIRST_DIGIT)
_correct_answers = [int(str(right.item())[0]) for right in train_task_switch.to_answers] if FIRST_DIGIT else train_task_switch.from_answers
print(f"Task Switch Accuracy (first digit): {np.mean(np.array(generated_answers) == np.array(_correct_answers))}")

# %% Get module name list

NUM_LAYERS = len(model.model.layers)
modules = [
    [f"model.layers.{i}.self_attn.o_proj", f"model.layers.{i}.mlp"]
    for i in range(NUM_LAYERS)
]
modules = [item for sublist in modules for item in sublist]

# %% Save activations on alternative samples, to use for patching
from_activations = {}

for desid in desids:
    with TraceDict(model, modules, retain_input=True) as ret:
        _ = model(**desid.tokenized_from_samples.to(DEVICE))
        from_activations[desid] = ret

for desid in desids:
    for module in modules:
        if "self_attn" in module:
            # then we want to get the input to o_proj instead of the output
            from_activations[desid][module] = from_activations[desid][module].input
        else:
            # we want the output of the mlp
            from_activations[desid][module] = from_activations[desid][module].output
# %%

# helper funcs

def get_highest_tokens(logits):
    probs = torch.softmax(logits, dim=-1).mean(0)
    topk = torch.topk(probs, k=10)
    topk_tokens = tokenizer.convert_ids_to_tokens(topk.indices)
    return topk_tokens

def get_avg_target_prob(logits, idxs=train_value_switch.from_answers_logit_idxs):
    probs = torch.softmax(logits[:, -1], dim=-1)
    return probs.gather(1, idxs.unsqueeze(1)).mean()

# %%

# original_logits = model(x)["logits"].detach()

# %%
# helper func for the convex combo patching, and expanded module name list (now with head numbers!)

modules_w_heads = []
for module in modules:
    if "self_attn" in module:
        for head in range(NUM_HEADS):
            modules_w_heads.append(f"{module}.{head}")
    else:
        modules_w_heads.append(module)

mask_dict = {module:i for i, module in enumerate(modules_w_heads)}

# train mask
TARGET_TOKEN_IDX = -1

def edit_output(inputs=None, output=None, layer=None, mask=None, from_activations=None):
    if "self_attn" in layer:
        inp = inputs[0]
        # individually ablated each of the attention_heads at the given token_idx
        mod_heads = []
        for head_idx in range(NUM_HEADS):
            head_start = head_idx * HEAD_SIZE
            head_end = (head_idx + 1) * HEAD_SIZE    
            abl_amt = mask[mask_dict[f"{layer}.{head_idx}"]]
            mod_head = abl_amt*inp[:, TARGET_TOKEN_IDX, head_start:head_end] + (1-abl_amt)*from_activations[layer][:, TARGET_TOKEN_IDX, head_start:head_end]
            mod_heads.append(mod_head)
        mod_inp_last_token = torch.cat(mod_heads, dim=-1)

        # if not torch.allclose(mod_inp_last_token, output[:, TARGET_TOKEN_IDX]
        mod_inp = torch.cat([inp[:, :-1], mod_inp_last_token.unsqueeze(1)], dim=1)
        # print(input.shape, from_activations[module][:, TARGET_TOKEN_IDX, :].shape)

        weights = model.state_dict()[f"{layer}.weight"]
        mod_output = torch.einsum("bsh,oh->bso", mod_inp, weights) # weight is out x in, and there's no bias
        return mod_output
    elif "mlp" in layer:
        abl_amt = mask[mask_dict[layer]]
        mod_out_last_token = abl_amt*output[:, TARGET_TOKEN_IDX] + (1-abl_amt)*from_activations[layer][:, TARGET_TOKEN_IDX]
        mod_out = torch.cat([output[:, :-1], mod_out_last_token.unsqueeze(1)], dim=1) # this cat is one way to get around the problems of overwriting operations & gradients
        return mod_out
    else:
        assert False, "shouldn't be here"

# %%

lambs1 = [0.01,0.015,0.02,0.015,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.06, 0.065, 0.07, 0.075, 0.08, 0.09, 0.1, 0.11, 0.12]
lambs2 = [0.5,0.1,0.01,0.001]

masks = {}
masks[100] = torch.ones(len(modules_w_heads), device=DEVICE, dtype=torch.float)

for lamb1 in tqdm(lambs1):
    for lamb2 in tqdm(lambs2):
        mask = torch.ones(len(modules_w_heads), requires_grad=True, device=DEVICE, dtype=torch.float)
        optimizer = torch.optim.Adam([mask], lr=1e-2)

        for i in range(100):
            optimizer.zero_grad()
            mask.data.clamp_(0, 1)

            with TraceDict(model, modules, edit_output=partial(edit_output, mask=mask, from_activations=from_activations[train_value_switch])) as ret:
                logits = model(train_value_switch.tokenized_to_samples.input_ids)["logits"]
                value_loss = train_value_switch(logits)

            # loss = 5 * value_loss + lamb*torch.sum((1.001-mask)**0.5)
            loss = 5 * value_loss + lamb1*torch.sum(torch.abs(1-mask)) * (i//20+1)/5 + lamb2 * torch.sum(mask * (1 - mask)) * (i//20+1)/5
            loss.backward()

            # print(f"For value switch: Heads patched: {1640 - sum(mask)}")
            # print(f"Value switch loss: {loss.item()}")
            # print(f"Avg. from target prob: {get_avg_target_prob(logits, train_value_switch.from_answers_logit_idxs)}")
            optimizer.step()

            optimizer.zero_grad()
            mask.data.clamp_(0, 1)

            with TraceDict(model, modules, edit_output=partial(edit_output, mask=mask, from_activations=from_activations[train_task_switch])) as ret:
                logits = model(train_task_switch.tokenized_to_samples.input_ids)["logits"]
                task_loss = train_task_switch(logits)

            loss = task_loss
            loss.backward()

            # print(f"For task switch: Heads patched: {1640 - sum(mask)}")
            # print(f"Task switch loss: {loss.item()}")
            # print(f"Avg. to target prob: {get_avg_target_prob(logits, train_task_switch.to_answers_logit_idxs)}")

            optimizer.step()
            if i%10 == 9 or i%10 == 4:
                mask.data.clamp_(0, 1)
                masks[(lamb1,lamb2, i)] = mask.detach().cpu()


# %%

# import pickle
# # unpickle
# with open("masks.pkl", "rb") as f:
#     double_masks = pickle.load(f)   

# with open("single_masks.pkl", "rb") as f:     
#     single_masks = pickle.load(f)
rounded = [torch.round(mask) for mask in masks.values()]

# rounded = double_masks

# %%

# Evaluate masks on held-out test sets

# load in test data and save original logits and alternative-sequence activations

test_value_switch = desideratum.ValueSwitchDesideratum(tokenizer, operations = ["+", "-"], num_samples = 50, var_values = range(10, 100), device = DEVICE)
test_task_switch = desideratum.TaskSwitchDesideratum(tokenizer, operations = ["+", "-"], num_samples = 50, var_values = range(10, 100), device = DEVICE)


# %%
with torch.no_grad():
    test_value_switch.set_target_data(model(**test_value_switch.tokenized_to_samples)["logits"])
    test_task_switch.set_target_data(model(**test_task_switch.tokenized_to_samples)["logits"])

with TraceDict(model, modules, retain_input=True) as ret:
    _ = model(**test_value_switch.tokenized_from_samples.to(DEVICE))
    from_activations[test_value_switch] = ret

with TraceDict(model, modules, retain_input=True) as ret:
    _ = model(**test_task_switch.tokenized_from_samples.to(DEVICE))
    from_activations[test_task_switch] = ret
    

for module in tqdm(modules):
    if "self_attn" in module:
        # then we want to get the input to o_proj instead of the output
        from_activations[test_value_switch][module] = from_activations[test_value_switch][module].input
        from_activations[test_task_switch][module] = from_activations[test_task_switch][module].input
    else:
        # we want the output of the mlp
        from_activations[test_value_switch][module] = from_activations[test_value_switch][module].output
        from_activations[test_task_switch][module] = from_activations[test_task_switch][module].output

# %%

# preserve mask order & give each mask a readable name
ordered_mask_items = list(masks.items())
rounded_named = OrderedDict(
    (("baseline_all_ones" if (isinstance(k, int) and k == 100) else f"lambda_{k}"),
     torch.round(v).detach().cpu())
    for k, v in ordered_mask_items
)

# raw collectors (BEFORE any dict bucketing/sorting/averaging)
val_pairs_raw  = []   # list of (mask_name, k, acc)
task_pairs_raw = []   # list of (mask_name, k, acc)
# --- END OF NEW LINES ---

# run the evaluation to get accuracies
val_results, task_results = {}, {}

with torch.no_grad():
    # for round_mask in rounded: # change mask set if you want!
    for mname, round_mask in rounded_named.items():
        # get top k locations in mask
        k = int(1640 - sum(round_mask))

        print(f"Out of 1640 components, {k} are patched in.")
        FIRST_DIGIT = True
        with TraceDict(model, modules, edit_output=partial(edit_output, mask=round_mask, from_activations=from_activations[test_value_switch])) as ret:
            generated_answers = predict_answers(model, tokenizer, test_value_switch.to_samples, first_digit_only=FIRST_DIGIT)
            _correct_answers = [int(str(right.item())[0]) for right in test_value_switch.from_answers] if FIRST_DIGIT else test_value_switch.from_answers
            acc = np.mean(np.array(generated_answers) == np.array(_correct_answers))
            val_results[k] = acc
            val_pairs_raw.append((str(mname), int(k), float(acc)))
            print(acc)

        with TraceDict(model, modules, edit_output=partial(edit_output, mask=round_mask, from_activations=from_activations[test_task_switch])) as ret:
            generated_answers = predict_answers(model, tokenizer, test_task_switch.to_samples, first_digit_only=FIRST_DIGIT)
            _correct_answers = [int(str(right.item())[0]) for right in test_task_switch.to_answers] if FIRST_DIGIT else test_task_switch.to_answers
            acc = np.mean(np.array(generated_answers) == np.array(_correct_answers))
            task_results[k] = acc
            task_pairs_raw.append((str(mname), int(k), float(acc)))
            print(acc)
