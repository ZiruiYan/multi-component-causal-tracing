import numpy as np

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn.functional as F
import torch.optim as optim


def evaluate(model, z, batch_texts, batch_texts_gender, tokenizer, device= "cpu"):
    eval_batch_size = 64
    odds_all = []
    he_index = tokenizer.encode(" he", add_special_tokens=False)[0]
    she_index = tokenizer.encode(" she", add_special_tokens=False)[0]
    for bb in range(0, len(batch_texts), eval_batch_size ):
        batch_chunk = batch_texts[bb:bb + eval_batch_size ]  # Take a batch of 10 texts
        gender_chuch = batch_texts_gender[int(bb/2):int(bb/2) + int(eval_batch_size/2)]

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

            # Keep track of the hook handles in a list
            hook_handles = []

            # For GPT2LMHeadModel, the blocks are in `model.transformer.h`
            for block in model.model.layers:
                handle = block.mlp.register_forward_hook(hook_fn)
                hook_handles.append(handle)

            batch_outputs = model(**batch_inputs)
            batch_hidden_states = batch_outputs.hidden_states

            # print(mlp_outputs)

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

            odds_all.extend(modified_odds_ratios.tolist())

    odds_average = np.mean(odds_all)

    return logits_heshe_1, logits_heshe_2, logits_heshe_mix, odds_all, odds_average
