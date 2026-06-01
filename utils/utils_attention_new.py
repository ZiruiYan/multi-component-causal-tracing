import torch
import os 
from datetime import datetime

import itertools

import statistics
import math

import pandas as pd


import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def accelerated_candidate_probs(contexts, candidates, model, device):
    # Pre-process contexts: remove EOS tokens (50256) and convert to tensors.
    processed_contexts = [
        torch.tensor([token for token in context if token != 50256], dtype=torch.long)
        for context in contexts
    ]

    all_mean_probs = []
    
    # Process each candidate group (each candidate group is a list with one candidate per context)
    for candidate_this in candidates:
        combined_sequences = []   # To hold context + candidate (minus one token for prediction)
        candidate_tokens = []     # To hold candidate token tensors (for scoring)
        context_lens = []         # Save lengths of each (cleaned) context
        
        # For each context, build the combined sequence and candidate tensor.
        for context, candidate in zip(processed_contexts, candidate_this):
            # Save the (cleaned) context length.
            context_lens.append(len(context))
            
            # Convert candidate to tensor.
            cand_tensor = torch.tensor(candidate, dtype=torch.long)
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

        # print(input_ids.shape)
        
        # Run the model in one batch forward pass.
        # Assume the model returns logits with shape (batch_size, seq_len, vocab_size)
        # print("Processing", input_ids)

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
        all_mean_probs.append(mean_probs_for_this_batch)
    
    return all_mean_probs



def get_probabilities_for_examples_multitoken_batch_grad( 
                                                    contexts, 
                                                    candidates, 
                                                    model,
                                                    device='cpu'
                                                    ):
    """
    Calculate probabilities for multi-token candidates given multiple contexts.

    Args:
        model: Language model that returns logits given a batch of token IDs
        contexts: List of lists of token IDs, one list per example
        candidates: List of lists (length 2) of token ID lists.
                    i.e. candidates[i] = [cand1, cand2] for contexts[i]
        device: 'cuda' or 'cpu'

    Returns:
        all_mean_probs: list of lists, shape: [num_contexts][num_candidates_per_context]
            each sub-list contains the probability (normalized by number of tokens) 
            of each candidate for that context
    """

    all_mean_probs = []

        # For each candidate in this context
    for candidate_this in candidates:

        # Combine context + candidate. Exclude the last token when predicting the next one
        # combined = [ context + candidate for (context,candidate) in zip(contexts,candidate_this)]
        combined = [ [token for token in context if token != 50256] + candidate for (context,candidate) in zip(contexts,candidate_this)]
        combined_filtered = [ c[:-1] for c in combined]
        max_len = max(len(lst) for lst in combined_filtered)
        padded_sequences = [
            lst + [50256]*(max_len - len(lst))  # pad with zeros (or any other value)
            for lst in combined_filtered
        ]
        input_ids = torch.tensor(padded_sequences).unsqueeze(dim=0).to(device)
        # Run the model forward
        logits = model(input_ids)[0]
        # logits shape: (batch_size=1, seq_len, vocab_size)
        # We just have one example in the batch, so index out batch dim:
        logits = logits[0]  # shape => (seq_len, vocab_size)

        # Convert logits to log probs
        log_probs = F.log_softmax(logits, dim=-1)

        # Indices for where the continuation starts/ends
        context_end_pos = [len([token for token in context if token != 50256]) - 1 for context in contexts]  # last index from context
        continuation_end_pos = [   e+len(l) for (e,l) in zip(context_end_pos,candidate_this) ] 
        # continuation_end_pos = context_end_pos + len(candidate)

        mean_probs_for_this_batch = []
        # for each context
        for idx in range(len(context_end_pos)):
            token_log_probs = []
            # For each position in the continuation, get the log prob of the next token
            for i in range(context_end_pos[idx], continuation_end_pos[idx]):
                next_token_id = combined[idx][i+1]
                next_token_log_prob = log_probs[idx, i, next_token_id]
                token_log_probs.append(next_token_log_prob)

            # Average log-prob over the tokens in the candidate, then exponentiate
            mean_token_log_prob = torch.mean(torch.stack(token_log_probs))
            mean_token_prob = torch.exp(mean_token_log_prob)

            mean_probs_for_this_batch.append(mean_token_prob)

        # Collect the probabilities for this "candidates" for all context
        all_mean_probs.append(mean_probs_for_this_batch)

    return all_mean_probs

def get_probabilities_for_examples_multitoken_batch( 
                                                    contexts, 
                                                    candidates, 
                                                    model,
                                                    tokenizer,
                                                    device='cpu'
                                                    ):
    """
    Calculate probabilities for multi-token candidates given multiple contexts.

    Args:
        model: Language model that returns logits given a batch of token IDs
        contexts: List of lists of token IDs, one list per example
        candidates: List of lists (length 2) of token ID lists.
                    i.e. candidates[i] = [cand1, cand2] for contexts[i]
        device: 'cuda' or 'cpu'

    Returns:
        all_mean_probs: list of lists, shape: [num_contexts][num_candidates_per_context]
            each sub-list contains the probability (normalized by number of tokens) 
            of each candidate for that context
    """

    all_mean_probs = []

    # For each candidate in this context
    for candidate_this in candidates:

        # Combine context + candidate. Exclude the last token when predicting the next one
        # combined = [ context + candidate for (context,candidate) in zip(contexts,candidate_this)]
        combined = [ [token for token in context if token != 50256] + candidate for (context,candidate) in zip(contexts,candidate_this)]
        combined_filtered = [ c[:-1] for c in combined]
        max_len = max(len(lst) for lst in combined_filtered)
        padded_sequences = [
            lst + [50256]*(max_len - len(lst))  # pad with zeros (or any other value)
            for lst in combined_filtered
        ]
        # input_ids = torch.tensor(padded_sequences).unsqueeze(dim=0).to(device)
        input_ids = torch.tensor(padded_sequences).to(device)


        # print("input_ids.shape")
        # print(input_ids.shape)
        # Run the model forward
        logits = model(input_ids)[0]
        # logits shape: (batch_size=1, seq_len, vocab_size)

        # Convert logits to log probs
        log_probs = F.log_softmax(logits, dim=-1)

        # Indices for where the continuation starts/ends
        context_end_pos = [len([token for token in context if token != 50256]) - 1 for context in contexts]  # last index from context
        continuation_end_pos = [   e+len(l) for (e,l) in zip(context_end_pos,candidate_this) ] 
        # continuation_end_pos = context_end_pos + len(candidate)

        mean_probs_for_this_batch = []
        # for each context
        for idx in range(len(context_end_pos)):
            token_log_probs = []
            # For each position in the continuation, get the log prob of the next token
            for i in range(context_end_pos[idx], continuation_end_pos[idx]):
                next_token_id = combined[idx][i+1]
                next_token_log_prob = log_probs[idx, i, next_token_id].item()
                token_log_probs.append(next_token_log_prob)

            # Average log-prob over the tokens in the candidate, then exponentiate
            mean_token_log_prob = statistics.mean(token_log_probs)
            mean_token_prob = math.exp(mean_token_log_prob)

            mean_probs_for_this_batch.append(mean_token_prob)

        # Collect the probabilities for this "candidates" for all context
        all_mean_probs.append(mean_probs_for_this_batch)

    return all_mean_probs



class MyAttentionWrapper(torch.nn.Module):
    def __init__(self, orig_attn, attn_override=None, attn_override_mask=None, override_seq_len = None, layer_idx=None):
        """
        Args:
            original_attention: GPT-2 attention layer to override.
            corrupted_weights: Tensor of corrupted attention weights [batch_size, num_heads, seq_len, seq_len] (optional).
            intervention_indices: Indices to intervene on (optional). If None, replace all weights.
        """
        super().__init__()
        self.original_attention = orig_attn
        self.attn_override = attn_override
        self.attn_override_mask = attn_override_mask
        # self.override_seq_len = self.attn_override_mask.shape[-1]
        self.override_seq_len = override_seq_len

        self.layer_idx = layer_idx

        # print("self.attn_override_mask.shape", self.attn_override_mask.shape)
        # print("self.attn_override", self.attn_override.shape)

    def forward(self,
                hidden_states,
                position_embeddings,
                attention_mask,
                past_key_value= None,
                cache_position = None,
                **kwargs
    ):
        # 1) reconstruct shape info
        # batch_seq, seq_len, _ = hidden_states.size()       # (batch, seq_len)
        # head_dim  = self.original_attention.head_dim
        # hidden_shape = (*batch_seq, -1, head_dim)  # (batch, seq_len, num_kv_heads, head_dim)

        # # 2) recompute key_states exactly like Qwen2Attention does:
        # k = self.original_attention.k_proj(hidden_states)       # → (batch, seq_len, kv_heads * head_dim)
        # k = k.view(hidden_shape)                                   # → (batch, seq_len, kv_heads, head_dim)
        # key_states = k.transpose(1, 2)                           # → (batch, kv_heads, seq_len, head_dim)




        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.original_attention.head_dim)


        # 2) recompute raw value states
        #    [batch, seq_len, num_kv_heads*head_dim]
        value_layer  = self.original_attention.v_proj(hidden_states).view(hidden_shape).transpose(1,2)

        num_q_heads  = self.original_attention.config.num_attention_heads   # 12
        num_kv_heads = value_layer.size(1)                                  # 2
        group_size   = num_q_heads // num_kv_heads                         # 6

        # tile each KV head group across the query heads
        value_layer = value_layer.repeat_interleave(group_size, dim=1)
        # now: [batch, 12, seq_len, head_dim]

        #    → [batch, seq_len, num_kv_heads, head_dim]
        # v = v.view(hidden_shape)
        # #    → [batch, num_kv_heads, seq_len, head_dim]
        # value_states = v.transpose(1, 2)

        # # 3) if you have caching turned on, Qwen2Attention does:
        # cos, sin = position_embeddings
        # if past_key_value is not None:
        #     key_states, value_states = past_key_value.update(
        #         key_states=None,             # we only need value here
        #         value_states=value_states,
        #         layer_idx=self.layer_idx,
        #         cache_kwargs={"sin": sin, "cos": cos, "cache_position": cache_position},
        #     )

        # now `value_states` is exactly what Qwen would have used internally

        kwargs.pop("output_attentions", None)
        kwargs.pop("use_cache", None)
        # 4) call the real attention to get its output & weights
        attn_output, original_weights = self.original_attention(
            hidden_states,
            # layer_past=layer_past,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=True,
            # head_mask=head_mask,
            use_cache=False,
            **kwargs  # <-- Just in case there are other args passed
        )



        # self.original_attention(
        #     hidden_states,
        #     position_embeddings=position_embeddings,
        #     attention_mask=attention_mask,
        #     cache_position=cache_position,
        #     **kwargs,
        # )

        # print("attn_output")
        # print(attn_output)

        # print("original_weights")
        # print(original_weights)

        if self.attn_override is not None and self.attn_override_mask is not None:
            batch_size, _, seq_len, _ = value_layer.size()
            num_heads = self.original_attention.config.num_attention_heads
            head_dim = self.original_attention.head_dim
            
            modified_weights = original_weights.clone()


            # print("layer ", self.layer_idx)
            # print("original_weights.shape")
            # print(original_weights.shape)
            # print("val_l:",   value_layer.shape,      value_layer.dtype)


            for i in range(modified_weights.shape[0]):
                modified_weights[i, :, :self.override_seq_len[i], :self.override_seq_len[i]] = (
                        self.attn_override_mask[i, :, :self.override_seq_len[i], :self.override_seq_len[i]] * self.attn_override[i, :, :self.override_seq_len[i], :self.override_seq_len[i]]
                        + (1 - self.attn_override_mask[i, :, :self.override_seq_len[i], :self.override_seq_len[i]]) * original_weights[i, :, :self.override_seq_len[i], :self.override_seq_len[i]]
                )

            # print("mod_w:", modified_weights.shape, modified_weights.dtype)


            # Recompute the attention output using modified weights
            attn_output = torch.matmul(modified_weights, value_layer)  # [batch_size, num_heads, seq_len, head_dim]
            attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)  # Reshape to original shape


            # Final projection
            attn_output = self.original_attention.o_proj(attn_output)

            # Return outputs with the modified attention
            # print("END")
            return attn_output, modified_weights

        else:
            return outputs

# class MyAttentionWrapper(torch.nn.Module):
#     def __init__(self, orig_attn, attn_override=None, attn_override_mask=None):
#         super().__init__()
#         self.orig_attn = orig_attn
#         self.override = attn_override
#         self.override_mask = attn_override_mask
#         # get the correct rotary dim straight from the config
#         # self.rotary_dim = orig_attn.config.rotary_embedding_dim

#     def forward(self, hidden_states, layer_past=None,
#                 attention_mask=None, head_mask=None,
#                 use_cache=None, output_attentions=None,
#                 position_ids=None,  # <-- Add this
#                 **kwargs
#                 ):
#         # Let the original attention build cos/sin with its own rotary_dim
#         return self.orig_attn(
#             hidden_states,
#             layer_past=layer_past,
#             attention_mask=attention_mask,
#             head_mask=head_mask,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             position_ids=position_ids,  # <-- Forward it
#             # pass through your override tensors if needed
#             attn_override=self.override,
#             attn_override_mask= self.override_mask,
#             **kwargs  # <-- Just in case there are other args passed
#         )


# class CounterfactualAttentionGPT2(torch.nn.Module):
#     def __init__(self, original_attention, attn_override=None, attn_override_mask=None, override_seq_len = None):
#         """
#         Args:
#             original_attention: GPT-2 attention layer to override.
#             corrupted_weights: Tensor of corrupted attention weights [batch_size, num_heads, seq_len, seq_len] (optional).
#             intervention_indices: Indices to intervene on (optional). If None, replace all weights.
#         """
#         super().__init__()
#         self.original_attention = original_attention
#         self.attn_override = attn_override
#         self.attn_override_mask = attn_override_mask
#         # self.override_seq_len = self.attn_override_mask.shape[-1]
#         self.override_seq_len = override_seq_len

#         # print("self.attn_override_mask.shape", self.attn_override_mask.shape)
#         # print("self.attn_override", self.attn_override.shape)

#     def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, use_cache=None, output_attentions=True):
#         # Compute the original attention
#         # print("Gathering debuging info")
#         # print("hidden_states.shape",hidden_states.shape)
#         # print("layer_past",layer_past)
#         # print("attention_mask",attention_mask)
#         # print("head_mask",head_mask)
#         # print("use_cache",use_cache)
#         # print("output_attentions",output_attentions)
#         # print("End gathering info")
#         outputs = self.original_attention(
#             hidden_states,
#             layer_past=layer_past,
#             attention_mask=attention_mask,
#             head_mask=head_mask,
#             use_cache=True,
#             output_attentions=True
#         )
#         if self.attn_override is not None and self.attn_override_mask is not None:
#             # Extract original attention weights (tensor)
#             # print(type(outputs))
#             # print(outputs[0].shape)
#             # # print(len(outputs[1]))
#             # print(outputs[2].shape)
#             original_weights = outputs[2]  # Outputs[1] contains the attention weights tensor
#             # Compute the new attention output using modified weights
#             value_layer = outputs[1][1]  # Value tensor [batch_size, num_heads, seq_len, num_heads * head_dim]
#             # print("value_layer.size")
#             # print(value_layer.size())
#             batch_size, _, seq_len, _ = value_layer.size()
#             num_heads = self.original_attention.num_heads
#             head_dim = self.original_attention.head_dim
            
#             modified_weights = original_weights.clone()


#             for i in range(modified_weights.shape[0]):
#                 modified_weights[i, :, :self.override_seq_len[i], :self.override_seq_len[i]] = (
#                         self.attn_override_mask[i, :, :self.override_seq_len[i], :self.override_seq_len[i]] * self.attn_override[i, :, :self.override_seq_len[i], :self.override_seq_len[i]]
#                         + (1 - self.attn_override_mask[i, :, :self.override_seq_len[i], :self.override_seq_len[i]]) * original_weights[i, :, :self.override_seq_len[i], :self.override_seq_len[i]]
#                 )

#             # Recompute the attention output using modified weights
#             attn_output = torch.matmul(modified_weights, value_layer)  # [batch_size, num_heads, seq_len, head_dim]
#             attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)  # Reshape to original shape


#             # Final projection
#             attn_output = self.original_attention.c_proj(attn_output)

#             # Return outputs with the modified attention
#             return (attn_output, modified_weights) + outputs[2:]

#         else:
#             return outputs


def evaluate(model, z, batch_texts, batch_candidates, tokenizer, original_attentions, device= "cpu"):
    eval_batch_size = 64
    odds_all = []
    num_layers = len(model.model.layers)

    for bb in range(0, len(batch_texts), eval_batch_size):

        batch_chunk = batch_texts[bb:bb + eval_batch_size ]  # Take a batch of 10 texts
        candidate_chuck = batch_candidates[bb:bb + eval_batch_size]

        # Tokenize the current batch and move it to the device
        batch_inputs = tokenizer(batch_chunk, return_tensors="pt", padding=True, truncation=True)
        batch_lengths = batch_inputs["attention_mask"].sum(dim=1)
        batch_lengths = batch_lengths[::2]
        # batch_inputs = tokenizer(batch_chunk, return_tensors="pt", padding=False, truncation=True, add_special_tokens=False)
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

        # 3) Reassign the original attentions when you're done.
        for i, layer in enumerate(model.model.layers):
            layer.self_attn = original_attentions[i]

        # Run the batch through the model
        with torch.no_grad():



            # print("*************************")
            # print("in batch_outputs")
            # print("*************************")

            batch_outputs = model(**batch_inputs, output_attentions=True)
            attention_weights = batch_outputs.attentions  # Extract original attention weights

            original_attention_weights = tuple(w[::2, :, :] for w in attention_weights)
            conterfactual_attention_weights = tuple(w[1::2, :, :] for w in attention_weights)

            # print("*************************")
            # print("in result base")
            # print("*************************")


            result_base = get_probabilities_for_examples_multitoken_batch(batch_inputs["input_ids"][::2].tolist(), batch_candidates_idx,model,tokenizer,device)

            first, second = result_base

            # Elementwise division: (first_list[i] / second_list[i])
            odds_base = [s / f for f, s in zip(first, second)]

            attention_override_mask = [ z[layer_].view(1, -1, 1, 1).expand(conterfactual_attention_weights[layer_].shape) for layer_ in range(num_layers) ]

            # Replace GPT-2 attention layers with counterfactual attention layers
            for i, layer in enumerate(model.model.layers):
                model.model.layers[i].self_attn = MyAttentionWrapper(
                    orig_attn=original_attentions[i],
                    attn_override=conterfactual_attention_weights[i],
                    attn_override_mask=attention_override_mask[i],
                    override_seq_len = batch_lengths,
                    layer_idx = i,
                )
                # model.transformer.h[i].attn = CounterfactualAttentionGPT2(
                #     original_attention=original_attentions[i],
                #     attn_override=conterfactual_attention_weights[i],
                #     attn_override_mask=attention_override_mask[i],
                #     override_seq_len = batch_lengths
                # )

            # print("*************************")
            # print("result base")
            # print("*************************")

            results = get_probabilities_for_examples_multitoken_batch(batch_inputs["input_ids"][::2].tolist(), batch_candidates_idx,model,tokenizer, device)

            first, second = results

            odds_ratio = [s / f for f, s in zip(first, second)]

            effect_head = [ (o_r - o_b) / o_b for (o_r,o_b) in zip(odds_ratio, odds_base)]

            odds_all += effect_head

    odds_average = np.mean(odds_all)

    return odds_all, odds_average

