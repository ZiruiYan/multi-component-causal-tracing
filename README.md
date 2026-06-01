# Multi-Component Causal Tracing

This repository contains the runnable code for the proposed PGB-CT experiments from the paper. It keeps the original training-script style while removing plotting-only code, old machine-specific paths, proxy settings, baseline-only files, and unused cleanup artifacts.

There are no YAML configs. Experiments are configured through command-line arguments in the training scripts.

## Contents

- `train_attention_winogender.py`, `train_attention_winobias.py`: GPT-2 attention-head PGB-CT experiments.
- `train_attention_winogender_qwen.py`, `train_attention_winobias_qwen.py`: Qwen attention experiments.
- `train_attention_winogender_llama.py`, `train_attention_winobias_llama.py`: Llama attention experiments.
- `train_mlp.py`, `train_mlp_qwen.py`, `train_mlp_llama.py`: MLP PGB-CT experiments on the Professions prompts.
- `train_factual.py`: MLP PGB-CT experiment on CounterFact factual edits.
- `binding/variable_binding.py`: notebook-style Variable Binding Desiderata experiment.
- `utils/`: original utility/model-intervention code used by the training scripts.
- `datasets/`: loaders for Winogender, Winobias, Professions, and CounterFact.
- `data/`: datasets used by the included experiments.
- `results/`: empty output folder; generated checkpoints and logs are ignored by git.

## Setup

Use Python 3.9 or newer.

```bash
pip install -r requirements.txt
```

`transformers>=4.51,<4.52` is intentional: Qwen3 loading needs the 4.51 API, while staying on the 4.51 line avoids the later GPT-2 attention/cache API changes observed in newer Transformers releases.

The Variable Binding Desiderata code uses a separate environment because it was checked with a different Transformers line.

```bash
conda create -n vbd python=3.9
conda activate vbd
pip install -r binding/requirements-vbd.txt
```

The VBD requirements pin `transformers==4.56.2`, `tokenizers==0.22.1`, and `torch==2.8.0`, matching the working VBD environment. Do not replace this with the main `requirements.txt` environment unless you retest `binding/variable_binding.py`: older Transformers versions may not load newer Llama-family models cleanly, while later major/minor releases can change model internals used by activation patching.

## Running

Run commands from the repository root.

```bash
python train_attention_winogender.py --model gpt2 --device cuda
python train_attention_winobias.py --split dev --model gpt2 --device cuda
```

```bash
python train_attention_winogender_qwen.py --device cuda
python train_attention_winobias_qwen.py --split dev --device cuda
```

```bash
python train_attention_winogender_llama.py --device cuda
python train_attention_winobias_llama.py --split dev --device cuda
```

```bash
python train_mlp.py --model gpt2 --device cuda
python train_mlp_qwen.py --device cuda
python train_mlp_llama.py --device cuda
python train_factual.py --model distilgpt2 --device cuda
```

The Qwen scripts default to `Qwen/Qwen3-1.7B-Base`, and the Llama scripts default to `unsloth/Llama-3.2-1B`. Use `--model_name` to choose a different Hugging Face model ID or a local model path. Use `--cache_dir` only if you want to point Transformers to a specific local cache.

For VBD:

```bash
VBD_MODEL_PATH=/path/to/llama-model python -m binding.variable_binding
```

Run VBD from the repository root. `VBD_MODEL_PATH` should point to a local or Hugging Face Llama-style model whose modules match the paths used in `binding/variable_binding.py`, such as `model.layers.{i}.self_attn.o_proj` and `model.layers.{i}.mlp`. The script uses CUDA when available and loads the model in half precision, so running it on CPU is expected to be slow.

## Outputs

Each training script writes under `results/`, including:

- `output/args.txt`
- `output/log.csv`
- `output/evaluate.csv`
- `output/losses.npy`
- `output/sparsities.npy`
- `ckpt/z_step_*.pt`

Plot generation was removed from the final code folder.
