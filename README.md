# Tiny Character-Level LLM with `taeda`

A minimal "dumb LLM" that trains a character-level language model through
`taeda.RAMVRAMModel`, which keeps the master weights in system RAM and streams
each layer to the GPU only while it is computing. It is a small end-to-end demo
that RAM-to-VRAM offloading works for training (forward, backward, optimizer
step) and for text generation.

## What it does

1. **Tokenizes** a small built-in text into single characters.
2. **Builds training pairs**: the previous 16 characters are the input, the next
   character is the target.
3. **Trains** a small `nn.Sequential` network wrapped in `taeda.RAMVRAMModel`
   with Adam and cross-entropy loss.
4. **Generates** text one character at a time by sampling from the softmax.
5. **Prints a memory report** for the wrapped model.

## Requirements

- Python 3.9+
- PyTorch (a CUDA build if you want GPU offloading)
- the `taeda` package, which provides `RAMVRAMModel`

If CUDA is not available, the script falls back to `device="cpu"`. It runs, but
nothing is offloaded.

## Usage

```bash
python RVR-TinyStories.py
```

## Model

| Layer | Shape |
|---|---|
| `Embedding` | `vocab -> 32` |
| `Flatten` | `(B, 16, 32) -> (B, 512)` |
| `Linear` + `ReLU` | `512 -> 1024` |
| `Linear` + `ReLU` | `1024 -> 512` |
| `Linear` | `512 -> vocab` |

About 1.06 million parameters (about 4 MB of master weights in RAM).

## Configuration

All settings are constants near the top of the script.

| Name | Default | Meaning |
|---|---|---|
| `CTX` | `16` | Context window in characters |
| `EMB` | `32` | Embedding size. `CTX * EMB` must equal the first `Linear` input size |
| `STEPS` | `1500` | Number of training steps |
| `BATCH` | `64` | Batch size |
| `lr` | `2e-3` | Adam learning rate |
| `max_vram_percent` | `80` | VRAM budget passed to `RAMVRAMModel` |
| `temperature` | `0.7` | Sampling temperature in `generate()` (lower is more conservative) |

## Example output

```
Step    1: loss 3.0593
Step  100: loss 0.0002
...
Step 1500: loss 0.0000

=== Generated text ===
the robot d ads about cats and dogs. the cat and the dog are good friends. ...

=== RAMVRAMModel ===
Layers: 7
Device: cuda
VRAM budget: 80 %
Parameters: 1,061,557
Master weights in RAM: 4.05 MB
```

The starting loss of about 3.06 is roughly `ln(vocab_size)`, which is what a
model guessing uniformly would score.

## Important: this model memorizes

The corpus is tiny and every 16-character context is unique, so the model
reaches a loss of about 0 by memorizing the text. Generated output is a
remix of the training sentences, not new language. The first few characters can
be garbled when the prompt, padded with spaces, is a context never seen in
training.

To make it generalize:

- Train on a much larger text, for example `corpus = open("book.txt", encoding="utf-8").read()`.
- Hold out about 10% of the data and watch the validation loss. If training loss
  falls while validation loss rises, the model is memorizing.
- Add `nn.Dropout(0.2)` after the `ReLU` layers.
- Increase `CTX` (and adjust `EMB` so that `CTX * EMB` matches the first `Linear`).

## Limitations

- `RAMVRAMModel` currently supports **`nn.Sequential` models only**. Each layer
  must take a single tensor and return a single tensor.
- Training is slower than a plain GPU model, because every layer copies its
  weights and activations between CPU and GPU. This is the cost of fitting models
  larger than VRAM.
- The model has no attention and a fixed context window, so it is a demo of the
  offloading mechanism, not a practical language model.

## Troubleshooting

**`RuntimeError: Expected all tensors to be on the same device`**
Make sure you are using a version of `ramtovram.py` that moves the activations
back to the GPU at the start of every layer (`x = x.to(self.device)` inside the
layer loop).

**Loss stays near `ln(num_classes)`**
The model is not learning. Check that you are not training on random data and
that the optimizer was built from `model.parameters()` of the wrapped model.
