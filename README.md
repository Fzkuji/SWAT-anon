<div align="center">

# SWAT: Sliding Window Attention Training for Efficient Large Language Models

</div>

This repository contains the implementation of **SWAT (Sliding Window Attention Training)**, a training method that makes sliding window attention work for long-context language modeling.

Existing pretrained LLMs collapse under sliding window attention (SWA) inference: they suffer from the attention sink and lose information once tokens are evicted from the window. SWAT addresses these problems at training time rather than patching them at inference. It replaces the softmax with a sigmoid formulation to enlarge each token's information capacity, and adds position-dependent biases that rebalance how recent and distant context share that capacity. With only a 128-token window, a SWAT model matches a full-context Transformer when inferring on sequences of up to 8,192 tokens, while keeping linear complexity and a constant KV cache.

## Where the SWAT code is

| Component | Path |
| --- | --- |
| SWAT attention layer (sigmoid + window position bias) | [`fla/layers/swattn.py`](fla/layers/swattn.py) |
| SWAT model definition | [`fla/models/swat/modeling_swat.py`](fla/models/swat/modeling_swat.py) |
| SWAT configuration | [`fla/models/swat/configuration_swat.py`](fla/models/swat/configuration_swat.py) |

## Installation

```bash
pip install -e .
# Flash Attention is recommended for the sliding-window kernels:
pip install flash-attn --no-build-isolation
```

## Usage

```python
from fla.models.swat import SWATConfig
from fla.models.swat.modeling_swat import SWATForCausalLM

config = SWATConfig(
    hidden_size=1024,
    num_hidden_layers=24,
    num_heads=16,
    window_size=128,   # sliding window size used at training and inference
)
model = SWATForCausalLM(config)
```

Key configuration options specific to SWAT:

- `window_size`: the sliding window size (e.g. 128). Tokens outside this window are evicted, and the model learns to compress their information into deeper layers.
- The sigmoid activation and the bidirectional position biases (WiPE) are built into the SWAT attention layer; see `fla/layers/swattn.py`.

## Training and Evaluation

Replace the model config with `SWATConfig` to train a SWAT model using the training and evaluation scripts in this repository. We pre-train SWAT at 340M and 760M parameters and compare against recurrent and Transformer baselines on commonsense reasoning, language modeling, and long-context retention. SWAT is competitive on short-context reasoning and clearly stronger on tasks that require retrieving information from far back in the sequence.
