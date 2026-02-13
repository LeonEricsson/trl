# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TRL (Transformer Reinforcement Learning) is a library for post-training foundation models using techniques like SFT, GRPO, DPO, RLOO, KTO, and reward modeling. Built on top of the Hugging Face Transformers ecosystem (extends `transformers.Trainer`).

## Code Guidelines

- Avoid try/except blocks unless it's really necessary.  It's fine that a program fails if something goes wrong as this helps us to catch non-obvious bugs and unforeseen side-effects earlier. You can add try catch on code that explicitly aims to be fault tolerant like adding retry mechanisms or explicit and intentional robustness. 

- Do not add unnecessary comments. Especially do not try to explain code change that reflect your work process, do not refer to old code. "The code used to do that but now we are doing this" is not a pattern we want. Instead prefer to use targeted comments sparingly to explain ambiguous code.

## Zen of Python
remember the zen of python when writing code.

```
Beautiful is better than ugly.
Explicit is better than implicit.
Simple is better than complex.
Complex is better than complicated.
Flat is better than nested.
Sparse is better than dense.
Readability counts.
Special cases aren't special enough to break the rules.
Although practicality beats purity.
Errors should never pass silently.
Unless explicitly silenced.
In the face of ambiguity, refuse the temptation to guess.
There should be one-- and preferably only one --obvious way to do it.
Although that way may not be obvious at first unless you're Dutch.
Now is better than never.
Although never is often better than *right* now.
If the implementation is hard to explain, it's a bad idea.
If the implementation is easy to explain, it may be a good idea.
Namespaces are one honking great idea -- let's do more of those!
```

## Common Commands
All code is run with `uv run commands`

```bash
# Install for development
uv pip install -e ".[dev]"

# Run tests
uv run pytest tests/test_sft_trainer.py -v

# Run a single test
uv run pytest tests/test_sft_trainer.py::TestClassName::test_method -v

# Linting and formatting (ruff-based, line length 119; pre-commit hooks run automatically on commit)
uv run ruff check .
uv run ruff format .
```

## Architecture

### Trainer Pattern
Every trainer follows the same structure:
- **Config class** (e.g., `SFTConfig`) extends `transformers.TrainingArguments`
- **Trainer class** (e.g., `SFTTrainer`) extends `BaseTrainer` which extends `transformers.Trainer`
- Located in `trl/trainer/` with separate `*_config.py` and `*_trainer.py` files

Stable trainers: `SFTTrainer`, `DPOTrainer`, `GRPOTrainer`, `RLOOTrainer`, `RewardTrainer`, `KTOTrainer`

### Key Modules
- `trl/trainer/` — All trainer implementations and their configs
- `trl/data_utils.py` — Dataset processing, chat template application, packing strategies, multimodal support
- `trl/scripts/` — CLI training scripts (used by `trl` CLI command via `trl/cli.py`)
- `trl/rewards/` — Reward functions (accuracy, format, etc.)
- `trl/generation/` — vLLM integration for generation
- `trl/experimental/` — Research-stage trainers (18+ methods: PPO, CPO, ORPO, etc.) — excluded from default tests
- `trl/trainer/callbacks.py` — Training callbacks (logging, progress, model sync)
- `trl/trainer/utils.py` — Shared trainer utilities

### Import System
Uses `_LazyModule` for lazy loading in `__init__.py`. Unused imports in `__init__.py` are intentional re-exports (ruff F401 is suppressed there).

### Distributed Training
Predefined accelerate configs in `trl/accelerate_configs/` (single_gpu, multi_gpu, DeepSpeed ZeRO 1/2/3, FSDP 1/2).

## Testing

- Test utilities in `tests/testing_utils.py` provide skip decorators: `require_peft`, `require_torch_accelerator`, `require_vllm`, `require_bitsandbytes`, etc.
- `TrlTestCase` base class provides `tmp_dir` fixture
- Markers: `@pytest.mark.slow`, `@pytest.mark.low_priority`
- Distributed tests live in `tests/distributed/`
- CI runs on `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` with `uv` for package management

## Style

- Ruff for linting and formatting, line length 119, target Python 3.10
- `print()` allowed in `examples/` and `scripts/` but not in library code
- Copyright headers required (enforced by `scripts/add_copyrights.py`)
- Doc style enforced by `doc-builder style` with max line length 119
