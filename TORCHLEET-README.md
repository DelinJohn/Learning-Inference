# TorchLeet practice setup

This folder also holds a local clone of [TorchLeet](https://github.com/Exorust/TorchLeet) (in `Torchleet/`),
prepared for practising PyTorch and LLM building blocks **without looking at solutions**.
The notes index lives in `README.md`; this file only covers the TorchLeet work.

## Credit and license

**TorchLeet was created by Chandrahas Aroori ([@Exorust](https://github.com/Exorust)) and its contributors.**
All problems, notebooks, the grader and the problem metadata belong to them:
<https://github.com/Exorust/TorchLeet>. It is released under the MIT License
(Copyright (c) 2024-2026 Chandrahas Aroori and TorchLeet contributors).

If you use it in teaching or research, the repo asks you to cite it (see its `CITATION.cff`).
The edits described below are small local fixes to that project. They are not part of the upstream repo.

If you ever copy TorchLeet files into this repo, or publish the fixed notebooks, keep the upstream `LICENSE` file
with them. The MIT License requires the copyright and permission notice to stay with copies of the software.
The checklist in `torchleet-checklist.md` only links to the problems and lists their titles.

Files for this work:

| File | Purpose |
|---|---|
| `Torchleet/` | the cloned repo (shallow clone, `--depth 1`), with local edits listed below |
| `torchleet-checklist.md` | your progress checklist: 68 problems in 7 phases, with time budgets |
| `~/.torchleet-solutions/` | the 66 solution notebooks, moved out of the repo (outside this folder) |

## Git

This folder is a git repo. `Torchleet/` is **not tracked** (see `.gitignore`): it is a third-party clone with its own `.git`,
and nesting one repo in another causes problems. What is tracked: the notes, `index.json`, the checklist and the READMEs.

- On a new machine, fetch the clone again:
  ```bash
  git clone --depth 1 https://github.com/Exorust/Torchleet.git
  ```
  A fresh clone **does not contain the local fixes** listed below, and it brings the solution notebooks back.
- The links in `torchleet-checklist.md` point into `Torchleet/`, so they only work where that folder exists.

## How to work

- Environment: the existing `timeseriesllm` micromamba env (Python 3.11, torch 2.5.1+cu124, Triton 3.1, JupyterLab).
  The GPU is an RTX 3050 laptop (4 GB). Use the `timeseriesllm` Jupyter kernel.
- Start Jupyter from the repo folder:
  ```bash
  cd Torchleet && /home/delin/micromamba/envs/timeseriesllm/bin/jupyter lab
  ```
- Check a problem with the grader. Example for the sinusoidal problem:
  ```python
  from torchleet import check
  check("sinusoidal-positional-embedding", SinusoidalPositionalEmbedding)
  ```
  The first cell of each notebook shows the problem id. The grader is `python/torchleet/` in the repo.
- Not installed yet: `torchvision` (needed by the CNN, augmentation, AlexNet, ViT and Swin problems).
  Install a version that matches torch 2.5.1 (0.20.1) and check that torch is not changed.

## What changed

### Environment
- Installed the grader into `timeseriesllm`: `pip install --no-deps -e Torchleet/python`. Torch was not touched.
- I started a separate `.venv` and deleted it again (partial install, 16 MB). It no longer exists.

### Solutions hidden
- Moved all 66 solution notebooks (every `*_SOLN.ipynb`, the solution files named in the manifests, and the kv-cache
  `.ipynb_checkpoints` copy) to `~/.torchleet-solutions/`, with the same folder layout as the repo.
- Nothing was deleted. The original files are also in the clone's git history.
- To bring them back:
  ```bash
  cp -rn ~/.torchleet-solutions/* Torchleet/
  ```
- Do **not** run `git checkout -- .` unless you want every local edit below reverted.

### Notebook fixes (question notebooks)
| Notebook | Problem found | Change |
|---|---|---|
| `llm/Byte-Pair-Encoder/BPE-q3.ipynb` | description was copied from the Transformer problem; title typo "Pain" | rewrote the description from the stubs and tests |
| `llm/Multi-Head-Attention/...-Question.ipynb` | heading said "Implement Attention from Scratch" | fixed heading |
| `llm/Grouped-Query-Attention/...-Question.ipynb` | same wrong heading | fixed heading |
| `llm/Sinusoidal-Positional-Embedding/...-Question.ipynb` | wrong heading; test cell swapped the constructor arguments and passed a `(1, 50)` tensor | fixed heading; test cell now uses `(max_seq_len, d_model)` and a `(1, seq_len, d_model)` input |
| `llm/RMS-Norm/rms-norm.ipynb` | the notebook already contained the finished `forward`; no description | blanked `forward`, added a description, added checks (RMS about 1, scale applied, extra leading dims) |
| `llm/SmolLM/smollm-q12-Question.ipynb` | no description | added one from the stubs and grader hints |
| `llm/Flash-Attention/flash-attention.ipynb` | contained the working Triton kernel and the working PyTorch function | removed the Triton solution cells, blanked `flash_attention_pytorch` (docstring kept), added a "Part 2" heading |
| `v3/llm-inference/` kv-cache, speculative-decoding, continuous-batching, inference-engine | titles started "Question 12:", "Question 18:" and so on | removed the prefix |

### Repo metadata
- `llm/Flash-Attention/problem.toml` and `llm/Create-Embeddings-out-of-an-LLM/problem.toml`: added description, category and
  `tracks = ["llm-path"]` (they were in no track). Fixed "Llm" to "LLM" in the second title.
- Regenerated with the repo's own script (`python scripts/generate.py json | readme | studyplan`):
  `problems.json`, `space/problems.json`, `PROBLEMS.md`, `STUDYPLAN.md`. The LLM path is now 23 problems.

### Git state of `Torchleet/`
- 17 modified files (the ones above), 66 deleted files (the moved solutions), and 4 untracked `__pycache__` files
  created by running the grader. The `__pycache__` files are harmless.

## Known caveats
- `python scripts/generate.py check` now reports 65 errors. Every one is a solution file I moved on purpose.
  There are no other failures.
- `PROBLEMS.md` and `problems.json` still link to the moved solution files.
- RMS Norm, Flash Attention and Create Embeddings never had separate solution files.
- The LLM-path numbering skips positions 4, 11 and 13. I could not tell what belonged there, so I left it.
- The repo's text mismatches were found by comparing titles and headings with the code, not by reading every description.
  More may exist.
- The notebooks were checked for valid JSON and for stubs that fail until implemented. I ran the grader only on the
  sinusoidal problem, which passed all four checks on the version you wrote.

## Notes about this README layout
`README.md` (the notes index) lists `llm-inference-prefill-decode-notes.md` and `xtract-@waterloo_intern.md` as
preserved originals in its file layout, but those two files are not in this folder now. I did not edit `README.md`.
