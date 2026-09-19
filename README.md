# bert-lora-finetuning

![CI](https://github.com/shreyansh0714/bert-lora-finetuning/actions/workflows/ci.yml/badge.svg)

Fine-tuning `bert-base-uncased` for 77-class intent classification ([Banking77](https://huggingface.co/datasets/PolyAI/banking77)),
comparing **full fine-tuning** against **LoRA** (Low-Rank Adaptation) — via HuggingFace's `peft` library
and via a from-scratch implementation, with an empirical rank sweep.

## Why this project

Full fine-tuning updates all ~110M of BERT's parameters per task. That is expensive to train — Adam
keeps two optimizer states per trainable parameter, so memory scales with the number of parameters you
are actually updating — and expensive to deploy, since every task needs its own full checkpoint.

LoRA freezes the pretrained weights and represents the task-specific update as the product of two small
matrices, `ΔW = (alpha/r)·BA`. The claim is that most of the accuracy survives while the trainable
parameter count drops by orders of magnitude. This project measures that claim directly rather than
citing it: same model, same data, same protocol, both methods, real numbers.

## Results

### Stage 1 — full fine-tuning vs. LoRA at r=8

From `notebooks/01_baseline_full_finetune_vs_lora.ipynb`:

| Method | Trainable params | % of total | Accuracy | Macro F1 | Epochs | Train time | Peak GPU mem |
|---|---|---|---|---|---|---|---|
| Full fine-tune | 109,541,453 | 100% | 92.7% | 0.927 | 10 | 790s | 2,155 MB |
| LoRA (r=8) | 354,125 | 0.32% | 85.4% | 0.852 | 20 | 1,222s | 1,306 MB |

![Baseline comparison](results/01_baseline_comparison.png)

**The trade-off.** LoRA retained **92% of full fine-tuning's accuracy** (85.4% against 92.7%, a gap of
7.3 points) while training **0.32% of the parameters** — 309× fewer — and using **39% less peak GPU
memory** (1,306 MB against 2,155 MB).

It was *slower* in wall clock: 1,222s against 790s, or 1.55×. That number is easy to misread. Per epoch
LoRA is about 23% faster (61s against 79s), because there are far fewer gradients to compute and far
fewer optimizer states to update. It simply needed twice as many epochs to get where it got. The saving
is in memory and parameter count, not in time-to-result.

![Convergence](results/01_training_curves.png)

The convergence plot is where the remaining accuracy gap becomes legible. Full fine-tuning is
essentially flat from epoch 5 onward. LoRA is still improving at epoch 20, where its budget ran out —
so **85.4% is a lower bound, not a converged value**, and some unknown fraction of that 7.3-point gap is
training budget rather than a limit of the method.

## Rank sweep: how much LoRA capacity is actually needed?

Notebook 01 measures LoRA at a single rank, `r=8`. The sweep in `notebooks/02_lora_rank_sweep.ipynb`
turns that one point into a curve. It took two runs to get a curve worth reading, and the first run is
part of the result rather than something we discarded quietly.

### Run 1 — four epochs, and why we did not report its ranking

The first sweep gave every rank four epochs, matching the budget notebook 01 originally used.

| Rank | Trainable params | Accuracy |
|---|---|---|
| 4 | 206,669 | 30.8% |
| 8 | 354,125 | 44.8% |
| 16 | 649,037 | 54.5% |
| 32 | 1,238,861 | 61.4% |

![First sweep, four epochs](results/02_rank_sweep_4ep.png)

Read at face value this says "rank matters enormously, keep increasing it." Two features of the data
said we were measuring the wrong variable:

1. **The curve never flattens.** It is still climbing steeply at `r=32`, 31 points below full
   fine-tuning's 92.7%. A capacity ceiling shows up as a curve bending over. This one does not bend.
2. **Validation accuracy was still rising at the final epoch for every rank.** None of the models had
   finished learning with the parameters they already had.

Both symptoms point at the training budget, not at rank. At a short fixed budget the two are
confounded: a larger adapter absorbs a fixed number of steps faster, so higher rank looks better for a
reason that has nothing to do with capacity.

The fix was therefore to remove the confound — train each rank to *its own* ceiling — not to keep
adding rank.

### Run 2 — twenty epochs with early stopping

Same ranks, same learning rate, same target modules. The only change is the budget: 20 epochs with
early stopping on validation accuracy, so each rank stops when it stops improving.

| Rank | Alpha | Trainable params | Accuracy | Macro F1 | Epochs | Train time |
|---|---|---|---|---|---|---|
| 4 | 8 | 206,669 | 82.2% | 0.819 | 20 | 1,227s |
| 8 | 16 | 354,125 | 84.3% | 0.841 | 20 | 1,223s |
| 16 | 32 | 649,037 | 86.7% | 0.867 | 20 | 1,230s |
| 32 | 64 | 1,238,861 | 87.9% | 0.879 | 16 | 983s |

![Rank sweep](results/02_rank_sweep.png)

### What the two runs prove together

Holding rank fixed and changing only the budget moved `r=8` from 44.8% to 84.3% — **39.5 points from
training time alone**. The decisive comparison is this one:

| Configuration | Trainable params | Accuracy |
|---|---|---|
| `r=32`, 4 epochs | 1,238,861 | 61.4% |
| `r=4`, 20 epochs | 206,669 | **82.2%** |

The smallest adapter trained properly beats the largest adapter trained briefly by **20.9 points using
6× fewer parameters**. Rank cannot substitute for training budget, which is why run 1's ranking was a
statement about convergence speed rather than about capacity.

### The finding

With every rank trained toward its ceiling, the diminishing return is visible — and it is in
*efficiency*, not in raw accuracy:

| Step | Accuracy gained | Parameters added | Points per 100k params |
|---|---|---|---|
| `r=4` → `r=8` | +2.08 | 147,456 | 1.41 |
| `r=8` → `r=16` | +2.40 | 294,912 | 0.81 |
| `r=16` → `r=32` | +1.14 | 589,824 | 0.19 |

Every doubling still buys accuracy, but the yield per added parameter falls roughly 7× across the
sweep. At `r=32`, LoRA reaches **94.8% of full fine-tuning's accuracy while training 1.13% of the
parameters**.

**Three caveats before citing that table:**

- **Only `r=32` converged.** It early-stopped at epoch 16 with its best checkpoint at epoch 12. `r=4`,
  `r=8` and `r=16` all hit the 20-epoch ceiling with validation accuracy still rising, so those three
  are lower bounds. The efficiency decline is partly confounded by this: the lower ranks remain
  budget-limited, not only capacity-limited.
- **The run-to-run noise floor is about one point.** Notebook 01 and this sweep both trained `r=8` with
  identical settings for 5,640 steps and produced 85.4% and 84.3% — 1.04 points apart from classifier
  head initialisation alone. The `r=16` → `r=32` gain of 1.14 points sits barely above that, so `r=32`
  leading `r=16` is suggestive, not established.
- **Peak memory does not separate the ranks** — 1,304 MB to 1,325 MB across a 6× parameter range. It is
  dominated by the frozen base model and the activations, not by the adapters.

### Results dashboard

`dashboard.html` renders these numbers from `results/*.json` — it has no build step and nothing in it is
typed by hand, so it updates whenever the notebooks are re-run and the results re-committed. Serve the
repo root over HTTP and open it:

```bash
python -m http.server 8000
# then open http://localhost:8000/dashboard.html
```

Opening the file directly as `file://` will not work — the browser blocks the `fetch` calls and every
panel reads as empty.

## Protocol

The two methods are **not** given a matched epoch budget, and that is deliberate. LoRA optimizes a much
smaller parameter space and needs more steps to reach the same place; holding epochs equal understates
it. An earlier version of this repo did exactly that, gave both arms 4 epochs, and reported LoRA at
46.9% — a configuration artifact, not a finding.

Instead each method trains until it stops improving:

- Banking77 ships train/test only, so a stratified **90/10 split is carved out of train** for validation.
  The test split is untouched until the final evaluation.
- Both arms use **early stopping on validation accuracy** (`load_best_model_at_end`), and report the
  best checkpoint against the held-out test set.
- **Steps to convergence is recorded as a result**, not held constant.
- Learning rates differ by method, as they should: 2e-5 for full fine-tuning, 2e-4 for LoRA, both with
  6% warmup. A fresh 77-way classification head at 2e-4 diverges in the first epoch without it.

## Repo structure

```
bert-lora-finetuning/
├── notebooks/
│   ├── 01_baseline_full_finetune_vs_lora.ipynb   # main comparison, run this first
│   ├── 02_lora_rank_sweep.ipynb                   # r = 4, 8, 16, 32, independent of 01
│   └── 03_lora_from_scratch_demo.ipynb            # no GPU needed, demos src/ directly
├── src/
│   └── lora_from_scratch.py    # minimal LoRA linear layer, no peft dependency
├── tests/
│   └── test_lora_from_scratch.py   # run in CI on every push
├── results/                    # CSV/JSON/PNG written by the notebooks
├── dashboard.html              # reads results/*.json, no build step
├── requirements.txt
└── .github/workflows/ci.yml
```

## How to run

1. **Notebooks 01 and 02** need a GPU — upload to [Colab](https://colab.research.google.com) or
   [Kaggle](https://kaggle.com/code) with a GPU runtime. Each installs its own dependencies in its first
   cell. Budget roughly 35 minutes for `01` and 1.5–2 hours for `02`.
   - `01` and `02` do not depend on each other's execution and can run in parallel on two machines.
     `02` picks up the full-fine-tune reference line automatically from `results/01_baseline_comparison.json`
     if that file is present, and omits the line if it isn't.
   - Download the generated `results/` files from the runtime's output panel and commit them; that is
     what the README tables and the dashboard read.
2. **Notebook 03** and the test suite run anywhere, no GPU:
   ```bash
   pip install -r requirements.txt
   pytest tests/ -v
   ```

## Limitations — read before citing these numbers

- **LoRA did not converge.** It hit the 20-epoch ceiling while validation accuracy was still rising, so
  85.4% is a floor. The honest version of the headline is "at least 92% of full fine-tuning accuracy,"
  not "exactly 92%." Re-running with a larger budget would narrow the gap by an unknown amount.
- **Single seed.** One training run per configuration, not an average. Treat the numbers as directional,
  not statistically validated — a 7.3-point gap from one seed each is suggestive, not conclusive.
- **One dataset, one base model.** Banking77 and `bert-base-uncased` only. Nothing here shows the result
  generalizes to other tasks or model sizes.
- **`target_modules=["query", "value"]` only** — not the FFN or output projections. This follows the
  original LoRA paper's attention-only setting; it is a convention, not a tuned choice.
- **No hyperparameter search** beyond the rank sweep. Learning rate, alpha and dropout are fixed.
- **Memory is measured as peak allocation**, via `torch.cuda.max_memory_allocated`, taken as the max
  across visible devices. An earlier version read device 0 only and reported the two methods backwards.
- This is comparison code, not production code — no error handling, config management, or serving
  beyond the `merge()` demonstration in notebook 03.

## License

MIT — see [LICENSE](LICENSE).