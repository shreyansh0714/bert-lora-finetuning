"""Regenerate the README figures from the committed files in results/.

Run from anywhere: python scripts/make_figures.py
No GPU needed; reads only CSV/JSON already in results/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
FULL, LORA, RUN1, PARAMS = "#2a78d6", "#eb6834", "#9a9a94", "#1baf7a"

run1 = pd.read_csv(RESULTS / "02_rank_sweep_4ep.csv").sort_values("rank")
run2 = pd.read_csv(RESULTS / "02_rank_sweep.csv").sort_values("rank")
baseline = {r["method"]: r for r in json.loads((RESULTS / "01_baseline_comparison.json").read_text())}
full = baseline["full_finetune"]
FULL_ACC = full["accuracy"]
ranks = run2["rank"].tolist()


def full_line(ax):
    ax.axhline(FULL_ACC, color=FULL, linestyle="--", label=f"Full fine-tune ({FULL_ACC:.1%})")


def rank_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks, [str(r) for r in ranks])
    ax.minorticks_off()
    ax.set_xlabel("LoRA rank (r)")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(RESULTS / name, dpi=150)
    plt.close(fig)
    print("wrote", name)


# 01: Stage 1 convergence (same plot as notebook 01's cell).
curves = json.loads((RESULTS / "01_training_curves.json").read_text())
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot([h["epoch"] for h in curves["full_finetune"]], [h["accuracy"] for h in curves["full_finetune"]],
        marker="o", color=FULL, label="Full fine-tune")
ax.plot([h["epoch"] for h in curves["lora_r8"]], [h["accuracy"] for h in curves["lora_r8"]],
        marker="o", color=LORA, label="LoRA r=8")
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation accuracy")
ax.set_title("Convergence: full fine-tuning vs LoRA r=8")
ax.legend()
save(fig, "01_training_curves.png")

# Run 1 alone.
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(run1["rank"], run1["accuracy"], marker="o", color=RUN1, lw=2, label="Run 1 (4 epochs, no warmup)")
full_line(ax)
rank_axis(ax)
ax.set_ylim(0.25, 1.0)
ax.set_ylabel("Test accuracy")
ax.set_title("Run 1: four epochs, still climbing, far below full fine-tuning")
ax.legend(loc="center left")
save(fig, "02_rank_sweep_4ep.png")

# Run 2 accuracy by rank.
fig, ax = plt.subplots(figsize=(7, 4.5))
full_line(ax)
ax.plot(run2["rank"], run2["accuracy"], marker="o", color=LORA, lw=2, label="Run 2 (up to 20 epochs)")
for r, a in zip(run2["rank"], run2["accuracy"]):
    ax.annotate(f"{a:.1%}", (r, a), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9)
rank_axis(ax)
ax.set_ylim(0.80, 0.94)
ax.set_ylabel("Test accuracy")
ax.set_title("Run 2: test accuracy vs rank")
ax.legend(loc="lower right")
save(fig, "02_rank_sweep_accuracy.png")

# Run 2 trainable parameters by rank.
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(run2["rank"], run2["trainable_params"], marker="o", color=PARAMS, lw=2)
for r, p in zip(run2["rank"], run2["trainable_params"]):
    ax.annotate(f"{p:,}", (r, p), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
rank_axis(ax)
ax.set_yscale("log")
ax.set_ylim(top=run2["trainable_params"].max() * 1.5)
ax.set_ylabel("Trainable parameters")
ax.set_title("Run 2: trainable parameters vs rank")
save(fig, "02_rank_sweep_params.png")

# Run 2 per-epoch validation accuracy.
sweep_curves = json.loads((RESULTS / "02_rank_sweep_curves.json").read_text())
fig, ax = plt.subplots(figsize=(8, 4.8))
for r, hist in sorted(sweep_curves.items(), key=lambda kv: int(kv[0])):
    ax.plot([h["epoch"] for h in hist], [h["accuracy"] for h in hist], marker=".", label=f"r={r}")
ax.set_xticks(range(0, 21, 2))
ax.set_ylim(top=1.0)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation accuracy")
ax.set_title("Run 2: convergence by rank (r=32 stopped early at epoch 16)")
ax.legend(loc="lower right")
save(fig, "02_rank_sweep_convergence.png")

# Run 1 vs Run 2 on one axis.
fig, ax = plt.subplots(figsize=(7.2, 4.8))
full_line(ax)
ax.plot(run2["rank"], run2["accuracy"], marker="o", color=LORA, lw=2, label="Run 2: fixed training setup")
ax.plot(run1["rank"], run1["accuracy"], marker="o", color=RUN1, lw=2, ls=":", label="Run 1: 4 epochs, no warmup")
a1, a2 = run1.set_index("rank")["accuracy"][8], run2.set_index("rank")["accuracy"][8]
ax.annotate("", xy=(8, a2), xytext=(8, a1), arrowprops=dict(arrowstyle="<->", color="#444"))
ax.text(8.4, (a1 + a2) / 2, f"+{(a2 - a1) * 100:.1f} pts at the same rank\n(training setup changed)",
        va="center", fontsize=9, color="#444")
rank_axis(ax)
ax.set_ylim(0.25, 1.0)
ax.set_ylabel("Test accuracy")
ax.set_title("How LoRA was trained moved accuracy far more than rank did")
ax.legend(loc="lower right", fontsize=9)
save(fig, "02_run_comparison.png")

# Full fine-tune vs LoRA r=32, Run 1 and Run 2.
best1 = run1.set_index("rank").loc[32]
best2 = run2.set_index("rank").loc[32]
labels = ["Full fine-tune", "LoRA r=32\nRun 1 (4 ep)", f"LoRA r=32\nRun 2 ({int(best2['epochs_run'])} ep)"]
colors = [FULL, RUN1, LORA]
panels = [
    ("Trainable parameters (log)", "trainable_params", lambda v: f"{v / 1e6:.2f}M"),
    ("Test accuracy", "accuracy", lambda v: f"{v:.1%}"),
    ("Training time (s)", "train_time_s", lambda v: f"{v:.0f}s"),
    ("Peak GPU memory (MB)", "peak_mem_mb", lambda v: f"{v:,.0f}"),
]
fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))
for ax, (title, key, fmt) in zip(axes, panels):
    vals = [full[key], best1[key], best2[key]]
    bars = ax.bar(labels, vals, color=colors)
    ax.bar_label(bars, [fmt(v) for v in vals], padding=2, fontsize=9)
    ax.set_title(title)
    if key == "trainable_params":
        ax.set_yscale("log")
        ax.set_ylim(1e6, max(vals) * 4)
    elif key == "accuracy":
        ax.set_ylim(0, 1.0)
    else:
        ax.set_ylim(0, max(vals) * 1.12)
fig.suptitle("Full fine-tuning vs LoRA r=32, before and after fixing the training setup")
save(fig, "02_final_comparison.png")
