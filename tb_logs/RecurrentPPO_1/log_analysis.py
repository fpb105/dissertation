import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── CONFIG ──────────────────────────────────────────────────────────────────
LOG_DIR = "."          # run this script from inside tb_logs/RecurrentPPO_1/
OUTPUT  = "training_metrics.png"
SMOOTH  = 50           # rolling-average window (set to 1 to disable)
# ────────────────────────────────────────────────────────────────────────────

def smooth(values, window):
    if window <= 1 or len(values) < window:
        return values
    out = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out

ea = EventAccumulator(LOG_DIR)
ea.Reload()
tags = ea.Tags()['scalars']
print("Found tags:", tags)

def get(tag):
    events = ea.Scalars(tag)
    steps  = [e.step  for e in events]
    values = [e.value for e in events]
    return steps, values

# Panels: (title, tag, y-label, invert_good_direction)
panels = [
    ("Explained Variance",    "train/explained_variance", "Explained Variance",  False),
    ("Value Loss",            "train/value_loss",          "Loss",                True),
    ("Total Loss",            "train/loss",                "Loss",                True),
    ("Entropy",               "train/entropy_loss",        "Entropy Loss",        False),
    ("Action Std Dev",        "train/std",                 "Std Dev",             True),
    ("Approx KL Divergence",  "train/approx_kl",           "KL Divergence",       True),
    ("Clip Fraction",         "train/clip_fraction",       "Clip Fraction",       False),
    ("FPS",                   "time/fps",                  "Steps / Second",      False),
]

# Filter to tags that actually exist
panels = [(t, tag, yl, inv) for t, tag, yl, inv in panels if tag in tags]

cols = 2
rows = (len(panels) + 1) // cols

fig = plt.figure(figsize=(14, rows * 3.5))
fig.suptitle("RecurrentPPO Training Metrics", fontsize=15, fontweight='bold', y=1.01)
gs  = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.55, wspace=0.35)

COLOUR      = "#4C72B0"
SMOOTH_COL  = "#C44E52"

for idx, (title, tag, ylabel, _) in enumerate(panels):
    ax = fig.add_subplot(gs[idx // cols, idx % cols])
    steps, values = get(tag)

    # subsample to max 2000 points so the plot isn't huge
    if len(steps) > 2000:
        stride = len(steps) // 2000
        steps  = steps[::stride]
        values = values[::stride]

    steps_m = [s / 1_000_000 for s in steps]   # convert to millions

    ax.plot(steps_m, values, color=COLOUR, alpha=0.3, linewidth=0.8, label="Raw")
    sv = smooth(values, SMOOTH)
    ax.plot(steps_m, sv, color=SMOOTH_COL, linewidth=1.6, label=f"Smoothed (w={SMOOTH})")

    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel("Training Steps (millions)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="best")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # annotate start and end smoothed values
    ax.annotate(f"{sv[0]:.3f}", xy=(steps_m[0], sv[0]),
                fontsize=6, color='grey', ha='left')
    ax.annotate(f"{sv[-1]:.3f}", xy=(steps_m[-1], sv[-1]),
                fontsize=6, color='grey', ha='right')

plt.savefig(OUTPUT, dpi=200, bbox_inches='tight')
print(f"\nSaved to {OUTPUT}")
plt.show()