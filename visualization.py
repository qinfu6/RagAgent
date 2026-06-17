import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASELINE_FILE = "result/baseline/benchmark_20260519_145123.json"
AGENT_FILE = "result/agent/benchmark_20260521_005847.json"
MULTI_AGENT_FILE = "result/multi-agent/benchmark_20260520_223505.json"

files = {
    "Baseline": BASELINE_FILE,
    "Agent": AGENT_FILE,
    "Multi-Agent": MULTI_AGENT_FILE,
}

data = {}
for name, filepath in files.items():
    with open(filepath, "r", encoding="utf-8") as f:
        content = json.load(f)
    data[name] = [
        content["accuracy_1"],
        content["accuracy_2"],
        content["accuracy_3"],
        content["accuracy_4"],
    ]

categories = ["Accuracy_1", "Accuracy_2", "Accuracy_3", "Accuracy_4"]
x = np.arange(len(categories))
width = 0.25
multiplier = 0

fig, ax = plt.subplots(figsize=(12, 6))
colors = ["#4C72B0", "#DD8452", "#55A868"]

for (name, values), color in zip(data.items(), colors):
    offset = width * multiplier
    rects = ax.bar(x + offset, values, width, label=name, color=color)
    for rect in rects:
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    multiplier += 1

ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.15)
ax.set_xticks(x + width)
ax.set_xticklabels(categories)
ax.legend(loc="upper right", fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.5)

output_path = Path("result/accuracy_comparison.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Chart saved to {output_path}")

plt.show()
