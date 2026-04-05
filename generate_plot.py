import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load TSV
df = pd.read_csv("results.tsv", sep="\t")

# Normalize types
df["mean_us"] = pd.to_numeric(df["mean_us"], errors="coerce")
df["std_us"] = pd.to_numeric(df["std_us"], errors="coerce")
df["tracking_mean_us"] = pd.to_numeric(df["tracking_mean_us"], errors="coerce")
df["status"] = df["status"].str.strip().str.upper()

print(f"Total runs: {len(df)}")
print(f"Columns: {list(df.columns)}\n")

print(df.head(10))


# --------------------------------------------------
# Outcome breakdown
# --------------------------------------------------
counts = df["status"].value_counts()
print("\nRun outcomes:")
print(counts.to_string())

n_keep = counts.get("KEEP", 0)
n_discard = counts.get("DISCARD", 0)
n_decided = n_keep + n_discard

if n_decided > 0:
    print(f"\nKeep rate: {n_keep}/{n_decided} = {n_keep / n_decided:.1%}")


# --------------------------------------------------
# Show kept runs
# --------------------------------------------------
kept = df[df["status"] == "KEEP"].copy()

print(f"\nKEPT runs ({len(kept)} total):\n")
for i, row in kept.iterrows():
    print(
        f"  #{i:3d}  mean={row['mean_us']:.3f}us  "
        f"std={row['std_us']:.3f}  "
        f"tracking={row['tracking_mean_us']:.1f}  "
        f"{row['description']}"
    )


# --------------------------------------------------
# Progress plot
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 8))

valid = df.copy().reset_index(drop=True)

baseline = valid.loc[0, "mean_us"]

# Focus near baseline (interesting region)
window = valid[valid["mean_us"] <= baseline * 1.02]

# Discarded
disc = window[window["status"] == "DISCARD"]
ax.scatter(
    disc.index,
    disc["mean_us"],
    c="#cccccc",
    s=12,
    alpha=0.5,
    label="Discarded",
)

# Kept
kept_v = window[window["status"] == "KEEP"]
ax.scatter(
    kept_v.index,
    kept_v["mean_us"],
    c="#2ecc71",
    s=50,
    edgecolors="black",
    linewidths=0.5,
    label="Kept",
)

# Running best
kept_mask = valid["status"] == "KEEP"
kept_idx = valid.index[kept_mask]
kept_vals = valid.loc[kept_mask, "mean_us"]

running_min = kept_vals.cummin()

ax.step(
    kept_idx,
    running_min,
    where="post",
    color="#27ae60",
    linewidth=2,
    alpha=0.7,
    label="Running best",
)

# Annotate
for idx, val in zip(kept_idx, kept_vals):
    desc = str(valid.loc[idx, "description"]).strip()
    if len(desc) > 50:
        desc = desc[:47] + "..."

    ax.annotate(
        desc,
        (idx, val),
        textcoords="offset points",
        xytext=(6, 6),
        fontsize=8,
        rotation=30,
        alpha=0.9,
    )

ax.set_xlabel("Run #")
ax.set_ylabel("Mean latency (µs, lower is better)")
ax.set_title(f"Autotuning Progress: {len(df)} Runs, {len(kept)} Kept")

ax.legend()
ax.grid(alpha=0.2)

best = kept_vals.min()
margin = (baseline - best) * 0.15
ax.set_ylim(best - margin, baseline + margin)

plt.tight_layout()
plt.savefig("progress.png", dpi=150)
plt.show()

print("\nSaved plot to progress.png")


# --------------------------------------------------
# Summary stats
# --------------------------------------------------
baseline = df.iloc[0]["mean_us"]
best = kept["mean_us"].min()
best_row = kept.loc[kept["mean_us"].idxmin()]

print("\nSummary:")
print(f"Baseline mean:   {baseline:.3f} µs")
print(f"Best mean:       {best:.3f} µs")
print(
    f"Improvement:     {baseline - best:.3f} µs "
    f"({(baseline - best) / baseline * 100:.2f}%)"
)
print(f"Best run:        {best_row['description']}")


# --------------------------------------------------
# Improvement ranking
# --------------------------------------------------
kept = kept.copy()
kept["prev"] = kept["mean_us"].shift(1)
kept["delta"] = kept["prev"] - kept["mean_us"]

hits = kept.iloc[1:].copy()
hits = hits.sort_values("delta", ascending=False)

print("\nTop improvements:")
print(f"{'Rank':>4}  {'Delta(us)':>10}  {'Mean(us)':>10}  Description")
print("-" * 80)

for rank, (_, row) in enumerate(hits.iterrows(), 1):
    print(
        f"{rank:4d}  {row['delta']:+10.3f}  "
        f"{row['mean_us']:10.3f}  {row['description']}"
    )

print(f"\nTotal improvement: {hits['delta'].sum():.3f} µs")


# --------------------------------------------------
# Optional: tracking vs update tradeoff
# --------------------------------------------------
print("\nTracking vs Update tradeoffs (kept only):")
for _, row in kept.iterrows():
    print(
        f"{row['mean_us']:8.2f} us  |  "
        f"{row['tracking_mean_us']:8.2f} us  |  "
        f"{row['description']}"
    )