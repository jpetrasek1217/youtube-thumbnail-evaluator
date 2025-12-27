import re
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read CSV safely
df = pd.read_csv(
    "video_data.csv",
    encoding="utf-8",
    encoding_errors="replace"
)

# -------------------------
# Helper: duration parsing
# -------------------------
def duration_to_seconds(s):
    s = str(s)

    h = re.search(r'(\d+)H', s)
    m = re.search(r'(\d+)M', s)
    sec = re.search(r'(\d+)S', s)

    total = 0
    if h:
        total += int(h.group(1)) * 60 * 24
    if m:
        total += int(m.group(1)) * 8
    if sec:
        total += int(sec.group(1))
        
    if total == 0:
        total = 1

    return total


# -------------------------
# Data extraction
# -------------------------

# Column indices preserved from your original code:
# 2 = title
# 3 = thumbnail_url
# 5 = language
# 6 = duration
# 7 = made_for_kids
# 8 = views

title_lengths = df.iloc[:, 2].astype(str).apply(len)
languages = df.iloc[:, 5].astype(str).str.partition("-")[0]
language_counts = languages.value_counts()
length_in_seconds = df.iloc[:, 6].apply(duration_to_seconds)
views = pd.to_numeric(df.iloc[:, 8], errors="coerce").fillna(1).astype(int).clip(lower=1)
thumbnail_urls = df.iloc[:, 3].astype(str)

# -------------------------
# Terminal output
# -------------------------

print("\n========== DATASET SUMMARY ==========")
print(f"Total videos: {len(df)}")

print("\n--- Title Lengths ---")
print(f"Average title length: {title_lengths.mean():.2f}")
print(f"Min title length: {title_lengths.min()}")
print(f"Max title length: {title_lengths.max()}")

print("\n--- Thumbnails ---")
live_thumbnail_urls = thumbnail_urls.str.endswith("/default_live.jpg").sum()
print(f"Number of default live thumbnails: {live_thumbnail_urls} out of {len(df.iloc[:, 2].astype(str))}")

print("\n--- Video Lengths (seconds) ---")
print(f"Average length: {length_in_seconds.mean():.2f}")
print(f"Average length: {length_in_seconds.mean():.2f}")
print(f"Median length: {length_in_seconds.median()}")
print(f"Standard deviation of length: {length_in_seconds.std():.2f}")
print(f"Geometric mean of length: {statistics.geometric_mean(length_in_seconds):.2f}")
print(f"Min length: {length_in_seconds.min()}")
print(f"Max length: {length_in_seconds.max()}")

print("\n--- Top 10 Languages ---")
print(language_counts.head(10).to_string())

print("\n--- Views ---")
print(f"Average views: {views.mean():.2f}")
print(f"Median views: {views.median()}")
print(f"Standard deviation of views: {views.std():.2f}")
print(f"Geometric mean of views: {statistics.geometric_mean(views):.2f}")
print(f"Min views: {views.min()}")
print(f"Max views: {views.max()}")


# -------------------------
# Plots
# -------------------------

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Video Dataset Overview", fontsize=16)

# -------------------------
# 1. Title length histogram
# -------------------------
axes[0, 0].hist(title_lengths, bins=30)
axes[0, 0].set_xlabel("Title Length (characters)")
axes[0, 0].set_ylabel("Number of Videos")
axes[0, 0].set_title("Title Length Distribution")
axes[0, 0].grid(True, alpha=0.5)

# -------------------------
# 2. Language bar chart
# -------------------------
top_n = 50
top_languages = language_counts.head(top_n)
axes[0, 1].bar(top_languages.index, top_languages.values)
axes[0, 1].set_xlabel("Language")
axes[0, 1].set_ylabel("Number of Videos")
axes[0, 1].set_title(f"Top {top_n} Languages")
axes[0, 1].tick_params(axis="x", rotation=45)
axes[0, 1].grid(True, alpha=0.5)

# -------------------------
# 3. Video length histogram (log x-axis)
# -------------------------
log_bins_length = np.logspace(
    np.log10(length_in_seconds.min()),
    np.log10(length_in_seconds.max()),
    num=50
)

axes[1, 0].hist(length_in_seconds, bins=log_bins_length)
axes[1, 0].set_xscale("log")
axes[1, 0].set_xlabel("Video Length (seconds) [log scale]")
axes[1, 0].set_ylabel("Number of Videos")
axes[1, 0].set_title("Video Length Distribution")
axes[1, 0].grid(True, which="both", alpha=0.5)

# -------------------------
# 4. Views histogram (log x-axis)
# -------------------------
views_nonzero = views[views > 0]
log_bins_views = np.logspace(
    np.log10(views_nonzero.min()),
    np.log10(views_nonzero.max()),
    num=50
)

axes[1, 1].hist(views_nonzero, bins=log_bins_views)
axes[1, 1].set_xscale("log")
axes[1, 1].set_xlabel("Views [log scale]")
axes[1, 1].set_ylabel("Number of Videos")
axes[1, 1].set_title("Views Distribution")
axes[1, 1].grid(True, which="both", alpha=0.5)

# Layout fix
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()