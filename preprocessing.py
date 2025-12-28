import pandas as pd
import re
# -------------------------
# Load CSV into DataFrame
# -------------------------
df = pd.read_csv("video_data.csv", encoding="utf-8")

# -------------------------
# Column names (adjust if needed)
# -------------------------
VIDEO_ID = "video_id"
CHANNEL_ID = "channel_id"
TITLE = "title"
THUMBNAIL_URL = "thumbnail_url"
THUMBNAIL_URL_DEFAULT = "thumbnail_url_default"
THUMBNAIL_URL_MEDIUM = "thumbnail_url_medium"
TAGS = "tags"
DEFAULT_LANGUAGE = "default_language"
DURATION = "duration"
VIEWS = "views"
SUBSCRIBER_COUNT = "subscriber_count"
CHANNEL_VIEWS = "channel_views"
VIDEO_COUNT = "video_count"

# -------------------------
# Duration parsing function
# -------------------------
def duration_to_seconds(s):
    if pd.isna(s):
        return 1

    s = str(s)

    d = re.search(r'(\d+)D', s)
    h = re.search(r'(\d+)H', s)
    m = re.search(r'(\d+)M', s)
    sec = re.search(r'(\d+)S', s)

    total = 0
    if d:
        total += int(d.group(1)) * 86400
    if h:
        total += int(h.group(1)) * 3600
    if m:
        total += int(m.group(1)) * 60
    if sec:
        total += int(sec.group(1))

    return max(total, 1)

# -------------------------
# Remove live thumbnails
# -------------------------
df = df[~df[THUMBNAIL_URL_DEFAULT].str.contains("_live.jpg", na=False)]
df = df[~df[THUMBNAIL_URL_MEDIUM].str.contains("_live.jpg", na=False)]

# -------------------------
# Clean default language
# -------------------------
df[DEFAULT_LANGUAGE] = (
    df[DEFAULT_LANGUAGE]
    .astype(str)
    .str.split("-")
    .str[0]
)

# -------------------------
# Convert duration
# -------------------------
df[DURATION] = df[DURATION].apply(duration_to_seconds)

# -------------------------
# Clamp numeric columns to >= 1
# -------------------------
numeric_cols = [
    VIEWS,
    SUBSCRIBER_COUNT,
    CHANNEL_VIEWS,
    VIDEO_COUNT
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(1).astype(int)
    df[col] = df[col].clip(lower=1)

# -------------------------
# Save cleaned data
# -------------------------
df.to_csv("video_data_cleaned.csv", index=False, encoding="utf-8")