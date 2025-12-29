import pandas as pd
import re
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import torch
from torchvision import transforms
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------
# Load CSV into DataFrame
# -------------------------
df = pd.read_csv("video_data.csv", encoding="utf-8")

# -------------------------
# Column names (adjust if needed)
# -------------------------
VIDEO_ID = "video_id"
CHANNEL_ID = "channel_id"
THUMBNAIL_URL = "thumbnail_url"
THUMBNAIL_URL_DEFAULT = "thumbnail_url_default"
TAGS = "tags"
DEFAULT_LANGUAGE = "default_language"
UPLOAD_TIME = "upload_time"
CREATED_CHANNEL_AT = "created_channel_at"

THUMBNAIL_URL_MEDIUM = "thumbnail_url_medium"
TITLE = "title"
DURATION = "duration"
IS_SHORT = "is_short"
TITLE_LENGTH = "title_length"
VID_HOUR_OF_DAY = "vid_hour_of_day"
VID_DAY_OF_WEEK = "vid_day_of_week"
SUBSCRIBER_COUNT = "subscriber_count"
CHANNEL_VIEWS = "channel_views"
VIDEO_COUNT = "video_count"
CHANNEL_AGE_YEARS = "channel_age_years"
VIEWS = "views"

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

def insert_publish_time_features(
    df: pd.DataFrame,
    timestamp_col: str,
    start_pos: int = 9
) -> pd.DataFrame:
    """
    Inserts hour_of_day and day_of_week columns derived from an ISO 8601 timestamp.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    timestamp_col : str
        Column containing ISO timestamps (e.g. '2012-10-01T15:27:35Z')
    start_pos : int
        Column index where insertion should begin (default: 7)

    Returns
    -------
    pd.DataFrame
        DataFrame with new columns inserted
    """

    # Parse timestamp (UTC-safe)
    timestamps = pd.to_datetime(df[timestamp_col].astype(str), utc=True, errors="coerce")

    hour_of_day = timestamps.dt.hour
    day_of_week = timestamps.dt.dayofweek  # Monday=0, Sunday=6
    year = df[timestamp_col].astype(str).str.split("-", n=1).str[0].astype(int)

    # Insert in order
    if start_pos == 9:
        df.insert(start_pos, VID_HOUR_OF_DAY, hour_of_day)
        df.insert(start_pos + 1, VID_DAY_OF_WEEK, day_of_week)
    else:
        df.insert(start_pos, CHANNEL_AGE_YEARS, 2026-year)

    return df

# -------------------------
# Remove live thumbnails
# -------------------------
df = df[~df[THUMBNAIL_URL_DEFAULT].str.contains("_live.jpg", na=False)]
df = df[~df[THUMBNAIL_URL_MEDIUM].str.contains("_live.jpg", na=False)]

# -------------------------
# Clean default language then remove any language that is not english
# -------------------------
df[DEFAULT_LANGUAGE] = (
    df[DEFAULT_LANGUAGE]
    .astype(str)
    .str.split("-")
    .str[0]
)
df = df[df[DEFAULT_LANGUAGE].str.contains("en", na=False)]

# -------------------------
# Convert duration and add more cols
# -------------------------
df[DURATION] = df[DURATION].apply(duration_to_seconds)
df = insert_publish_time_features(df, UPLOAD_TIME)
df = insert_publish_time_features(df, CREATED_CHANNEL_AT, 16)
df.insert(7, IS_SHORT, (df[DURATION] < 60).astype(int))
df[TITLE_LENGTH] = df[TITLE].astype(str).apply(len)

# -------------------------
# Clamp numeric columns to >= 1
# -------------------------
numeric_cols = [
    VIEWS,
    SUBSCRIBER_COUNT,
    CHANNEL_VIEWS,
    VIDEO_COUNT
]

# for col in numeric_cols:
#     df[col] = pd.to_numeric(df[col], errors="coerce").fillna(1).astype(int)
#     df[col] = df[col].clip(lower=1)

# normalize the following columns with log scale
log_cols = [
    DURATION,
    SUBSCRIBER_COUNT,
    CHANNEL_VIEWS,
    VIEWS,
]
for col in log_cols:
    df[col] = df[col].apply(lambda x: np.log1p(x))

# normalize without log scale
# duration_sec, title_length, vid_hour_of_day, vid_day_of_week, video_count, channel_age_years
norm_cols = [
    DURATION,
    IS_SHORT,
    TITLE_LENGTH,
    VID_HOUR_OF_DAY,
    VID_DAY_OF_WEEK,
    SUBSCRIBER_COUNT,
    CHANNEL_VIEWS,
    VIDEO_COUNT,
    CHANNEL_AGE_YEARS,
    VIEWS
]

feature_cols = [THUMBNAIL_URL_MEDIUM, TITLE] + norm_cols

for col in norm_cols:
    min_val = df[col].min()
    max_val = df[col].max()
    df[col] = (df[col] - min_val) / (max_val - min_val + 1e-9)

# Keep columns that are not in norm_cols
other_cols = [col for col in df.columns if col not in feature_cols]

# Reorder dataframe: other columns first, then norm_cols
df = df[other_cols + feature_cols]

# -------------------------
# Save cleaned data
# -------------------------
df.to_csv("video_data_cleaned.csv", index=False, encoding="utf-8")

# Define your transform
transform = transforms.Compose([
    transforms.Resize(224),           # resize shorter side to 224
    transforms.CenterCrop(224),       # crop to 224x224
    transforms.ToTensor()
])

# Function to download and convert one image
def download_image(url):
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return transform(img)
    except Exception as e:
        print(f"Failed to process {url}: {e}")
        # Return a dummy tensor in case of failure
        return torch.zeros(3, 224, 224)

# Function to process all images and save tensors
def process_column(urls, save_path="thumbnail_tensors.pt", max_workers=8):
    if os.path.exists(save_path):
        print(f"Loading tensors from {save_path}...")
        data = torch.load(save_path)
        return data

    print("Downloading and processing images...")
    tensors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_image, url): url for url in urls}
        for future in as_completed(futures):
            tensors.append(future.result())

    # Stack tensors into one tensor and save
    tensors_stack = torch.stack(tensors)
    torch.save(tensors_stack, save_path)
    print(f"Tensors saved to {save_path}")
    return tensors_stack

thumbnail_tensors = process_column(df[THUMBNAIL_URL_MEDIUM].tolist())
print(thumbnail_tensors.shape)  # [num_images, 3, 224, 224]