import torch
import pandas as pd
from hybrid_nn import HybridEvaluator


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
# Example Usage
# -------------------------
df = pd.read_csv("video_data.csv", encoding="utf-8")
device = "cuda" if torch.cuda.is_available() else "cpu"

# inputs
batch_size = 32
thumbnail_tensors_stack = torch.load("thumbnail_tensors.pt")
titles = df[TITLE].astype(str).tolist()[:batch_size]  # List of strings

channel_num = df[SUBSCRIBER_COUNT, CHANNEL_VIEWS, VIDEO_COUNT, CHANNEL_AGE_YEARS] # log_subs, channel_age_year, video_count, channel_views
video_num = df[DURATION, IS_SHORT, TITLE_LENGTH, VID_HOUR_OF_DAY, VID_DAY_OF_WEEK] # duration_sec, title_length, vid_hour, vid_day_of_week, is_short

df[VIEWS] = pd.cut(
    df[VIEWS],
    bins=8,
    labels=False,
    include_lowest=True
)

# Model - get it to work with updated features
model = HybridEvaluator(num_numeric_features=10, num_classes=8, device=device).to(device)

outputs = model(
    thumbnail_tensors_stack.to(device),
    titles,
    channel_num.to(device),
    video_num.to(device)
)

print("\n\noutput to HYBRID EVALUATOR", outputs)  # torch.Size([batch_size, 8])
