import os
import googleapiclient.discovery
import pandas as pd
import random
import re

api_service_name = "youtube"
api_version = "v3"
api_key = os.environ["YOUTUBE_DATA_API_KEY"]
list_params = ["snippet","contentDetails","statistics"]
# Get credentials and create an API client
youtube = googleapiclient.discovery.build(
    api_service_name, api_version, developerKey=api_key
)

data = {
    "video_id": [], # id
    "channel_id": [], # snippet.channelId
    "title": [], # snippet.title
    "thumbnail_url_default": [], # snippet.thumbnails.default.url
    "thumbnail_url_medium": [], # snippet.thumbnails.medium.url
    "tags": [], # snippet.tags
    "default_language": [], # snippet.defaultLanguage
    "duration": [], # contentDetails.duration
    "views": [], # statistics.viewCount, TARGET
    "subscriber_count": [], # from channel statistics
    "channel_views": [], # from channel statistics
    "video_count": [], # from channel statistics
}

file = open("video_ids_list.txt", "r")
video_ids_list = file.read().split(",")
random.shuffle(video_ids_list)
file.close()

for i in range(len(video_ids_list)//50 + 1):
    batch = ",".join(video_ids_list[i*50:(i+1)*50])
    request_vid = youtube.videos().list(part=",".join(list_params), id=batch)
    response_vid = request_vid.execute()
    req_channel_data_list = []
    
    for item in response_vid["items"]:
        print("getting all data for video id:", item["id"])
        data["video_id"].append(item["id"])
        data["channel_id"].append(item["snippet"].get("channelId", ""))
        req_channel_data_list.append(item["snippet"].get("channelId", ""))
        data["title"].append(item["snippet"].get("title", ""))
        data["thumbnail_url_default"].append(item["snippet"]["thumbnails"]["default"].get("url", ""))
        data["thumbnail_url_medium"].append(item["snippet"]["thumbnails"]["medium"].get("url", ""))
        data["tags"].append(", ".join(item["snippet"].get("tags", [])))
        data["default_language"].append(item["snippet"].get("defaultLanguage", ""))
        data["duration"].append(item["contentDetails"].get("duration", ""))
        data["views"].append(item["statistics"].get("viewCount", 0))

    request_channel = youtube.channels().list(part="statistics", id=",".join(req_channel_data_list))
    response_channel = request_channel.execute()
    for channel_id in req_channel_data_list:
        for item in response_channel["items"]:
            if channel_id == item["id"]:
                data["subscriber_count"].append(item["statistics"].get("subscriberCount", 0))
                data["channel_views"].append(item["statistics"].get("viewCount", 0))
                data["video_count"].append(item["statistics"].get("videoCount", 0))
    print("Completed batch", i+1, "of",
        len(video_ids_list)//50 + 1, "----",
        len(data["video_id"]),
        len(data["channel_id"]),
        len(data["title"]),
        len(data["thumbnail_url_default"]),
        len(data["thumbnail_url_medium"]),
        len(data["tags"]),
        len(data["default_language"]),
        len(data["duration"]),
        len(data["views"]),
        len(data["subscriber_count"]),
        len(data["channel_views"]),
        len(data["video_count"])
    )

df = pd.DataFrame(data)
df.to_csv("video_data.csv", index=False)
