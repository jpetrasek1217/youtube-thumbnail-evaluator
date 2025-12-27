import os
import googleapiclient.discovery
import pandas as pd

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
    "thumbnail_url": [], # snippet.thumbnails.default.url
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
        data["thumbnail_url"].append(item["snippet"]["thumbnails"]["default"].get("url", ""))
        data["tags"].append(", ".join(item["snippet"].get("tags", [])))
        data["default_language"].append(item["snippet"].get("defaultLanguage", ""))
        data["duration"].append(item["contentDetails"].get("duration", ""))
        data["views"].append(item["statistics"].get("viewCount", "0"))
    
    request_channel = youtube.channels().list(part="statistics", id=",".join(req_channel_data_list))
    response_channel = request_channel.execute()
    for item in response_channel["items"]:
        data["subscriber_count"].append(item["statistics"].get("subscriberCount", 0))
        data["channel_views"].append(item["statistics"].get("viewCount", 0))
        data["video_count"].append(item["statistics"].get("videoCount", 0))
    print("Completed batch", i+1, "of", len(video_ids_list)//50 + 1)

df = pd.DataFrame(data)
df.to_csv("video_data.csv", index=False)