import os
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("YT_API_KEY")

youtube = build("youtube", "v3", developerKey=API_KEY)
resp = youtube.search().list(
    part="snippet",
    q="Преспански договор",
    type="video",
    maxResults=10,
    regionCode="MK",
    relevanceLanguage="mk"
).execute()

for i, item in enumerate(resp.get("items", []), 1):
    print(i, item["id"]["videoId"], "-", item["snippet"]["title"])
