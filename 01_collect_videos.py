import os, time, json
from collections import defaultdict
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from keywords import SEARCH_TERMS

load_dotenv()
API_KEY = os.getenv("YT_API_KEY")
youtube = build("youtube", "v3", developerKey=API_KEY)

def search_videos(query, per_page=50, max_pages=2):
    """Враќа листа на (videoId, snippet) за даден query.
       max_pages=2 ~ до 100 резултати по клучен збор (пилот)."""
    out = []
    page_token = None
    for _ in range(max_pages):
        try:
            resp = youtube.search().list(
                part="snippet",
                q=query,
                type="video",
                maxResults=per_page,
                regionCode="MK",
                relevanceLanguage="mk",
                order="relevance",
                pageToken=page_token
            ).execute()
        except HttpError as e:
            print("HTTP error on search:", e)
            break
        out.extend(resp.get("items", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.2)  # нежно кон квотата
    return out

def fetch_video_stats(video_ids):
    """Враќа dict video_id -> statistics (views/likes/comments)."""
    stats = {}
    # YouTube дозволува до 50 ID-а одеднаш
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i+50]
        try:
            resp = youtube.videos().list(
                part="snippet,statistics",
                id=",".join(chunk)
            ).execute()
        except HttpError as e:
            print("HTTP error on videos.list:", e)
            continue
        for it in resp.get("items", []):
            vid = it["id"]
            sn = it.get("snippet", {})
            st = it.get("statistics", {})
            has_comments = "commentCount" in st and int(st.get("commentCount", 0)) > 0
            comments_enabled = not st.get("commentCount") == "0"
            stats[vid] = {
                "title": sn.get("title"),
                "description": sn.get("description"),
                "channelTitle": sn.get("channelTitle"),
                "publishedAt": sn.get("publishedAt"),
                "viewCount": int(st.get("viewCount", 0)),
                "likeCount": int(st.get("likeCount", 0)),
                "commentCount": int(st.get("commentCount", 0)),
                "comments_enabled": comments_enabled,
                "has_comments": has_comments,
                "defaultLanguage": sn.get("defaultLanguage"),
                "defaultAudioLanguage": sn.get("defaultAudioLanguage"),
            }
        time.sleep(0.2)
    return stats

def main():
    # Агрегираме преку сите клучни зборови и чуваме матчинзи
    hits = defaultdict(lambda: {"matched_terms": set(), "seed": None})
    print("Searching videos across keywords...")
    for term in tqdm(SEARCH_TERMS):
        items = search_videos(term, per_page=50, max_pages=2)
        for it in items:
            vid = it["id"]["videoId"]
            hits[vid]["matched_terms"].add(term)
            if not hits[vid]["seed"]:
                hits[vid]["seed"] = term

    video_ids = list(hits.keys())
    print(f"Found {len(video_ids)} unique videos. Fetching stats...")
    stats = fetch_video_stats(video_ids)

    rows = []
    for vid in video_ids:
        s = stats.get(vid, {})
        rows.append({
            "video_id": vid,
            "title": s.get("title"),
            "description": s.get("description"),
            "channel": s.get("channelTitle"),
            "published_at": s.get("publishedAt"),
            "views": s.get("viewCount"),
            "likes": s.get("likeCount"),
            "comments": s.get("commentCount"),
            "matched_terms": "; ".join(sorted(hits[vid]["matched_terms"])),
            "seed_term": hits[vid]["seed"],
            "lang": s.get("defaultLanguage") or s.get("defaultAudioLanguage"),
            "video_url": f"https://www.youtube.com/watch?v={vid}"
        })

    df = pd.DataFrame(rows).sort_values(["comments", "views"], ascending=[False, False])
    df.to_csv("datasets/videos.csv", index=False, encoding="utf-8-sig")
    print("Saved videos.csv with", len(df), "rows.")

if __name__ == "__main__":
    main()
