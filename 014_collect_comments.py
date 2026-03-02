#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
022_collect_comments_full_replies.py
YouTube → Collect ALL comments (top-level + replies) with resume, summary, and backoff.

Влез:
    videos_clean.csv (или videos.csv) со колона video_id

Излез:
    comments.csv            — сите коментари + replies
    comments_per_video.csv  — број по видео
    skipped_videos.csv      — видеа со грешки/нема коментари
"""

import os, time, csv, sys
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ── Конфигурација ─────────────────────────────────────────────────────────────
SOURCE_CSV = "datasets/videos_clean.csv"    # може и videos.csv / videos_clean.csv
OUT_COMMENTS = "datasets/comments.csv"
OUT_SUMMARY  = "datasets/comments_per_video.csv"
OUT_SKIPPED  = "datasets/skipped_videos.csv"

ORDER = "relevance"                # или "time" ако сакаш хронолошки
MAX_RESULTS = 100                  # YouTube page size
SLEEP = 0.15                       # пауза меѓу повици
MAX_TOTAL_COMMENTS_PER_VIDEO = None  # None = сите, или број (пр. 500)
RESUME = True                      # прескокни веќе обработени видеа
BACKOFF_BASE = 1.2                 # за quotaExceeded → експоненцијален backoff
MAX_REPLIES_PER_THREAD = None      # None = сите, или број (пр. 200)

# ── Setup ─────────────────────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("YT_API_KEY")
if not API_KEY:
    raise RuntimeError("❌ Missing YT_API_KEY in .env")
youtube = build("youtube", "v3", developerKey=API_KEY)

# ── Utility функции ───────────────────────────────────────────────────────────
def safe_int(x, default=0):
    try: return int(x)
    except Exception: return default

def ensure_csv_header(path: str, header: List[str]):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(header)

def append_rows(path: str, rows: List[List[Any]]):
    if not rows: return
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerows(rows)

def load_done_video_ids(comments_csv: str) -> set:
    if not os.path.exists(comments_csv): return set()
    try:
        df = pd.read_csv(comments_csv, usecols=["video_id"])
        return set(df["video_id"].astype(str).unique())
    except Exception:
        return set()

def append_summary(video_id: str, total: int, reason: Optional[str] = None):
    append_rows(OUT_SUMMARY, [[video_id, total, reason or "ok"]])

def append_skipped(video_id: str, reason: str):
    append_rows(OUT_SKIPPED, [[video_id, reason]])

# ── Fetch функции ─────────────────────────────────────────────────────────────
def fetch_comment_threads(video_id: str, order=ORDER):
    """Враќа топ-ниво threads (со неколку почетни replies)."""
    page_token = None
    while True:
        resp = youtube.commentThreads().list(
            part="snippet,replies",
            videoId=video_id,
            maxResults=MAX_RESULTS,
            pageToken=page_token,
            textFormat="plainText",
            order=order
        ).execute()
        for item in resp.get("items", []):
            yield item
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
        time.sleep(SLEEP)

def fetch_replies(parent_id: str):
    """Ги влече СИТЕ replies за даден top-level коментар (parent_id)."""
    page_token = None
    total = 0
    while True:
        resp = youtube.comments().list(
            part="snippet",
            parentId=parent_id,
            maxResults=MAX_RESULTS,
            pageToken=page_token,
            textFormat="plainText"
        ).execute()
        for item in resp.get("items", []):
            yield item
            total += 1
            if isinstance(MAX_REPLIES_PER_THREAD, int) and total >= MAX_REPLIES_PER_THREAD:
                return
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
        time.sleep(SLEEP)

# ── Row builders ──────────────────────────────────────────────────────────────
def row_from_top_level(video_id: str, thread: Dict[str, Any]) -> List[Any]:
    top = thread.get("snippet", {}).get("topLevelComment", {}) or {}
    sn  = top.get("snippet", {}) or {}
    text = sn.get("textDisplay") or sn.get("textOriginal") or ""
    return [
        video_id,
        thread.get("id"),            # thread_id
        top.get("id"),               # comment_id
        None,                        # parent_id
        False,                       # is_reply
        sn.get("authorDisplayName"),
        (sn.get("authorChannelId") or {}).get("value"),
        text,
        len(text),
        safe_int(sn.get("likeCount")),
        sn.get("publishedAt"),
        sn.get("updatedAt"),
    ]

def row_from_reply(video_id: str, thread_id: str, reply: Dict[str, Any]) -> List[Any]:
    sn = reply.get("snippet", {}) or {}
    text = sn.get("textDisplay") or sn.get("textOriginal") or ""
    return [
        video_id,
        thread_id,
        reply.get("id"),
        sn.get("parentId"),
        True,                        # is_reply = True
        sn.get("authorDisplayName"),
        (sn.get("authorChannelId") or {}).get("value"),
        text,
        len(text),
        safe_int(sn.get("likeCount")),
        sn.get("publishedAt"),
        sn.get("updatedAt"),
    ]

# ── Главна функција за едно видео ─────────────────────────────────────────────
def collect_for_video(video_id: str, max_total: Optional[int]) -> int:
    """Собира сите top-level + replies коментари за едно видео."""
    written = 0
    for thread in fetch_comment_threads(video_id):
        # --- Top-level ---
        row_top = row_from_top_level(video_id, thread)
        append_rows(OUT_COMMENTS, [row_top])
        written += 1

        # --- Replies ---
        top_comment_id = (
            thread.get("snippet", {})
                  .get("topLevelComment", {})
                  .get("id")
        )
        reply_count = safe_int(thread.get("snippet", {}).get("totalReplyCount"))
        if reply_count > 0 and top_comment_id:
            replies_found = 0
            for reply in fetch_replies(top_comment_id):
                row_reply = row_from_reply(video_id, thread["id"], reply)
                append_rows(OUT_COMMENTS, [row_reply])
                written += 1
                replies_found += 1
            if replies_found == 0:
                print(f"⚠️ No replies returned for {video_id}/{top_comment_id}")

        if isinstance(max_total, int) and written >= max_total:
            return written
        time.sleep(SLEEP)
    return written

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ensure_csv_header(OUT_COMMENTS, [
        "video_id", "thread_id", "comment_id", "parent_id", "is_reply",
        "author_display_name", "author_channel_id", "text", "text_length",
        "like_count", "published_at", "updated_at"
    ])
    ensure_csv_header(OUT_SUMMARY, ["video_id", "total_comments_downloaded", "reason"])
    ensure_csv_header(OUT_SKIPPED, ["video_id", "reason"])

    done = load_done_video_ids(OUT_COMMENTS) if RESUME else set()
    vids_df = pd.read_csv(SOURCE_CSV, usecols=["video_id"])
    all_ids = vids_df["video_id"].dropna().astype(str).unique().tolist()
    todo_ids = [v for v in all_ids if v not in done]

    print(f"▶️ Collecting ALL comments (top-level + replies) for {len(todo_ids)} videos...")

    for vid in tqdm(todo_ids):
        backoff = BACKOFF_BASE
        while True:
            try:
                total = collect_for_video(vid, MAX_TOTAL_COMMENTS_PER_VIDEO)
                append_summary(vid, total, None)
                break
            except HttpError as e:
                msg = str(e)
                if "commentsDisabled" in msg:
                    append_skipped(vid, "commentsDisabled")
                    append_summary(vid, 0, "commentsDisabled")
                    break
                if "videoNotFound" in msg or "notFound" in msg:
                    append_skipped(vid, "notFound")
                    append_summary(vid, 0, "notFound")
                    break
                if "forbidden" in msg and "commentsDisabled" not in msg:
                    append_skipped(vid, "forbidden")
                    append_summary(vid, 0, "forbidden")
                    break
                if any(x in msg for x in ["quotaExceeded","rateLimitExceeded","userRateLimitExceeded"]):
                    sleep_s = min(60, backoff * 5)
                    print(f"[{vid}] quota/rate limit → sleep {sleep_s:.1f}s then retry…")
                    time.sleep(sleep_s)
                    backoff *= BACKOFF_BASE
                    continue
                append_skipped(vid, f"HttpError:{e}")
                append_summary(vid, 0, "HttpError")
                break
            finally:
                time.sleep(SLEEP)

    print("✅ Done. Check output files:")
    print(f" - {OUT_COMMENTS}")
    print(f" - {OUT_SUMMARY}")
    print(f" - {OUT_SKIPPED}")

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user. Progress saved so far.")
        sys.exit(130)
