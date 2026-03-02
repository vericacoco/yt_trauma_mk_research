#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
017_translator_videos.py  —  Unified videos pipeline for EN<->MK + Latin->Cyr + cleaning (Windows-safe cache).

- Cleans description boilerplate (URLs, emails, subscribe/author/©/credits...)
- Detects English vs Macedonian (heuristic)
- EN -> MK (Cyrillic) translation
- MK Latin -> MK Cyrillic transliteration
- MK -> EN translation (to keep an English view too)
- Builds:
    title_mk, description_mk
    title_en, description_en
    title_description_mk, title_description_en
    text_mk_norm  (auditing/preproc)

Usage:
    python 017_translator_videos.py --src videos.csv --dst videos_mk_en.csv
"""

import argparse
import json
import os
import re
import time
from collections import OrderedDict
from typing import Iterable, List, Dict, Tuple

import pandas as pd
from tqdm import tqdm

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None
    print("⚠ deep_translator не е инсталиран. Инсталирај: pip install deep-translator")

# ---------------------------
# Regex / cleaning
# ---------------------------

URL_RE        = re.compile(r'https?://\S+|www\.\S+', flags=re.IGNORECASE)
EMAIL_RE      = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
HANDLE_RE     = re.compile(r'@\w+')
HASHTAG_RE    = re.compile(r'#(\w+)\b')
WHITESPACE_RE = re.compile(r'\s+')
CYRILLIC_RE   = re.compile(r'[Ѐ-ԧ]')
LATIN_RE      = re.compile(r'[A-Za-z]')

SOCIAL_RE = re.compile(r"(subscribe|follow\s+me|instagram|facebook|twitter|x\.com|tiktok|patreon|donate|like\s+and\s+share)", re.IGNORECASE)
COPY_RE   = re.compile(r"(copyright|©|all rights reserved|no copyright|music by|video by|produced by|credits|cast|director|shot on)", re.IGNORECASE)

TAG_RE        = re.compile(r'(?:<URL>|<USER>)')
ONLY_TAGS_RE  = re.compile(r'^(?:\W|_|\d|<URL>|<USER>)+$', re.IGNORECASE)

MERGE_SEP_DEFAULT = " — "

def clean_description(text: str) -> str:
    """Drop URLs/emails/social/copyright-y lines; collapse spaces."""
    if not isinstance(text, str): return ""
    lines, t = [], str(text)
    for ln in t.splitlines():
        raw = ln.strip()
        if not raw: continue
        if URL_RE.search(raw):   continue
        if EMAIL_RE.search(raw): continue
        if SOCIAL_RE.search(raw):continue
        if COPY_RE.search(raw):  continue
        lines.append(raw)
    out = " ".join(lines)
    out = WHITESPACE_RE.sub(" ", out).strip()
    return out

def clean_light(text: str) -> str:
    """Keep content but normalize tokens."""
    if not text: return ""
    t = str(text)
    t = URL_RE.sub('<URL>', t)
    t = HANDLE_RE.sub('<USER>', t)
    t = HASHTAG_RE.sub(r'\1', t)
    t = t.replace('\u200b', '').replace('\ufeff', '')
    t = WHITESPACE_RE.sub(' ', t).strip()
    return t

# ---------------------------
# Language/script heuristics
# ---------------------------

def looks_cyrillic(s: str) -> bool:
    return bool(CYRILLIC_RE.search(str(s)))

def looks_latin_only(s: str) -> bool:
    s = str(s)
    return bool(LATIN_RE.search(s)) and not looks_cyrillic(s)

EN_HINTS = {" the "," and "," you "," is "," are "," with "," for "," on "," to "," in "," of "," i ", " we ", " it "}

def looks_english(s: str) -> bool:
    t = " " + str(s).lower() + " "
    latin_ratio = sum(ch.isalpha() and 'a' <= ch.lower() <= 'z' for ch in t) / (len(t) + 1e-9)
    if latin_ratio < 0.6:
        return False
    return any(k in t for k in EN_HINTS)

# ---------------------------
# Latin -> Macedonian Cyrillic
# ---------------------------

MAPPING = [
    ("dzh","џ"),("dž","џ"),
    ("gj","ѓ"),("kj","ќ"),
    ("zh","ж"),("ch","ч"),("sh","ш"),
    ("lj","љ"),("nj","њ"),
    ("ts","ц"),
    ("č","ч"),("ć","ч"),
    ("š","ш"),("ž","ж"),
    ("j","ј"),("c","ц"),
    ("x","кс"),("w","в"),("q","к"),
    # ако 'y->ј' ти пречи за EN, коментирај ја следнава линија:
    ("y","ј"),
]
MAPPING_SORTED = sorted(MAPPING, key=lambda kv: len(kv[0]), reverse=True)
ALT_RE = re.compile("(" + "|".join(re.escape(k) for k,_ in MAPPING_SORTED) + ")", re.IGNORECASE)
DZ_RE  = re.compile(r"(?i)\bdz(?!h)")

def _pc(src: str, dst: str) -> str:
    if src.isupper(): return dst.upper()
    if len(src)>=2 and src[0].isupper() and src[1:].islower():
        return dst[0].upper() + (dst[1:] if len(dst)>1 else "")
    return dst.lower()

def latin_to_cyr_mk(text: str) -> str:
    if not text: return text
    t = DZ_RE.sub(lambda m: _pc(m.group(0), "ѕ"), str(text))
    def repl(m):
        src = m.group(0); low = src.lower()
        for k,v in MAPPING_SORTED:
            if low == k: return _pc(src, v)
        return src
    return ALT_RE.sub(repl, t)

def preprocess_for_mk(text: str) -> str:
    """Light-clean + transliterate if Latin-heavy."""
    t = clean_light(text)
    if looks_latin_only(t):
        t = latin_to_cyr_mk(t)
    return t

def looks_like_placeholders(text: str) -> bool:
    t = TAG_RE.sub('', str(text))
    t = WHITESPACE_RE.sub(' ', t).strip()
    return (not t) or bool(ONLY_TAGS_RE.match(str(text)))

# ---------------------------
# Windows-safe cached translator
# ---------------------------

def default_cache_dir() -> str:
    # Use LOCALAPPDATA on Windows; fallback to user home elsewhere
    root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    path = os.path.join(root, "yt_mk_cache")
    os.makedirs(path, exist_ok=True)
    return path

class Translator:
    def __init__(self, source_lang="mk", target_lang="en",
                 cache_path: str = None,
                 max_retries: int = 5, base_sleep: float = 0.8,
                 chunk_limit: int = 4000):
        self.source_lang = source_lang
        self.target_lang = target_lang
        if cache_path is None:
            cache_path = os.path.join(default_cache_dir(), f"{source_lang}_to_{target_lang}.json")
        self.cache_path = cache_path
        self.max_retries = max_retries
        self.base_sleep = base_sleep
        self.chunk_limit = chunk_limit
        self.cache: Dict[str, str] = OrderedDict()

        if GoogleTranslator is None:
            self.translator = None
        else:
            self.translator = GoogleTranslator(source=source_lang, target=target_lang)

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self.cache.update(json.load(f))
            except Exception:
                self.cache = OrderedDict()

    def save_cache(self):
        # ensure parent dir exists
        cache_dir = os.path.dirname(os.path.abspath(self.cache_path)) or "."
        os.makedirs(cache_dir, exist_ok=True)

        tmp = self.cache_path + f".tmp.{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass

        last_err = None
        for i in range(7):  # retries with backoff
            try:
                os.replace(tmp, self.cache_path)
                return
            except PermissionError as e:
                last_err = e
                time.sleep(0.2 * (2 ** i))
        # fallback: write backup if locked
        backup = self.cache_path + f".backup.{int(time.time())}.json"
        try:
            os.replace(tmp, backup)
            print(f"⚠️ Could not replace '{self.cache_path}' (locked). Wrote backup: {backup}")
        except Exception as e2:
            try: os.remove(tmp)
            except: pass
            print(f"⚠️ Cache save failed: {last_err or e2}")

    def _split_chunks(self, text: str) -> List[str]:
        t = str(text)
        if len(t) <= self.chunk_limit:
            return [t]
        chunks, start = [], 0
        while start < len(t):
            end = min(start + self.chunk_limit, len(t))
            cut = t.rfind('. ', start, end)
            if cut == -1:
                cut = t.rfind(' ', start, end)
            if cut == -1 or cut <= start + int(self.chunk_limit*0.6):
                cut = end
            chunks.append(t[start:cut].strip())
            start = cut
        return [c for c in chunks if c]

    def _translate_raw(self, text: str) -> str:
        if self.translator is None:
            return str(text)
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                out = self.translator.translate(text) or ""
                if out.strip() == str(text).strip():
                    if looks_like_placeholders(text):
                        return str(text)
                    raise RuntimeError("Translator returned identical input (likely fail).")
                return out
            except Exception as e:
                last_err = e
                time.sleep(self.base_sleep * (2 ** (attempt - 1)))
        return ""

    def translate_one(self, text: str) -> str:
        key = f"[{self.source_lang}->{self.target_lang}] " + str(text)
        if key in self.cache and str(self.cache[key]).strip():
            return self.cache[key]

        time.sleep(0.12)  # gentle pacing
        parts = self._split_chunks(text)
        translated_parts = []
        for p in parts:
            out = self._translate_raw(p)
            if not str(out).strip():
                return ""  # no-cache empty
            translated_parts.append(out)
        full = " ".join(translated_parts).strip()
        if full:
            self.cache[key] = full
            self.save_cache()
        return full

def translate_list(texts: Iterable[str], translator: Translator, desc: str = "batch") -> List[str]:
    out = []
    for t in tqdm(texts, desc=f"Translating {translator.source_lang}->{translator.target_lang} ({desc})"):
        txt = "" if t is None else str(t)
        if not txt.strip():
            out.append("")
            continue
        out_txt = translator.translate_one(txt)
        out.append(out_txt if out_txt.strip() else "")
    translator.save_cache()
    return out

# ---------------------------
# Field conversion logic
# ---------------------------

def to_mk_and_en(title: str, description: str,
                 merge_sep: str = MERGE_SEP_DEFAULT,
                 mk2en: Translator = None,
                 en2mk: Translator = None) -> Tuple[Dict, Dict]:
    """
    Returns two dicts:
      mk_fields = {title_mk, description_mk, title_description_mk}
      en_fields = {title_en, description_en, title_description_en}

    Rules per field:
      - If looks English -> EN->MK for mk_fields; keep EN as-is in en_fields
      - Else (MK):
          * if Latin-heavy -> translit to Cyrillic
          * keep original (if already Cyrillic)
        Then mk2en -> en_fields
    """
    title_raw = str(title or "")
    desc_raw  = str(description or "")

    # clean description boilerplate
    desc_clean = clean_description(desc_raw)

    # Light-clean for language tests (not for final)
    title_l = clean_light(title_raw)
    desc_l  = clean_light(desc_clean)

    def field_to_mk_and_en(txt: str) -> Tuple[str, str]:
        if looks_english(txt):
            # EN -> MK (final mk value), keep EN as en_value
            mk_val = en2mk.translate_one(txt) if en2mk else txt
            en_val = txt
        else:
            # MK path: Latin-heavy -> translit
            if looks_latin_only(txt):
                mk_val = latin_to_cyr_mk(txt)
            else:
                mk_val = txt
            # Also produce EN from MK
            en_val = mk2en.translate_one(mk_val) if mk2en else mk_val
        return mk_val, en_val

    title_mk, title_en = field_to_mk_and_en(title_l)
    desc_mk,  desc_en  = field_to_mk_and_en(desc_l)

    def join_nonempty(a, b, sep=merge_sep):
        a = str(a or "").strip(); b = str(b or "").strip()
        if a and b: return f"{a}{sep}{b}"
        return a or b

    mk_merged = join_nonempty(title_mk, desc_mk)
    en_merged = join_nonempty(title_en,  desc_en)

    mk_fields = {
        "title_mk": title_mk,
        "description_mk": desc_mk,
        "title_description_mk": mk_merged,
    }
    en_fields = {
        "title_en": title_en,
        "description_en": desc_en,
        "title_description_en": en_merged,
    }
    return mk_fields, en_fields

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Videos EN<->MK pipeline + Latin->Cyr + cleaning (Windows-safe cache).")
    ap.add_argument("--src", default="datasets/videos_clean.csv", help="Input CSV")
    ap.add_argument("--dst", default="datasets/videos_mk_en.csv", help="Output CSV")
    ap.add_argument("--title_col", default="title", help="Title column name")
    ap.add_argument("--desc_col",  default="description", help="Description column name")
    ap.add_argument("--cache_mk2en", default=None, help="Cache path for MK->EN (default: LOCALAPPDATA/yt_mk_cache/mk_to_en.json)")
    ap.add_argument("--cache_en2mk", default=None, help="Cache path for EN->MK (default: LOCALAPPDATA/yt_mk_cache/en_to_mk.json)")
    ap.add_argument("--merge_sep", default=MERGE_SEP_DEFAULT, help="Separator for merged fields")
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    if args.title_col not in df.columns: df[args.title_col] = ""
    if args.desc_col  not in df.columns: df[args.desc_col]  = ""

    # Prepare translators (Windows-safe cache locations by default)
    mk2en_cache = args.cache_mk2en or os.path.join(default_cache_dir(), "mk_to_en.json")
    en2mk_cache = args.cache_en2mk or os.path.join(default_cache_dir(), "en_to_mk.json")

    mk2en = Translator(source_lang="mk", target_lang="en", cache_path=mk2en_cache)
    en2mk = Translator(source_lang="en", target_lang="mk", cache_path=en2mk_cache)

    # Build text_mk_norm (auditing): MK-ified versions (title/desc), before translation directions
    titles_raw = df[args.title_col].fillna("").astype(str).tolist()
    descs_raw  = df[args.desc_col].fillna("").astype(str).tolist()

    titles_norm = [preprocess_for_mk(t) for t in titles_raw]
    descs_norm  = [preprocess_for_mk(clean_description(d)) for d in descs_raw]
    text_mk_norm = [
        (titles_norm[i] + (args.merge_sep if titles_norm[i] and descs_norm[i] else "") + descs_norm[i])
        for i in range(len(titles_norm))
    ]

    # Convert fields per row
    mk_title, mk_desc, mk_merged = [], [], []
    en_title, en_desc, en_merged = [], [], []
    for t, d in tqdm(zip(titles_raw, descs_raw), total=len(df), desc="Processing rows"):
        mk_f, en_f = to_mk_and_en(t, d, merge_sep=args.merge_sep, mk2en=mk2en, en2mk=en2mk)
        mk_title.append(mk_f["title_mk"]);     mk_desc.append(mk_f["description_mk"]);     mk_merged.append(mk_f["title_description_mk"])
        en_title.append(en_f["title_en"]);     en_desc.append(en_f["description_en"]);     en_merged.append(en_f["title_description_en"])

    # Assign
    df["title_mk"]  = mk_title
    df["description_mk"] = mk_desc
    df["title_description_mk"] = mk_merged

    df["title_en"]  = en_title
    df["description_en"] = en_desc
    df["title_description_en"] = en_merged

    df["text_mk_norm"] = text_mk_norm  # auditing/preproc

    # Save
    df.to_csv(args.dst, index=False, encoding="utf-8-sig")
    print(f"✅ Saved '{args.dst}' with MK & EN fields + merged columns.")
    print(f"ℹ️ Caches: mk2en -> {mk2en_cache} | en2mk -> {en2mk_cache}")

    # Mini report
    n_en_titles = sum(bool(str(x).strip()) for x in df["title_en"])
    n_en_descs  = sum(bool(str(x).strip()) for x in df["description_en"])
    print(f"ℹ️ Non-empty EN fields -> title_en: {n_en_titles} | description_en: {n_en_descs}")


    # ───────────────────────────────────────────────────────────────────────────────
    # Split by comment availability (if comments_per_video.csv exists)
    # ───────────────────────────────────────────────────────────────────────────────
    summary_path = "datasets/comments_per_video.csv"
    if os.path.exists(summary_path):
        try:
            df_summary = pd.read_csv(summary_path, names=["video_id", "total_comments_downloaded", "reason"], header=0)
            # нормализирај ID за join
            df["video_id"] = df["video_id"].astype(str)
            df_summary["video_id"] = df_summary["video_id"].astype(str)

            # Cleaning duplicates
            df_summary = df_summary.groupby("video_id", as_index=False).first()
            print(f"✅ Задржан само првиот запис по video_id → {len(df_summary)} уникатни видеа.")

            df_merged = df.merge(df_summary, on="video_id", how="left")
            df_with_comments = df_merged[df_merged["total_comments_downloaded"].fillna(0) > 0].copy()
            df_no_comments   = df_merged[df_merged["total_comments_downloaded"].fillna(0) == 0].copy()

            df_with_comments.to_csv("datasets/videos_mk_en_with_comments.csv", index=False, encoding="utf-8-sig")
            df_no_comments.to_csv("datasets/videos_mk_en_no_comments.csv", index=False, encoding="utf-8-sig")

            print(f"💾 Зачувано {len(df_with_comments)} видеа со коментари → videos_mk_en_with_comments.csv")
            print(f"💾 Зачувано {len(df_no_comments)} видеа без коментари → videos_mk_en_no_comments.csv")




            combined_path = "datasets/videos_mk_en_combined.csv"
            df_combined = pd.concat([df_with_comments, df_no_comments], ignore_index=True)
            df_combined.to_csv(combined_path, index=False, encoding="utf-8-sig")
            print(f"💾 Зачувана комбинирана табела → {combined_path} ({len(df_combined)} редови вкупно)")
        except Exception as e:
            print(f"⚠️ Не успеав да ги раздвојам видеата по коментари: {e}")
    else:
        print("ℹ️ comments_per_video.csv не е најден — нема поделба со/без коментари.")

if __name__ == "__main__":
    main()
