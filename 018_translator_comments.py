#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
018_translator_comments.py
Normalize all comments to Macedonian Cyrillic (MK) from a 'text' column, then produce English versions.

✔ Cleans URLs, @handles, hashtags
✔ Full Latin (incl. mixed Latin+Cyr) → MK Cyrillic (forced)
✔ Serbian/Bulgarian Cyrillic → MK Cyrillic normalization
✔ MK → EN translation with Windows-safe cache (fsync + retry)
✔ Robust CSV reader that skips bad lines

Usage:
    python 018_translator_comments.py --src comments.csv --dst comments_mk_en.csv
"""

import argparse, json, os, re, time
from collections import OrderedDict
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None
    print("⚠ deep_translator не е инсталиран. Инсталирај:  pip install deep-translator")

# --------------------------------------------------------------------
# Light cleaning
# --------------------------------------------------------------------
URL_RE        = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
HANDLE_RE     = re.compile(r'(?<=\s)@\w+|^@\w+')
HASHTAG_RE    = re.compile(r'(?<=\s)#(\w+)\b|^#(\w+)\b')
WHITESPACE_RE = re.compile(r'\s+')

def clean_light(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = URL_RE.sub('<URL>', text)
    t = HANDLE_RE.sub('<USER>', t)
    t = HASHTAG_RE.sub(lambda m: (m.group(1) or m.group(2) or ""), t)
    t = t.replace('\u200b', '').replace('\ufeff', '')
    t = WHITESPACE_RE.sub(' ', t).strip()
    return t

# --------------------------------------------------------------------
# Cyrillic / Latin detection
# --------------------------------------------------------------------
CYR_BLOCKS = [(0x0400,0x04FF),(0x0500,0x052F),(0x2DE0,0x2DFF),(0xA640,0xA69F),(0x1C80,0x1C8F)]
def _in_cyrillic(ch: str) -> bool:
    cp = ord(ch)
    return any(a <= cp <= b for a,b in CYR_BLOCKS)
def has_cyrillic(s: str) -> bool:
    return any(_in_cyrillic(ch) for ch in str(s))
def is_latin_char(ch: str) -> bool:
    cp = ord(ch)
    return (0x0041 <= cp <= 0x007A) or (0x00C0 <= cp <= 0x024F)

# --------------------------------------------------------------------
# Full Latin -> MK Cyrillic transliteration
# --------------------------------------------------------------------
DIGRAPHS = [
    ("dzh","џ"), ("dž","џ"),
    ("gj","ѓ"),  ("kj","ќ"),
    ("zh","ж"),  ("ch","ч"), ("sh","ш"),
    ("lj","љ"),  ("nj","њ"),
    ("ts","ц"),
    ("č","ч"), ("ć","ч"), ("š","ш"), ("ž","ж"), ("đ","џ"),
]
DZ_ONLY_RE = re.compile(r"(?i)\bdz(?!h)")
SINGLE_MAP = {
    "a":"а","b":"б","c":"ц","d":"д","e":"е","f":"ф","g":"г","h":"х",
    "i":"и","j":"ј","k":"к","l":"л","m":"м","n":"н","o":"о","p":"п",
    "q":"к","r":"р","s":"с","t":"т","u":"у","v":"в","w":"в","x":"кс",
    "y":"ј","z":"з",
}
def _pc(src: str, dst: str) -> str:
    if src.isupper(): return dst.upper()
    if len(src)>=2 and src[0].isupper() and src[1:].islower():
        return dst[0].upper() + (dst[1:] if len(dst)>1 else "")
    return dst.lower()

def latin_to_mk_cyr(text: str) -> str:
    if not text:
        return text
    t = str(text)
    t = DZ_ONLY_RE.sub(lambda m: _pc(m.group(0), "ѕ"), t)
    out, i, n = [], 0, len(t)
    d_sorted = sorted(DIGRAPHS, key=lambda kv: len(kv[0]), reverse=True)
    while i < n:
        matched = False
        for src, dst in d_sorted:
            L = len(src)
            if i+L <= n and t[i:i+L].lower() == src:
                out.append(_pc(t[i:i+L], dst))
                i += L
                matched = True
                break
        if matched:
            continue
        ch = t[i]; low = ch.lower()
        if low in SINGLE_MAP:
            out.append(_pc(ch, SINGLE_MAP[low]))
        else:
            out.append(ch)
        i += 1
    return "".join(out)

# --------------------------------------------------------------------
# SR/BG Cyrillic -> Macedonian Cyrillic normalization
# --------------------------------------------------------------------
SR_BG_TABLE = {
    "ђ":"ѓ","Ђ":"Ѓ","ћ":"ќ","Ћ":"Ќ",
    "щ":"шт","Щ":"ШТ","ю":"ју","Ю":"ЈУ",
    "я":"ја","Я":"ЈА","й":"ј","Й":"Ј",
    "ъ":"а","Ъ":"А","ь":"",
    "ѝ":"и","Ѝ":"И","э":"е","Э":"Е",
}
def srb_bg_cyr_to_mk_cyr(text: str) -> str:
    return "".join(SR_BG_TABLE.get(ch, ch) for ch in str(text))

# --------------------------------------------------------------------
# Router: ANY -> MK Cyrillic
# --------------------------------------------------------------------
def any_to_mk_cyr(text: str) -> str:
    """
    Force transliteration if text has ANY Latin letter (even mixed),
    then normalize SR/BG to MK.
    """
    t = clean_light(str(text or ""))
    if any(is_latin_char(ch) for ch in t):
        t = latin_to_mk_cyr(t)
    t = srb_bg_cyr_to_mk_cyr(t)
    return t.strip()

# --------------------------------------------------------------------
# Translator (MK -> EN) with Windows-safe cache
# --------------------------------------------------------------------
def default_cache_dir() -> str:
    root = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    path = os.path.join(root, "yt_mk_cache")
    os.makedirs(path, exist_ok=True)
    return path

class Translator:
    def __init__(self, source_lang="mk", target_lang="en",
                 cache_path=None, max_retries=5, base_sleep=0.8,
                 chunk_limit=4000):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.cache_path = cache_path or os.path.join(default_cache_dir(), f"{source_lang}_to_{target_lang}.json")
        self.max_retries = max_retries
        self.base_sleep = base_sleep
        self.chunk_limit = chunk_limit
        self.cache: Dict[str,str] = OrderedDict()
        self.translator = GoogleTranslator(source=source_lang, target=target_lang) if GoogleTranslator else None
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path,"r",encoding="utf-8") as f:
                    self.cache.update(json.load(f))
            except Exception:
                self.cache = OrderedDict()

    def save_cache(self):
        cache_dir = os.path.dirname(os.path.abspath(self.cache_path)) or "."
        os.makedirs(cache_dir, exist_ok=True)
        tmp = self.cache_path + f".tmp.{os.getpid()}"
        with open(tmp,"w",encoding="utf-8") as f:
            json.dump(self.cache,f,ensure_ascii=False)
            f.flush()
            try: os.fsync(f.fileno())
            except Exception: pass
        for i in range(7):
            try:
                os.replace(tmp,self.cache_path)
                return
            except PermissionError:
                time.sleep(0.2*(2**i))
        try: os.remove(tmp)
        except: pass

    def _split_chunks(self,text:str)->List[str]:
        t=str(text)
        if len(t)<=self.chunk_limit: return [t]
        parts=[];start=0
        while start<len(t):
            end=min(start+self.chunk_limit,len(t))
            cut=t.rfind(". ",start,end)
            if cut==-1: cut=t.rfind(" ",start,end)
            if cut==-1 or cut<=start+int(self.chunk_limit*0.6): cut=end
            parts.append(t[start:cut].strip());start=cut
        return [p for p in parts if p]

    def _translate_raw(self,text:str)->str:
        if self.translator is None:
            return str(text)
        for attempt in range(1,self.max_retries+1):
            try:
                out=self.translator.translate(text) or ""
                if out.strip()==str(text).strip():
                    if re.fullmatch(r"[\W_0-9<>\s]+",str(text)):
                        return str(text)
                    raise RuntimeError("translator returned identical")
                return out
            except Exception:
                time.sleep(self.base_sleep*(2**(attempt-1)))
        return str(text)

    def translate_one(self,text:str)->str:
        key=f"[{self.source_lang}->{self.target_lang}] "+str(text)
        if key in self.cache and self.cache[key].strip():
            return self.cache[key]
        time.sleep(0.12)
        outs=[]
        for p in self._split_chunks(text):
            tr=self._translate_raw(p)
            outs.append(tr if tr.strip() else p)
        full=" ".join(outs).strip()
        if full:
            self.cache[key]=full
            self.save_cache()
        return full

# --------------------------------------------------------------------
# Robust CSV reader
# --------------------------------------------------------------------
def read_csv_robust(path:str)->pd.DataFrame:
    trials=[
        dict(engine="c",sep=",",encoding="utf-8"),
        dict(engine="python",sep=None,encoding="utf-8",on_bad_lines="skip"),
        dict(engine="python",sep="\t",encoding="utf-8",on_bad_lines="skip"),
    ]
    for opt in trials:
        try: return pd.read_csv(path,**opt)
        except Exception: continue
    raise RuntimeError("Unable to read CSV")

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
def main():
    ap=argparse.ArgumentParser(description="Normalize comments to MK Cyrillic + EN translation")
    ap.add_argument("--src",default="datasets/comments_fixed.csv")
    ap.add_argument("--dst",default="datasets/comments_mk_en.csv")
    ap.add_argument("--text_col",default="text")
    ap.add_argument("--cache_mk2en",default=None)
    args=ap.parse_args()

    df=read_csv_robust(args.src)
    if args.text_col not in df.columns:
        df[args.text_col]=""

    mk2en=Translator("mk","en",cache_path=args.cache_mk2en)
    texts=df[args.text_col].fillna("").astype(str).tolist()

    text_mk=[]; text_en=[]
    for t in tqdm(texts,desc="Comments → MK(Cyr) → EN"):
        mk_val=any_to_mk_cyr(t)
        en_val=mk2en.translate_one(mk_val) if mk_val.strip() else ""
        text_mk.append(mk_val)
        text_en.append(en_val)

    df["text_mk"]=text_mk
    df["text_en"]=text_en

    for attempt in range(5):
        try:
            df.to_csv(args.dst,index=False,encoding="utf-8-sig")
            break
        except PermissionError:
            print(f"⚠️ File '{args.dst}' open — close Excel & retrying in 3 s...")
            time.sleep(3)
    print(f"✅ Saved '{args.dst}' with text_mk + text_en")
    print(f"ℹ️ Cache: {mk2en.cache_path}")

if __name__=="__main__":
    main()
