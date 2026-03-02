import pandas as pd

SRC = "datasets/videos_clean.csv"  # резултат од авто-филтерот
DST = "datasets/videos_manual.csv"  # финален излез по рачна селекција

df = pd.read_csv(SRC)
n = len(df)
print(f"Вкупно редови во {SRC}: {n}")

keep_rows = []
i = 0

HELP = (
    "\nКоманди: [Enter]=Keep, d=Delete, b=Back, q=Save&Quit, h=Help\n"
)

print(HELP)

while 0 <= i < n:
    row = df.iloc[i]
    video_id = row.get("video_id", "")
    title = str(row.get("title", ""))
    channel = str(row.get("channel", ""))
    views = int(row.get("views") or 0)
    comments = int(row.get("comments") or 0)
    url = row.get("video_url") or f"https://www.youtube.com/watch?v={video_id}"

    print("\n--------------------------------------")
    print(f"[{i+1}/{n}]")
    print(f"Title   : {title}")
    print(f"Channel : {channel}")
    print(f"Views   : {views}, Comments: {comments}")
    print(f"URL     : {url}")
    print("--------------------------------------")

    choice = input("Keep=[Enter], d=Delete, b=Back, q=Save&Quit, h=Help : ").strip().lower()

    if choice == "d":
        i += 1
    elif choice == "b":
        if keep_rows:
            keep_rows.pop()
        i = max(i - 1, 0)
    elif choice == "q":
        break
    elif choice == "h":
        print(HELP)
    else:
        keep_rows.append(row)
        i += 1

print("\nФилтрирањето заврши.")
print(f"Задржани редови: {len(keep_rows)}")

df_keep = pd.DataFrame(keep_rows)
df_keep.to_csv(DST, index=False, encoding="utf-8-sig")
print(f"✅ Зачувано во {DST}")

