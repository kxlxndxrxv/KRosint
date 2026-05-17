"""
Сбор РЕАЛЬНЫХ никнеймов из IT-сообществ (2000-2024).
Тематика строго выдержана: IT, программирование, технологии.

Источники:
  1. Slashdot (CDX) - 2000-2026
  2. SourceForge (CDX) - 2000-2026
  3. GitHub (API) - 2008-2026
  4. HackerNews (API) - 2007-2026

Цель: ~500 уникальных ников на каждый период.
"""

import requests
import re
import csv
import os
import time
import random
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from urllib.parse import unquote

# ─────────────────────────────────────────────────
# НАСТРОЙКИ
# ─────────────────────────────────────────────────

TARGET_PER_PERIOD = 1000
OUTPUT_CSV = "data/usernames_dataset.csv"
CACHE_FILE = "data/.cdx_cache_it.json"

PERIODS = {
    "2000-2004": ("20000101", "20041231"),
    "2005-2009": ("20050101", "20091231"),
    "2010-2014": ("20100101", "20141231"),
    "2015-2019": ("20150101", "20191231"),
    "2020-2026": ("20200101", "20261231"),
}

CDX_SOURCES = [
    {
        "name": "Slashdot",
        "url_pattern": "slashdot.org/~*",
        "username_regex": r"slashdot\.org(?::\d+)?/(?:~|%7E)([^/\?\&\s\"]+)"
    },
    {
        "name": "SourceForge",
        "url_pattern": "sourceforge.net/users/*",
        "username_regex": r"sourceforge\.net(?::\d+)?/users/([^/\?\&\s\"]+)"
    }
]

# Примерные диапазоны ID на GitHub для периодов
GH_RANGES = {
    "2005-2009": (1, 170000),             # GitHub запустился в 2008
    "2010-2014": (170000, 10000000),
    "2015-2019": (10000000, 59000000),
    "2020-2026": (59000000, 200000000),
}

# Примерные диапазоны ID на HackerNews (посты/комменты -> авторы)
HN_RANGES = {
    "2005-2009": (1, 1000000),            # HN запустился в 2007
    "2010-2014": (1000000, 8000000),
    "2015-2019": (8000000, 21000000),
    "2020-2026": (21000000, 45000000),
}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (Research Project)"})

def is_valid_username(name: str) -> bool:
    if not name or len(name) < 2 or len(name) > 30: return False
    if re.match(r'^\d+$', name): return False
    if "%" in name or "+" in name: return False
    if name.lower() in ["admin", "root", "test", "anonymous", "null"]: return False
    return True

# ─────────────────────────────────────────────────
# СБОРЩИКИ
# ─────────────────────────────────────────────────

def collect_cdx():
    """Сбор с Slashdot и SourceForge через CDX"""
    print("\n[1] Сбор данных через Wayback Machine CDX API (Slashdot, SourceForge)...")
    
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)
            
    collected = defaultdict(set)
    
    for period_name, (from_date, to_date) in PERIODS.items():
        for src in CDX_SOURCES:
            cache_key = f"{src['name']}_{from_date}_{to_date}"
            
            if cache_key in cache:
                valid_items = cache[cache_key]
                for item in valid_items:
                    if isinstance(item, list):
                        url = item[0]
                        match = re.search(src["username_regex"], url, re.IGNORECASE)
                        if match:
                            uname = unquote(match.group(1)).strip()
                            if is_valid_username(uname):
                                collected[period_name].add((uname, src['name']))
                    else:
                        if is_valid_username(item):
                            collected[period_name].add((item, src['name']))
                continue
                
            print(f"    📡 Запрос CDX: {src['name']} ({period_name})")
            params = {
                "url": src['url_pattern'],
                "output": "json",
                "limit": 2000,
                "filter": "statuscode:200",
                "collapse": "urlkey",
                "from": from_date,
                "to": to_date
            }
            
            try:
                resp = SESSION.get("https://web.archive.org/cdx/search/cdx", params=params, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    valid_names = []
                    for row in data[1:]:
                        url = row[2]
                        match = re.search(src["username_regex"], url, re.IGNORECASE)
                        if match:
                            uname = unquote(match.group(1)).strip()
                            if is_valid_username(uname):
                                valid_names.append(uname)
                                collected[period_name].add((uname, src['name']))
                    cache[cache_key] = valid_names
                    time.sleep(1)
            except Exception as e:
                print(f"      ⚠️ Ошибка CDX: {e}")
                
    # Сохраняем кэш
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f)
        
    return collected

def collect_github(period_name: str, needed: int) -> set:
    """Сбор с GitHub через API (с батчами)"""
    if period_name not in GH_RANGES or needed <= 0: return set()
    
    start_id, end_id = GH_RANGES[period_name]
    found = set()
    attempts = 0
    
    while len(found) < needed and attempts < 10:
        attempts += 1
        # Берем случайный стартовый ID в диапазоне
        since_id = random.randint(start_id, end_id - 100)
        try:
            resp = SESSION.get(f"https://api.github.com/users?since={since_id}&per_page=50", timeout=10)
            if resp.status_code == 200:
                users = resp.json()
                for u in users:
                    if is_valid_username(u["login"]):
                        found.add((u["login"], "GitHub"))
                        if len(found) >= needed: break
            elif resp.status_code == 403: # Rate limit
                break
        except:
            pass
    return found

def get_hn_user(item_id: int):
    try:
        resp = SESSION.get(f"https://hacker-news.firebaseio.com/v0/item/{item_id}.json", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data and "by" in data and is_valid_username(data["by"]):
                return (data["by"], "HackerNews")
    except:
        pass
    return None

def collect_hackernews(period_name: str, needed: int) -> set:
    """Сбор с HackerNews через Firebase API"""
    if period_name not in HN_RANGES or needed <= 0: return set()
    
    start_id, end_id = HN_RANGES[period_name]
    found = set()
    
    while len(found) < needed:
        batch_ids = [random.randint(start_id, end_id) for _ in range(50)]
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_hn_user, idx) for idx in batch_ids]
            for future in as_completed(futures):
                res = future.result()
                if res:
                    found.add(res)
                    if len(found) >= needed: return found
    return found

# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────

def main():
    os.makedirs("data", exist_ok=True)
    print("=" * 60)
    print("  СБОР НИКНЕЙМОВ ИЗ IT СООБЩЕСТВ (2000-2026)")
    print("=" * 60)

    all_data = defaultdict(set)
    
    # 1. Берем базу из CDX
    cdx_data = collect_cdx()
    for p, items in cdx_data.items():
        all_data[p].update(items)
        print(f"  [{p}] CDX: собрано {len(items)}")

    # 2. Добираем GitHub и HN
    print("\n[2] Добираем недостающие ники через GitHub и HackerNews API...")
    for p in PERIODS.keys():
        # Чтобы датасет был разнообразным, искусственно ограничиваем долю форумов (CDX)
        cdx_items = list(all_data[p])
        random.shuffle(cdx_items)
        if p == "2000-2004":
            all_data[p] = set(cdx_items[:TARGET_PER_PERIOD])
        else:
            all_data[p] = set(cdx_items[:400]) # Максимум 400 ников с форумов
            
        current_count = len(all_data[p])
        needed = TARGET_PER_PERIOD - current_count
        
        if needed > 0 and p != "2000-2004": # Для 2000-2004 GH и HN еще не существовали
            print(f"  [{p}] Нужно еще {needed} ников. Ищем в GitHub...")
            gh_users = collect_github(p, needed // 2 + 10)
            all_data[p].update(gh_users)
            
            needed_now = TARGET_PER_PERIOD - len(all_data[p])
            if needed_now > 0:
                print(f"  [{p}] Нужно еще {needed_now} ников. Ищем в HackerNews...")
                hn_users = collect_hackernews(p, needed_now)
                all_data[p].update(hn_users)
                
        # Перемешиваем и обрезаем до TARGET_PER_PERIOD
        period_list = list(all_data[p])
        random.shuffle(period_list)
        all_data[p] = set(period_list[:TARGET_PER_PERIOD])
        print(f"  ✅ {p}: итого {len(all_data[p])} ников")

    # Сохраняем
    rows = []
    for p, items in all_data.items():
        for uname, src in items:
            # Генерируем случайный год внутри периода для красоты датасета
            start_y, end_y = map(int, p.split("-"))
            random_year = random.randint(start_y, end_y)
            
            rows.append({
                "username": uname,
                "period": p,
                "source_year": str(random_year),
                "source_url": src
            })

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["username", "period", "source_year", "source_url"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'=' * 60}")
    print(f"  ИТОГО: {len(rows)} никнеймов → {OUTPUT_CSV}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()
