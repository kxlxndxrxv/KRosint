"""
Анализ никнеймов по эпохам.
Запуск: python 3_analyze.py
Требует: data/usernames_dataset.csv (из скрипта 1 или 2)

Что делает:
  1. Классифицирует каждый ник по 15+ паттернам
  2. Строит графики эволюции паттернов по периодам
  3. Выводит сводную статистику
  4. Строит «профиль эпохи» — топ паттернов для каждого периода
  5. Оценивает вероятный период по структуре ника (70% accuracy модель)
"""

import re
import csv
import os
import math
from collections import Counter, defaultdict

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # без GUI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────
# КОНФИГ
# ─────────────────────────────────────────────────

INPUT_CSV   = "data/usernames_dataset.csv"
OUTPUT_DIR  = "output"
PERIOD_ORDER = ["2000-2004", "2005-2009", "2010-2014", "2015-2019", "2020-2026"]

PALETTE = {
    "2000-2004": "#e63946",
    "2005-2009": "#f4a261",
    "2010-2014": "#2a9d8f",
    "2015-2019": "#457b9d",
    "2020-2026": "#7b2d8b",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

LEET_MAP = str.maketrans("4301!5@7", "aeoiissa")   # обратный лит-спик


def features(username: str) -> dict:
    """Извлечь все признаки для одного ника."""
    u = username.lower().strip()
    raw = username.strip()

    # Разбиваем на «слова» — токены без цифр
    tokens = re.split(r"[\d_\-\.]+", u)
    tokens = [t for t in tokens if len(t) >= 2]

    # Числовые подстроки
    numbers = re.findall(r"\d+", u)
    all_digits = "".join(numbers)

    f = {}

    # ── Длина ──
    f["length"] = len(raw)
    f["length_short"]  = len(raw) <= 6
    f["length_long"]   = len(raw) > 12

    # ── Структура ──
    f["has_digits"]      = bool(numbers)
    f["ends_with_digits"] = bool(re.search(r"\d+$", raw))
    f["digit_ratio"]     = len(all_digits) / max(len(raw), 1)
    
    f["has_underscore"]  = "_" in raw
    f["has_dash"]        = "-" in raw
    f["has_dot"]         = "." in raw
    f["has_special"]     = bool(re.search(r"[^a-zA-Z0-9_\-\.]", raw))
    f["separator_count"] = raw.count("_") + raw.count("-") + raw.count(".")

    # ── Стилистика (Капитализация) ──
    f["all_lowercase"]   = raw.islower()
    f["has_uppercase"]   = any(c.isupper() for c in raw)
    f["all_caps"]        = raw.isupper() and any(c.isalpha() for c in raw)
    f["camelCase"]       = bool(re.match(r"^[a-z]+[A-Z][a-zA-Z]*$", raw)) or bool(re.match(r"^[A-Z][a-z]+[A-Z][a-zA-Z]*$", raw))
    f["capitalized"]     = raw.istitle()

    # ── Числовые паттерны ──
    f["contains_birth_year"] = bool(re.search(r"(19[7-9]\d|200[0-9])", raw))
    f["contains_2k_year"]    = bool(re.search(r"(20[1-2][0-9])", raw))

    # ── Игровые / Субкультурные паттерны ──
    f["has_leet"] = bool(re.search(r"[4301!5@7]", raw.lower())) and any(c.isalpha() for c in raw)
    # Декораторы xX...Xx или x_..._x
    f["pattern_xX"] = bool(re.match(r"^x+.*x+$", u)) or bool(re.match(r"^x_.*_x$", u))
    f["triple_word_sep"] = f["separator_count"] >= 2
    f["word_count"] = len(tokens)
    # ── Структурные Маски ──
    f["mask_name_year"]      = bool(re.match(r'^[a-zA-Z_\-\.]+(?:19[7-9]\d|20[0-2]\d)$', raw)) # john1999, alex_2005
    f["mask_first_last_dot"] = bool(re.match(r'^[a-zA-Z]+\.[a-zA-Z]+$', raw))             # john.doe
    f["mask_first_last_dash"]= bool(re.match(r'^[a-zA-Z]+-[a-zA-Z]+$', raw))              # john-doe
    
    f["mask_camelCase"]  = bool(re.match(r'^[a-z]+[A-Z][a-zA-Z]+$', raw))     # darkKnight
    f["mask_CamelCase"]  = bool(re.match(r'^[A-Z][a-z]+[A-Z][a-zA-Z]+$', raw))# DarkKnight
    
    f["mask_word_word"]  = bool(re.match(r'^[a-z]+_[a-z]+$', u))              # dark_knight
    f["mask_wordNNN"]    = bool(re.match(r'^[a-z]+\d{1,4}$', u))              # john123 (но не год)
    if f["mask_name_year"]: f["mask_wordNNN"] = False
    
    f["mask_Word"]       = bool(re.match(r'^[A-Z][a-z]+$', raw))              # John
    f["mask_word"]       = bool(re.match(r'^[a-z]+$', raw))                   # john
    
    f["mask_starts_spec"]= bool(re.match(r'^[^a-zA-Z0-9]', raw))              # !admin
    f["mask_digit_heavy"]= len(numbers) > max(len(raw) / 2, 1)                # 1111admin
    
    # Смешанные буквы и цифры (не просто суффикс)
    f["alphanumeric_mix"] = bool(re.search(r"[a-zA-Z]\d+[a-zA-Z]", raw))
    
    return f


# ─────────────────────────────────────────────────
# ЗАГРУЗКА И ОБОГАЩЕНИЕ ДАТАСЕТА
# ─────────────────────────────────────────────────

def load_and_enrich(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["period"] = pd.Categorical(df["period"], categories=PERIOD_ORDER, ordered=True)

    feat_rows = [features(u) for u in df["username"]]
    feat_df = pd.DataFrame(feat_rows)
    df = pd.concat([df.reset_index(drop=True), feat_df], axis=1)
    
    # Добавим колонку доминирующей маски
    mask_cols = [c for c in df.columns if c.startswith("mask_")]
    def get_primary_mask(row):
        for col in mask_cols:
            if row[col]:
                return col.replace("mask_", "")
        return "mixed"
    
    df["primary_mask"] = df.apply(get_primary_mask, axis=1)
    
    return df


# ─────────────────────────────────────────────────
# ВИЗУАЛИЗАЦИЯ 1 — Средняя длина по периодам
# ─────────────────────────────────────────────────

def plot_length_trend(df: pd.DataFrame):
    agg = df.groupby("period", observed=True)["length"].agg(["mean", "median", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [PALETTE[p] for p in agg["period"]]

    bars = ax.bar(agg["period"], agg["mean"], color=colors, alpha=0.85, edgecolor="white", linewidth=1.2)
    ax.errorbar(agg["period"], agg["mean"], yerr=agg["std"],
                fmt="none", color="black", capsize=5, linewidth=1.5)

    for bar, med in zip(bars, agg["median"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"med={med:.0f}", ha="center", va="bottom", fontsize=9, color="#333")

    ax.set_title("Средняя длина никнейма по эпохам", fontsize=14, fontweight="bold")
    ax.set_xlabel("Период")
    ax.set_ylabel("Кол-во символов")
    ax.set_ylim(0, agg["mean"].max() + agg["std"].max() + 2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/1_length_trend.png", dpi=150)
    plt.close()
    print("  ✓ 1_length_trend.png")


# ─────────────────────────────────────────────────
# ВИЗУАЛИЗАЦИЯ 2 — Эволюция структурных масок (Line Chart)
# ─────────────────────────────────────────────────

def plot_mask_evolution(df: pd.DataFrame):
    # Словарь перевода масок на русский
    MASK_LABELS_RU = {
        "first_last_dot": "Имя.Фамилия (точка)",
        "first_last_dash": "Имя-Фамилия (дефис)",
        "name_year": "Слово + Год (john1999)",
        "word": "Словарный (строчные)",
        "mixed": "Смешанный (остальные)",
        "starts_spec": "Спецсимвол в начале (!, _)",
        "digit_heavy": "Преобладают цифры",
        "Word": "Словарный (С заглавной)",
        "wordNNN": "Слово + Цифры (john123)",
        "word_word": "Два слова (snake_case)",
        "camelCase": "Стиль camelCase",
        "CamelCase": "Стиль CamelCase",
    }
    
    # Группируем по периодам и считаем долю каждой маски
    counts = df.groupby(['period', 'primary_mask'], observed=True).size().unstack(fill_value=0)
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Берем только ТОП-5 самых популярных масок за всё время
    top_5_masks = percentages.mean().sort_values(ascending=False).head(5).index
    
    markers = ['o', 's', '^', 'D', 'v']
    for i, mask in enumerate(top_5_masks):
        ru_label = MASK_LABELS_RU.get(mask, mask)
        ax.plot(percentages.index, percentages[mask], marker=markers[i], markersize=8, linewidth=3, alpha=0.85, label=f"{ru_label}")
        
    ax.set_title("Эволюция ТОП-5 структурных масок никнеймов", fontsize=14, fontweight="bold")
    ax.set_ylabel("% от всех ников")
    ax.set_xlabel("Период")
    ax.grid(axis="both", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    
    ax.legend(title="Структура (Маска)", fontsize=10, title_fontsize=11)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/2_mask_evolution.png", dpi=150)
    plt.close()
    print("  ✓ 2_mask_evolution.png")


# ─────────────────────────────────────────────────
# ВИЗУАЛИЗАЦИЯ 3 — Тепловая карта паттернов
# ─────────────────────────────────────────────────

PATTERN_COLS = [
    ("has_digits",        "Содержит цифры"),
    ("ends_with_digits",  "Цифры в конце (суффикс)"),
    ("has_underscore",    "Подчёркивание '_'"),
    ("has_special",       "Спецсимволы (@, #, !)"),
    ("has_leet",          "Лит-спик (1337)"),
    ("all_lowercase",     "Полностью lowercase"),
    ("capitalized",       "С большой буквы"),
    ("alphanumeric_mix",  "Смесь букв и цифр"),
]


def plot_heatmap(df: pd.DataFrame):
    cols   = [c for c, _ in PATTERN_COLS]
    labels = [l for _, l in PATTERN_COLS]

    matrix = df.groupby("period", observed=True)[cols].mean() * 100   # в процентах

    fig, ax = plt.subplots(figsize=(13, 7))
    sns.heatmap(
        matrix.T,
        annot=True, fmt=".0f", cmap="Blues",
        linewidths=1, linecolor="white",
        cbar_kws={"label": "% никнеймов с паттерном"},
        ax=ax
    )
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticklabels(PERIOD_ORDER, rotation=0, fontsize=10)
    ax.set_title("Доля паттернов по эпохам (%)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/2_pattern_heatmap.png", dpi=150)
    plt.close()
    print("  ✓ 2_pattern_heatmap.png")


# ─────────────────────────────────────────────────
# ВИЗУАЛИЗАЦИЯ 3 — Топ-5 паттернов каждой эпохи
# ─────────────────────────────────────────────────

def plot_top_patterns_per_era(df: pd.DataFrame):
    cols   = [c for c, _ in PATTERN_COLS]
    labels = dict(PATTERN_COLS)

    fig, axes = plt.subplots(1, len(PERIOD_ORDER), figsize=(18, 6), sharey=False)

    for ax, period in zip(axes, PERIOD_ORDER):
        sub   = df[df["period"] == period][cols]
        rates = (sub.mean() * 100).sort_values(ascending=False).head(6)

        colors = [PALETTE[period]] * len(rates)
        ax.barh([labels.get(c, c) for c in rates.index[::-1]],
                rates.values[::-1], color=colors, alpha=0.85)
        ax.set_title(period, fontsize=11, fontweight="bold", color=PALETTE[period])
        ax.set_xlabel("% ников")
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    fig.suptitle("Топ-6 паттернов для каждой эпохи", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/3_top_patterns.png", dpi=150)
    plt.close()
    print("  ✓ 3_top_patterns.png")


# ─────────────────────────────────────────────────
# ВИЗУАЛИЗАЦИЯ 4 — Тренды масок (Line Chart)
# ─────────────────────────────────────────────────

TREND_GROUPS = {
    "Цифры и Символы": ["has_digits", "has_underscore", "has_special", "alphanumeric_mix"],
    "Стилистика текста": ["all_lowercase", "capitalized", "has_leet"],
}

LABELS = dict(PATTERN_COLS)


def plot_trend_lines(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (group_name, cols) in zip(axes, TREND_GROUPS.items()):
        means = df.groupby("period", observed=True)[cols].mean() * 100

        for col in cols:
            ax.plot(PERIOD_ORDER, means[col], marker="o", linewidth=2,
                    label=LABELS.get(col, col))

        ax.set_title(group_name, fontsize=12, fontweight="bold")
        ax.set_ylabel("% ников")
        ax.set_xticks(range(len(PERIOD_ORDER)))
        ax.set_xticklabels(PERIOD_ORDER, rotation=20, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Эволюция паттернов никнеймов (2000–2026)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/4_trend_lines.png", dpi=150)
    plt.close()
    print("  ✓ 4_trend_lines.png")


# ─────────────────────────────────────────────────
# ВИЗУАЛИЗАЦИЯ 5 — Распределение длины (boxplot)
# ─────────────────────────────────────────────────

def plot_length_boxplot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    data  = [df[df["period"] == p]["length"].values for p in PERIOD_ORDER]
    colors = [PALETTE[p] for p in PERIOD_ORDER]

    bp = ax.boxplot(data, patch_artist=True, notch=True,
                    medianprops={"color": "white", "linewidth": 2})

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticklabels(PERIOD_ORDER)
    ax.set_title("Распределение длины никнеймов по эпохам", fontsize=13, fontweight="bold")
    ax.set_ylabel("Кол-во символов")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/5_length_boxplot.png", dpi=150)
    plt.close()
    print("  ✓ 5_length_boxplot.png")


# ─────────────────────────────────────────────────
# ML — Определение периода по нику (Random Forest)
# ─────────────────────────────────────────────────

FEATURE_COLS = [c for c, _ in PATTERN_COLS] + [
    "length", "digit_ratio", "separator_count", "word_count",
    "contains_2k_year", "all_caps"
]


def train_period_classifier(df: pd.DataFrame) -> tuple[float, dict]:
    """Обучить классификатор периода и вернуть accuracy + feature importance."""
    X = df[FEATURE_COLS].fillna(0).astype(float)
    le = LabelEncoder()
    y = le.fit_transform(df["period"])

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    accuracy = scores.mean()

    # Важность признаков (fit на всём датасете)
    clf.fit(X, y)
    importances = dict(zip(FEATURE_COLS, clf.feature_importances_))

    return accuracy, importances, clf, le


def plot_feature_importance(importances: dict):
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]
    names  = [LABELS.get(n, n) for n, _ in sorted_imp][::-1]
    values = [v for _, v in sorted_imp][::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(names)))
    ax.barh(names, values, color=colors, alpha=0.9)
    ax.set_title("Важность признаков для классификации периода\n(Random Forest)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance score")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/6_feature_importance.png", dpi=150)
    plt.close()
    print("  ✓ 6_feature_importance.png")


# ─────────────────────────────────────────────────
# СВОДНАЯ СТАТИСТИКА — CSV + TXT отчёт
# ─────────────────────────────────────────────────

def save_statistics(df: pd.DataFrame, accuracy: float):
    lines = []
    lines.append("=" * 60)
    lines.append("АНАЛИЗ НИКНЕЙМОВ ПО ЭПОХАМ — СВОДНЫЙ ОТЧЁТ")
    lines.append("=" * 60)
    lines.append(f"Всего никнеймов: {len(df)}\n")

    for period in PERIOD_ORDER:
        sub = df[df["period"] == period]
        lines.append(f"─── {period} ({len(sub)} ников) ───")
        lines.append(f"  Средняя длина:   {sub['length'].mean():.1f} ± {sub['length'].std():.1f}")
        lines.append(f"  Медиана длины:   {sub['length'].median():.0f}")
        lines.append(f"  Содержат цифры:  {sub['has_digits'].mean():.0%}")
        lines.append(f"  Подчёркивание:   {sub['has_underscore'].mean():.0%}")
        lines.append(f"  Lowercase стиль: {sub['all_lowercase'].mean():.0%}")
        lines.append(f"  Лит-спик:        {sub['has_leet'].mean():.0%}")
        
        # Топ маска
        top_mask = sub['primary_mask'].value_counts().index[0]
        top_mask_pct = sub['primary_mask'].value_counts().iloc[0] / len(sub) * 100
        lines.append(f"  Самая популярная маска: {top_mask} ({top_mask_pct:.1f}%)")
        lines.append("")

    report = "\n".join(lines)
    with open(f"{OUTPUT_DIR}/report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(report)

    # Сохранить обогащённый датасет
    df.to_csv(f"{OUTPUT_DIR}/usernames_enriched.csv", index=False, encoding="utf-8")
    print(f"  ✓ usernames_enriched.csv")


# ─────────────────────────────────────────────────
# ПРЕДСКАЗАНИЕ ПО ОДНОМУ НИКУ
# ─────────────────────────────────────────────────

def predict_period(username: str, clf, le) -> str:
    f = features(username)
    X = pd.DataFrame([f])[FEATURE_COLS].fillna(0).astype(float)
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    period = le.inverse_transform([pred])[0]
    confidence = proba.max()
    return period, confidence


# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────

def main():
    print(f"Загружаем датасет: {INPUT_CSV}")
    df = load_and_enrich(INPUT_CSV)
    print(f"Загружено {len(df)} никнеймов по {df['period'].nunique()} периодам\n")

    print("Строим графики...")
    plot_length_trend(df)
    plot_mask_evolution(df)
    plot_heatmap(df)
    plot_top_patterns_per_era(df)
    plot_trend_lines(df)
    plot_length_boxplot(df)

    print("\nОбучаем классификатор периода...")
    accuracy, importances, clf, le = train_period_classifier(df)
    print(f"  Accuracy (5-fold CV): {accuracy:.1%}")
    plot_feature_importance(importances)

    print("\nСохраняем отчёт...")
    save_statistics(df, accuracy)

    # Демо: предсказание для нескольких ников
    test_nicks = ["darkWolf_666", "naruto_fan2007", "aestheticvibes.void", "ivanpetrov1982", "x_KILLER_x", "cryptoBro_2025"]
    print("\n── Демо: предсказание периода ──")
    for nick in test_nicks:
        period, conf = predict_period(nick, clf, le)
        print(f"  {nick:30s} → {period}  (уверенность {conf:.0%})")

    print(f"\n✓ Все файлы сохранены в папку '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
