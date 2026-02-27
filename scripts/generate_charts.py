"""
Harvard PLL Course Catalog — Business Intelligence Charts
Generates 8 charts saved to charts/ directory.
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "courses.csv"
OUT_DIR = ROOT / "charts"
OUT_DIR.mkdir(exist_ok=True)

# ── Brand palette ─────────────────────────────────────────────────────────────
CRIMSON  = "#A51C30"   # Harvard crimson – primary
NAVY     = "#1B3A5C"
SLATE    = "#4A6785"
STEEL    = "#7BA0C4"
TEAL     = "#2E9688"
AMBER    = "#D97B2B"
SAND     = "#E8C88A"
LIGHT    = "#F2F4F7"
GRAY     = "#9AA5B4"

MODALITY_COLORS = {
    "Online":      STEEL,
    "Online Live": TEAL,
    "In-Person":   CRIMSON,
    "Blended":     AMBER,
}

TIER_COLORS = {
    "Free":         TEAL,
    "Under $500":   STEEL,
    "$500–$2K":     SAND,
    "$2K–$10K":     AMBER,
    "$10K+":        CRIMSON,
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def save(name: str) -> None:
    path = OUT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved -> {path.name}")


def style_ax(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=14, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlabel(xlabel, fontsize=10, color=NAVY)
    ax.set_ylabel(ylabel, fontsize=10, color=NAVY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRAY)
    ax.tick_params(colors=NAVY, labelsize=9)
    ax.set_facecolor(LIGHT)


def parse_price(p) -> float:
    """Return numeric midpoint; 0 for free; NaN if unparseable."""
    s = str(p).strip()
    if s in ("Free*", "Free", "$0", "0", ""):
        return 0.0
    nums = re.findall(r"[\d]+", s.replace(",", ""))
    if not nums:
        return float("nan")
    vals = [int(n) for n in nums]
    return sum(vals) / len(vals)


def reg_category(r: str) -> str:
    s = str(r).strip()
    if s == "Available now":
        return "Available Now"
    if s.startswith("Register by"):
        return "Register by Date"
    if s.startswith("Starts"):
        return "Starting Soon"
    if s.startswith("Opens"):
        return "Opening Soon"
    return "Other"


# ── Load & enrich data ────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["price_num"] = df["price"].apply(parse_price)
    df["is_free"] = df["price_num"] == 0
    df["reg_cat"] = df["registration"].apply(reg_category)
    df["is_ai"] = df["title"].str.contains(
        r"\bAI\b|Artificial Intelligence|Machine Learning|Data Science",
        case=False, na=False
    )

    def tier(p):
        if p == 0:           return "Free"
        if p < 500:          return "Under $500"
        if p < 2000:         return "$500–$2K"
        if p < 10000:        return "$2K–$10K"
        return "$10K+"

    df["price_tier"] = df["price_num"].apply(tier)

    # Collapse small subjects for readability
    keep = ["Business", "Health & Medicine", "Social Sciences",
            "Data Science", "Humanities", "Computer Science",
            "Art & Design", "Education & Teaching"]
    df["subject_grp"] = df["subject"].apply(
        lambda s: s if s in keep else "Other"
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Chart 1 – Course Volume by Subject
# ══════════════════════════════════════════════════════════════════════════════
def chart_volume(df: pd.DataFrame) -> None:
    counts = (
        df["subject_grp"]
        .value_counts()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor(LIGHT)
    bars = ax.barh(counts.index, counts.values, color=CRIMSON, height=0.6, zorder=2)
    ax.bar_label(bars, fmt="%d", padding=4, fontsize=9, color=NAVY)
    ax.grid(axis="x", color="white", linewidth=1.2, zorder=1)
    ax.set_xlim(0, counts.max() * 1.15)
    style_ax(ax,
             "Course Catalog Volume by Subject Area",
             "Number of Courses")
    fig.suptitle(
        f"Total catalog: {len(df):,} courses across {df['subject'].nunique()} subjects",
        fontsize=9, color=GRAY, y=0.01
    )
    plt.tight_layout()
    save("01_course_volume_by_subject.png")


# ══════════════════════════════════════════════════════════════════════════════
# Chart 2 – Delivery Modality Mix by Subject (stacked %)
# ══════════════════════════════════════════════════════════════════════════════
def chart_modality_mix(df: pd.DataFrame) -> None:
    order = df.groupby("subject_grp")["title"].count().sort_values(ascending=False).index
    pivot = (
        df.groupby(["subject_grp", "modality"])
        .size()
        .unstack(fill_value=0)
        .loc[order]
    )
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor(LIGHT)
    bottom = np.zeros(len(pivot_pct))
    for mod in ["Online", "Online Live", "In-Person", "Blended"]:
        if mod not in pivot_pct.columns:
            continue
        vals = pivot_pct[mod].values
        bars = ax.bar(pivot_pct.index, vals, bottom=bottom,
                      label=mod, color=MODALITY_COLORS[mod], width=0.6, zorder=2)
        for bar, v in zip(bars, vals):
            if v >= 8:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.0f}%", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold"
                )
        bottom += vals

    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(axis="y", color="white", linewidth=1.2, zorder=1)
    ax.legend(title="Delivery Mode", bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=9, frameon=False)
    ax.set_xticks(range(len(pivot_pct.index)))
    ax.set_xticklabels(pivot_pct.index, rotation=30, ha="right")
    style_ax(ax,
             "Delivery Modality Mix by Subject Area",
             ylabel="Share of Courses (%)")
    plt.tight_layout()
    save("02_modality_mix_by_subject.png")


# ══════════════════════════════════════════════════════════════════════════════
# Chart 3 – Free vs. Paid Split by Subject (stacked count)
# ══════════════════════════════════════════════════════════════════════════════
def chart_free_vs_paid(df: pd.DataFrame) -> None:
    order = df.groupby("subject_grp")["title"].count().sort_values(ascending=False).index
    pivot = (
        df.groupby(["subject_grp", "is_free"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={True: "Free", False: "Paid"})
        .loc[order]
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor(LIGHT)
    cols = {"Paid": NAVY, "Free": TEAL}
    bottom = np.zeros(len(pivot))
    for label, color in cols.items():
        if label not in pivot.columns:
            continue
        vals = pivot[label].values
        bars = ax.bar(pivot.index, vals, bottom=bottom,
                      label=label, color=color, width=0.6, zorder=2)
        for bar, v in zip(bars, vals):
            if v >= 3:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(int(v)), ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold"
                )
        bottom += vals

    ax.grid(axis="y", color="white", linewidth=1.2, zorder=1)
    ax.legend(title="Pricing", bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=9, frameon=False)
    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    style_ax(ax,
             "Free vs. Paid Course Split by Subject Area",
             ylabel="Number of Courses")
    plt.tight_layout()
    save("03_free_vs_paid_by_subject.png")


# ══════════════════════════════════════════════════════════════════════════════
# Chart 4 – Price Tier Breakdown by Subject (stacked %)
# ══════════════════════════════════════════════════════════════════════════════
def chart_price_tiers(df: pd.DataFrame) -> None:
    tier_order = ["Free", "Under $500", "$500–$2K", "$2K–$10K", "$10K+"]
    subj_order = df.groupby("subject_grp")["title"].count().sort_values(ascending=False).index

    pivot = (
        df.groupby(["subject_grp", "price_tier"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[t for t in tier_order if t in df["price_tier"].unique()],
                 fill_value=0)
        .loc[subj_order]
    )
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor(LIGHT)
    bottom = np.zeros(len(pivot_pct))
    for tier in tier_order:
        if tier not in pivot_pct.columns:
            continue
        vals = pivot_pct[tier].values
        bars = ax.bar(pivot_pct.index, vals, bottom=bottom,
                      label=tier, color=TIER_COLORS[tier], width=0.6, zorder=2)
        for bar, v in zip(bars, vals):
            if v >= 8:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.0f}%", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold"
                )
        bottom += vals

    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(axis="y", color="white", linewidth=1.2, zorder=1)
    ax.legend(title="Price Tier", bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=9, frameon=False)
    ax.set_xticks(range(len(pivot_pct.index)))
    ax.set_xticklabels(pivot_pct.index, rotation=30, ha="right")
    style_ax(ax,
             "Price Tier Distribution by Subject Area",
             ylabel="Share of Courses (%)")
    plt.tight_layout()
    save("04_price_tiers_by_subject.png")


# ══════════════════════════════════════════════════════════════════════════════
# Chart 5 – Median Price by Modality (paid courses only)
# ══════════════════════════════════════════════════════════════════════════════
def chart_price_by_modality(df: pd.DataFrame) -> None:
    paid = df[df["price_num"] > 0]
    stats = (
        paid.groupby("modality")["price_num"]
        .agg(median="median", q25=lambda x: x.quantile(0.25),
             q75=lambda x: x.quantile(0.75), count="count")
        .sort_values("median", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor(LIGHT)
    colors = [MODALITY_COLORS.get(m, SLATE) for m in stats.index]
    bars = ax.bar(stats.index, stats["median"], color=colors, width=0.5, zorder=2)

    # IQR whiskers
    for bar, (_, row) in zip(bars, stats.iterrows()):
        x = bar.get_x() + bar.get_width() / 2
        ax.plot([x, x], [row["q25"], row["q75"]],
                color=NAVY, linewidth=2.5, zorder=3)
        ax.plot([x - 0.07, x + 0.07], [row["q25"], row["q25"]],
                color=NAVY, linewidth=2, zorder=3)
        ax.plot([x - 0.07, x + 0.07], [row["q75"], row["q75"]],
                color=NAVY, linewidth=2, zorder=3)

    ax.bar_label(bars, labels=[f"${v:,.0f}" for v in stats["median"]],
                 padding=6, fontsize=10, fontweight="bold", color=NAVY)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(axis="y", color="white", linewidth=1.2, zorder=1)
    ax.set_ylim(0, stats["q75"].max() * 1.4)

    # n= labels below bars
    for i, (mod, row) in enumerate(stats.iterrows()):
        ax.text(i, -stats["median"].max() * 0.06,
                f"n={int(row['count'])}", ha="center", fontsize=8, color=GRAY)

    style_ax(ax,
             "Median Course Price by Delivery Modality",
             ylabel="Median Price (USD)  |  bars = IQR range")
    fig.suptitle("Paid courses only · IQR bars show middle 50% price range",
                 fontsize=8, color=GRAY, y=0.01)
    plt.tight_layout()
    save("05_median_price_by_modality.png")


# ══════════════════════════════════════════════════════════════════════════════
# Chart 6 – Median Price by Subject (paid courses, top 8)
# ══════════════════════════════════════════════════════════════════════════════
def chart_price_by_subject(df: pd.DataFrame) -> None:
    paid = df[df["price_num"] > 0]
    stats = (
        paid.groupby("subject_grp")["price_num"]
        .agg(median="median", count="count")
        .query("count >= 5")
        .sort_values("median", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor(LIGHT)
    bars = ax.barh(stats.index, stats["median"], color=NAVY, height=0.6, zorder=2)
    ax.bar_label(bars, labels=[f"${v:,.0f}" for v in stats["median"]],
                 padding=5, fontsize=9, color=NAVY)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(axis="x", color="white", linewidth=1.2, zorder=1)
    ax.set_xlim(0, stats["median"].max() * 1.2)
    style_ax(ax,
             "Median Paid Course Price by Subject Area",
             xlabel="Median Price (USD)")
    fig.suptitle("Subjects with ≥ 5 paid courses · mid-range of each price band used for ranges",
                 fontsize=8, color=GRAY, y=0.01)
    plt.tight_layout()
    save("06_median_price_by_subject.png")


# ══════════════════════════════════════════════════════════════════════════════
# Chart 7 – AI Course Penetration by Subject
# ══════════════════════════════════════════════════════════════════════════════
def chart_ai_penetration(df: pd.DataFrame) -> None:
    grp = df.groupby("subject_grp").agg(
        total=("title", "count"),
        ai=("is_ai", "sum")
    )
    grp["non_ai"] = grp["total"] - grp["ai"]
    grp["ai_pct"] = grp["ai"] / grp["total"] * 100
    grp = grp.sort_values("ai_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor(LIGHT)
    ax.barh(grp.index, grp["non_ai"], color=SLATE, height=0.6,
            label="Standard Courses", zorder=2)
    bars = ax.barh(grp.index, grp["ai"], left=grp["non_ai"],
                   color=AMBER, height=0.6, label="AI-Related Courses", zorder=2)

    for bar, (_, row) in zip(bars, grp.iterrows()):
        if row["ai"] > 0:
            pct_label = f"{row['ai_pct']:.0f}%"
            ax.text(row["non_ai"] + row["ai"] + 2, bar.get_y() + bar.get_height() / 2,
                    pct_label, va="center", fontsize=9, color=AMBER, fontweight="bold")

    ax.grid(axis="x", color="white", linewidth=1.2, zorder=1)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    style_ax(ax,
             "AI-Related Course Penetration by Subject Area",
             xlabel="Number of Courses")
    fig.suptitle(
        f"AI-related = titles containing 'AI', 'Artificial Intelligence', 'Machine Learning', or 'Data Science'",
        fontsize=8, color=GRAY, y=0.01
    )
    plt.tight_layout()
    save("07_ai_penetration_by_subject.png")


# ══════════════════════════════════════════════════════════════════════════════
# Chart 8 – Registration / Availability Status
# ══════════════════════════════════════════════════════════════════════════════
def chart_availability(df: pd.DataFrame) -> None:
    STATUS_COLORS = {
        "Available Now":    TEAL,
        "Register by Date": AMBER,
        "Starting Soon":    CRIMSON,
        "Opening Soon":     SLATE,
        "Other":            GRAY,
    }
    order = df.groupby("subject_grp")["title"].count().sort_values(ascending=False).index
    pivot = (
        df.groupby(["subject_grp", "reg_cat"])
        .size()
        .unstack(fill_value=0)
        .loc[order]
    )
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    status_order = [s for s in STATUS_COLORS if s in pivot_pct.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor(LIGHT)
    bottom = np.zeros(len(pivot_pct))
    for status in status_order:
        vals = pivot_pct[status].values
        bars = ax.bar(pivot_pct.index, vals, bottom=bottom,
                      label=status, color=STATUS_COLORS[status], width=0.6, zorder=2)
        for bar, v in zip(bars, vals):
            if v >= 8:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:.0f}%", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold"
                )
        bottom += vals

    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.grid(axis="y", color="white", linewidth=1.2, zorder=1)
    ax.legend(title="Enrollment Status", bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=9, frameon=False)
    ax.set_xticks(range(len(pivot_pct.index)))
    ax.set_xticklabels(pivot_pct.index, rotation=30, ha="right")
    style_ax(ax,
             "Course Enrollment Availability Status by Subject Area",
             ylabel="Share of Courses (%)")
    plt.tight_layout()
    save("08_availability_status_by_subject.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"  {len(df):,} courses | {df['subject'].nunique()} subjects\n")

    charts = [
        ("01 Course Volume by Subject",             chart_volume),
        ("02 Modality Mix by Subject",              chart_modality_mix),
        ("03 Free vs. Paid Split",                  chart_free_vs_paid),
        ("04 Price Tier Breakdown",                 chart_price_tiers),
        ("05 Median Price by Modality",             chart_price_by_modality),
        ("06 Median Price by Subject",              chart_price_by_subject),
        ("07 AI Course Penetration",                chart_ai_penetration),
        ("08 Enrollment Availability Status",       chart_availability),
    ]

    for label, fn in charts:
        print(f"Generating: {label}")
        fn(df)

    print(f"\nAll charts saved to: {OUT_DIR}")
