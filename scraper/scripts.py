import csv
import re
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path

BASE_URL = "https://pll.harvard.edu/catalog"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "courses.csv"
TOTAL_PAGES = 20
DELAY = 1.0  # seconds between requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

CSV_FIELDS = [
    "title",
    "url",
    "subject",
    "modality",
    "price",
    "duration",
    "registration",
    "description",
]


def fetch_page(page: int) -> BeautifulSoup:
    response = requests.get(BASE_URL, params={"page": page}, headers=HEADERS, timeout=15)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def clean(el) -> str:
    return " ".join(el.get_text().split()).strip() if el else ""


def parse_courses(soup: BeautifulSoup) -> list[dict]:
    courses = []

    for card in soup.select("div.views-row"):
        # Title & URL
        title_el = card.select_one("div.field--name-title a, h3 a")
        if not title_el:
            continue
        title = clean(title_el)
        href = title_el.get("href", "")
        url = f"https://pll.harvard.edu{href}" if href.startswith("/") else href

        # Subject tags
        subject_els = card.select(
            "div.topics--teaser a, "
            "div[class*='extra-field-pll-extra-field-subject'] a"
        )
        subject = ", ".join(clean(el) for el in subject_els)

        # Modality
        modality_el = card.select_one("div.field--name-field-modality .field__item")
        modality = clean(modality_el)

        # Price
        price_el = card.select_one(
            "div[class*='extra-field-pll-extra-field-price'] .field__item"
        )
        price = clean(price_el)

        # Registration deadline (prefer <time> text, fall back to container)
        reg_el = card.select_one(
            "div[class*='extra-field-pll-extra-field-registration-date'] time, "
            "div[class*='extra-field-pll-extra-field-registration-date'] .field__item"
        )
        registration = clean(reg_el)

        # Duration — stored in data-course-length attribute on the datalayer div
        # Raw value looks like "5weeks" or "3days"; insert space before unit.
        data_el = card.select_one("div.datalayer-values")
        duration = ""
        if data_el:
            raw = data_el.get("data-course-length", "").strip()
            duration = re.sub(r"(\d+)(week|day)", r"\1 \2", raw)

        # Description / summary
        desc_el = card.select_one(
            "div.field--name-field-summary p, "
            "div.text-teaser p"
        )
        description = clean(desc_el)

        courses.append(
            {
                "title": title,
                "url": url,
                "subject": subject,
                "modality": modality,
                "price": price,
                "duration": duration,
                "registration": registration,
                "description": description,
            }
        )

    return courses


def scrape_all() -> list[dict]:
    all_courses = []
    for page in range(TOTAL_PAGES):
        print(f"Scraping page {page + 1}/{TOTAL_PAGES} ...", flush=True)
        soup = fetch_page(page)
        courses = parse_courses(soup)
        print(f"  -> {len(courses)} courses found", flush=True)
        all_courses.extend(courses)
        if page < TOTAL_PAGES - 1:
            time.sleep(DELAY)
    return all_courses


def save_csv(courses: list[dict]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(courses)
    print(f"\nSaved {len(courses)} courses -> {OUTPUT_PATH}")


if __name__ == "__main__":
    courses = scrape_all()
    save_csv(courses)
