#!/usr/bin/env python3
"""prepare_data.py -- build Sextant training / calibration / dev records from public datasets.

Downloads permissively licensed public datasets (Hugging Face Hub + two GitHub upstreams),
verifies each dataset's license from its card / LICENSE file at download time, and converts a
seeded, label-balanced subsample into the project's JSONL record format under
``data/processed/<dataset>/{train,calib,dev}.jsonl``.  It also appends one provenance line per
dataset to ``data/provenance.jsonl`` and regenerates ``data/README.md``.

Nothing here touches any benchmark corpus: the ``FORBIDDEN_SOURCES`` list below is a hard guard
that refuses to run a dataset whose source id matches a benchmark-contaminated corpus.

Usage
-----
    python3 scripts/prepare_data.py                      # everything
    python3 scripts/prepare_data.py --datasets snips,snli --limit 20   # quick smoke test
    python3 scripts/prepare_data.py --list               # show dataset keys
    python3 scripts/prepare_data.py --verify-only        # re-validate existing outputs

Requirements: ``pip install requests huggingface_hub pandas pyarrow``.

Idempotent: raw files are cached in ``data/raw/`` (skipped when present), outputs are rewritten
deterministically (fixed seed), and the provenance file replaces the entry of a re-run dataset.
Large parquet shards are NOT downloaded in full: the script reads an evenly spaced subset of
their row groups over HTTP range requests and caches that subset locally.
"""
from __future__ import annotations

import argparse
import ast
import collections
import csv
import datetime as dt
import hashlib
import io
import json
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

try:
    import pandas as pd
    import pyarrow.parquet as pq
    import requests
    from huggingface_hub import HfApi
except ImportError as exc:  # pragma: no cover
    sys.exit(f"missing dependency: {exc}. Run: pip install requests huggingface_hub pandas pyarrow")

# --------------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------------
SEED = 20260922
CAPS = {"train": 1200, "calib": 400, "dev": 400}
MAX_STATE_CHARS = 4000
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
OUT_DIR = REPO_ROOT / "data" / "processed"
PROVENANCE_PATH = REPO_ROOT / "data" / "provenance.jsonl"
README_PATH = REPO_ROOT / "data" / "README.md"
MODEL = "sextant-1"
TIER = "external_dev"
QID = "q"
HTTP_TIMEOUT = 120
USER_AGENT = "sextant-prepare-data/1.0 (+https://github.com/; python-requests)"

# Card license tags we accept as permissive (Hugging Face `license:` front-matter ids).
PERMISSIVE_TAGS = {
    "cc-by-2.0", "cc-by-3.0", "cc-by-4.0", "cc-by-sa-3.0", "cc-by-sa-4.0", "cc0-1.0",
    "mit", "apache-2.0", "bsd", "bsd-2-clause", "bsd-3-clause", "cdla-sharing-1.0", "odc-by",
}

# Benchmark-contamination guard: any source id containing one of these fragments is refused.
FORBIDDEN_SOURCES = [
    "banking77", "clinc", "massive", "lex_glue", "ledgar", "go_emotions", "mmlu", "ai2_arc",
    "multi_nli", "mnli", "/glue", "chaos", "sst", "rotten_tomatoes", "yelp", "helpsteer2",
    "stsb", "sts_benchmark", "stsbenchmark", "measuring-hate-speech", "boolq", "fever",
    "vitaminc", "paws", "civil_comments", "jigsaw", "sms_spam", "strategyqa", "jev-bench",
    "jevbench", "jev_bench", "system-one", "auto-model-router",
]

# --------------------------------------------------------------------------------------------
# authored option descriptions (written for this project; not part of the source datasets)
# --------------------------------------------------------------------------------------------
HWU64_INTENTS = {
    "alarm_query": "Ask about existing alarms: which alarms are set or when they will go off",
    "alarm_remove": "Cancel or delete an existing alarm",
    "alarm_set": "Set a new alarm or wake-up call for a given time",
    "audio_volume_down": "Lower the speaker volume",
    "audio_volume_mute": "Mute or silence the audio",
    "audio_volume_up": "Raise the speaker volume",
    "calendar_query": "Ask about calendar events, the schedule or upcoming appointments",
    "calendar_remove": "Cancel or delete a calendar event, meeting or reminder",
    "calendar_set": "Create a calendar event, meeting or reminder",
    "cooking_recipe": "Ask for a recipe or how to cook a dish",
    "datetime_convert": "Convert a time between time zones or ask what time it is elsewhere",
    "datetime_query": "Ask the current time or date, or the date or day of an event",
    "email_addcontact": "Add a new contact or email address to the address book",
    "email_query": "Check the inbox, ask about received emails or have them read out",
    "email_querycontact": "Look up a contact's details such as their email address or phone number",
    "email_sendemail": "Send, compose or reply to an email",
    "general_affirm": "Affirmative reply: yes, correct, that is right",
    "general_commandstop": "Tell the assistant to stop, cancel what it is doing or be quiet",
    "general_confirm": "Ask the assistant to confirm that it understood or did something",
    "general_dontcare": "Express indifference: any option is fine, you choose",
    "general_explain": "Ask the assistant to explain, clarify or elaborate on what it said",
    "general_joke": "Ask the assistant to tell a joke or say something funny",
    "general_negate": "Negative reply: no, that is wrong, that is not what I meant",
    "general_praise": "Praise or thank the assistant: good job, thanks, well done",
    "general_quirky": "Small talk, chit-chat or quirky questions to the assistant (how are you, what is your favourite colour)",
    "general_repeat": "Ask the assistant to repeat what it just said",
    "iot_cleaning": "Start or control the robot vacuum cleaner",
    "iot_coffee": "Make or brew coffee with the smart coffee machine",
    "iot_hue_lightchange": "Change the colour of the smart lights",
    "iot_hue_lightdim": "Dim the smart lights, make them less bright",
    "iot_hue_lightoff": "Turn the smart lights off",
    "iot_hue_lighton": "Turn the smart lights on",
    "iot_hue_lightup": "Brighten the smart lights, increase their brightness",
    "iot_wemo_off": "Turn off a smart plug or the appliance connected to it",
    "iot_wemo_on": "Turn on a smart plug or the appliance connected to it",
    "lists_createoradd": "Create a new list or add an item to a list (shopping list, to-do list)",
    "lists_query": "Ask what is on a list or have a list read out",
    "lists_remove": "Remove an item from a list or delete a list",
    "music_likeness": "Say that you like the music playing or ask to save or favourite it",
    "music_query": "Ask about the music that is playing (which song, artist or album)",
    "music_settings": "Change music playback settings such as shuffle or repeat",
    "news_query": "Ask for the news headlines or news about a topic",
    "play_audiobook": "Play or resume an audiobook",
    "play_game": "Play a game with the assistant",
    "play_music": "Play a song, artist, album, genre or playlist",
    "play_podcasts": "Play or resume a podcast episode",
    "play_radio": "Play a radio station",
    "qa_currency": "Ask about currency exchange rates or convert an amount of money",
    "qa_definition": "Ask for the definition or meaning of a word or thing",
    "qa_factoid": "Ask a general factual question (who, what, where, how many)",
    "qa_maths": "Ask for a mathematical calculation to be done",
    "qa_stock": "Ask about a stock price or the value of shares",
    "recommendation_events": "Ask for recommendations of events or things to do nearby",
    "recommendation_locations": "Ask for recommendations of places such as restaurants, shops or bars",
    "recommendation_movies": "Ask for movie recommendations",
    "social_post": "Post a message, complaint or update on social media",
    "social_query": "Check social media notifications or ask what people are posting",
    "takeaway_order": "Order takeaway food or book a food delivery",
    "takeaway_query": "Ask about the status of a takeaway order or whether a place delivers",
    "transport_query": "Ask about train or bus times, routes or directions",
    "transport_taxi": "Book or call a taxi or ride",
    "transport_ticket": "Book or buy a train, bus or plane ticket",
    "transport_traffic": "Ask about traffic conditions on a route",
    "weather_query": "Ask about the weather or the forecast",
}
assert len(HWU64_INTENTS) == 64

SNIPS_INTENTS = {
    "AddToPlaylist": "Add a song, album or artist to a playlist",
    "BookRestaurant": "Book a table or reservation at a restaurant",
    "GetWeather": "Ask about the weather or forecast for a place or time",
    "PlayMusic": "Play music: a song, album, artist, genre or playlist, possibly on a named service",
    "RateBook": "Give a rating to a book, novel, essay or other written work",
    "SearchCreativeWork": "Find a creative work such as a movie, TV show, song, book, game or painting by name",
    "SearchScreeningEvent": "Find movie showtimes, screenings or where a film is playing",
}

DBPEDIA_CLASSES = [  # order = ClassLabel order on the card
    ("Company", "A business, corporation, firm or other commercial organization"),
    ("EducationalInstitution", "A school, college, university or other educational institution"),
    ("Artist", "A person known for creative work: musician, band member, singer, painter, actor, writer or other artist"),
    ("Athlete", "A sportsperson: player, competitor or athlete in any sport"),
    ("OfficeHolder", "A politician or public official who holds or held an office (mayor, senator, judge, minister)"),
    ("MeanOfTransportation", "A vehicle or craft: ship, aircraft, car model, locomotive, spacecraft"),
    ("Building", "A building or man-made structure: church, house, hotel, stadium, station, museum"),
    ("NaturalPlace", "A natural geographic feature: river, mountain, lake, island, glacier, valley"),
    ("Village", "A village or small rural settlement"),
    ("Animal", "An animal species or genus: mammal, bird, fish, insect, mollusc, reptile"),
    ("Plant", "A plant species or genus: tree, flower, herb, shrub, grass"),
    ("Album", "A music album or recorded collection of songs"),
    ("Film", "A movie or film"),
    ("WrittenWork", "A written work: book, novel, magazine, journal, newspaper, comic, poem"),
]

SNLI_LABELS = ["entailment", "neutral", "contradiction"]  # ClassLabel order on the card
SNLI_OPTIONS = {
    "entailment": "Reading the premise guarantees that the hypothesis holds",
    "neutral": "The hypothesis might be true: the premise neither confirms nor contradicts it",
    "contradiction": "The premise makes the hypothesis impossible",
}

AMAZON_LABELS = ["negative", "positive"]  # ClassLabel order on the card
AMAZON_OPTIONS = {
    "positive": "The reviewer is satisfied: the review praises or recommends the product (4-5 stars)",
    "negative": "The reviewer is dissatisfied: the review criticizes or warns against the product (1-2 stars)",
}

TWEET_LABELS = ["negative", "neutral", "positive"]  # ClassLabel order on the card
TWEET_OPTIONS = {
    "negative": "The tweet expresses a negative opinion or feeling",
    "neutral": "The tweet is neutral or factual, with no clear positive or negative sentiment",
    "positive": "The tweet expresses a positive opinion or feeling",
}

PROSOCIAL_LEVELS = [  # dataset's ordered safety_label scale (index = score level)
    ("__casual__", "Casual: a benign, everyday utterance that raises no safety concern"),
    ("__possibly_needs_caution__", "Possibly needs caution: mildly questionable; one of three annotators flagged it"),
    ("__probably_needs_caution__", "Probably needs caution: questionable; two of three annotators flagged it"),
    ("__needs_caution__", "Needs caution: clearly problematic, offensive or unethical; all annotators flagged it"),
    ("__needs_intervention__", "Needs intervention: potentially harmful or dangerous; a response must intervene"),
]

BITEXT_INTENTS = {
    "cancel_order": "Cancel an order that was placed",
    "change_order": "Change or modify the items or details of an existing order",
    "change_shipping_address": "Change the shipping or delivery address of an order",
    "check_cancellation_fee": "Ask about the fees charged for cancelling",
    "check_invoice": "Ask about or look up an invoice",
    "check_payment_methods": "Ask which payment methods are accepted",
    "check_refund_policy": "Ask about the refund policy or the conditions for a refund",
    "complaint": "File a complaint about the company or its service",
    "contact_customer_service": "Ask how to contact customer service (hours, phone, email)",
    "contact_human_agent": "Ask to talk to a human agent or a live person",
    "create_account": "Open or register a new account",
    "delete_account": "Delete, close or remove an account",
    "delivery_options": "Ask about the available delivery or shipping options",
    "delivery_period": "Ask how long delivery takes or when an order will arrive",
    "edit_account": "Edit or update account details or personal information",
    "get_invoice": "Request a copy of an invoice or ask where to find it",
    "get_refund": "Request a refund of money",
    "newsletter_subscription": "Subscribe to or unsubscribe from the newsletter",
    "payment_issue": "Report a problem with a payment (declined, error, charged wrongly)",
    "place_order": "Place a new order or ask how to buy an item",
    "recover_password": "Recover or reset a forgotten password or regain account access",
    "registration_problems": "Report a problem while registering or signing up",
    "review": "Leave a review or feedback about a product or the company",
    "set_up_shipping_address": "Set up or add a new shipping or delivery address",
    "switch_account": "Switch to a different account type or plan (for example free to premium)",
    "track_order": "Track the status or estimated delivery of an order",
    "track_refund": "Check the status of a refund that was already requested",
}
BITEXT_CATEGORIES = {  # category names as they appear in the CSV (the card lists older names)
    "ACCOUNT": "Account management: creating, editing, deleting or switching accounts, passwords, registration",
    "CANCEL": "Questions about cancellation fees",
    "CONTACT": "Contacting customer service or a human agent",
    "DELIVERY": "Delivery options and delivery times",
    "FEEDBACK": "Complaints and reviews",
    "INVOICE": "Invoices: checking and obtaining them",
    "ORDER": "Orders: placing, changing, cancelling and tracking them",
    "PAYMENT": "Payment methods and payment problems",
    "REFUND": "Refunds: policy, requesting and tracking refunds",
    "SHIPPING": "Shipping addresses: setting up or changing them",
    "SUBSCRIPTION": "Newsletter subscription",
}

# CFPB `Product` values (the taxonomy changed over the years; only values present are used).
CFPB_PRODUCTS = {
    "Credit reporting or other personal consumer reports": "Credit reports, credit scores, background or tenant screening reports: wrong information, disputes, identity-theft entries",
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reports, credit scores, credit repair services or other consumer reports: wrong information, disputes, identity-theft entries",
    "Credit reporting": "Credit reports and credit bureaus: wrong information, disputes",
    "Debt collection": "A debt collector contacting the consumer: debt not owed, harassment, threats, failure to validate the debt",
    "Mortgage": "Home mortgages: application, payments, escrow, servicing, loan modification, foreclosure",
    "Checking or savings account": "Bank deposit accounts: fees, overdrafts, deposits and withdrawals, account closure, unauthorized transactions",
    "Bank account or service": "Bank accounts and banking services: checking, savings, fees, deposits",
    "Credit card": "Credit cards: billing disputes, fees, interest, rewards, fraud on the card",
    "Credit card or prepaid card": "Credit cards or prepaid cards: billing disputes, fees, fraud, card problems",
    "Prepaid card": "Prepaid or gift cards: loading funds, fees, blocked cards",
    "Student loan": "Federal or private student loans: repayment, servicing, forgiveness, collections",
    "Vehicle loan or lease": "Auto loans or leases: payments, repossession, titles, lease terms",
    "Consumer Loan": "Consumer loans such as auto, installment or pawn loans",
    "Payday loan, title loan, personal loan, or advance loan": "Payday, title, personal or cash-advance loans: charges, repayment, collection practices",
    "Payday loan, title loan, or personal loan": "Payday, title or personal loans: charges, repayment, collection practices",
    "Payday loan": "Payday loans: charges, repayment, collection practices",
    "Money transfer, virtual currency, or money service": "Money transfers, remittances, mobile payment apps, cryptocurrency, check cashing or other money services",
    "Money transfers": "Domestic or international money transfers",
    "Virtual currency": "Cryptocurrency or virtual currency services",
    "Other financial service": "Other financial services: check cashing, debt settlement, credit repair, foreign exchange",
    "Debt or credit management": "Debt settlement, credit counselling or credit repair services",
}
CFPB_MIN_ROWS_PER_PRODUCT = 30
# The CFPB product taxonomy changed several times; complaints are mapped onto the current (2023+) names
# where the mapping is unambiguous. "Credit card or prepaid card" (2017-2023) is split using Sub-product.
CFPB_CANONICAL = {
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting or other personal consumer reports",
    "Credit reporting": "Credit reporting or other personal consumer reports",
    "Bank account or service": "Checking or savings account",
    "Payday loan, title loan, or personal loan": "Payday loan, title loan, personal loan, or advance loan",
    "Payday loan": "Payday loan, title loan, personal loan, or advance loan",
    "Money transfers": "Money transfer, virtual currency, or money service",
    "Virtual currency": "Money transfer, virtual currency, or money service",
}
CFPB_UNMAPPABLE = {"Consumer Loan", "Other financial service"}  # mixed products; rows dropped


def cfpb_canonical_product(product: str, sub_product: str) -> str | None:
    """Map a (Product, Sub-product) pair onto the current CFPB product taxonomy; None = drop the row."""
    if product in CFPB_UNMAPPABLE:
        return None
    if product == "Credit card or prepaid card":
        sp = (sub_product or "").lower()
        if "credit card" in sp or "charge card" in sp:
            return "Credit card"
        if any(w in sp for w in ("prepaid", "gift card", "payroll card", "benefit card", "student card", "transit card", "id prepaid", "other special purpose card")):
            return "Prepaid card"
        return None
    return CFPB_CANONICAL.get(product, product)

MEDABS_CLASSES = {  # keyed by the `labels` config condition_name (snake_case)
    "neoplasms": "Cancers, tumours and other neoplasms, benign or malignant",
    "digestive_system_diseases": "Diseases of the digestive system: stomach, intestines, liver, pancreas, gallbladder, oesophagus",
    "nervous_system_diseases": "Diseases of the nervous system: brain, spinal cord, nerves, epilepsy, neuropathy, dementia",
    "cardiovascular_diseases": "Diseases of the heart and blood vessels: coronary disease, hypertension, arrhythmia, heart failure",
    "general_pathological_conditions": "General pathological conditions not specific to one organ system: infection, inflammation, injury, metabolic disorders",
}

# --------------------------------------------------------------------------------------------
# small utilities
# --------------------------------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, flush=True)


def norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def truncate_text(text: str, limit: int) -> tuple[str, bool]:
    """Cut `text` to at most `limit` chars at a sentence boundary (falling back to a word boundary)."""
    if len(text) <= limit:
        return text, False
    marker = " [truncated]"
    cut = text[: limit - len(marker)]
    end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "), cut.rfind(".\n"), cut.rfind("\n"))
    if end < limit // 2:
        end = cut.rfind(" ")
    if end < limit // 2:
        end = len(cut) - 1
    return cut[: end + 1].rstrip() + marker, True


def fit_state(state: Any, long_field: str | None, stats: dict) -> Any:
    """Keep the serialized state under MAX_STATE_CHARS by truncating `long_field` (sentence-aware)."""
    if isinstance(state, str):
        out, cut = truncate_text(state, MAX_STATE_CHARS)
    else:
        if long_field is None:  # default to the longest string field
            long_field = max(state, key=lambda k: len(str(state[k])))
        other = sum(len(str(v)) + len(k) + 8 for k, v in state.items() if k != long_field)
        budget = max(500, MAX_STATE_CHARS - other)
        out = dict(state)
        out[long_field], cut = truncate_text(state[long_field], budget)
    if cut:
        stats["truncated"] = stats.get("truncated", 0) + 1
    return out


def http_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def http_get_text(session: requests.Session, url: str) -> str:
    for attempt in range(4):
        try:
            r = session.get(url, timeout=HTTP_TIMEOUT)
            r.raise_for_status()
            return r.text
        except requests.RequestException as exc:
            if attempt == 3:
                raise RuntimeError(f"GET {url} failed after 4 attempts: {exc}") from exc
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def hf_url(hf_id: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{hf_id}/resolve/{revision}/{path}"


def raw_dir_for(source_key: str) -> Path:
    return RAW_DIR / source_key.replace("/", "__")


# --------------------------------------------------------------------------------------------
# downloading (full files and parquet row-group subsets over HTTP range requests)
# --------------------------------------------------------------------------------------------
def download_file(session: requests.Session, url: str, dest: Path, force: bool = False) -> dict:
    """Stream `url` to `dest` (skipped when cached). Returns a provenance entry for the file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        log(f"    cached   {dest.relative_to(REPO_ROOT)} ({dest.stat().st_size // 1024} KB)")
    else:
        log(f"    download {url}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        for attempt in range(3):
            try:
                with session.get(url, stream=True, timeout=HTTP_TIMEOUT) as r:
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                break
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(f"download failed after 3 attempts: {url}: {exc}") from exc
                log(f"      retry {attempt + 1}: {exc}")
                time.sleep(2 * (attempt + 1))
        tmp.replace(dest)
        log(f"    saved    {dest.relative_to(REPO_ROOT)} ({dest.stat().st_size // 1024} KB)")
    return {"path": str(dest.relative_to(REPO_ROOT)), "url": url, "mode": "full",
            "bytes": dest.stat().st_size, "sha256": sha256_file(dest)}


class RangeHTTPFile(io.RawIOBase):
    """Minimal seekable read-only file over HTTP range requests (enough for pyarrow.parquet)."""

    def __init__(self, session: requests.Session, url: str):
        self.session, self.url, self.pos = session, url, 0
        self._resolve()

    def _resolve(self) -> None:
        r = self.session.head(self.url, allow_redirects=True, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        if "Content-Length" not in r.headers:
            raise RuntimeError(f"server did not report Content-Length for {self.url}")
        self.size = int(r.headers["Content-Length"])
        self.final_url = r.url
        self.requests_made, self.bytes_read = 0, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size) - 1
        for attempt in range(4):
            try:
                r = self.session.get(self.final_url, headers={"Range": f"bytes={self.pos}-{end}"},
                                     timeout=HTTP_TIMEOUT)
                if r.status_code == 206:
                    data = r.content
                    self.pos += len(data)
                    self.requests_made += 1
                    self.bytes_read += len(data)
                    return data
                raise RuntimeError(f"range request not honoured (HTTP {r.status_code})")
            except Exception as exc:
                if attempt == 3:
                    raise RuntimeError(f"range read failed for {self.url}: {exc}") from exc
                time.sleep(2 * (attempt + 1))
                self._resolve()  # signed CDN URLs expire; re-resolve and retry
        return b""  # unreachable

    def readinto(self, b) -> int:  # type: ignore[override]
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def evenly_spaced(total: int, n: int) -> list[int]:
    if n >= total:
        return list(range(total))
    if n == 1:
        return [0]
    return sorted({round(i * (total - 1) / (n - 1)) for i in range(n)})


def fetch_parquet_subset(session: requests.Session, url: str, dest: Path, n_row_groups: int,
                         columns: list[str], force: bool = False) -> dict:
    """Read `n_row_groups` evenly spaced row groups of a remote parquet file and cache them at `dest`."""
    sidecar = dest.with_suffix(".json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sidecar.exists() and not force:
        meta = json.loads(sidecar.read_text())
        if meta.get("n_row_groups") == n_row_groups and meta.get("columns") == columns:
            log(f"    cached   {dest.relative_to(REPO_ROOT)} ({meta['rows']} rows, row groups {meta['row_groups'][:3]}...)")
            return meta["provenance"]
    log(f"    range-read {url}")
    f = RangeHTTPFile(session, url)
    pf = pq.ParquetFile(f)
    total_rg = pf.metadata.num_row_groups
    groups = evenly_spaced(total_rg, n_row_groups)
    tables = [pf.read_row_group(g, columns=columns) for g in groups]
    import pyarrow as pa
    table = pa.concat_tables(tables)
    tmp = dest.with_suffix(".part")
    pq.write_table(table, tmp)
    tmp.replace(dest)
    prov = {"path": str(dest.relative_to(REPO_ROOT)), "url": url, "mode": "parquet_row_group_subset",
            "source_bytes": f.size, "source_rows": pf.metadata.num_rows, "source_row_groups": total_rg,
            "row_groups_read": groups, "rows": table.num_rows, "columns": columns,
            "bytes_transferred": f.bytes_read, "sha256": sha256_file(dest)}
    sidecar.write_text(json.dumps({"n_row_groups": n_row_groups, "columns": columns, "rows": table.num_rows,
                                   "row_groups": groups, "provenance": prov}, indent=1))
    log(f"    saved    {dest.relative_to(REPO_ROOT)} ({table.num_rows} rows from {len(groups)}/{total_rg} row groups, "
        f"{f.bytes_read // 1024} KB transferred)")
    return prov


# --------------------------------------------------------------------------------------------
# license verification
# --------------------------------------------------------------------------------------------
def verify_hf_license(api: HfApi, session: requests.Session, hf_id: str, revision: str,
                      accept_tags: set[str], text_marker: str | None = None) -> dict:
    """Read the card at `revision`; pass if its license tag is permissive and/or `text_marker` is present."""
    info = api.dataset_info(hf_id, revision=revision)
    card = info.card_data.to_dict() if info.card_data else {}
    tags = card.get("license")
    tags = tags if isinstance(tags, list) else ([tags] if tags else [])
    tags = [str(t).lower() for t in tags]
    evidence = []
    tag_ok = any(t in accept_tags and t in PERMISSIVE_TAGS for t in tags)
    if tag_ok:
        evidence.append(f"card license tag {tags}")
    marker_ok = False
    if text_marker:
        readme = http_get_text(session, hf_url(hf_id, revision, "README.md"))
        marker_ok = text_marker in readme
        if marker_ok:
            evidence.append(f"card text: {text_marker!r}")
    if not (tag_ok or marker_ok):
        raise RuntimeError(f"license check failed for {hf_id}@{revision}: card tag {tags}, "
                           f"marker {'absent' if text_marker else 'not configured'}; refusing to use it")
    return {"revision": info.sha, "tags": tags, "evidence": "; ".join(evidence)}


def verify_license_file(session: requests.Session, url: str, expected_start: str) -> str:
    text = http_get_text(session, url)
    head = text.lstrip()[:400]
    if expected_start.lower() not in head.lower():
        raise RuntimeError(f"LICENSE at {url} does not start with {expected_start!r} (got {head[:80]!r})")
    return f"LICENSE file {url} starts with {expected_start!r}"


# --------------------------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------------------------
def balanced_sample(rows: list[dict], label_fn: Callable[[dict], Any], n: int, rng: random.Random) -> list[dict]:
    """Round-robin over labels so the sample is as label-balanced as the pool allows."""
    by_label: dict[Any, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_label[label_fn(r)].append(r)
    labels = sorted(by_label, key=str)
    for lab in labels:
        rng.shuffle(by_label[lab])
    out: list[dict] = []
    cursor = {lab: 0 for lab in labels}
    while len(out) < n:
        progressed = False
        for lab in labels:
            if len(out) >= n:
                break
            if cursor[lab] < len(by_label[lab]):
                out.append(by_label[lab][cursor[lab]])
                cursor[lab] += 1
                progressed = True
        if not progressed:
            break
    rng.shuffle(out)
    return out


def dedupe(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        if r["_key"] in seen:
            continue
        seen.add(r["_key"])
        out.append(r)
    return out


def make_splits(train_rows: list[dict], dev_rows: list[dict] | None, label_fn, caps: dict, rng: random.Random) -> dict:
    """Return {'train','calib','dev'} row lists. calib is carved from the train source; dev too when no dev source.

    When the pool is smaller than the summed caps, the carved splits shrink proportionally so that
    train is not starved (e.g. PubMedQA's ~890 usable rows -> ~178 / 178 / 534)."""
    train_rows = dedupe(train_rows)
    if dev_rows is not None:
        dev_rows = dedupe(dev_rows)
        dev_keys = {r["_key"] for r in dev_rows}
        train_rows = [r for r in train_rows if r["_key"] not in dev_keys]  # no text leakage into dev
        dev = balanced_sample(dev_rows, label_fn, caps["dev"], rng)
        pool = train_rows
        scale = min(1.0, len(pool) / max(1, caps["calib"] + caps["train"]))
    else:
        scale = min(1.0, len(train_rows) / max(1, sum(caps.values())))
        dev = balanced_sample(train_rows, label_fn, round(caps["dev"] * scale), rng)
        taken = {r["_key"] for r in dev}
        pool = [r for r in train_rows if r["_key"] not in taken]
    calib = balanced_sample(pool, label_fn, round(caps["calib"] * scale), rng)
    taken = {r["_key"] for r in calib}
    pool = [r for r in pool if r["_key"] not in taken]
    train = balanced_sample(pool, label_fn, caps["train"], rng)
    return {"train": train, "calib": calib, "dev": dev}


# --------------------------------------------------------------------------------------------
# record construction
# --------------------------------------------------------------------------------------------
class Emitter:
    """Collects records per split and writes them to data/processed/<dataset>/<split>.jsonl."""

    def __init__(self, dataset: str, source: str, license_str: str):
        self.dataset, self.source, self.license = dataset, source, license_str
        self.records: dict[str, list[dict]] = {"train": [], "calib": [], "dev": []}
        self.stats: dict[str, Any] = {}

    def add(self, split: str, row: dict, family: str, transformation: str, state: Any, question: dict,
            gold: Any, extra: dict | None = None, long_field: str | None = None) -> None:
        state = fit_state(state, long_field, self.stats)
        n = len(self.records[split])
        rec = {
            "id": f"{self.dataset}/{split}/{n}",
            "source": self.source,
            "license": self.license,
            "split": split,
            "tier": TIER,
            "family": family,
            "synthetic": False,
            "transformation": transformation,
            "group": f"{self.dataset}/{row['_src']}/{row['_i']}",
            "request": {"model": MODEL, "state": state, "questions": {QID: question}},
            "gold": {QID: gold},
        }
        if extra:
            rec.update(extra)
        self.records[split].append(rec)

    def write(self) -> dict:
        out_dir = OUT_DIR / self.dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        counts = {}
        for split, recs in self.records.items():
            path = out_dir / f"{split}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[split] = len(recs)
        return counts


def choice_q(instructions: str, options: dict[str, str]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


def noul_q(instructions: str, true_desc: str | None = None, false_desc: str | None = None) -> dict:
    q: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if true_desc or false_desc:
        q["criteria"] = {"true": true_desc, "false": false_desc}
    return q


def score_q(instructions: str, levels: list[str]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


# --------------------------------------------------------------------------------------------
# dataset builders -- each returns a provenance dict (files, splits, notes) and fills an Emitter
# --------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, args, api: HfApi, session: requests.Session, caps: dict):
        self.args, self.api, self.session, self.caps = args, api, session, caps

    def rng(self, name: str) -> random.Random:
        return random.Random(f"{self.args.seed}:{name}")

    def hf_file(self, hf_id: str, revision: str, path: str) -> tuple[Path, dict]:
        dest = raw_dir_for(hf_id) / path
        return dest, download_file(self.session, hf_url(hf_id, revision, path), dest, self.args.force_download)

    def hf_subset(self, hf_id: str, revision: str, path: str, n_row_groups: int, columns: list[str]) -> tuple[Path, dict]:
        dest = raw_dir_for(hf_id) / (path + f".subset{n_row_groups}.parquet")
        prov = fetch_parquet_subset(self.session, hf_url(hf_id, revision, path), dest, n_row_groups, columns,
                                    self.args.force_download)
        return dest, prov

    def github_file(self, repo: str, branch: str, path: str) -> tuple[Path, dict]:
        url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
        dest = raw_dir_for("github/" + repo) / path
        return dest, download_file(self.session, url, dest, self.args.force_download)


def split_counts(splits: dict, label_fn) -> dict:
    return {s: dict(collections.Counter(str(label_fn(r)) for r in rows)) for s, rows in splits.items()}


# ---- 1. HWU64 -------------------------------------------------------------------------------
def build_hwu64(ctx: Ctx, em: Emitter) -> dict:
    repo, branch = "xliuhw/NLU-Evaluation-Data", "master"
    lic = verify_license_file(ctx.session, f"https://raw.githubusercontent.com/{repo}/{branch}/LICENSE",
                              "Attribution 4.0 International")
    path, fprov = ctx.github_file(repo, branch, "AnnotatedData/NLU-Data-Home-Domain-Annotated-All.csv")
    rows, dropped = [], collections.Counter()
    with open(path, encoding="utf-8", newline="") as f:
        for n, r in enumerate(csv.DictReader(f, delimiter=";")):
            status = (r.get("status") or "").strip().upper()
            intent = f"{r['scenario']}_{r['intent']}"
            text = re.sub(r"\s+", " ", (r.get("answer") or "").strip())
            if status.startswith("IRR"):
                dropped["status_IRR*"] += 1
                continue
            if intent not in HWU64_INTENTS:
                dropped[f"not_in_64:{intent}"] += 1
                continue
            if not text:
                dropped["empty_answer"] += 1
                continue
            # (userid, answerid) is not unique upstream, so the CSV row number identifies the source row
            rows.append({"_src": "annotated_all", "_i": f"row{n}", "_key": norm_key(text), "text": text, "label": intent})
    splits = make_splits(rows, None, lambda r: r["label"], ctx.caps, ctx.rng("hwu64"))
    all_intents = list(HWU64_INTENTS)
    instructions = "Which intent does this user request to a home assistant express?"
    for split, srows in splits.items():
        for r in srows:
            em.add(split, r, "intent", "full_options", r["text"], choice_q(instructions, HWU64_INTENTS), r["label"])
            vrng = random.Random(f"{ctx.args.seed}:hwu64:10opt:{r['_i']}")
            others = vrng.sample([i for i in all_intents if i != r["label"]], 9)
            keys = others + [r["label"]]
            vrng.shuffle(keys)
            em.add(split, r, "intent", "10_options", r["text"],
                   choice_q(instructions, {k: HWU64_INTENTS[k] for k in keys}), r["label"])
    return {
        "source": f"https://github.com/{repo}", "license": "CC BY 4.0 (Attribution 4.0 International)",
        "license_url": f"https://github.com/{repo}/blob/{branch}/LICENSE", "license_evidence": lic,
        "revision": f"{branch} (file sha256 recorded)", "files": [fprov],
        "splits_used": {"train": "AnnotatedData CSV (seeded stratified carve)", "calib": "same", "dev": "same (no canonical single split upstream; the 10-fold CV files are shuffled regenerations)"},
        "label_counts": split_counts(splits, lambda r: r["label"]),
        "notes": "Filtered: status starting with IRR (irrelevant) dropped, only the standard 64 intents kept "
                 "(audio_volume_other, cooking_query, general_greet, music_dislikeness excluded), empty answers dropped, "
                 f"exact-duplicate utterances removed. Dropped counts: {dict(dropped)}. Two records per utterance: "
                 "full_options (64) and 10_options (gold + 9 seeded random intents), linked via `group`.",
    }


# ---- 2. SNIPS -------------------------------------------------------------------------------
def build_snips(ctx: Ctx, em: Emitter) -> dict:
    repo, branch = "sonos/nlu-benchmark", "master"
    lic = verify_license_file(ctx.session, f"https://raw.githubusercontent.com/{repo}/{branch}/LICENSE", "CC0 1.0 Universal")
    files, train_rows, dev_rows = [], [], []
    for intent in SNIPS_INTENTS:
        for kind, name, target in (("train", f"train_{intent}_full.json", train_rows), ("validate", f"validate_{intent}.json", dev_rows)):
            path, fprov = ctx.github_file(repo, branch, f"2017-06-custom-intent-engines/{intent}/{name}")
            files.append(fprov)
            raw = path.read_bytes()
            try:
                data = json.loads(raw.decode("utf-8"))
            except UnicodeDecodeError:
                data = json.loads(raw.decode("latin-1"))  # train_PlayMusic_full.json is ISO-8859-1 upstream
            for i, item in enumerate(data[intent]):
                text = re.sub(r"\s+", " ", "".join(chunk["text"] for chunk in item["data"])).strip()
                if text:
                    target.append({"_src": kind, "_i": f"{intent}:{i}", "_key": norm_key(text), "text": text, "label": intent})
    splits = make_splits(train_rows, dev_rows, lambda r: r["label"], ctx.caps, ctx.rng("snips"))
    q = choice_q("Which intent does this voice-assistant request express?", SNIPS_INTENTS)
    for split, srows in splits.items():
        for r in srows:
            em.add(split, r, "intent", "full_options", r["text"], q, r["label"])
    return {
        "source": f"https://github.com/{repo}", "license": "CC0 1.0 Universal",
        "license_url": f"https://github.com/{repo}/blob/{branch}/LICENSE", "license_evidence": lic,
        "revision": f"{branch} (file sha256 recorded)", "files": files,
        "splits_used": {"train": "train_<Intent>_full.json", "calib": "carved from train_<Intent>_full.json", "dev": "validate_<Intent>.json"},
        "label_counts": split_counts(splits, lambda r: r["label"]),
        "notes": "2017-06-custom-intent-engines, 7 intents; entity spans flattened to plain text; duplicates removed.",
    }


# ---- 3. DBpedia 14 --------------------------------------------------------------------------
def build_dbpedia(ctx: Ctx, em: Emitter) -> dict:
    hf_id, rev = "fancyzhx/dbpedia_14", ctx.args.revisions["dbpedia_14"]
    lic = verify_hf_license(ctx.api, ctx.session, hf_id, rev, {"cc-by-sa-3.0"},
                            "licensed under the terms of the Creative Commons Attribution-ShareAlike License")
    cols = ["label", "title", "content"]
    tr_path, tr_prov = ctx.hf_subset(hf_id, rev, "dbpedia_14/train-00000-of-00001.parquet", 84, cols)
    te_path, te_prov = ctx.hf_subset(hf_id, rev, "dbpedia_14/test-00000-of-00001.parquet", 28, cols)
    names = [n for n, _ in DBPEDIA_CLASSES]

    def load(path: Path, src: str) -> list[dict]:
        df = pd.read_parquet(path)
        out = []
        for i, r in enumerate(df.itertuples(index=False)):
            content = re.sub(r"\s+", " ", str(r.content)).strip()
            title = str(r.title).strip()
            if content:
                out.append({"_src": src, "_i": i, "_key": norm_key(title + "|" + content), "title": title,
                            "content": content, "label": names[int(r.label)]})
        return out

    splits = make_splits(load(tr_path, "train"), load(te_path, "test"), lambda r: r["label"], ctx.caps, ctx.rng("dbpedia_14"))
    q = choice_q("Which category best describes the entity described in `content` (titled `title`)?", dict(DBPEDIA_CLASSES))
    for split, srows in splits.items():
        for r in srows:
            em.add(split, r, "topic", "full_options", {"title": r["title"], "content": r["content"]}, q, r["label"], long_field="content")
    return {
        "source": hf_id, "license": "cc-by-sa-3.0 (Creative Commons Attribution-ShareAlike 3.0)",
        "license_url": f"https://huggingface.co/datasets/{hf_id}", "license_evidence": lic["evidence"], "revision": lic["revision"],
        "files": [tr_prov, te_prov], "splits_used": {"train": "train (row-group subset)", "calib": "carved from train", "dev": "test (row-group subset)"},
        "label_counts": split_counts(splits, lambda r: r["label"]),
        "notes": "The source parquet files are class-sorted; evenly spaced row groups were read to cover all 14 classes.",
    }


# ---- 4. SNLI --------------------------------------------------------------------------------
def build_snli(ctx: Ctx, em: Emitter) -> dict:
    hf_id, rev = "stanfordnlp/snli", ctx.args.revisions["snli"]
    lic = verify_hf_license(ctx.api, ctx.session, hf_id, rev, {"cc-by-sa-4.0"},
                            "Creative Commons Attribution-ShareAlike 4.0 International License")
    tr_path, tr_prov = ctx.hf_file(hf_id, rev, "plain_text/train-00000-of-00001.parquet")
    va_path, va_prov = ctx.hf_file(hf_id, rev, "plain_text/validation-00000-of-00001.parquet")
    dropped = collections.Counter()

    def load(path: Path, src: str) -> list[dict]:
        df = pd.read_parquet(path)
        out = []
        for i, r in enumerate(df.itertuples(index=False)):
            if int(r.label) < 0:
                dropped[f"{src}:label_-1"] += 1
                continue
            p, h = str(r.premise).strip(), str(r.hypothesis).strip()
            if p and h:
                out.append({"_src": src, "_i": i, "_key": norm_key(p) + "||" + norm_key(h), "premise": p, "hypothesis": h,
                            "label": SNLI_LABELS[int(r.label)]})
        return out

    splits = make_splits(load(tr_path, "train"), load(va_path, "validation"), lambda r: r["label"], ctx.caps, ctx.rng("snli"))
    qc = choice_q("Given `premise`, how should `hypothesis` be classified?", SNLI_OPTIONS)
    qn = noul_q("Does `premise` guarantee that `hypothesis` is true?",
                "The premise entails the hypothesis: it must be true", "The hypothesis is not guaranteed by the premise (neutral or contradicted)")
    for split, srows in splits.items():
        for r in srows:
            state = {"premise": r["premise"], "hypothesis": r["hypothesis"]}
            em.add(split, r, "compatibility", "choice", state, qc, r["label"])
            em.add(split, r, "compatibility", "noul", state, qn, r["label"] == "entailment")
    return {
        "source": hf_id, "license": "cc-by-sa-4.0 (Creative Commons Attribution-ShareAlike 4.0 International)",
        "license_url": f"https://huggingface.co/datasets/{hf_id}", "license_evidence": lic["evidence"], "revision": lic["revision"],
        "files": [tr_prov, va_prov], "splits_used": {"train": "train", "calib": "carved from train", "dev": "validation"},
        "label_counts": split_counts(splits, lambda r: r["label"]),
        "notes": f"Rows with label -1 (no gold) dropped: {dict(dropped)}. Two records per pair (choice + noul; noul gold = entailment, "
                 "so about one third of noul golds are true), linked via `group`.",
    }


# ---- 5. Amazon polarity ---------------------------------------------------------------------
def build_amazon(ctx: Ctx, em: Emitter) -> dict:
    hf_id, rev = "fancyzhx/amazon_polarity", ctx.args.revisions["amazon_polarity"]
    lic = verify_hf_license(ctx.api, ctx.session, hf_id, rev, {"apache-2.0"}, "Apache License 2.0")
    cols = ["label", "title", "content"]
    tr_path, tr_prov = ctx.hf_subset(hf_id, rev, "amazon_polarity/train-00000-of-00004.parquet", 24, cols)
    te_path, te_prov = ctx.hf_subset(hf_id, rev, "amazon_polarity/test-00000-of-00001.parquet", 8, cols)

    def load(path: Path, src: str) -> list[dict]:
        df = pd.read_parquet(path)
        out = []
        for i, r in enumerate(df.itertuples(index=False)):
            content, title = re.sub(r"\s+", " ", str(r.content)).strip(), str(r.title).strip()
            if content:
                out.append({"_src": src, "_i": i, "_key": norm_key(content), "title": title, "content": content,
                            "label": AMAZON_LABELS[int(r.label)]})
        return out

    splits = make_splits(load(tr_path, "train"), load(te_path, "test"), lambda r: r["label"], ctx.caps, ctx.rng("amazon_polarity"))
    qn = noul_q("Is this review positive?", "The review expresses a positive opinion of the product",
                "The review expresses a negative opinion of the product")
    qc = choice_q("What is the sentiment of this product review (`title` and `content`)?", AMAZON_OPTIONS)
    for split, srows in splits.items():
        for r in srows:
            state = {"title": r["title"], "content": r["content"]}
            em.add(split, r, "sentiment", "noul", state, qn, r["label"] == "positive", long_field="content")
            em.add(split, r, "sentiment", "choice", state, qc, r["label"], long_field="content")
    return {
        "source": hf_id, "license": "apache-2.0 (Apache License 2.0)", "license_url": f"https://huggingface.co/datasets/{hf_id}",
        "license_evidence": lic["evidence"], "revision": lic["revision"], "files": [tr_prov, te_prov],
        "splits_used": {"train": "train shard 0 of 4 (row-group subset)", "calib": "carved from train", "dev": "test (row-group subset)"},
        "label_counts": split_counts(splits, lambda r: r["label"]),
        "notes": "Two records per review (noul + choice), linked via `group`.",
    }


# ---- 6. TweetEval sentiment -----------------------------------------------------------------
def build_tweet_eval(ctx: Ctx, em: Emitter) -> dict:
    hf_id, rev = "cardiffnlp/tweet_eval", ctx.args.revisions["tweet_eval"]
    marker = "Sentiment: [Creative Commons Attribution 3.0 Unported License]"
    lic = verify_hf_license(ctx.api, ctx.session, hf_id, rev, set(), marker)  # card tag is `unknown`; the sentiment config is CC BY 3.0
    tr_path, tr_prov = ctx.hf_file(hf_id, rev, "sentiment/train-00000-of-00001.parquet")
    va_path, va_prov = ctx.hf_file(hf_id, rev, "sentiment/validation-00000-of-00001.parquet")

    def load(path: Path, src: str) -> list[dict]:
        df = pd.read_parquet(path)
        return [{"_src": src, "_i": i, "_key": norm_key(str(r.text)), "text": str(r.text).strip(), "label": TWEET_LABELS[int(r.label)]}
                for i, r in enumerate(df.itertuples(index=False)) if str(r.text).strip()]

    splits = make_splits(load(tr_path, "train"), load(va_path, "validation"), lambda r: r["label"], ctx.caps, ctx.rng("tweet_eval"))
    q = choice_q("What is the sentiment of this tweet?", TWEET_OPTIONS)
    for split, srows in splits.items():
        for r in srows:
            em.add(split, r, "sentiment", "full_options", r["text"], q, r["label"])
    return {
        "source": f"{hf_id} (config: sentiment)", "license": "Creative Commons Attribution 3.0 Unported (sentiment config only; Twitter ToS apply)",
        "license_url": f"https://huggingface.co/datasets/{hf_id}#licensing-information", "license_evidence": lic["evidence"],
        "revision": lic["revision"], "files": [tr_prov, va_prov],
        "splits_used": {"train": "sentiment/train", "calib": "carved from sentiment/train", "dev": "sentiment/validation"},
        "label_counts": split_counts(splits, lambda r: r["label"]),
        "notes": "Only the `sentiment` config is used (other configs are Undefined / need permission).",
    }


# ---- 7. ProsocialDialog ---------------------------------------------------------------------
def build_prosocial(ctx: Ctx, em: Emitter) -> dict:
    hf_id, rev = "allenai/prosocial-dialog", ctx.args.revisions["prosocial_dialog"]
    lic = verify_hf_license(ctx.api, ctx.session, hf_id, rev, {"cc-by-4.0"})
    tr_path, tr_prov = ctx.hf_file(hf_id, rev, "train.json")
    va_path, va_prov = ctx.hf_file(hf_id, rev, "valid.json")
    level_index = {name: i for i, (name, _) in enumerate(PROSOCIAL_LEVELS)}
    dropped = collections.Counter()

    def load(path: Path, src: str) -> list[dict]:
        out = []
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["safety_label"] not in level_index:
                    dropped[f"unknown_label:{r['safety_label']}"] += 1
                    continue
                votes = r["safety_annotations"]
                if isinstance(votes, str):
                    votes = ast.literal_eval(votes)
                context, response = r["context"].strip(), r["response"].strip()
                if not context:
                    dropped["empty_context"] += 1
                    continue
                out.append({"_src": src, "_i": f"{r['dialogue_id']}:{r['response_id']}", "_key": norm_key(context) + "||" + norm_key(response),
                            "context": context, "response": response, "label": level_index[r["safety_label"]],
                            "votes": dict(collections.Counter(votes))})
        return out

    splits = make_splits(load(tr_path, "train"), load(va_path, "valid"), lambda r: r["label"], ctx.caps, ctx.rng("prosocial_dialog"))
    q = score_q("How much caution does the utterance in `context` call for? (`response` is the reply it received.)",
                [f"{name}: {desc}" for name, desc in PROSOCIAL_LEVELS])
    for split, srows in splits.items():
        for r in srows:
            em.add(split, r, "severity", "score", {"context": r["context"], "response": r["response"]}, q, r["label"],
                   extra={"annotator_votes": r["votes"]}, long_field="context")
    return {
        "source": hf_id, "license": "cc-by-4.0", "license_url": f"https://huggingface.co/datasets/{hf_id}",
        "license_evidence": lic["evidence"], "revision": lic["revision"], "files": [tr_prov, va_prov],
        "splits_used": {"train": "train.json", "calib": "carved from train.json", "dev": "valid.json"},
        "label_counts": split_counts(splits, lambda r: r["label"]),
        "notes": "Score levels = the dataset's ordered safety_label scale (0 casual .. 4 needs intervention). The three per-annotator votes "
                 "use a 3-way scale (casual / needs caution / needs intervention) from which the 5-level label is derived deterministically, "
                 "so no annotator distribution over the 5 levels exists; the raw vote counts are kept in the extra field `annotator_votes` "
                 "instead of `gold_probs`. Contexts were GPT-3-generated, labels are human (per the card). "
                 f"Dropped: {dict(dropped)}.",
    }


# ---- 8. PubMedQA ----------------------------------------------------------------------------
def build_pubmedqa(ctx: Ctx, em: Emitter) -> dict:
    hf_id, rev = "qiaojin/PubMedQA", ctx.args.revisions["pubmedqa"]
    lic = verify_hf_license(ctx.api, ctx.session, hf_id, rev, {"mit"})
    upstream = verify_license_file(ctx.session, "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/LICENSE", "MIT License")
    path, fprov = ctx.hf_file(hf_id, rev, "pqa_labeled/train-00000-of-00001.parquet")
    df = pd.read_parquet(path)
    rows, dropped = [], collections.Counter()
    for r in df.itertuples(index=False):
        decision = str(r.final_decision).strip().lower()
        if decision not in ("yes", "no"):
            dropped[f"final_decision:{decision}"] += 1
            continue
        contexts = [str(c).strip() for c in list(r.context["contexts"])]
        context = " ".join(c for c in contexts if c)
        question = str(r.question).strip()
        if not context or not question:
            dropped["empty"] += 1
            continue
        rows.append({"_src": "pqa_labeled", "_i": str(r.pubid), "_key": str(r.pubid), "question": question, "context": context, "label": decision == "yes"})
    splits = make_splits(rows, None, lambda r: r["label"], ctx.caps, ctx.rng("pubmedqa"))
    q = noul_q("Based on `context`, is the answer to `question` yes?", "The context supports answering the question with yes",
               "The context supports answering the question with no")
    for split, srows in splits.items():
        for r in srows:
            em.add(split, r, "factual_yes_no", "noul", {"question": r["question"], "context": r["context"]}, q, r["label"], long_field="context")
    return {
        "source": f"{hf_id} (config: pqa_labeled)", "license": "mit (MIT License)", "license_url": "https://github.com/pubmedqa/pubmedqa/blob/master/LICENSE",
        "license_evidence": lic["evidence"] + "; " + upstream, "revision": lic["revision"], "files": [fprov],
        "splits_used": {"train": "pqa_labeled (seeded carve)", "calib": "same", "dev": "same (pqa_labeled has a single split of 1,000 expert-labeled rows)"},
        "label_counts": split_counts(splits, lambda r: r["label"]),
        "notes": f"Only yes/no rows kept (maybe dropped): {dict(dropped)}. Gold = final_decision == yes; context = abstract sections joined with a space. "
                 "dev/calib are yes/no balanced; train keeps the remaining rows (yes-heavy).",
    }


# ---- 9. SQuAD v2 ----------------------------------------------------------------------------
def build_squad_v2(ctx: Ctx, em: Emitter) -> dict:
    hf_id, rev = "rajpurkar/squad_v2", ctx.args.revisions["squad_v2"]
    lic = verify_hf_license(ctx.api, ctx.session, hf_id, rev, {"cc-by-sa-4.0"}, "The dataset is distributed under the CC BY-SA 4.0 license")
    tr_path, tr_prov = ctx.hf_file(hf_id, rev, "squad_v2/train-00000-of-00001.parquet")
    va_path, va_prov = ctx.hf_file(hf_id, rev, "squad_v2/validation-00000-of-00001.parquet")
    dropped = collections.Counter()

    def load(path: Path, src: str) -> list[dict]:
        df = pd.read_parquet(path)
        out = []
        for r in df.itertuples(index=False):
            question, context = re.sub(r"\s+", " ", str(r.question)).strip(), str(r.context).strip()
            if not question or not context or len(question) > 500:
                dropped[f"{src}:empty_or_overlong_question"] += 1
                continue
            out.append({"_src": src, "_i": str(r.id), "_key": norm_key(question) + "||" + norm_key(context[:200]), "question": question,
                        "context": context, "label": len(list(r.answers["text"])) > 0})
        return out

    splits = make_splits(load(tr_path, "train"), load(va_path, "validation"), lambda r: r["label"], ctx.caps, ctx.rng("squad_v2"))
    q = noul_q("Can `question` be answered from `context`?", "The passage states the information needed to answer the question",
               "The passage does not contain the answer, even if the question looks related to it")
    for split, srows in splits.items():
        for r in srows:
            em.add(split, r, "adequacy", "noul", {"question": r["question"], "context": r["context"]}, q, r["label"], long_field="context")
    return {
        "source": hf_id, "license": "cc-by-sa-4.0 (CC BY-SA 4.0)", "license_url": f"https://huggingface.co/datasets/{hf_id}",
        "license_evidence": lic["evidence"], "revision": lic["revision"], "files": [tr_prov, va_prov],
        "splits_used": {"train": "train", "calib": "carved from train", "dev": "validation"},
        "label_counts": split_counts(splits, lambda r: r["label"]),
        "notes": f"Gold = question has at least one answer span. Balanced answerable/unanswerable. Dropped: {dict(dropped)}.",
    }


# ---- 10. Bitext customer support ------------------------------------------------------------
def build_bitext(ctx: Ctx, em: Emitter) -> dict:
    hf_id, rev = "bitext/Bitext-customer-support-llm-chatbot-training-dataset", ctx.args.revisions["bitext_support"]
    lic = verify_hf_license(ctx.api, ctx.session, hf_id, rev, {"cdla-sharing-1.0"})
    path, fprov = ctx.hf_file(hf_id, rev, "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
    df = pd.read_csv(path)
    unknown_intents = sorted(set(df["intent"]) - set(BITEXT_INTENTS))
    unknown_cats = sorted(set(df["category"]) - set(BITEXT_CATEGORIES))
    if unknown_intents or unknown_cats:
        raise RuntimeError(f"bitext taxonomy changed; add descriptions for intents {unknown_intents} / categories {unknown_cats}")
    rows = []
    for i, r in enumerate(df.itertuples(index=False)):
        text = re.sub(r"\s+", " ", str(r.instruction)).strip()
        if text:
            rows.append({"_src": "csv", "_i": i, "_key": norm_key(text), "text": text, "intent": str(r.intent), "category": str(r.category)})
    splits = make_splits(rows, None, lambda r: r["intent"], ctx.caps, ctx.rng("bitext_support"))
    qi = choice_q("Which intent best matches this customer-support message?", BITEXT_INTENTS)
    qc = choice_q("Route this customer-support message to the right category.", BITEXT_CATEGORIES)
    for split, srows in splits.items():
        for r in srows:
            em.add(split, r, "intent", "intent_options", r["text"], qi, r["intent"])
            em.add(split, r, "routing", "category_options", r["text"], qc, r["category"])
    return {
        "source": hf_id, "license": "cdla-sharing-1.0 (Community Data License Agreement - Sharing 1.0)",
        "license_url": f"https://huggingface.co/datasets/{hf_id}", "license_evidence": lic["evidence"], "revision": lic["revision"],
        "files": [fprov], "splits_used": {"train": "single CSV (seeded carve, balanced over intent)", "calib": "same", "dev": "same"},
        "label_counts": {"intent": split_counts(splits, lambda r: r["intent"]), "category": split_counts(splits, lambda r: r["category"])},
        "notes": "Hybrid synthetic (NLG-expanded from natural seeds, per the card); {{placeholders}} kept verbatim; exact duplicates removed. "
                 "Two records per utterance (27-intent choice + 11-category choice), linked via `group`.",
    }


# ---- 11. CFPB consumer complaints -----------------------------------------------------------
def build_cfpb(ctx: Ctx, em: Emitter) -> dict:
    hf_id, rev = "BEE-spoke-data/consumer-finance-complaints", ctx.args.revisions["cfpb_complaints"]
    lic = verify_hf_license(ctx.api, ctx.session, hf_id, rev, {"cc0-1.0"})
    cols = ["Product", "Sub-product", "Consumer complaint narrative", "Complaint ID", "Date received"]
    path, fprov = ctx.hf_subset(hf_id, rev, "has-text/train-00000-of-00003.parquet", 40, cols)
    df = pd.read_parquet(path)
    rows, dropped, mapped = [], collections.Counter(), collections.Counter()
    for r in df.itertuples(index=False):
        product, sub_product, narrative, cid = r[0], r[1], r[2], r[3]
        narrative = re.sub(r"[ \t]+", " ", str(narrative or "")).strip()
        if not narrative or narrative.lower() in ("none", "nan") or not product or str(product) in ("None", "nan"):
            dropped["no_narrative_or_product"] += 1
            continue
        label = cfpb_canonical_product(str(product), str(sub_product or ""))
        if label is None:
            dropped[f"unmappable:{product}"] += 1
            continue
        if label != str(product):
            mapped[f"{product} -> {label}"] += 1
        rows.append({"_src": "has-text-shard0", "_i": str(cid), "_key": norm_key(narrative), "text": narrative, "label": label})
    rows = dedupe(rows)
    counts = collections.Counter(r["label"] for r in rows)
    products = sorted(p for p, n in counts.items() if n >= CFPB_MIN_ROWS_PER_PRODUCT)
    rare = {p: n for p, n in counts.items() if n < CFPB_MIN_ROWS_PER_PRODUCT}
    rows = [r for r in rows if r["label"] in products]
    options = {}
    for p in products:
        if p not in CFPB_PRODUCTS:
            log(f"    WARNING: no authored description for CFPB product {p!r}; using the label itself")
        options[p] = CFPB_PRODUCTS.get(p, p)
    splits = make_splits(rows, None, lambda r: r["label"], ctx.caps, ctx.rng("cfpb_complaints"))
    q = choice_q("Which financial product is this consumer complaint about?", options)
    for split, srows in splits.items():
        for r in srows:
            em.add(split, r, "routing", "product_options", r["text"], q, r["label"])
    return {
        "source": f"{hf_id} (config: has-text)", "license": "cc0-1.0 (CC0 1.0 Universal; US government data)",
        "license_url": f"https://huggingface.co/datasets/{hf_id}", "license_evidence": lic["evidence"], "revision": lic["revision"],
        "files": [fprov], "splits_used": {"train": "has-text shard 0 of 3, row-group subset (seeded carve)", "calib": "same", "dev": "same"},
        "label_counts": split_counts(splits, lambda r: r["label"]),
        "notes": f"Only rows with a narrative. Product values were mapped onto the current CFPB taxonomy (mapped: {dict(mapped)}); "
                 f"options = the {len(products)} canonical products with >= {CFPB_MIN_ROWS_PER_PRODUCT} rows in the subset "
                 f"(rare products dropped: {rare}); dropped rows: {dict(dropped)}. Narratives keep the CFPB XXXX redactions.",
    }


# ---- 12. Medical abstracts (optional) -------------------------------------------------------
def build_medical_abstracts(ctx: Ctx, em: Emitter) -> dict:
    hf_id, rev = "TimSchopf/medical_abstracts", ctx.args.revisions["medical_abstracts"]
    lic = verify_hf_license(ctx.api, ctx.session, hf_id, rev, {"cc-by-sa-3.0"})
    upstream = verify_license_file(ctx.session, "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/LICENSE",
                                   "Attribution-ShareAlike 3.0 Unported")
    lab_path, lab_prov = ctx.hf_file(hf_id, rev, "labels/train-00000-of-00001.parquet")
    tr_path, tr_prov = ctx.hf_file(hf_id, rev, "data/train-00000-of-00001.parquet")
    te_path, te_prov = ctx.hf_file(hf_id, rev, "data/test-00000-of-00001.parquet")
    labels = {int(r.condition_label): re.sub(r"\s+", "_", str(r.condition_name).strip().lower()) for r in pd.read_parquet(lab_path).itertuples(index=False)}
    missing = sorted(set(labels.values()) - set(MEDABS_CLASSES))
    if missing:
        raise RuntimeError(f"medical_abstracts label names changed; add descriptions for {missing}")

    def load(path: Path, src: str) -> list[dict]:
        df = pd.read_parquet(path)
        return [{"_src": src, "_i": i, "_key": norm_key(str(r.medical_abstract)), "text": str(r.medical_abstract).strip(), "label": labels[int(r.condition_label)]}
                for i, r in enumerate(df.itertuples(index=False)) if str(r.medical_abstract).strip()]

    splits = make_splits(load(tr_path, "train"), load(te_path, "test"), lambda r: r["label"], ctx.caps, ctx.rng("medical_abstracts"))
    q = choice_q("Which class of patient condition does this medical abstract describe?", MEDABS_CLASSES)
    for split, srows in splits.items():
        for r in srows:
            em.add(split, r, "topic", "full_options", r["text"], q, r["label"])
    return {
        "source": hf_id, "license": "cc-by-sa-3.0 (Creative Commons Attribution-ShareAlike 3.0 Unported)",
        "license_url": "https://github.com/sebischair/Medical-Abstracts-TC-Corpus/blob/main/LICENSE",
        "license_evidence": lic["evidence"] + "; " + upstream, "revision": lic["revision"], "files": [lab_prov, tr_prov, te_prov],
        "splits_used": {"train": "train", "calib": "carved from train", "dev": "test"},
        "label_counts": split_counts(splits, lambda r: r["label"]),
        "notes": "Exact-duplicate abstracts removed (the source has ~18% duplicates) and dev texts excluded from the train pool.",
    }


# --------------------------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------------------------
# Pinned Hub revisions (resolved on 2026-09-22). `--latest` resolves `main` instead.
PINNED_REVISIONS = {
    "dbpedia_14": "9abd46cf7fc8b4c64290f26993c540b92aa145ac",
    "snli": "cdb5c3d5eed6ead6e5a341c8e56e669bb666725b",
    "amazon_polarity": "9d9c45c18f8c3cf1b23a3c27917b60cbf28f3289",
    "tweet_eval": "b3a375baf0f409c77e6bc7aa35102b7b3534f8be",
    "prosocial_dialog": "d77d7ad3c624c51030f2f32c83e892b3d620b3d4",
    "pubmedqa": "9001f2853fb87cab8d220904e0de81ac6973b318",
    "squad_v2": "3ffb306f725f7d2ce8394bc1873b24868140c412",
    "bitext_support": "430d1a89bd93bd1fa23c16f29dd53e73f0087443",
    "cfpb_complaints": "088cc7308d4afc2a880f2329d08e2f7a09188ec6",
    "medical_abstracts": "d6afbd7ff4eb215db2470980023d7d7a3dfee7dc",
}

DATASETS: dict[str, dict] = {  # key -> {source, family, build}
    "hwu64": {"source": "https://github.com/xliuhw/NLU-Evaluation-Data", "family": "intent", "build": build_hwu64},
    "snips": {"source": "https://github.com/sonos/nlu-benchmark", "family": "intent", "build": build_snips},
    "dbpedia_14": {"source": "fancyzhx/dbpedia_14", "family": "topic", "build": build_dbpedia},
    "snli": {"source": "stanfordnlp/snli", "family": "compatibility", "build": build_snli},
    "amazon_polarity": {"source": "fancyzhx/amazon_polarity", "family": "sentiment", "build": build_amazon},
    "tweet_eval_sentiment": {"source": "cardiffnlp/tweet_eval", "family": "sentiment", "build": build_tweet_eval},
    "prosocial_dialog": {"source": "allenai/prosocial-dialog", "family": "severity", "build": build_prosocial},
    "pubmedqa": {"source": "qiaojin/PubMedQA", "family": "factual_yes_no", "build": build_pubmedqa},
    "squad_v2": {"source": "rajpurkar/squad_v2", "family": "adequacy", "build": build_squad_v2},
    "bitext_support": {"source": "bitext/Bitext-customer-support-llm-chatbot-training-dataset", "family": "intent+routing", "build": build_bitext},
    "cfpb_complaints": {"source": "BEE-spoke-data/consumer-finance-complaints", "family": "routing", "build": build_cfpb},
    "medical_abstracts": {"source": "TimSchopf/medical_abstracts", "family": "topic", "build": build_medical_abstracts},
}


# --------------------------------------------------------------------------------------------
# provenance + README
# --------------------------------------------------------------------------------------------
def load_provenance() -> list[dict]:
    if not PROVENANCE_PATH.exists():
        return []
    return [json.loads(l) for l in PROVENANCE_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]


def upsert_provenance(entry: dict) -> None:
    entries = [e for e in load_provenance() if e.get("dataset") != entry["dataset"]]
    entries.append(entry)
    entries.sort(key=lambda e: list(DATASETS).index(e["dataset"]) if e["dataset"] in DATASETS else 999)
    PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROVENANCE_PATH, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def write_readme(entries: list[dict]) -> None:
    lines = [
        "# data/",
        "",
        "Datasets used to train, calibrate and sanity-check Sextant (a local decision engine).",
        "",
        "| directory | contents | committed? |",
        "|---|---|---|",
        "| `synthetic/` | synthetic records generated by `scripts/synth/generate.py` (labels verifiable by construction) | yes |",
        "| `raw/` | downloaded source files (cache) | **no** (gitignored) |",
        "| `processed/<dataset>/{train,calib,dev}.jsonl` | records built from public datasets by `scripts/prepare_data.py` | **no** (gitignored) |",
        "| `provenance.jsonl` | one line per dataset: source, revision, license, files, splits, row counts, notes | yes |",
        "",
        "**Raw and processed real-data files are not committed.** They are re-created deterministically (seed "
        f"{SEED}) by running `python3 scripts/prepare_data.py` (requires `pip install requests huggingface_hub pandas pyarrow` "
        "and outbound HTTPS). Each dataset's license is re-verified from its dataset card / LICENSE file at download time; "
        "the script refuses datasets whose license cannot be verified as permissive and hard-blocks every benchmark corpus "
        "listed in `FORBIDDEN_SOURCES`.",
        "",
        "## Regenerating",
        "",
        "```bash",
        "python3 scripts/prepare_data.py                         # all datasets",
        "python3 scripts/prepare_data.py --datasets snips,snli  # a subset",
        "python3 scripts/prepare_data.py --limit 20             # quick smoke test (caps every split at 20 rows)",
        "python3 scripts/prepare_data.py --verify-only          # re-validate data/processed without downloading",
        "```",
        "",
        "Large parquet shards are not downloaded in full: the script reads an evenly spaced subset of their row groups over HTTP "
        "range requests and caches that subset under `data/raw/`. Hub revisions are pinned in `PINNED_REVISIONS` (`--latest` "
        "overrides). Sampling is label-balanced where the source allows it, with caps of "
        f"{CAPS['train']} / {CAPS['calib']} / {CAPS['dev']} source rows per split (train / calib / dev); datasets with several "
        "transformations emit one record per transformation for each sampled row, linked by the `group` field.",
        "",
        "## Record format",
        "",
        "One JSON object per line: `{id, source, license, split, tier: \"external_dev\", family, synthetic: false, transformation, group, "
        "request: {model: \"sextant-1\", state, questions: {q: {type: choice|score|noul, instructions, criteria}}}, gold: {q: ...}}`. "
        "Choice `criteria` maps option key to an authored description, score `criteria` is the ordered list of levels (gold = level index), "
        "noul gold is a boolean. States are kept under ~4000 characters (long fields are cut at a sentence boundary and end with "
        "` [truncated]`). Option descriptions were written for this project and are not part of the source datasets.",
        "",
        "## Datasets",
        "",
        "| dataset | source | license | family | transformations | train / calib / dev (records) | notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        rc = e.get("records_produced", {})
        counts = f"{rc.get('train', 0)} / {rc.get('calib', 0)} / {rc.get('dev', 0)}"
        rows = e.get("rows_sampled", {})
        if rows and rows != rc:
            counts += f" ({rows.get('train', 0)} / {rows.get('calib', 0)} / {rows.get('dev', 0)} rows)"
        lines.append(f"| `{e['dataset']}` | {e['source']} | {e['license']} | {e.get('family', '')} | "
                     f"{', '.join(e.get('transformations', []))} | {counts} | {e.get('notes', '').replace('|', '/')} |")
    lines += ["", f"Generated by `scripts/prepare_data.py` on {dt.date.today().isoformat()}. See `provenance.jsonl` for revisions, file hashes and label counts.", ""]
    README_PATH.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------------------------
# verification of outputs
# --------------------------------------------------------------------------------------------
def verify_outputs(keys: list[str], show_examples: bool = True) -> bool:
    ok = True
    for key in keys:
        d = OUT_DIR / key
        if not d.exists():
            log(f"  [{key}] missing output directory")
            ok = False
            continue
        shown = False
        for split in ("train", "calib", "dev"):
            path = d / f"{split}.jsonl"
            if not path.exists():
                log(f"  [{key}] missing {split}.jsonl")
                ok = False
                continue
            n, ids, problems = 0, set(), collections.Counter()
            with open(path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    n += 1
                    if rec["id"] in ids:
                        problems["duplicate id"] += 1
                    ids.add(rec["id"])
                    for field in ("id", "source", "license", "split", "tier", "family", "transformation", "request", "gold"):
                        if field not in rec:
                            problems[f"missing {field}"] += 1
                    q = rec["request"]["questions"][QID]
                    gold = rec["gold"][QID]
                    if q["type"] == "choice" and gold not in q["criteria"]:
                        problems["choice gold not an option"] += 1
                    elif q["type"] == "score" and not (isinstance(gold, int) and 0 <= gold < len(q["criteria"])):
                        problems["score gold out of range"] += 1
                    elif q["type"] == "noul" and not isinstance(gold, bool):
                        problems["noul gold not bool"] += 1
                    if len(json.dumps(rec["request"]["state"], ensure_ascii=False)) > MAX_STATE_CHARS + 200:
                        problems["state too long"] += 1
                    if rec["split"] != split:
                        problems["split mismatch"] += 1
                    if show_examples and not shown and split == "dev":
                        shown = True
                        ex = json.dumps(rec, ensure_ascii=False)
                        log(f"  [{key}] example: {ex[:700]}{'...' if len(ex) > 700 else ''}")
            status = "ok" if not problems else f"PROBLEMS {dict(problems)}"
            log(f"  [{key}] {split}: {n} records, {status}")
            if problems or n == 0:
                ok = False
    return ok


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default="all", help="comma-separated dataset keys (default: all); see --list")
    p.add_argument("--limit", type=int, default=None, help="cap every split at N source rows (quick test)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--latest", action="store_true", help="resolve Hub revision `main` instead of the pinned shas")
    p.add_argument("--force-download", action="store_true", help="ignore the raw cache and download again")
    p.add_argument("--list", action="store_true", help="list dataset keys and exit")
    p.add_argument("--verify-only", action="store_true", help="only validate existing data/processed outputs")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        for k, v in DATASETS.items():
            print(f"{k:22s} {v['family']:16s} {v['source']}")
        return 0
    keys = list(DATASETS) if args.datasets == "all" else [k.strip() for k in args.datasets.split(",") if k.strip()]
    unknown = [k for k in keys if k not in DATASETS]
    if unknown:
        sys.exit(f"unknown dataset key(s): {unknown}; valid keys: {', '.join(DATASETS)}")
    if args.verify_only:
        log("Verifying outputs...")
        return 0 if verify_outputs(keys) else 1

    for k in keys:  # contamination guard
        src = DATASETS[k]["source"].lower()
        hit = [f for f in FORBIDDEN_SOURCES if f in src]
        if hit:
            sys.exit(f"refusing dataset {k}: source {src!r} matches forbidden benchmark corpus fragment {hit}")

    caps = {s: (min(c, args.limit) if args.limit else c) for s, c in CAPS.items()}
    api, session = HfApi(), http_session()
    args.revisions = dict(PINNED_REVISIONS)
    if args.latest:
        args.revisions = {k: "main" for k in PINNED_REVISIONS}
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(args, api, session, caps)
    t0 = time.time()
    results: dict[str, str] = {}
    for key in keys:
        spec = DATASETS[key]
        log(f"\n=== {key} ({spec['source']}) ===")
        t1 = time.time()
        try:
            em = Emitter(key, spec["source"], "")
            prov = spec["build"](ctx, em)
            for recs in em.records.values():  # fill the verified license string into every record
                for rec in recs:
                    rec["license"] = prov["license"]
            em.license = prov["license"]
            counts = em.write()
            rows_sampled = {s: len({r["group"] for r in recs}) for s, recs in em.records.items()}
            transformations = sorted({r["transformation"] for recs in em.records.values() for r in recs})
            entry = {
                "dataset": key, "family": spec["family"], "date": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "seed": args.seed, "caps": caps, **prov, "transformations": transformations,
                "rows_sampled": rows_sampled, "records_produced": counts, "truncated_states": em.stats.get("truncated", 0),
                "output": {s: str((OUT_DIR / key / f"{s}.jsonl").relative_to(REPO_ROOT)) for s in counts},
                "script": "scripts/prepare_data.py",
            }
            upsert_provenance(entry)
            results[key] = f"ok  train={counts['train']} calib={counts['calib']} dev={counts['dev']} records ({time.time() - t1:.0f}s)"
            log(f"  -> {results[key]}")
        except Exception as exc:  # keep going; report at the end
            tb = traceback.extract_tb(exc.__traceback__)[-1]
            results[key] = f"SKIPPED: {type(exc).__name__}: {exc} (at {Path(tb.filename).name}:{tb.lineno} in {tb.name})"
            log(f"  !! {key} skipped: {results[key]}")
    write_readme(load_provenance())
    log("\nVerifying outputs...")
    ok = verify_outputs([k for k, r in results.items() if r.startswith("ok")])
    log(f"\nSummary ({time.time() - t0:.0f}s):")
    for k, r in results.items():
        log(f"  {k:22s} {r}")
    log(f"provenance: {PROVENANCE_PATH.relative_to(REPO_ROOT)}   readme: {README_PATH.relative_to(REPO_ROOT)}")
    failed = [k for k, r in results.items() if not r.startswith("ok")]
    return 1 if (failed or not ok) else 0


if __name__ == "__main__":
    sys.exit(main())
