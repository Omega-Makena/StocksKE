"""
importer.py — Download Innova price-list XLS files, compile per-security XLSX files,
and export a unified prices CSV ready for the prediction pipeline.

Typical workflow:
    # 1. Download raw XLS files from Innova
    python importer.py download --start 2024-01-01 --end 2024-12-31

    # 2. Compile into one XLSX per security
    python importer.py compile --in downloaded_price_lists --out compiled_securities

    # 3. Export a unified prices CSV (ticker, date, price) for the pipeline
    python importer.py build-csv --in compiled_securities --out prices.csv

    # Or do all three in one shot:
    python importer.py all --start 2024-01-01 --end 2024-12-31 --prices-csv prices.csv
"""

import argparse
import csv
import logging
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Source URL + filename date format. Both are env-overridable so that when the
# Innova endpoint changes you fix it with a flag/var, not a code edit:
#   IMPORTER_BASE_URL     e.g. https://host/path/Price%20List%20-
#   IMPORTER_DATE_FORMAT  strftime for the date portion (default "%d-%B-%Y")
#   IMPORTER_FILE_EXT     ".xls" (default) or ".xlsx"
#
# NOTE: the live path includes a per-site GUID folder that may rotate. The value
# below was verified working (returns real .xls). If it starts 404-ing, find the
# current URL from the price-list page's Network tab and confirm with:
#   python importer/importer.py probe --date <a-trading-day>
BASE_URL = os.environ.get(
    "IMPORTER_BASE_URL",
    "https://www.innova.co.ke/DBECCB7F-3FDA-4FA1-B075-9BD11288CFF9/Price%20List%20-",
)
DATE_FORMAT = os.environ.get("IMPORTER_DATE_FORMAT", "%d-%B-%Y")
FILE_EXT = os.environ.get("IMPORTER_FILE_EXT", ".xls")
DEFAULT_DOWNLOAD_DIR = "downloaded_price_lists"
DEFAULT_COMPILED_DIR = "compiled_securities"
REQUEST_TIMEOUT = 20
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0

# Magic bytes: OLE2 (legacy .xls) and ZIP (.xlsx / OOXML).
_XLS_MAGIC = b"\xd0\xcf\x11\xe0"
_XLSX_MAGIC = b"PK\x03\x04"


def _is_spreadsheet(content: bytes) -> bool:
    """True if the bytes look like a real .xls/.xlsx file (not an HTML error page)."""
    return content[:4] == _XLS_MAGIC or content[:4] == _XLSX_MAGIC


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_one(session: requests.Session, url: str, dest: Path) -> bool:
    """Download a single URL to dest. Returns True only if the response is a
    genuine spreadsheet — a 200 that is actually an HTML error page (as the
    current Innova site returns) is rejected instead of silently saved."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                if _is_spreadsheet(resp.content):
                    dest.write_bytes(resp.content)
                    return True
                ct = resp.headers.get("Content-Type", "?")
                logger.warning(
                    "200 but NOT a spreadsheet (Content-Type=%s, %d bytes) — likely an "
                    "error page, not saving: %s", ct, len(resp.content), url)
                return False
            if resp.status_code == 404:
                logger.debug("Not found (404): %s", url)
                return False
            logger.warning("HTTP %s for %s", resp.status_code, url)
        except requests.RequestException as exc:
            logger.warning("Request error (attempt %d/%d) for %s: %s", attempt + 1, MAX_RETRIES, url, exc)
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF_BASE ** attempt)
    return False


def build_url(date: datetime, base_url: str = BASE_URL,
              date_format: str = DATE_FORMAT, ext: str = FILE_EXT) -> str:
    """Construct the price-list URL for a date (single place, so probe and
    download always agree)."""
    return f"{base_url}{date.strftime(date_format)}{ext}"


def probe_url(date: datetime, base_url: str = BASE_URL,
              date_format: str = DATE_FORMAT, ext: str = FILE_EXT) -> dict:
    """Fetch one date's URL and report exactly what came back — use this from
    inside the Innova network to discover/confirm the real working URL."""
    url = build_url(date, base_url, date_format, ext)
    info = {"url": url}
    try:
        with requests.Session() as s:
            s.headers.update({"User-Agent": "NSE-Research-Bot/1.0"})
            r = s.get(url, timeout=REQUEST_TIMEOUT)
        info.update({
            "status": r.status_code,
            "content_type": r.headers.get("Content-Type", "?"),
            "bytes": len(r.content),
            "is_spreadsheet": _is_spreadsheet(r.content),
        })
    except requests.RequestException as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def download_price_lists(
    start_date: datetime,
    end_date: datetime,
    output_folder: str = DEFAULT_DOWNLOAD_DIR,
    base_url: str = BASE_URL,
    date_format: str = DATE_FORMAT,
    ext: str = FILE_EXT,
) -> list[Path]:
    """
    Download XLS price lists for every calendar day in [start_date, end_date].
    Returns list of successfully saved file paths.
    """
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)

    days = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(days)]

    saved: list[Path] = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": "NSE-Research-Bot/1.0"})
        for date in dates:
            formatted = date.strftime(date_format)
            url = build_url(date, base_url, date_format, ext)
            dest = out / f"{formatted}{ext}"
            if dest.exists():
                logger.info("Already exists, skipping: %s", dest.name)
                saved.append(dest)
                continue
            logger.info("Downloading: %s", url)
            if _download_one(session, url, dest):
                logger.info("Saved: %s", dest.name)
                saved.append(dest)
            else:
                logger.warning("Failed: %s", url)

    logger.info("Download complete. %d/%d files saved to '%s'.", len(saved), len(dates), output_folder)
    return saved


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------

# Column positions in the current Innova "Price-List" sheet (read with
# header=None, so these are raw column indices). Layout as of 2024:
#   0,1 = 52wk High/Low | 2 = security name | 3 = ISIN | 4 = status (xd/cd)
#   5 = High | 6 = Low | 7 = VWAP | 8 = Previous | 9 = Volume
_C_NAME, _C_ISIN, _C_STATUS = 2, 3, 4
_C_HIGH, _C_LOW, _C_VWAP, _C_PREV, _C_VOL = 5, 6, 7, 8, 9


def _cell(row, i):
    return row.iloc[i] if i < len(row) else None


def _parse_row(row, date: str) -> dict | None:
    """Extract one security record from a Price-List row. Returns None for header
    and sector-title rows (which have no 'Ord' in the name column)."""
    name_cell = _cell(row, _C_NAME)
    if not isinstance(name_cell, str):
        return None
    security = name_cell.strip()
    if "Ord" not in security:          # sector headers / titles lack "Ord X.XX"
        return None

    isin = _cell(row, _C_ISIN)
    isin = str(isin).strip() if not pd.isna(isin) else ""
    return {
        "Security": security,
        "Date": date,
        "ISIN": isin,
        "Highest Price": _cell(row, _C_HIGH),
        "Lowest Price": _cell(row, _C_LOW),
        "Weighted Price": _cell(row, _C_VWAP),   # VWAP — the day's price
        "Previous Price": _cell(row, _C_PREV),   # fallback when not traded
        "Volume": _cell(row, _C_VOL),
    }


def _read_sheet(path: Path) -> pd.DataFrame | None:
    """Read the price-list sheet with header=None so column positions are stable.
    Prefers the 'Price-List' sheet (current format), falling back to older names."""
    for sheet in ("Price-List", "Sheet1", 0):
        try:
            df = pd.read_excel(path, sheet_name=sheet, header=None)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
    logger.warning("Could not read any sheet from %s", path.name)
    return None


def compile_securities(
    input_folder: str,
    output_folder: str = DEFAULT_COMPILED_DIR,
) -> dict[str, Path]:
    """
    Parse all XLS/XLSX files in input_folder, compile one XLSX per security.
    Returns dict {security_name: output_path}.
    """
    inp = Path(input_folder)
    if not inp.exists():
        raise FileNotFoundError(f"Input folder not found: {inp}")

    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)

    securities_data: dict[str, list[dict]] = defaultdict(list)

    for file in sorted(inp.glob("*.xls*")):
        date = file.stem
        df = _read_sheet(file)
        if df is None:
            continue
        for _, row in df.iterrows():
            record = _parse_row(row, date)
            if record:
                securities_data[record["Security"]].append(record)

    written: dict[str, Path] = {}
    for security, records in securities_data.items():
        df = pd.DataFrame(records)
        df.sort_values("Date", inplace=True)
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in security)
        dest = out / f"{safe_name}.xlsx"
        df.to_excel(dest, index=False)
        written[security] = dest

    logger.info("Compiled %d securities to '%s'.", len(written), output_folder)
    return written


# ---------------------------------------------------------------------------
# Export unified prices CSV
# ---------------------------------------------------------------------------

_DATE_FORMATS = ("%d-%B-%Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y")


def _normalise_date(raw: str) -> str | None:
    """
    Convert any of the date formats produced by the importer (e.g. '01-June-2020')
    or already-ISO dates to 'YYYY-MM-DD'. Returns None if unparseable.
    """
    raw = str(raw).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return None


def _strip_ord_suffix(name: str) -> str:
    """'KCB Group Plc Ord 1_00' → 'KCB Group Plc'  (handles _ from safe-name encoding)."""
    return re.sub(r"\s+[Oo]rd[\s._].*$", "", name).strip()


# Generic corporate tokens that don't identify a company — stripped before
# fuzzy matching so 'Unga Group Ltd' matches registry 'Unga Group Plc', etc.
_CORP_STOP = {"plc", "ltd", "limited", "co", "company", "the", "group",
              "holdings", "kenya", "k"}


def _tokens(s: str, drop_stop: bool = False) -> set[str]:
    toks = set(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())
    return toks - _CORP_STOP if drop_stop else toks


def _resolve_ticker(clean_name: str, ticker_map: dict[str, str]) -> str | None:
    """Resolve a security name to a ticker. Exact match first, then a token-subset
    match on distinctive tokens (generic 'Plc/Ltd/Group/Kenya…' stripped), so the
    registry name ('Kapchorua Tea') matches the file's fuller name ('Kapchorua
    Tea Kenya Plc'). Prefers the most-specific (longest) registry match."""
    key = clean_name.lower().strip()
    if key in ticker_map:
        return ticker_map[key]
    sec = _tokens(clean_name, drop_stop=True)
    if not sec:
        return None
    best, best_n = None, 0
    for name, ticker in ticker_map.items():
        ct = _tokens(name, drop_stop=True)
        if not ct or not (ct <= sec):
            continue
        # A single distinctive token (e.g. {equity}) must match EXACTLY, not as a
        # subset — otherwise "Equity Afia Ltd" would wrongly resolve to Equity
        # Group. Multi-token registry names may match as a subset.
        if len(ct) == 1 and ct != sec:
            continue
        if len(ct) > best_n:
            best, best_n = ticker, len(ct)
    return best


def build_prices_csv(
    compiled_dir: str,
    output_csv: str,
    ticker_map: dict[str, str],
) -> Path:
    """
    Read all per-security XLSX files from compiled_dir, resolve each security
    name to an NSE ticker using ticker_map, and write a single unified CSV:

        Stock Code, Date, Day's Final Price

    'Day's Final Price' is the Weighted Price (VWAP) from the compiled XLSX.

    ticker_map must be {normalised_company_name_lowercase: ticker}, e.g.:
        {"kcb group plc": "KCB", "safaricom plc": "SCOM", ...}

    Files whose names cannot be resolved to a ticker are skipped (with a warning).
    Returns the path to the written CSV.
    """
    inp = Path(compiled_dir)
    if not inp.exists():
        raise FileNotFoundError(f"Compiled securities directory not found: {inp}")

    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    skipped_files = []

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Stock Code", "Date", "Day's Final Price"])

        for xlsx in sorted(inp.glob("*.xlsx")):
            # Recover company name from safe filename:
            # "KCB Group Plc Ord 1_00.xlsx" → strip "Ord..." → "KCB Group Plc"
            raw_name = xlsx.stem          # e.g. "KCB Group Plc Ord 1_00"
            clean = _strip_ord_suffix(raw_name)
            ticker = _resolve_ticker(clean, ticker_map)

            if not ticker:
                logger.warning("No ticker found for '%s' (file: %s) — skipped", clean, xlsx.name)
                skipped_files.append(xlsx.name)
                continue

            try:
                df = pd.read_excel(xlsx)
            except Exception as exc:
                logger.warning("Failed to read %s: %s", xlsx.name, exc)
                continue

            for _, row in df.iterrows():
                raw_date = row.get("Date", "")
                # VWAP is the traded price; fall back to Previous when untraded
                price = row.get("Weighted Price")
                if pd.isna(price):
                    price = row.get("Previous Price")
                if pd.isna(price) or raw_date == "":
                    continue
                iso_date = _normalise_date(str(raw_date))
                if not iso_date:
                    logger.debug("Unparseable date '%s' in %s — skipped row", raw_date, xlsx.name)
                    continue
                try:
                    writer.writerow([ticker, iso_date, float(price)])
                    rows_written += 1
                except (TypeError, ValueError):
                    pass

    if skipped_files:
        logger.warning(
            "%d file(s) skipped (no ticker match): %s",
            len(skipped_files),
            ", ".join(skipped_files[:5]) + ("…" if len(skipped_files) > 5 else ""),
        )
    logger.info("Prices CSV written: %d rows → %s", rows_written, out)
    return out


# ---------------------------------------------------------------------------
# Ticker map helpers
# ---------------------------------------------------------------------------

def _load_ticker_map(companies_json: str | None = None) -> dict[str, str]:
    """
    Return {normalised_company_name_lower: ticker}.

    Priority:
    1. If companies_json path given → load from that JSON file.
    2. Try to import pipeline.companies (auto-discovery when running from
       the project root or when pipeline/ is on sys.path).
    3. Fall back to an empty dict (all securities will be skipped with warnings).
    """
    import json as _json
    import sys as _sys

    if companies_json:
        try:
            with open(companies_json, encoding="utf-8") as fh:
                raw: dict = _json.load(fh)
            return {k.lower(): v for k, v in raw.items()}
        except Exception as exc:
            logger.warning("Could not load ticker map from %s: %s", companies_json, exc)

    # Auto-discover pipeline/companies.py
    pipeline_dir = Path(__file__).parent.parent / "pipeline"
    if pipeline_dir.exists() and str(pipeline_dir) not in _sys.path:
        _sys.path.insert(0, str(pipeline_dir))
    try:
        from companies import NAME_MAP  # {ticker: "Full Company Name"}
        return {name.lower(): ticker for ticker, name in NAME_MAP.items()}
    except ImportError:
        logger.warning(
            "Could not import pipeline/companies.py. "
            "Provide --companies-json or ensure pipeline/ is on PYTHONPATH."
        )
        return {}


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="NSE Innova price-list importer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    dl = sub.add_parser("download", help="Download XLS files from Innova")
    dl.add_argument("--start", required=True, type=_parse_date, metavar="YYYY-MM-DD")
    dl.add_argument("--end",   required=True, type=_parse_date, metavar="YYYY-MM-DD")
    dl.add_argument("--out",   default=DEFAULT_DOWNLOAD_DIR)
    dl.add_argument("--base-url", default=BASE_URL)
    dl.add_argument("--date-format", default=DATE_FORMAT,
                    help=r"strftime for the filename date (default %%d-%%B-%%Y)")
    dl.add_argument("--ext", default=FILE_EXT, help="file extension (default .xls)")

    pr = sub.add_parser("probe", help="Test one date's URL and report what it returns")
    pr.add_argument("--date", required=True, type=_parse_date, metavar="YYYY-MM-DD")
    pr.add_argument("--base-url", default=BASE_URL)
    pr.add_argument("--date-format", default=DATE_FORMAT)
    pr.add_argument("--ext", default=FILE_EXT)

    cp = sub.add_parser("compile", help="Compile downloaded files into per-security XLSX")
    cp.add_argument("--in",  dest="input_folder",  required=True)
    cp.add_argument("--out", dest="output_folder", default=DEFAULT_COMPILED_DIR)

    bc = sub.add_parser("build-csv", help="Export unified prices CSV from compiled XLSX files")
    bc.add_argument("--in",  dest="compiled_dir", required=True,
                    help="Folder containing compiled per-security XLSX files")
    bc.add_argument("--out", dest="output_csv",   required=True,
                    help="Output CSV path, e.g. prices.csv")
    bc.add_argument(
        "--companies-json",
        default=None,
        help=(
            "Optional JSON file mapping lowercase company name → ticker. "
            "If omitted, the script tries to load pipeline/companies.py automatically."
        ),
    )

    al = sub.add_parser("all", help="Download, compile, then export unified prices CSV")
    al.add_argument("--start", required=True, type=_parse_date, metavar="YYYY-MM-DD")
    al.add_argument("--end",   required=True, type=_parse_date, metavar="YYYY-MM-DD")
    al.add_argument("--download-dir", default=DEFAULT_DOWNLOAD_DIR)
    al.add_argument("--compiled-dir", default=DEFAULT_COMPILED_DIR)
    al.add_argument("--prices-csv",   default="prices.csv",
                    help="Where to write the unified prices CSV (default: prices.csv)")
    al.add_argument("--base-url", default=BASE_URL)

    args = parser.parse_args()

    if args.cmd == "download":
        download_price_lists(args.start, args.end, args.out, args.base_url,
                             args.date_format, args.ext)

    elif args.cmd == "probe":
        info = probe_url(args.date, args.base_url, args.date_format, args.ext)
        print("\nPROBE RESULT")
        for k, v in info.items():
            print(f"  {k:14}: {v}")
        if info.get("is_spreadsheet"):
            print("\n  [OK] This URL returns a real spreadsheet - the importer will work.\n")
        else:
            print("\n  [FAIL] Not a spreadsheet. Try another --date-format / --base-url / --ext,\n"
                  "    or copy the real URL from your browser's Network tab on the page that\n"
                  "    downloads the price list, then pass it via --base-url.\n")

    elif args.cmd == "compile":
        compile_securities(args.input_folder, args.output_folder)

    elif args.cmd == "build-csv":
        ticker_map = _load_ticker_map(args.companies_json)
        build_prices_csv(args.compiled_dir, args.output_csv, ticker_map)

    elif args.cmd == "all":
        download_price_lists(args.start, args.end, args.download_dir, args.base_url)
        compile_securities(args.download_dir, args.compiled_dir)
        ticker_map = _load_ticker_map(None)
        build_prices_csv(args.compiled_dir, args.prices_csv, ticker_map)


if __name__ == "__main__":
    main()
