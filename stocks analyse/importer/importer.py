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
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://www.innova.co.ke/files/Price%20List%20-"
DEFAULT_DOWNLOAD_DIR = "downloaded_price_lists"
DEFAULT_COMPILED_DIR = "compiled_securities"
REQUEST_TIMEOUT = 20
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_one(session: requests.Session, url: str, dest: Path) -> bool:
    """Download a single URL to dest. Returns True on success."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                dest.write_bytes(resp.content)
                return True
            if resp.status_code == 404:
                logger.debug("Not found (404): %s", url)
                return False
            logger.warning("HTTP %s for %s", resp.status_code, url)
        except requests.RequestException as exc:
            logger.warning("Request error (attempt %d/%d) for %s: %s", attempt + 1, MAX_RETRIES, url, exc)
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF_BASE ** attempt)
    return False


def download_price_lists(
    start_date: datetime,
    end_date: datetime,
    output_folder: str = DEFAULT_DOWNLOAD_DIR,
    base_url: str = BASE_URL,
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
            formatted = date.strftime("%d-%B-%Y")
            url = f"{base_url}{formatted}.xls"
            dest = out / f"{formatted}.xls"
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

def _parse_row(row, date: str) -> dict | None:
    """Extract one securities record from a DataFrame row. Returns None to skip."""
    if pd.isna(row.iloc[0]) or not isinstance(row.iloc[0], str):
        return None

    security = row.iloc[0].strip()
    if "Ord" not in security:
        return None

    col1 = str(row.iloc[1]).strip() if not pd.isna(row.iloc[1]) else ""
    has_isin = col1.startswith("KE") and len(col1) == 12

    if has_isin:
        total_shares, market_cap, vwap, high, low = (
            row.iloc[2], row.iloc[3], row.iloc[4], row.iloc[5], row.iloc[6],
        )
    else:
        total_shares, market_cap, vwap, high, low = (
            row.iloc[1], row.iloc[2], row.iloc[3], row.iloc[4], row.iloc[5],
        )

    return {
        "Date": date,
        "Total Shares Issued": total_shares,
        "Market Cap": market_cap,
        "Weighted Price": vwap,
        "Highest Price": high,
        "Lowest Price": low,
    }


def _read_sheet(path: Path) -> pd.DataFrame | None:
    """Try to read the first available sheet from an XLS/XLSX file."""
    for sheet in ("Sheet1", 0):
        try:
            return pd.read_excel(path, sheet_name=sheet)
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
                securities_data[row.iloc[0].strip()].append(record)

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
            ticker = ticker_map.get(clean.lower())

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
                price = row.get("Weighted Price")
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
        download_price_lists(args.start, args.end, args.out, args.base_url)

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
