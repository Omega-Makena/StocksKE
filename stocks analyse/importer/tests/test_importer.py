"""Tests for importer.py"""
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from importer import (
    _parse_row,
    _read_sheet,
    compile_securities,
    download_price_lists,
    _is_spreadsheet,
    build_url,
    _download_one,
    _resolve_ticker,
)


class TestSpreadsheetValidation:
    def test_detects_xls_magic(self):
        assert _is_spreadsheet(b"\xd0\xcf\x11\xe0somexlsdata")

    def test_detects_xlsx_magic(self):
        assert _is_spreadsheet(b"PK\x03\x04rest-of-zip")

    def test_rejects_html_error_page(self):
        assert not _is_spreadsheet(b"<!DOCTYPE html><html>404 Not Found</html>")

    def test_rejects_empty(self):
        assert not _is_spreadsheet(b"")


class TestBuildUrl:
    def test_default_pattern(self):
        u = build_url(datetime(2025, 6, 16))
        assert u.endswith("16-June-2025.xls")

    def test_custom_format_and_ext(self):
        u = build_url(datetime(2025, 6, 16), base_url="https://x/P-",
                      date_format="%Y-%m-%d", ext=".xlsx")
        assert u == "https://x/P-2025-06-16.xlsx"


class TestDownloadRejectsErrorPages:
    def test_200_html_is_not_saved(self, tmp_path):
        # A 200 response whose body is an HTML error page must NOT be saved.
        resp = MagicMock(status_code=200, content=b"<html>not found</html>",
                         headers={"Content-Type": "text/html"})
        session = MagicMock()
        session.get.return_value = resp
        dest = tmp_path / "out.xls"
        assert _download_one(session, "http://x/f.xls", dest) is False
        assert not dest.exists()

    def test_200_real_xls_is_saved(self, tmp_path):
        resp = MagicMock(status_code=200, content=b"\xd0\xcf\x11\xe0DATA",
                         headers={"Content-Type": "application/vnd.ms-excel"})
        session = MagicMock()
        session.get.return_value = resp
        dest = tmp_path / "out.xls"
        assert _download_one(session, "http://x/f.xls", dest) is True
        assert dest.read_bytes().startswith(b"\xd0\xcf\x11\xe0")


class TestParseRow:
    def _row(self, name, isin="", high=None, low=None, vwap=None, prev=None,
             vol=None, wk_hi=None, wk_lo=None, status=None):
        # Current Innova "Price-List" layout (positional, header=None):
        # [0]52wkH [1]52wkL [2]name [3]ISIN [4]status [5]High [6]Low [7]VWAP [8]Prev [9]Vol
        return pd.Series([wk_hi, wk_lo, name, isin, status, high, low, vwap, prev, vol])

    def test_skips_non_string_name_column(self):
        assert _parse_row(self._row(None), "2024-03-01") is None
        assert _parse_row(self._row(123), "2024-03-01") is None

    def test_skips_sector_header_row(self):
        # sector titles (BANKING, AGRICULTURAL) have no "Ord" in the name
        assert _parse_row(self._row("BANKING"), "2024-03-01") is None
        assert _parse_row(self._row("Safaricom Plc"), "2024-03-01") is None

    def test_parses_security_row(self):
        row = self._row("Equity Group Holdings Plc Ord 0.50", isin="KE0000000554",
                        high=44.0, low=43.5, vwap=43.65, prev=43.5, vol=1_749_300)
        rec = _parse_row(row, "2024-06-04")
        assert rec is not None
        assert rec["Date"] == "2024-06-04"
        assert "Equity" in rec["Security"]
        assert rec["ISIN"] == "KE0000000554"
        assert rec["Weighted Price"] == 43.65
        assert rec["Highest Price"] == 44.0
        assert rec["Lowest Price"] == 43.5
        assert rec["Previous Price"] == 43.5
        assert rec["Volume"] == 1_749_300


class TestResolveTicker:
    MAP = {"equity group holdings plc": "EQTY", "kcb group plc": "KCB",
           "kapchorua tea": "KAPC", "unga group plc": "UNGA",
           "centum investment company": "CTUM"}

    def test_exact_match(self):
        assert _resolve_ticker("KCB Group Plc", self.MAP) == "KCB"

    def test_registry_name_is_subset_of_file_name(self):
        # file gives a fuller name than the registry
        assert _resolve_ticker("Kapchorua Tea Kenya Plc", self.MAP) == "KAPC"

    def test_ignores_generic_corporate_suffixes(self):
        assert _resolve_ticker("Unga Group Ltd", self.MAP) == "UNGA"          # Ltd vs Plc
        assert _resolve_ticker("Centum Investment Co Plc", self.MAP) == "CTUM"  # Co vs Company

    def test_unknown_name_returns_none(self):
        assert _resolve_ticker("Totally Unlisted Corp", self.MAP) is None

    def test_single_token_needs_exact_match_not_subset(self):
        # a NEW security sharing one distinctive token must NOT hijack a ticker
        # ('Equity' core is a single token -> requires exact core match)
        assert _resolve_ticker("Equity Afia Ltd", self.MAP) is None
        # the real one still resolves
        assert _resolve_ticker("Equity Group Holdings Plc", self.MAP) == "EQTY"


class TestCompileSecurities:
    def _make_xls(self, tmp_dir: Path, filename: str, rows: list[list]) -> Path:
        # write raw cells (no header) to a "Price-List" sheet, matching the source
        df = pd.DataFrame(rows)
        path = tmp_dir / filename
        df.to_excel(path, index=False, header=False, sheet_name="Price-List")
        return path

    def _sec_row(self, name, high=45.0, low=44.0, vwap=44.5):
        return [None, None, name, "KE0000000554", None, high, low, vwap, 44.0, 1000]

    def test_compiles_single_security(self, tmp_path):
        src = tmp_path / "src"; src.mkdir()
        dst = tmp_path / "dst"
        self._make_xls(src, "01-January-2024.xlsx", [
            [None, None, "BANKING", None, None, None, None, None, None, None],  # sector header
            self._sec_row("Equity Group Holdings Plc Ord 0.50"),
            [None] * 10,                                                          # blank row
        ])
        result = compile_securities(str(src), str(dst))
        assert len(result) == 1
        assert "Equity" in next(iter(result))

    def test_output_file_has_weighted_price(self, tmp_path):
        src = tmp_path / "src"; src.mkdir()
        dst = tmp_path / "dst"
        self._make_xls(src, "01-January-2024.xlsx",
                       [self._sec_row("KCB Group Plc Ord 1.00", vwap=50.0)])
        result = compile_securities(str(src), str(dst))
        for path in result.values():
            df = pd.read_excel(path)
            assert "Weighted Price" in df.columns

    def test_raises_for_missing_input_folder(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            compile_securities(str(tmp_path / "nonexistent"))

    def test_safe_name_sanitises_special_chars(self, tmp_path):
        src = tmp_path / "src"; src.mkdir()
        dst = tmp_path / "dst"
        self._make_xls(src, "01-January-2024.xlsx",
                       [self._sec_row("Co/Op Bank Ord 1.00")])
        result = compile_securities(str(src), str(dst))
        for path in result.values():
            assert "/" not in path.name


class TestDownloadPriceLists:
    def test_skips_existing_files(self, tmp_path):
        existing = tmp_path / "01-June-2020.xls"
        existing.write_bytes(b"data")
        with patch("importer.importer.requests.Session") as mock_session_cls:
            mock_session_cls.return_value.__enter__ = lambda s: s
            mock_session_cls.return_value.__exit__ = MagicMock(return_value=False)
            result = download_price_lists(
                datetime(2020, 6, 1), datetime(2020, 6, 1), str(tmp_path)
            )
        assert len(result) == 1
        assert result[0] == existing

    def test_saves_file_on_200(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        # must be a REAL spreadsheet (magic bytes) or the validator rejects it
        mock_resp.content = b"\xd0\xcf\x11\xe0xls-bytes"
        mock_resp.headers = {"Content-Type": "application/vnd.ms-excel"}

        with patch("importer.importer.requests.Session") as mock_session_cls:
            mock_sess = MagicMock()
            mock_sess.__enter__ = lambda s: s
            mock_sess.__exit__ = MagicMock(return_value=False)
            mock_sess.get.return_value = mock_resp
            mock_session_cls.return_value = mock_sess

            result = download_price_lists(
                datetime(2020, 6, 1), datetime(2020, 6, 1), str(tmp_path)
            )

        assert len(result) == 1
        assert result[0].read_bytes().startswith(b"\xd0\xcf\x11\xe0")

    def test_returns_empty_on_404(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("importer.importer.requests.Session") as mock_session_cls:
            mock_sess = MagicMock()
            mock_sess.__enter__ = lambda s: s
            mock_sess.__exit__ = MagicMock(return_value=False)
            mock_sess.get.return_value = mock_resp
            mock_session_cls.return_value = mock_sess

            result = download_price_lists(
                datetime(2020, 6, 1), datetime(2020, 6, 1), str(tmp_path)
            )

        assert result == []
