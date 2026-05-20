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
)


class TestParseRow:
    def _row(self, values: list):
        return pd.Series(values)

    def test_skips_non_string_first_column(self):
        assert _parse_row(self._row([None, 100, 200, 10, 5, 4]), "2024-03-01") is None
        assert _parse_row(self._row([123,  100, 200, 10, 5, 4]), "2024-03-01") is None

    def test_skips_row_without_ord_in_name(self):
        assert _parse_row(self._row(["Safaricom Plc", 100, 200, 10, 5, 4]), "2024-03-01") is None

    def test_parses_row_without_isin(self):
        row = self._row(["Equity Ord 0.50", 1000000, 5000000, 45.0, 46.0, 44.0])
        record = _parse_row(row, "2024-03-01")
        assert record is not None
        assert record["Date"] == "2024-03-01"
        assert record["Weighted Price"] == 45.0
        assert record["Highest Price"] == 46.0
        assert record["Lowest Price"] == 44.0
        assert record["Total Shares Issued"] == 1000000

    def test_parses_row_with_isin(self):
        row = self._row(["Equity Ord 0.50", "KE1000001234", 1000000, 5000000, 45.0, 46.0, 44.0])
        record = _parse_row(row, "2024-03-01")
        assert record is not None
        assert record["Weighted Price"] == 45.0
        assert record["Total Shares Issued"] == 1000000

    def test_isin_detection_requires_ke_prefix_and_12_chars(self):
        row_no_ke = self._row(["KCB Ord 1.00", "US1000001234", 1000000, 5000000, 45.0, 46.0, 44.0])
        record = _parse_row(row_no_ke, "2024-03-01")
        assert record["Total Shares Issued"] == "US1000001234"

        row_short = self._row(["KCB Ord 1.00", "KE10000", 1000000, 5000000, 45.0, 46.0, 44.0])
        record2 = _parse_row(row_short, "2024-03-01")
        assert record2["Total Shares Issued"] == "KE10000"


class TestCompileSecurities:
    def _make_xls(self, tmp_dir: Path, filename: str, rows: list[dict]) -> Path:
        df = pd.DataFrame(rows)
        path = tmp_dir / filename
        df.to_excel(path, index=False, sheet_name="Sheet1")
        return path

    def test_compiles_single_security(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        dst = tmp_path / "dst"
        self._make_xls(
            src,
            "01-January-2024.xlsx",
            [
                {0: "Equity Ord 0.50", 1: 1_000_000, 2: 50_000_000, 3: 45.0, 4: 46.0, 5: 44.0},
                {0: None, 1: None, 2: None, 3: None, 4: None, 5: None},
            ],
        )
        result = compile_securities(str(src), str(dst))
        assert len(result) == 1
        key = next(iter(result))
        assert "Equity Ord" in key

    def test_output_file_is_valid_excel(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        dst = tmp_path / "dst"
        self._make_xls(
            src,
            "01-January-2024.xlsx",
            [{0: "KCB Ord 1.00", 1: 500_000, 2: 25_000_000, 3: 50.0, 4: 51.0, 5: 49.0}],
        )
        result = compile_securities(str(src), str(dst))
        for path in result.values():
            df = pd.read_excel(path)
            assert "Weighted Price" in df.columns

    def test_raises_for_missing_input_folder(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            compile_securities(str(tmp_path / "nonexistent"))

    def test_safe_name_sanitises_special_chars(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        dst = tmp_path / "dst"
        self._make_xls(
            src,
            "01-January-2024.xlsx",
            [{0: "Co/Op Ord 1.00", 1: 500_000, 2: 25_000_000, 3: 10.0, 4: 11.0, 5: 9.0}],
        )
        result = compile_securities(str(src), str(dst))
        for path in result.values():
            assert "/" not in path.name


class TestDownloadPriceLists:
    def test_skips_existing_files(self, tmp_path):
        existing = tmp_path / "01-June-2020.xls"
        existing.write_bytes(b"data")
        with patch("importer.requests.Session") as mock_session_cls:
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
        mock_resp.content = b"xls-bytes"

        with patch("importer.requests.Session") as mock_session_cls:
            mock_sess = MagicMock()
            mock_sess.__enter__ = lambda s: s
            mock_sess.__exit__ = MagicMock(return_value=False)
            mock_sess.get.return_value = mock_resp
            mock_session_cls.return_value = mock_sess

            result = download_price_lists(
                datetime(2020, 6, 1), datetime(2020, 6, 1), str(tmp_path)
            )

        assert len(result) == 1
        assert result[0].read_bytes() == b"xls-bytes"

    def test_returns_empty_on_404(self, tmp_path):
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch("importer.requests.Session") as mock_session_cls:
            mock_sess = MagicMock()
            mock_sess.__enter__ = lambda s: s
            mock_sess.__exit__ = MagicMock(return_value=False)
            mock_sess.get.return_value = mock_resp
            mock_session_cls.return_value = mock_sess

            result = download_price_lists(
                datetime(2020, 6, 1), datetime(2020, 6, 1), str(tmp_path)
            )

        assert result == []
