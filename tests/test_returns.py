import csv
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from crypto_generative.data.returns import LogReturnBuilder


PANEL_COLUMNS = (
    "open_time_utc",
    "btc_close",
    "eth_close",
    "btc_missing",
    "btc_duration_valid",
    "eth_missing",
    "eth_duration_valid",
    "is_complete",
)


def panel_row(timestamp, btc_close, eth_close, complete="1"):
    incomplete = complete == "0"
    return {
        "open_time_utc": timestamp,
        "btc_close": btc_close,
        "eth_close": eth_close,
        "btc_missing": "1" if incomplete else "0",
        "btc_duration_valid": "0" if incomplete else "1",
        "eth_missing": "1" if incomplete else "0",
        "eth_duration_valid": "0" if incomplete else "1",
        "is_complete": complete,
    }


def write_panel(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PANEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class LogReturnBuilderTest(unittest.TestCase):
    def test_calculates_joint_log_returns(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "panel.csv"
            write_panel(
                path,
                [
                    panel_row("2020-01-01T00:00:00Z", "100", "50"),
                    panel_row("2020-01-01T06:00:00Z", "110", "40"),
                ],
            )

            rows, audit = LogReturnBuilder().build(path)

        self.assertEqual(rows[0]["returns_valid"], "0")
        self.assertEqual(rows[0]["invalid_reason"], "first_observation")
        self.assertAlmostEqual(float(rows[1]["btc_log_return"]), math.log(1.1))
        self.assertAlmostEqual(float(rows[1]["eth_log_return"]), math.log(0.8))
        self.assertEqual(audit.valid_return_rows, 1)

    def test_does_not_bridge_incomplete_candle(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "panel.csv"
            write_panel(
                path,
                [
                    panel_row("2020-01-01T00:00:00Z", "100", "50"),
                    panel_row("2020-01-01T06:00:00Z", "", "", complete="0"),
                    panel_row("2020-01-01T12:00:00Z", "120", "60"),
                    panel_row("2020-01-01T18:00:00Z", "132", "66"),
                ],
            )

            rows, audit = LogReturnBuilder().build(path)

        self.assertEqual(rows[1]["invalid_reason"], "current_candle_incomplete")
        self.assertEqual(
            rows[2]["invalid_reason"],
            "previous_candle_incomplete",
        )
        self.assertEqual(rows[3]["returns_valid"], "1")
        self.assertEqual(audit.valid_return_rows, 1)

    def test_rejects_irregular_calendar(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "panel.csv"
            write_panel(
                path,
                [
                    panel_row("2020-01-01T00:00:00Z", "100", "50"),
                    panel_row("2020-01-01T12:00:00Z", "110", "55"),
                ],
            )

            with self.assertRaisesRegex(ValueError, "Calendario no regular"):
                LogReturnBuilder().build(path)


if __name__ == "__main__":
    unittest.main()
