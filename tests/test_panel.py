import csv
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from crypto_generative.data.binance import OUTPUT_COLUMNS
from crypto_generative.data.panel import AssetInput, BtcEthPanelBuilder


def raw_row(timestamp, symbol, base_asset, close="11", duration_hours=6):
    open_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    close_time = (
        open_time + timedelta(hours=duration_hours) - timedelta(milliseconds=1)
    ).isoformat().replace("+00:00", "Z")
    return {
        "open_time_utc": timestamp,
        "close_time_utc": close_time,
        "open": "10",
        "high": "12",
        "low": "9",
        "close": close,
        "volume": "2",
        "quote_asset_volume": "21",
        "number_of_trades": "5",
        "taker_buy_base_volume": "1",
        "taker_buy_quote_volume": "10",
        "exchange": "Binance",
        "market_type": "spot",
        "symbol": symbol,
        "base_asset": base_asset,
        "quote_asset": "USDT",
        "interval": "6h",
    }


def write_raw(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class BtcEthPanelBuilderTest(unittest.TestCase):
    def test_builds_regular_panel_without_imputing_common_gap(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            btc_path, eth_path = root / "btc.csv", root / "eth.csv"
            timestamps = ["2020-01-01T00:00:00Z", "2020-01-01T12:00:00Z"]
            write_raw(btc_path, [raw_row(value, "BTCUSDT", "BTC") for value in timestamps])
            write_raw(eth_path, [raw_row(value, "ETHUSDT", "ETH") for value in timestamps])

            rows, audit = BtcEthPanelBuilder().build(
                (
                    AssetInput("btc", "BTCUSDT", "BTC", "USDT", btc_path),
                    AssetInput("eth", "ETHUSDT", "ETH", "USDT", eth_path),
                )
            )

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[1]["open_time_utc"], "2020-01-01T06:00:00Z")
        self.assertEqual(rows[1]["btc_close"], "")
        self.assertEqual(rows[1]["eth_close"], "")
        self.assertEqual(rows[1]["is_complete"], "0")
        self.assertEqual(audit.common_missing_rows, 1)
        self.assertEqual(audit.one_sided_missing_rows, 0)

    def test_marks_truncated_candle_without_discarding_its_values(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            btc_path, eth_path = root / "btc.csv", root / "eth.csv"
            write_raw(
                btc_path,
                [raw_row("2020-01-01T00:00:00Z", "BTCUSDT", "BTC", duration_hours=4)],
            )
            write_raw(
                eth_path,
                [raw_row("2020-01-01T00:00:00Z", "ETHUSDT", "ETH", duration_hours=4)],
            )

            rows, audit = BtcEthPanelBuilder().build(
                (
                    AssetInput("btc", "BTCUSDT", "BTC", "USDT", btc_path),
                    AssetInput("eth", "ETHUSDT", "ETH", "USDT", eth_path),
                )
            )

        self.assertEqual(rows[0]["btc_close"], "11")
        self.assertEqual(rows[0]["btc_source_close_time_utc"], "2020-01-01T03:59:59.999000Z")
        self.assertEqual(rows[0]["expected_close_time_utc"], "2020-01-01T05:59:59.999000Z")
        self.assertEqual(rows[0]["btc_duration_valid"], "0")
        self.assertEqual(rows[0]["is_complete"], "0")
        self.assertEqual(audit.common_invalid_duration_rows, 1)

    def test_rejects_inconsistent_market_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            btc_path, eth_path = root / "btc.csv", root / "eth.csv"
            write_raw(btc_path, [raw_row("2020-01-01T00:00:00Z", "BTCUSD", "BTC")])
            write_raw(eth_path, [raw_row("2020-01-01T00:00:00Z", "ETHUSDT", "ETH")])

            with self.assertRaisesRegex(ValueError, "Metadato invalido"):
                BtcEthPanelBuilder().build(
                    (
                        AssetInput("btc", "BTCUSDT", "BTC", "USDT", btc_path),
                        AssetInput("eth", "ETHUSDT", "ETH", "USDT", eth_path),
                    )
                )


if __name__ == "__main__":
    unittest.main()
