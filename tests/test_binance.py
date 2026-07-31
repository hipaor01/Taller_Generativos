import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from crypto_generative.data.binance import BinanceKlineClient


def kline(open_time, open_price="10", volume="2"):
    return [
        open_time,
        open_price,
        "12",
        "9",
        "11",
        volume,
        open_time + 21_599_999,
        "21",
        5,
        "1",
        "10",
        "0",
    ]


class BinanceKlineClientTest(unittest.TestCase):
    def test_fetch_paginates_without_repeating_boundary(self):
        interval_ms = 6 * 60 * 60 * 1000
        first_batch = [kline(index * interval_ms) for index in range(1000)]
        second_batch = [kline(1000 * interval_ms)]
        client = BinanceKlineClient()

        with patch.object(client, "_request_json", side_effect=[first_batch, second_batch]) as request:
            rows = client.fetch_klines("BTCUSDT", "6h", 0, 1001 * interval_ms)

        self.assertEqual(len(rows), 1001)
        self.assertEqual(request.call_args_list[1].args[1]["startTime"], 1000 * interval_ms)

    def test_audit_detects_gap_duplicate_and_invalid_values(self):
        interval_ms = 6 * 60 * 60 * 1000
        rows = [kline(0), kline(0), kline(2 * interval_ms, open_price="0", volume="-1")]

        audit = BinanceKlineClient().audit(rows, "6h")

        self.assertEqual(audit.expected_rows_between_first_and_last, 3)
        self.assertEqual(audit.duplicate_open_times, 1)
        self.assertEqual(audit.missing_candles, 1)
        self.assertEqual(audit.non_positive_ohlc_rows, 1)
        self.assertEqual(audit.negative_volume_rows, 1)

    def test_write_dataset_keeps_market_provenance(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "btc.csv"
            BinanceKlineClient().write_dataset(
                [kline(0)],
                output,
                symbol="BTCUSDT",
                base_asset="BTC",
                quote_asset="USDT",
                interval="6h",
            )

            text = output.read_text(encoding="utf-8")

        self.assertIn("open_time_utc", text)
        self.assertIn("Binance,spot,BTCUSDT,BTC,USDT,6h", text)


if __name__ == "__main__":
    unittest.main()
