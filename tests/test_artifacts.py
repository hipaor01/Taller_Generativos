import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from crypto_generative.data.artifacts import relative_or_absolute


class RelativeOrAbsoluteTest(unittest.TestCase):
    def test_relative_project_path_is_portable(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "tmp" / "btcusdt_6h.csv"

            self.assertEqual(
                relative_or_absolute(path, root),
                "tmp/btcusdt_6h.csv",
            )

    def test_external_path_remains_absolute(self):
        with TemporaryDirectory() as root_directory, TemporaryDirectory() as external:
            path = Path(external) / "btcusdt_6h.csv"

            self.assertEqual(
                relative_or_absolute(path, Path(root_directory)),
                str(path.resolve()),
            )


if __name__ == "__main__":
    unittest.main()
