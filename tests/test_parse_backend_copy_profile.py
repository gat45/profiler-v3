"""Tests pour le mode degrade de parse_backend_copy_profile.py (2026-09-08).

Avant ce fix, un CSV absent (patch non applique, voir PATCHES.md) et un run
non-mixte legitime (rien a rapporter) produisaient EXACTEMENT le meme
message -- ambigu. Verrouille la distinction.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import parse_backend_copy_profile as pbcp  # noqa: E402


class TestDegradedMode(unittest.TestCase):
    def test_missing_columns_raises_not_instrumented(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            bad_csv = Path(d) / "bad.csv"
            bad_csv.write_text("some,other,columns\n1,2,3\n", encoding="utf-8")
            with self.assertRaises(pbcp.NotInstrumented):
                pbcp.parse_csv(str(bad_csv))

    def test_well_formed_empty_csv_does_not_raise(self):
        """Colonnes correctes mais 0 ligne (run non-mixte) : PAS une erreur
        d'instrumentation, juste rien a rapporter."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            csv_path = Path(d) / "empty.csv"
            csv_path.write_text("tensor,src_backend,dst_backend,bytes,duration_ms\n",
                               encoding="utf-8")
            rows = pbcp.parse_csv(str(csv_path))
            self.assertEqual(rows, [])

    def test_well_formed_csv_parses(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            csv_path = Path(d) / "ok.csv"
            csv_path.write_text(
                "tensor,src_backend,dst_backend,bytes,duration_ms\n"
                "ffn_out-3,CPU,GPU,1024,0.5\n",
                encoding="utf-8")
            rows = pbcp.parse_csv(str(csv_path))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["layer"], 3)


if __name__ == "__main__":
    unittest.main()
