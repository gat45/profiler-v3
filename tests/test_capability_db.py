"""Tests pour sanity_check_output() (2026-09-08) -- detection de derive
anormale entre deux executions (meme prompt/seed), reponse a la critique
sur la corruption silencieuse (cf. bug gallocr #28448)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import capability_db as cdb  # noqa: E402


class TestSanityCheck(unittest.TestCase):
    def test_identical_output_is_ok(self):
        r = cdb.sanity_check_output("le chat mange une pomme", "le chat mange une pomme")
        self.assertTrue(r["ok"])
        self.assertEqual(r["similarity"], 1.0)

    def test_empty_output_flagged(self):
        r = cdb.sanity_check_output("le chat mange une pomme", "")
        self.assertFalse(r["ok"])
        self.assertTrue(any("courte" in a for a in r["alerts"]))

    def test_pathological_repetition_flagged(self):
        cand = "le " * 10
        r = cdb.sanity_check_output("le chat mange", cand)
        self.assertFalse(r["ok"])
        self.assertTrue(any("repetition" in a for a in r["alerts"]))

    def test_total_content_change_flagged(self):
        r = cdb.sanity_check_output(
            "le chat mange une pomme rouge dans le jardin",
            "quantum blockchain synergy paradigm shift disruption")
        self.assertFalse(r["ok"])
        self.assertTrue(any("similarite" in a for a in r["alerts"]))


if __name__ == "__main__":
    unittest.main()
