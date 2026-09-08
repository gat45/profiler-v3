"""Tests unitaires pour parse_hexagon_profile.py (ajout 2026-09-08).

Reponse directe a une critique recue : "l'absence de tests est aujourd'hui
la faille la plus bloquante pour toute adoption externe". Ce fichier ne
pretend pas couvrir tout le parseur -- il verrouille la REGRESSION precise
qui a motive le fix du 2026-09-08 (mapping 56.0% -> 99.99%/100%), pour que
personne ne la reintroduise sans s'en apercevoir.

N'utilise PAS les traces reelles de device_logs/ (trop grosses, seule
prof3_qwen09b.log — 80 Mo — dispose de trace-evt) : une fixture synthetique
minimale reproduit exactement le trou inter-op qui causait la perte
silencieuse d'evenements, sans dependance a un fichier volumineux.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import parse_hexagon_profile as php  # noqa: E402


def _write_fixture(path):
    """2 ops (couche 0 et couche 1) avec un TROU de cycles entre les deux
    (op0 finit a 1100, op1 commence a 1500) + 2 evenements trace-evt :
    - un a 1050 (a l'interieur de la fenetre [1000,1100] de op0)
    - un a 1300 (dans le TROU inter-op -- perdu par l'ancienne logique
      [start,end], mappe par la nouvelle [start, debut_op_suivant))
    """
    lines = [
        # OPBATCH : seed du CycleUnwrapper a cycles_start=1000
        "0.0.0 D ggml-hex: HTP0 profile-op OPBATCH|----|n-ops 2|----|----|----|"
        "usec 700 cycles 700 start 1000 mhz 1.0",
        # op couche 0 : fenetre [1000, 1100]
        "0.0.1 D ggml-hex: HTP0 profile-op MUL_MAT|blk.0.attn_q.weight|1:1|f32|1:1|"
        "hvx-tiled vtcm 1000|usec 10 cycles 100 start 1000 mhz 10.0",
        # trace-evt A : a l'interieur de la fenetre de op0 -> mappe de tout temps
        "0.0.2 D ggml-hex: HTP0 trace-evt HVX_COMP: thread 0 info 0 start 1050",
        # trace-evt B : dans le trou inter-op (1100 a 1500) -> AVANT le fix,
        # ne correspondait a la fenetre [start,end] d'AUCUN op -> perdu
        # silencieusement. APRES le fix, tombe dans [1000, 1500) -> couche 0.
        "0.0.3 D ggml-hex: HTP0 trace-evt DMA: thread 0 info 0 start 1300",
        # op couche 1 : fenetre [1500, 1700]
        "0.0.4 D ggml-hex: HTP0 profile-op MUL_MAT|blk.1.attn_q.weight|1:1|f32|1:1|"
        "hvx-tiled vtcm 1000|usec 10 cycles 200 start 1500 mhz 10.0",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestMappingRegression(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.TemporaryDirectory()
        self.log_path = Path(self.tmpdir.name) / "fixture.log"
        _write_fixture(self.log_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_gap_event_is_mapped(self):
        """LE test de non-regression : l'evenement tombant dans le trou
        inter-op (cyc=1300) doit etre mappe a la couche 0 -- avant le fix
        du 2026-09-08, il aurait ete silencieusement perdu (ni compte dans
        per_layer_engine, ni dans n_trace_before_first_op/after_last_op)."""
        events, batches, meta = php.parse_log(str(self.log_path))
        per_layer = meta["per_layer_engine"]
        self.assertIn(0, per_layer, "couche 0 doit avoir des evenements mappes")
        total_layer0 = sum(per_layer[0].values())
        # les 2 evenements (1050 dans la fenetre stricte + 1300 dans le trou)
        # doivent TOUS LES DEUX finir sur la couche 0.
        self.assertEqual(total_layer0, 2,
                         f"attendu 2 evenements sur la couche 0, obtenu {total_layer0} "
                         f"(per_layer_engine={per_layer}) -- la fenetre de matching "
                         f"a-t-elle regresse vers [start,end] au lieu de "
                         f"[start, debut_op_suivant) ?")

    def test_no_event_silently_lost(self):
        """Les 2 evenements de la fixture doivent etre soit mappes, soit
        comptes explicitement (avant/apres) -- jamais silencieusement perdus."""
        events, batches, meta = php.parse_log(str(self.log_path))
        n_mapped = meta["n_trace_evt_mapped"]
        n_before = meta["n_trace_before_first_op"]
        n_after = meta["n_trace_after_last_op"]
        self.assertEqual(n_mapped + n_before + n_after, 2,
                         "tout evenement trace-evt doit etre compte quelque part "
                         "(mappe, avant le premier op, ou apres le dernier)")

    def test_mapping_rate_above_99_percent_on_reference_trace(self):
        """Garde-fou global (pas juste la fixture synthetique) : si la trace
        de reference prof3_qwen09b.log est presente localement (pas
        versionnee, >2 Mo -- voir device_logs/.gitignore), le mapping doit
        rester >= 99%. Skip proprement si absente (ex: CI sans le fichier)."""
        ref = Path(__file__).resolve().parent.parent / "device_logs" / "prof3_qwen09b.log"
        if not ref.exists():
            self.skipTest(f"{ref} absent localement (log brut non versionne, normal en CI)")
        events, batches, meta = php.parse_log(str(ref))
        total = sum(meta["trace_evt_counts"].values())
        rate = meta["n_trace_evt_mapped"] / total if total else 0
        self.assertGreaterEqual(rate, 0.99,
                                f"mapping tombe a {rate*100:.1f}% sur la trace de reference "
                                f"(attendu >= 99%, mesure a 99.99% le 2026-09-08)")


if __name__ == "__main__":
    unittest.main()
