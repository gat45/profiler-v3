#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""predict_from_hf.py — predit RAM/debit/risque d'un modele HuggingFace SANS
telecharger les poids (2026-09-06), en combinant :
1. Lecture de l'en-tete GGUF a distance (HTTP Range, quelques Mo max, jamais
   le fichier complet) — l'architecture et les tenseurs sont dans l'en-tete,
   pas besoin des poids pour ca.
2. Le modele L3 existant (profile_model.py) pour la RAM et une base theorique
   de debit.
3. La CORRECTION DE ROUTING MOE calibree aujourd'hui sur 3 mesures REELLES
   (pas une formule inventee) — le L3 est structurellement aveugle a
   n_experts/top_k (verifie 2026-09-06,
   AUDIT_COMPLET_PROFILAGE_PAR_COUCHE_INEXISTANT_20260906.md), cette
   correction comble une partie de ce trou avec de la vraie donnee.
4. capability_db pour signaler les risques de crash/collapse connus AVANT
   tout telechargement.

BUG CORRIGE EN COURS DE ROUTE (2026-09-06) : pm.load_hf() suppose un repo au
format transformers (config.json a la racine) — echoue en HTTPError 404 sur
un repo GGUF pur (ex. mradermacher/*, notre cas d'usage principal toute la
journee). Ce module lit l'en-tete GGUF DIRECTEMENT depuis le fichier .gguf
distant via Range HTTP, sans dependre de config.json.

Usage :
  py predict_from_hf.py mradermacher/Some-Model-GGUF --file Some-Model.Q4_0.gguf
  py predict_from_hf.py mradermacher/Some-Model-GGUF   # liste les fichiers si --file omis
"""
import argparse
import io
import json
import struct
import sys
import urllib.error
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")
import profile_model as pm
import capability_db as cdb
import profiler as pv3  # noqa: E402  (plan de quant + prediction par couche)
# NOTE : alias "pv3", PAS "p" — "p" est deja utilise partout ci-dessous comme
# nom du dict de prediction (render_prediction(p), etc.). Bug reel introduit
# puis corrige le 2026-09-06 en relisant le fichier : une premiere version
# utilisait "import profiler as p" qui masquait silencieusement le parametre
# "p" des fonctions de rendu — aurait plante des l'usage.

# ---------------------------------------------------------------------------
# Correction de routing MoE — calibree sur 3 mesures REELLES 2026-09-06
# (Qwen3-0.9B-A0.6B 2exp/top1, Huihui-MoE 3exp/top1, Marco-Nano-V2 232exp/top8).
# Le facteur = tps_mesure / tps_L3_predit_identique(42.5). Interpolation
# UNIQUEMENT en log(n_experts), UNIQUEMENT entre points calibres — jamais
# d'extrapolation hors plage sans avertissement explicite (3 points ne
# suffisent pas a extrapoler de facon fiable).
# ---------------------------------------------------------------------------
ROUTING_CALIBRATION = [
    {"n_experts": 2, "top_k": 1, "tps_measured": 62.6, "factor": 62.6 / 42.5},
    {"n_experts": 3, "top_k": 1, "tps_measured": 54.1, "factor": 54.1 / 42.5},
    # AJOUT 2026-09-06 : 4e point, obtenu en testant REELLEMENT la prediction
    # sans-telechargement sur EuroMoE-2.6B (64 experts/top-8) — la prediction
    # interpolee avait donne 50.4 t/s (facteur x0.71), le REEL mesure (n=1)
    # etait 59.4 t/s, MIS A JOUR avec n=3 : [59.4, 62.7, 64.4] -> moyenne 62.2
    # (facteur reel 62.2/71.3=0.873). Ecart vs prediction initiale +23%, meme
    # ordre de grandeur mais pas exact — normal avec seulement 2-3 points de
    # calibration au moment de cette prediction. Utilise le vrai facteur
    # mesure ici, pas la prediction, pour ameliorer les FUTURES interpolations.
    {"n_experts": 64, "top_k": 8, "tps_measured": 62.2, "factor": 62.2 / 71.3,
     "n_runs": 3, "runs": [59.4, 62.7, 64.4]},
    {"n_experts": 232, "top_k": 8, "tps_measured": 19.9, "factor": 19.9 / 42.5},
]


# ---------------------------------------------------------------------------
# Correction dense (non-MoE) — calibree sur 2 mesures REELLES 2026-09-06
# (Qwen3-1.7B-Q4_0 et Qwen3-4B-Q4_0, deja presents sur le device, tires du
# device vers E:\oneplus\sweep\ pour l'analyse L3 locale). Le facteur =
# tps_mesure / tps_L3_predit (chacun sur son propre modele, PAS de point de
# reference commun contrairement a ROUTING_CALIBRATION car l3_cold_tps varie
# deja normalement avec la taille pour un modele dense). Cle d'interpolation :
# bytes/token (a["l3"]["total_bytes_gb"]), proxy direct de la taille active du
# modele, deja calcule par le pipeline L3 sans info supplementaire necessaire.
# SEULEMENT 2 points, tres proches (1.47 et 2.94 Go/token) — extrapolation
# hors de cette plage etroite marquee explicitement peu fiable.
# ---------------------------------------------------------------------------
DENSE_CALIBRATION = [
    {"bytes_per_token_gb": 1.472, "tps_measured": 38.2, "tps_l3_cold": 28.6,
     "factor": 38.2 / 28.6, "model": "Qwen3-1.7B-Q4_0"},
    {"bytes_per_token_gb": 2.935, "tps_measured": 20.0, "tps_l3_cold": 15.3,
     "factor": 20.0 / 15.3, "model": "Qwen3-4B-Q4_0"},
]


def predict_dense_factor(bytes_per_token_gb):
    """Meme logique que predict_routing_factor mais pour modeles denses,
    interpole en log(bytes/token). Seulement 2 points calibres (~1.3x les
    deux) -> confiance moderee meme en interpolation, peu fiable hors plage
    [1.47, 2.94] Go/token."""
    import math
    pts = sorted(DENSE_CALIBRATION, key=lambda p: p["bytes_per_token_gb"])
    for p in pts:
        if abs(bytes_per_token_gb - p["bytes_per_token_gb"]) < 1e-6:
            return p["factor"], "measured_exact", (
                f"bytes/token={bytes_per_token_gb:.3f} Go correspond EXACTEMENT "
                f"a {p['model']} — facteur mesure directement")
    lo, hi = pts[0], pts[-1]
    if bytes_per_token_gb < lo["bytes_per_token_gb"]:
        return lo["factor"], "extrapolated_unreliable", (
            f"bytes/token={bytes_per_token_gb:.3f} Go < point calibre le plus "
            f"bas ({lo['bytes_per_token_gb']:.3f}, {lo['model']}) — facteur du "
            "point le plus proche applique, PEU FIABLE hors plage (2 points "
            "seulement, tres proches)")
    if bytes_per_token_gb > hi["bytes_per_token_gb"]:
        return hi["factor"], "extrapolated_unreliable", (
            f"bytes/token={bytes_per_token_gb:.3f} Go > point calibre le plus "
            f"haut ({hi['bytes_per_token_gb']:.3f}, {hi['model']}) — facteur du "
            "point le plus proche applique, PEU FIABLE hors plage (2 points "
            "seulement, tres proches)")
    t = ((math.log(bytes_per_token_gb) - math.log(lo["bytes_per_token_gb"])) /
         (math.log(hi["bytes_per_token_gb"]) - math.log(lo["bytes_per_token_gb"])))
    factor = lo["factor"] + t * (hi["factor"] - lo["factor"])
    return factor, "interpolated", (
        f"interpole entre {lo['model']} (x{lo['factor']:.2f}) et {hi['model']} "
        f"(x{hi['factor']:.2f}) — SEULEMENT 2 points de calibration au total, "
        "tenir la confiance moderee meme en interpolation")


def predict_routing_factor(n_experts, top_k):
    """Retourne (factor, confidence, note). confidence = "interpolated" si
    n_experts tombe ENTRE deux points calibres, "extrapolated_unreliable" sinon
    (avec le facteur du point le plus proche, marque explicitement peu fiable)."""
    if not n_experts or n_experts <= 1:
        return 1.0, "n/a", "modele dense (pas de MoE), aucune correction necessaire"

    pts = sorted(ROUTING_CALIBRATION, key=lambda p: p["n_experts"])
    import math
    for p in pts:
        if n_experts == p["n_experts"]:
            return p["factor"], "measured_exact", (
                f"n_experts={n_experts} correspond EXACTEMENT a un point calibre "
                "— facteur mesure directement, pas une interpolation")
    lo = pts[0]
    hi = pts[-1]
    if n_experts < lo["n_experts"]:
        return lo["factor"], "extrapolated_unreliable", (
            f"n_experts={n_experts} < point calibre le plus bas "
            f"({lo['n_experts']}) — facteur du point le plus proche applique, "
            "PEU FIABLE hors plage")
    if n_experts > hi["n_experts"]:
        return hi["factor"], "extrapolated_unreliable", (
            f"n_experts={n_experts} > point calibre le plus haut "
            f"({hi['n_experts']}) — facteur du point le plus proche applique, "
            "PEU FIABLE hors plage")
    # interpolation lineaire en log(n_experts) entre les deux points encadrants
    for a, b in zip(pts, pts[1:]):
        if a["n_experts"] <= n_experts <= b["n_experts"]:
            t = ((math.log(n_experts) - math.log(a["n_experts"])) /
                 (math.log(b["n_experts"]) - math.log(a["n_experts"])))
            factor = a["factor"] + t * (b["factor"] - a["factor"])
            return factor, "interpolated", (
                f"interpole entre n_experts={a['n_experts']} (x{a['factor']:.2f}) "
                f"et n_experts={b['n_experts']} (x{b['factor']:.2f}) — SEULEMENT "
                "3 points de calibration au total, tenir la confiance moderee "
                "meme en interpolation")
    return 1.0, "unknown", "cas non gere (ne devrait pas arriver)"


# ---------------------------------------------------------------------------
# Lecture GGUF a distance — Range HTTP, PAS de telechargement des poids.
# ---------------------------------------------------------------------------
def list_hf_gguf_files(repo_id):
    """Liste les fichiers .gguf d'un repo HF via l'API (pas de download)."""
    url = f"https://huggingface.co/api/models/{repo_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "predict_from_hf"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.loads(r.read())
    return [s["rfilename"] for s in data.get("siblings", [])
            if s["rfilename"].lower().endswith(".gguf")]


def fetch_gguf_header_remote(repo_id, filename, max_header_bytes=8 * 1024 * 1024):
    """Lit l'en-tete GGUF (metadata + liste des tenseurs, PAS les poids) via
    UNE requete Range HTTP. 8 Mo par defaut est largement suffisant pour
    l'en-tete de n'importe quel modele teste a ce jour (les plus gros vus
    aujourd'hui : ~200 Ko pour un modele a 775 tenseurs + beaucoup de
    metadonnees). Retourne (meta, tensor_list) au meme format que
    pm.read_gguf_header() local."""
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "predict_from_hf",
        "Range": f"bytes=0-{max_header_bytes - 1}",
    })
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    f = io.BytesIO(data)
    try:
        return pm._read_gguf_header_inner(f)
    except (struct.error, ValueError) as e:
        raise RuntimeError(
            f"en-tete GGUF incomplet dans les {max_header_bytes} premiers "
            f"octets (modele avec beaucoup de tenseurs/metadonnees ?) — "
            f"augmenter max_header_bytes. Erreur : {e}")


def predict_no_download(repo_id, filename, budget_gb=9.0, safety_margin_gb=1.5):
    """Pipeline complet : en-tete distant -> analyse -> RAM -> L3 -> correction
    routing -> risque de crash. AUCUN octet de poids telecharge."""
    meta, tlist = fetch_gguf_header_remote(repo_id, filename)
    tensors = {}
    for name, dims, ttype, nb in tlist:
        ne = 1
        for d in dims:
            ne *= d
        tensors[name] = {"dtype": pm.V_TYPES.get(ttype, "?"), "shape": dims,
                         "bytes": nb, "elems": ne, "ttype": ttype}
    a = pm.analyze({}, tensors, gguf_meta=meta, is_gguf=True, ctx=4096)
    model_name = meta.get("general.name") or filename

    l3_cold_tps = a["l3"]["cold_tps"]
    n_experts = a["arch"].get("n_experts", 0)
    top_k = a["arch"].get("top_k", 0)
    if n_experts and n_experts > 1:
        factor, confidence, note = predict_routing_factor(n_experts, top_k)
    else:
        factor, confidence, note = predict_dense_factor(a["l3"]["total_bytes_gb"])
    corrected_tps = l3_cold_tps * factor

    # Risque de crash connu (capability_db) — verifie ici, PAS de nouveau test.
    fmt = "Q4_0" if "q4_0" in filename.lower() else (
        "Q8_0" if "q8_0" in filename.lower() else "?")
    risk = cdb.predict_crash_risk(fmt, "MUL_MAT", "HTP")

    total_gib_q4 = a["sizes_gib"].get("Q4_0", 0.0)
    fits = total_gib_q4 <= (budget_gb - safety_margin_gb)

    # AJOUT 2026-09-06 (demande explicite : "connaitre a l'avance comment le
    # modele serait quantifie par couche, pourquoi, la taille RAM, les
    # tokens/s") — reutilise le plan de quant + la prediction par couche de
    # profiler.py, DEJA CONSTRUITS aujourd'hui, mais jamais branches sur les
    # donnees distantes (seulement sur un fichier local jusqu'ici).
    quant_plan = pv3.quant_plan_safe(a, budget_gb, backend="HTP",
                                     safety_margin_gb=safety_margin_gb)
    rows = pv3.build_tensor_rows(tensors, is_gguf=True)
    layer_impact = pv3.predict_layer_quant_impact(
        rows, a, budget_gb, live=None, safety_margin_gb=safety_margin_gb)

    return {
        "model_name": model_name, "repo": repo_id, "file": filename,
        "arch": a["arch"], "n_experts": n_experts, "top_k": top_k,
        "size_q4_gib": total_gib_q4, "fits_budget": fits,
        "l3_cold_tps": l3_cold_tps,
        "routing_factor": factor, "routing_confidence": confidence,
        "routing_note": note, "corrected_tps_estimate": corrected_tps,
        "crash_risk": risk,
        "quant_plan": quant_plan, "layer_impact": layer_impact, "analysis": a,
    }


def render_prediction(p):
    L = [f"# Prediction SANS TELECHARGEMENT — {p['model_name']}", ""]
    L.append(f"- repo : {p['repo']} / fichier : {p['file']}")
    L.append(f"- architecture : {p['arch']['n_layer']} couches, hidden "
             f"{p['arch']['hidden']}, MoE={p['arch']['is_moe']} "
             f"({p['n_experts']} experts, top-{p['top_k']})")
    L.append(f"- taille Q4_0 estimee : {p['size_q4_gib']:.2f} GiB — "
             f"{'TIENT' if p['fits_budget'] else 'NE TIENT PAS'} dans le budget")
    L.append("")
    L.append(f"- t/s L3 (theorique, sans correction) : {p['l3_cold_tps']:.1f}")
    L.append(f"- facteur de correction routing (calibre sur 3 mesures reelles "
             f"2026-09-06) : x{p['routing_factor']:.2f} "
             f"[{p['routing_confidence']}]")
    L.append(f"  {p['routing_note']}")
    L.append(f"- **t/s ESTIME (avec correction) : {p['corrected_tps_estimate']:.1f}**")
    L.append("")
    L.append(cdb.render_crash_risk(p["crash_risk"]))
    L.append("")
    L.append(pv3.render_safe_plan(p["quant_plan"], p["analysis"]))
    L.append("")
    L.append(pv3.render_layer_quant_impact(p["layer_impact"]))
    L.append("")
    L.append("[LIMITE] Cette estimation n'a jamais ete verifiee par une mesure "
             "reelle SUR CE MODELE PRECIS — elle applique une correction "
             "calibree sur 3 AUTRES modeles. A confirmer par un vrai test "
             "device avant toute decision de deploiement. Le but ici est "
             "d'eliminer les candidats clairement mauvais AVANT de "
             "telecharger, pas de remplacer la mesure finale.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", help="repo HuggingFace, ex mradermacher/Model-GGUF")
    ap.add_argument("--file", help="nom exact du fichier .gguf (si omis, liste "
                    "les fichiers disponibles et quitte)")
    ap.add_argument("--budget-gb", type=float, default=9.0)
    args = ap.parse_args()

    if not args.file:
        try:
            files = list_hf_gguf_files(args.repo)
        except urllib.error.URLError as e:
            sys.exit(f"[erreur] impossible de lister {args.repo} : {e}")
        print(f"[{len(files)} fichier(s) .gguf trouve(s) dans {args.repo}]")
        for f in files:
            print(f"  {f}")
        print("\nRelancer avec --file <nom exact>")
        return

    try:
        p = predict_no_download(args.repo, args.file, budget_gb=args.budget_gb)
    except (urllib.error.URLError, RuntimeError) as e:
        sys.exit(f"[erreur] {e}")
    print(render_prediction(p))


if __name__ == "__main__":
    main()
