#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""adaptive_lever_corrige.py — copie corrigee de snapdragon_profiling/adaptive_lever.py

NE REMPLACE PAS L'ORIGINAL. Corrections apportees (2026-09-05), voir
bench_results/AUDIT_PREDICTEUR_PROFILAGE_PRECISION_20260905.md pour le detail :

1. BUG CRITIQUE CORRIGE : LEVER_TABLE recommandait encore ARGSORT a +48% pour
   MOE_ROUTING_SORT_DOMINATED (mesure du 2026-09-03). Reproduit aujourd'hui
   (2026-09-05) sur 3 builds differents (npu_repro_test, runtime_v2, rt_mempool)
   et sur 2 modeles MoE (Marco-Nano, Gemma) : le levier REGRESSE de -10% a -87%.
   Cause probable : un scheduler de fusion de graphe ("pass-4.5") introduit
   depuis, qui fusionne tout un bloc MoE en un seul dispatch HTP quand ARGSORT
   y reste, et le fragmente des qu'un op est exclu -> le cout de fragmentation
   depasse largement le gain de calcul. Voir RAPPORT_MERGE_PR28202_DEUX_SYSTEMES_
   FUSION_INCOMPATIBLES_20260905.md.

   Le levier ARGSORT est donc DESACTIVE par defaut ici (None, 0.0) jusqu'a
   nouvelle validation A/B sur le build exact utilise.

2. AJOUT : chaque entree de LEVER_TABLE porte maintenant une date de derniere
   validation (`validated_on`) et un `contradicted_on` optionnel. `recommend_lever`
   refuse d'appliquer un levier dont la derniere contradiction est plus recente
   que la derniere validation, et emet un avertissement explicite au lieu de
   recommander silencieusement.

3. AJOUT : `record_outcome()` — permet d'enregistrer le gain REEL observe apres
   un run avec le levier applique, pour eventuellement re-valider/re-contredire
   une entree sans editer le fichier a la main (ecrit dans
   adaptive_lever_outcomes.jsonl, append-only, jamais de perte d'historique).
"""
import argparse
import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_model as pm  # noqa: E402  (reutilise load/aggregate/classify)


# ---------------------------------------------------------------------------
# Table levier <- classe. Chaque entree : (env_var, gain_attendu, note,
# validated_on, contradicted_on). gain_attendu=None signifie "desactive /
# incertain", pas "zero mesure".
# ---------------------------------------------------------------------------
LEVER_TABLE = {
    "MOE_ROUTING_SORT_DOMINATED": {
        "env": None,  # <-- DESACTIVE (etait "ARGSORT"), voir docstring point 1
        "gain_attendu": None,
        "note": ("ARGSORT->CPU DESACTIVE : mesure 2026-09-03 (+48%/+49% Marco) "
                 "CONTREDITE le 2026-09-05 sur 3 builds (npu_repro_test, "
                 "runtime_v2, rt_mempool) et 2 modeles MoE (Marco -56% a -87%, "
                 "Gemma -10%). Ne pas appliquer sans re-A/B sur le build exact."),
        "validated_on": "2026-09-03",
        "contradicted_on": "2026-09-05",
    },
    "MOE_COMPUTE_DOMINATED": {
        "env": None, "gain_attendu": 0.10,
        "note": "matmul expert dominant (MUL_MAT_ID) : ARGSORT ne rapporte que +10% (Gemma) — garder sur HTP",
        "validated_on": "2026-09-03", "contradicted_on": None,
    },
    "DISPATCH_DOMINATED": {
        "env": None, "gain_attendu": 0.0,
        "note": "orchestration/micro-ops : pas d'ARGSORT en jeu (Qwen dense 0 MUL_MAT_ID) — levier OPBATCH/batching",
        "validated_on": "2026-09-03", "contradicted_on": None,
    },
    "DENSE_COMPUTE_DOMINATED": {
        "env": None, "gain_attendu": 0.0, "note": "dense pur : pas de routing MoE",
        "validated_on": "2026-09-03", "contradicted_on": None,
    },
    "MEMORY_MOVEMENT_DOMINATED": {
        "env": None, "gain_attendu": 0.0, "note": "CPY/CONCAT : levier DMA/layout, pas OPFILTER",
        "validated_on": "2026-09-03", "contradicted_on": None,
    },
    "VTCM_SPILL_DOMINATED": {
        "env": None, "gain_attendu": 0.0, "note": "spill VTCM : levier tiling/working-set, pas OPFILTER",
        "validated_on": "2026-09-03", "contradicted_on": None,
    },
    "MIXED": {
        "env": None, "gain_attendu": 0.0, "note": "aucun kernel > seuil — pas de levier OPFILTER applicable",
        "validated_on": "2026-09-03", "contradicted_on": None,
    },
    "UNKNOWN": {
        "env": None, "gain_attendu": 0.0, "note": "trace absente/vide : pas de decision",
        "validated_on": "2026-09-03", "contradicted_on": None,
    },
}

REQUIRE_LIVE_TRACE = True
OUTCOMES_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adaptive_lever_outcomes.jsonl")

# ---------------------------------------------------------------------------
# AJOUT 2026-09-06 : levier OPPOLL, absent de la table originale.
# Independant de la classe de goulot (contrairement a OPFILTER, gate par
# classify_bottleneck) : GGML_HEXAGON_OPPOLL agit sur le mecanisme d'attente
# AP-side du dispatch FastRPC, pas sur le routing MoE — recommande par
# defaut quelle que soit la classe, tant qu'il n'est pas contredit.
# Mesure : self-build-jz + build integre (gallocr diag + pass4.5), Marco-Nano
# Q4_0, HTP0 : 32.47 t/s (OPPOLL=1) vs 21.4-22.3 t/s (defaut/OPPOLL=0), soit
# ~+50%. UN SEUL modele/config valide a ce jour — a re-confirmer sur d'autres
# tailles/architectures avant de le considerer acquis universellement.
# ---------------------------------------------------------------------------
OPPOLL_LEVER = {
    "env": "GGML_HEXAGON_OPPOLL",
    "value": "1",
    "gain_attendu": 0.50,
    "note": ("dispatch AP-side (attente FastRPC) : +50% mesure sur Marco-Nano/"
             "self-build-jz (32.47 vs 21.4-22.3 t/s) — independant de la classe "
             "de goulot MoE, recommande par defaut sauf contradiction future"),
    "validated_on": "2026-09-06",
    "contradicted_on": None,
    "n_models_validated": 1,
}


def recommend_oppoll():
    """Verdict OPPOLL, separe de recommend_lever() (classe-dependant) car ce
    levier n'est pas gate par classify_bottleneck — il agit avant/independamment
    du routing MoE. Retourne (env, value, gain_attendu, detail)."""
    e = OPPOLL_LEVER
    if e.get("contradicted_on"):
        return None, None, 0.0, (f"OPPOLL contredit le {e['contradicted_on']} "
                                  "- desactive")
    warn = (f" [VALIDATION LIMITEE: {e['n_models_validated']} modele(s) testes "
            f"a ce jour, {e['validated_on']} — confirmer avant deploiement large]")
    return e["env"], e["value"], e["gain_attendu"], e["note"] + warn


def recommend_lever(live=None, cls=None, evidence=None, measured_tps=None):
    """Verdict runtime : (env_var, classe, gain_attendu, evidence, detail).

    Si l'entree est marquee contredite (contradicted_on > validated_on),
    env_var est force a None et le detail porte un avertissement explicite,
    meme si la table listait un levier auparavant.
    """
    if cls is None:
        if live is None:
            return None, "UNKNOWN", 0.0, {}, "pas de trace : pas de decision"
        cls, evidence = pm.classify_bottleneck(live)
    entry = LEVER_TABLE.get(cls, {"env": None, "gain_attendu": 0.0, "note": "classe non reconnue",
                                   "validated_on": None, "contradicted_on": None})
    env, gain, note = entry["env"], entry["gain_attendu"], entry["note"]
    warn = ""
    if entry.get("contradicted_on"):
        warn = (f" [ATTENTION: levier contredit le {entry['contradicted_on']}, "
                f"valide initialement le {entry.get('validated_on')} — DESACTIVE]")
    detail = note + warn
    if evidence:
        detail += " — preuves : " + json.dumps(
            {k: evidence.get(k) for k in ("argsort_share_kernel", "mmid_share_kernel",
                                          "kernel_top", "wall_ratio") if k in evidence},
            ensure_ascii=False)
    return env, cls, gain, (evidence or {}), detail


def record_outcome(cls, env_applied, baseline_tps, measured_tps, note=""):
    """Enregistre un resultat reel (baseline vs levier applique) sans jamais
    modifier LEVER_TABLE en place — append-only, pour audit/re-validation
    manuelle ulterieure plutot que mise a jour silencieuse."""
    gain_reel = (measured_tps - baseline_tps) / baseline_tps if baseline_tps else None
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "classe": cls, "env_applied": env_applied,
        "baseline_tps": baseline_tps, "measured_tps": measured_tps,
        "gain_reel": round(gain_reel, 4) if gain_reel is not None else None,
        "note": note,
    }
    with open(OUTCOMES_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def _classify_from_trace(path):
    events = pm.load_live_trace(path)
    live = pm.aggregate_live_trace(events)
    cls, evidence = pm.classify_bottleneck(live)
    return live, cls, evidence


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classify", metavar="TRACE.jsonl",
                    help="agrege la trace et imprime la classe + le verdict levier")
    ap.add_argument("--apply", metavar="TRACE.jsonl",
                    help="idem --classify mais imprime la ligne d'export si un levier est actif")
    ap.add_argument("--measured-tps", type=float, default=None,
                    help="debit wall reel (pour comparaison wall/L3)")
    ap.add_argument("--record-outcome", nargs=3, metavar=("CLASSE", "BASELINE_TPS", "MEASURED_TPS"),
                    help="enregistre un resultat reel sans toucher a LEVER_TABLE")
    args = ap.parse_args()

    if args.record_outcome:
        cls, base_tps, meas_tps = args.record_outcome
        rec = record_outcome(cls, LEVER_TABLE.get(cls, {}).get("env"), float(base_tps), float(meas_tps))
        print(json.dumps(rec, ensure_ascii=False, indent=2))
        return

    if not args.classify and not args.apply:
        ap.error("il faut --classify, --apply, ou --record-outcome")

    try:
        live, cls, evidence = _classify_from_trace(args.classify or args.apply)
    except Exception as e:
        sys.exit(f"echec classification : {e}")

    env, cls_r, gain, ev, note = recommend_lever(live=live, cls=cls, evidence=evidence)
    print(f"CLASSE        : {cls}")
    print(f"kernel_top    : {evidence.get('kernel_top')}")
    print(f"ARGSORT share : {evidence.get('argsort_share_kernel', 0.0):.1%}  "
          f"MUL_MAT_ID share : {evidence.get('mmid_share_kernel', 0.0):.1%}")
    print(f"VERDICT       : {note}")
    if env:
        print(f"LEVIER        : GGML_HEXAGON_OPFILTER={env}  (gain attendu ~{gain:+.0%} wall)")
        if args.apply:
            print(f"\nexport GGML_HEXAGON_OPFILTER={env}")
    else:
        print("LEVIER        : aucun (desactive ou inutile pour cette classe — voir VERDICT ci-dessus)")

    oppoll_env, oppoll_val, oppoll_gain, oppoll_note = recommend_oppoll()
    if oppoll_env:
        print(f"\nLEVIER OPPOLL : {oppoll_env}={oppoll_val}  (gain attendu ~{oppoll_gain:+.0%} wall)")
        print(f"  {oppoll_note}")
        if args.apply:
            print(f"export {oppoll_env}={oppoll_val}")

    journal = {"classe": cls, "kernel_top": evidence.get("kernel_top"),
               "argsort_share_kernel": evidence.get("argsort_share_kernel", 0.0),
               "mmid_share_kernel": evidence.get("mmid_share_kernel", 0.0),
               "opfilter": env, "gain_attendu": gain, "note": note,
               "trace": args.classify or args.apply}
    jpath = "adaptive_decision.json"
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(journal, f, indent=2, ensure_ascii=False)
    print(f"[journal] {jpath}")


if __name__ == "__main__":
    main()
