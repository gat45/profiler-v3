"""predictor_v1_corrige.py — copie corrigee de governor/predictor_v1.py

NE REMPLACE PAS L'ORIGINAL. Voir bench_results/AUDIT_PREDICTEUR_PROFILAGE_PRECISION_20260905.md
et bench_results/REVIEW_ARCHITECTURE_PREDICTOR_V1_20260905.md pour la justification complete
de chaque correctif. Resume des changements vs l'original :

1. "probability" -> "confidence" : ce n'etait pas une probabilite calibree (juste
   1 - max(risque)*0.5), le nom induisait en erreur sur ce que la valeur mesure.
2. "OPTIMAL" -> "CLEAN" : un regime sans degradation detectee n'est pas prouve
   optimal, juste non-degrade. Les cles de sortie/API qui utilisaient "OPTIMAL"
   acceptent aussi "CLEAN" en alias pour compat ascendante des consommateurs.
3. Etats composites : detect_regime() retourne maintenant aussi les regimes
   secondaires actifs simultanement (ex: DOZE_CAPPED + MEMORY_THRASH + THERMAL
   en meme temps ne masque plus les deux derniers).
4. Fallback modele inconnu : au lieu de retomber silencieusement sur
   "gemma_mixed3" (un MoE specifique), deux baselines neutres "default_dense"
   et "default_moe" avec des bornes tres larges, et le champ "model_matched"
   indique explicitement si le modele a ete reconnu ou si c'est un defaut.
5. post_run() : nouvelle fonction pour enregistrer prediction vs observation
   apres un run reel (erreur relative, dans/hors intervalle) — append-only
   dans predictor_v1_outcomes.jsonl, jamais de modification retroactive d'une
   prediction deja faite (pas de boucle de retroaction circulaire).
6. Bornes d'incertitude ajustees par regime : un regime a forte variance
   documentee (MEMORY_THRASH: 1.5-4.7 t/s observe) ne recoit plus la meme
   largeur PROPORTIONNELLE que OPTIMAL/CLEAN — la largeur relative est
   elargie d'un facteur explicite par regime (REGIME_UNCERTAINTY_MULT).
7. Marco-Nano baseline : NON changee automatiquement ici (29.3 t/s) car
   changer une constante calibree necessite une decision humaine sur quelle
   nouvelle valeur adopter (17-22 t/s mesures aujourd'hui, build-dependant) —
   mais le champ "baseline_age_days" et "baseline_needs_reverification" sont
   ajoutes pour rendre visible qu'elle n'a plus ete confirmee depuis sa
   derniere mesure, au lieu de la presenter avec la meme confiance qu'une
   valeur fraiche.
"""
import argparse
import json
import os
import re
import sys
import time

# --------------------------------------------------------------------------
# BASELINES calibrees (t/s) — source : rapports bench_results/ 2026-08/09
#   gemma_mixed3 : RAPPORT_TELEMIN (10.4 +/- 1.6, OPPOLL=0, n=48, fenetre propre)
#   marco q4_0   : baseline HTP0 29.3 +/- 0.9 ; ARGSORT->CPU 37-38.5
#     ATTENTION (2026-09-05) : re-mesure sur 3 builds differents donne
#     17.8-21.8 t/s, jamais proche de 29.3 — voir "baseline_needs_reverification"
#     ci-dessous, cette entree n'est PAS a considerer fiable telle quelle.
#   d2a          : ~16 t/s (HTP quasi complet + lm_head CPU)
# --------------------------------------------------------------------------
BASELINES = {
    "gemma_mixed3": {"tg": 10.4, "lo": 8.4, "hi": 12.0, "oppoll_penalty": 0.05,
                     "measured_on": "2026-09-02", "arch": "moe", "source": "measured"},
    "gemma_mixed":  {"tg": 10.4, "lo": 8.4, "hi": 12.0, "oppoll_penalty": 0.05,
                     "measured_on": "2026-09-02", "arch": "moe", "source": "measured"},
    "marco_nano":   {"tg": 29.3, "lo": 27.5, "hi": 31.0, "oppoll_penalty": 0.0,
                     "measured_on": "2026-09-02", "arch": "moe", "source": "measured",
                     "needs_reverification": True,
                     "reverification_note": ("re-mesure 2026-09-05 sur 3 builds "
                         "(npu_repro_test, runtime_v2, rt_mempool), meme protocole "
                         "(200 tokens, HTP0 pur, device propre) : 17.8-21.8 t/s. "
                         "Cette baseline (29.3) n'a jamais ete reproduite depuis.")},
    "marco":        {"tg": 29.3, "lo": 27.5, "hi": 31.0, "oppoll_penalty": 0.0,
                     "measured_on": "2026-09-02", "arch": "moe", "source": "measured",
                     "needs_reverification": True},
    "d2a":          {"tg": 16.0, "lo": 14.5, "hi": 17.5, "oppoll_penalty": 0.0,
                     "measured_on": "2026-09-02", "arch": "dense", "source": "measured"},
    # --- AJOUT : baselines neutres pour modele non reconnu, au lieu du
    # fallback silencieux vers gemma_mixed3 (voir point 4 du docstring) ---
    "default_dense": {"tg": 15.0, "lo": 5.0, "hi": 30.0, "oppoll_penalty": 0.0,
                      "measured_on": None, "arch": "dense", "is_default": True,
                      "source": "default"},
    "default_moe":   {"tg": 15.0, "lo": 4.0, "hi": 35.0, "oppoll_penalty": 0.0,
                      "measured_on": None, "arch": "moe", "is_default": True,
                      "source": "default"},
    # --- MIS A JOUR 2026-09-06, n=3 chacun (repete apres le premier passage
    # n=1) : mesure REELLE pour les 3 modeles prepares pour l'etude
    # d'isolation du cout de routing (cf. MODELES_PREPARES_MOE_ROUTING_
    # ISOLATION_20260906.md et RESULTATS_ROUTING_MOE_CONFIRME_20260906.md).
    # Protocole : llama-cli --single-turn -ngl 99, GGML_HEXAGON_OPPOLL=1,
    # HTP0 solo, build rt_clean, meme prompt, 32 tokens generes.
    #   marco_nano_v2   : [18.4, 20.8, 20.4] -> moyenne 19.9, min/max reels
    #   qwen3_0_9b_a0_6b: [62.3, 69.0, 56.5] -> moyenne 62.6, min/max reels
    #   huihui_moe_1_2b : [50.7, 54.4, 57.1] -> moyenne 54.1, min/max reels
    # Effet routing CONFIRME avec variance reelle (pas un artefact n=1) :
    # facteur x3.1 entre marco_nano_v2 (19.9) et qwen3_0_9b_a0_6b (62.6),
    # alors que le L3 predit 42.5 t/s identique pour les 3 -> le cout de
    # routing MoE (232/top-8 vs 2-3/top-1) est un terme reel et important,
    # absent du modele L3 actuel.
    "marco_nano_v2": {"tg": 19.9, "lo": 18.4, "hi": 20.8, "oppoll_penalty": 0.0,
                      "measured_on": "2026-09-06", "arch": "moe", "source": "measured",
                      "n_runs": 3,
                      "reverification_note": ("n=3 : [18.4, 20.8, 20.4] t/s — "
                          "cohorte de 3 modeles (232/top-8 vs 2-3/top-1) confirme "
                          "un facteur x3.1 vs qwen3_0_9b_a0_6b alors que le L3 "
                          "predit 42.5 t/s identique pour les 3 -> cout de routing "
                          "MoE reel et important, absent du modele L3 actuel.")},
    "qwen3_0_9b_a0_6b": {"tg": 62.6, "lo": 56.5, "hi": 69.0, "oppoll_penalty": 0.0,
                      "measured_on": "2026-09-06", "arch": "moe", "source": "measured",
                      "n_runs": 3},
    "huihui_moe_1_2b": {"tg": 54.1, "lo": 50.7, "hi": 57.1, "oppoll_penalty": 0.0,
                      "measured_on": "2026-09-06", "arch": "moe", "source": "measured",
                      "n_runs": 3},
    # AJOUT 2026-09-06 : 4e point de calibration MoE (64 experts/top-8),
    # d'abord teste via predict_from_hf.py (n=1, 59.4), puis confirme n=3 sur
    # device : [59.4, 62.7, 64.4] -> moyenne 62.2.
    "euromoe_2_6b_a0_6b": {"tg": 62.2, "lo": 59.4, "hi": 64.4, "oppoll_penalty": 0.0,
                      "measured_on": "2026-09-06", "arch": "moe", "source": "measured",
                      "n_runs": 3},
}

# Seuils materiels SM8850 (OnePlus 15)
MAX_P0 = 3628800
MAX_P6 = 4608000
THERMAL_HOT = 70.0          # degres C
LOAD_HIGH = 15.0            # loadavg
MEM_THRASH_GB = 1.5         # MemAvailable sous lequel on est en thrash probable
MEM_RISK_GB = 3.0           # sous lequel le risque swap monte fort
DOZE_CAPS_THRESHOLD = 0.95  # ratio caps/max sous lequel on classe DOZE_CAPPED
# LIMITE CONFIRMEE 2026-09-06 (premier test avec un VRAI fichier meta device,
# pas un dict construit a la main) : FAUX POSITIF reproduit. Device au repos
# normal (ecran Awake, PAS en veille Android reelle), scaling_max_freq lu a
# 2361600/1862400 (governor idle economise l'energie hors charge), caps_ratio
# = 0.404 -> DOZE_CAPPED detecte -> 17.2 t/s predit, alors que 62.3-69.0 t/s
# REELLEMENT mesures sur ce meme modele quelques minutes plus tot. Cause
# probable : detect_regime() ne combine PAS caps_ratio bas avec wakefulness
# != "Awake" -> un throttle CPU normal hors charge (rien a voir avec le Doze
# Android) declenche le meme regime qu'une vraie veille. PAS CORRIGE ICI
# (nécessiterait de determiner le bon seuil/combinaison sans casser la
# detection de vrais cas DOZE_CAPPED deja geres) — signale explicitement
# comme axe a fiabiliser avant de faire confiance a ce regime en prod.

# Elargissement de la largeur d'incertitude relative par regime (1.0 = pas de
# changement vs l'original ; >1.0 = intervalle plus large que la proportion
# naive tg/base_tg). Valeurs heuristiques V1, pas calibrees statistiquement.
REGIME_UNCERTAINTY_MULT = {
    "DOZE_CAPPED": 1.5,
    "MEMORY_THRASH": 3.0,   # 1.5-4.7 t/s observe = tres large, cf. docstring module original
    "THERMAL": 1.3,
    "SYSTEM_LOAD": 1.2,
    "CLEAN": 1.0,
    "OPTIMAL": 1.0,  # alias
}

OUTCOMES_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictor_v1_outcomes.jsonl")


def _f(d, k, default=0.0):
    try:
        v = d.get(k, default)
        return float(v) if v not in (None, "") else default
    except (TypeError, ValueError):
        return default


def detect_regime(f):
    """Retourne (regime, facteurs, evidence, secondary_conditions).

    Correctif #3 : les regimes secondaires actifs simultanement ne sont plus
    masques par la priorite causale — ils sont listes a part, le regime
    dominant reste la priorite la plus haute (logique inchangee sur ce point).
    """
    caps_p0 = _f(f, "caps_p0", MAX_P0)
    caps_p6 = _f(f, "caps_p6", MAX_P6)
    wake = str(f.get("wakefulness", "Awake"))
    mem_avail_gb = _f(f, "mem_available_kb", 11000000) / 1048576.0
    swap_free_kb = _f(f, "swap_free_kb", 12000000)
    swap_free_gb = swap_free_kb / 1048576.0 if swap_free_kb else 12.0
    t_ddr = _f(f, "thermal_ddr_c", 35.0)
    t_hvx = _f(f, "thermal_hvx_c", 35.0)
    t_hmx = _f(f, "thermal_hmx_c", 35.0)
    load = _f(f, "loadavg", 5.0)
    t_hot = max(t_ddr, t_hvx, t_hmx)

    caps_ratio = min(caps_p0 / MAX_P0, caps_p6 / MAX_P6) if MAX_P0 else 1.0
    caps_ratio = max(0.1, min(1.0, caps_ratio))

    # Evalue TOUTES les conditions de degradation (pas de retour anticipe) pour
    # pouvoir exposer les secondaires, puis choisit le regime dominant par la
    # meme priorite causale que l'original.
    conditions = []
    # CORRIGE 2026-09-06 : l'ancienne regle (wake != "Awake" OR caps_ratio bas)
    # confondait deux situations differentes — prouve par un test reel :
    # caps_ratio=0.404 mesure alors que wakefulness="Awake" (device au repos,
    # governor idle normal, ECRAN ALLUME) -> ancienne regle classait ca
    # DOZE_CAPPED et predisait 17.2 t/s, alors que 62.3-69.0 t/s ont ete
    # REELLEMENT mesures sur ce meme modele quelques minutes plus tot. Fix :
    # DOZE_CAPPED (vraie veille Android, penalite appliquee) exige maintenant
    # wake != "Awake" ; un caps_ratio bas EN ETANT Awake devient un regime
    # distinct CPU_IDLE_THROTTLE (informationnel, PAS de penalite tant que non
    # calibre — aucune preuve aujourd'hui que ce cas degrade le debit).
    if wake != "Awake":
        conditions.append(("DOZE_CAPPED", {"caps": caps_ratio, "thermal": 1.0, "mem": 1.0, "load": 1.0},
                           {"wakefulness": wake, "caps_p0": caps_p0, "caps_p6": caps_p6,
                            "caps_ratio": round(caps_ratio, 3)}))
    elif caps_ratio < DOZE_CAPS_THRESHOLD:
        conditions.append(("CPU_IDLE_THROTTLE", {"caps": 1.0, "thermal": 1.0, "mem": 1.0, "load": 1.0},
                           {"wakefulness": wake, "caps_p0": caps_p0, "caps_p6": caps_p6,
                            "caps_ratio": round(caps_ratio, 3),
                            "note": ("frequences CPU sous le max mais ecran Awake "
                                    "— probablement gouverneur idle normal, PAS "
                                    "une veille reelle ; aucune penalite appliquee "
                                    "faute de preuve que ca degrade le debit")}))
    if mem_avail_gb < MEM_THRASH_GB:
        conditions.append(("MEMORY_THRASH", {"caps": 1.0, "thermal": 1.0, "mem": 0.35, "load": 1.0},
                           {"mem_available_gb": round(mem_avail_gb, 2), "swap_free_gb": round(swap_free_gb, 2),
                            "note": "MemAvailable < 1.5 Go : regime 1.5-4.7 t/s observe"}))
    if t_hot > THERMAL_HOT:
        pen = 1.0 - min(0.45, (t_hot - THERMAL_HOT) * 0.03)
        conditions.append(("THERMAL", {"caps": 1.0, "thermal": pen, "mem": 1.0, "load": 1.0},
                           {"thermal_max_c": round(t_hot, 1)}))
    if load > LOAD_HIGH:
        pen = 1.0 - min(0.35, (load - LOAD_HIGH) * 0.02)
        conditions.append(("SYSTEM_LOAD", {"caps": 1.0, "thermal": 1.0, "mem": 1.0, "load": pen},
                           {"loadavg": round(load, 2)}))

    if conditions:
        regime, factors, evidence = conditions[0]
        secondary = [c[0] for c in conditions[1:]]
        return regime, factors, evidence, secondary

    mem_factor = 1.0
    if mem_avail_gb < MEM_RISK_GB:
        mem_factor = 0.85
    return ("CLEAN",
            {"caps": 1.0, "thermal": 1.0, "mem": mem_factor, "load": 1.0},
            {"mem_available_gb": round(mem_avail_gb, 2),
             "thermal_max_c": round(t_hot, 1), "loadavg": round(load, 2)},
            [])


def estimate_tg(f, base):
    regime, factors, evidence, secondary = detect_regime(f)
    tg = base["tg"]
    for k, v in factors.items():
        tg *= v
    oppoll = int(_f(f, "oppoll", 0))
    if oppoll and base.get("oppoll_penalty"):
        tg *= (1.0 - base["oppoll_penalty"])
    return regime, tg, factors, evidence, secondary


def risks(f, base, regime):
    mem_avail_gb = _f(f, "mem_available_kb", 11000000) / 1048576.0
    swap_free_gb = _f(f, "swap_free_kb", 12000000) / 1048576.0
    t_hot = max(_f(f, "thermal_ddr_c", 35.0), _f(f, "thermal_hvx_c", 35.0),
                _f(f, "thermal_hmx_c", 35.0))
    load = _f(f, "loadavg", 5.0)
    caps_ratio = min(_f(f, "caps_p0", MAX_P0) / MAX_P0,
                     _f(f, "caps_p6", MAX_P6) / MAX_P6)

    swap_risk = 0.02
    if mem_avail_gb < MEM_THRASH_GB:
        swap_risk = 0.90
    elif mem_avail_gb < MEM_RISK_GB:
        swap_risk = 0.55
    if swap_free_gb < 4.0:
        swap_risk = min(0.98, swap_risk + 0.15)

    cpu_risk = 0.04
    if caps_ratio < DOZE_CAPS_THRESHOLD:
        cpu_risk = 0.95
    elif load > LOAD_HIGH:
        cpu_risk = 0.60

    orch_risk = 0.08
    if load > LOAD_HIGH:
        orch_risk = min(0.85, 0.08 + (load - LOAD_HIGH) * 0.03)
    if regime == "MEMORY_THRASH":
        orch_risk = min(0.9, orch_risk + 0.3)

    therm_risk = 0.02
    if t_hot > THERMAL_HOT:
        therm_risk = min(0.95, (t_hot - THERMAL_HOT) * 0.04 + 0.3)

    return {"swap_reclaim": round(min(0.99, swap_risk), 2),
            "cpu_throttling": round(min(0.99, cpu_risk), 2),
            "orchestration": round(min(0.99, orch_risk), 2),
            "thermal": round(min(0.99, therm_risk), 2)}


def recommend(regime, f, base):
    """Recommandation d'ACTION uniquement (reveiller, refroidir, attendre) —
    la configuration runtime (backend/ngl/ctx) est deleguee a un planner
    separe (voir REVIEW_ARCHITECTURE_PREDICTOR_V1_20260905.md point 9 : ne pas
    melanger predicteur / planner / policy dans la meme fonction)."""
    model = f.get("model", "?")
    if regime == "DOZE_CAPPED":
        return ("Reveiller l'ecran (KEYCODE_WAKEUP + wm dismiss-keyguard) — caps "
                "CPU bloques a ~55% par le HAL oplus en doze. Re-verifier "
                "mWakefulness=Awake + caps au max AVANT le run.")
    if regime == "MEMORY_THRASH":
        return (f"Pression memoire critique : reduire footprint (ctx plus petit, "
                f"liberer RAM, re-run) — regime 1.5-4.7 t/s observe sur {model} "
                f"avec MemAvailable < 1.5 Go.")
    if regime == "THERMAL":
        return "Thermique > 70C : laisser refroidir (hysteresis) puis re-run."
    if regime == "SYSTEM_LOAD":
        return "Charge systeme elevee : attendre la fin du churn post-boot, tuer les processus de fond, re-run."
    if regime == "CPU_IDLE_THROTTLE":
        return ("Frequences CPU basses mais ecran Awake (pas une vraie veille) — "
               "probablement un gouverneur idle normal. Lancer le run normalement ; "
               "si le t/s observe est anormalement bas, alors seulement soupconner "
               "ce facteur (aucune penalite appliquee par defaut, non calibre).")
    return "Fenetre propre (CLEAN) — lancer le run. Voir le planner pour la config recommandee (backend/ngl/ctx)."


def predict(f):
    """Point d'entree : features -> prediction complete."""
    model = f.get("model", "gemma_mixed3")
    key = model
    model_matched = True
    if key not in BASELINES:
        model_matched = False
        matched = None
        for k in BASELINES:
            if BASELINES[k].get("is_default"):
                continue
            if k in model:
                matched = k
                break
        if matched:
            key = matched
            model_matched = True  # match par sous-chaine — moins fiable qu'un match exact, note ci-dessous
        else:
            # Correctif #4 : plus de fallback silencieux vers un modele MoE
            # specifique. Sans indice d'architecture, on prend "default_moe"
            # par prudence (bornes tres larges) mais on le signale explicitement.
            key = "default_moe"

    base = BASELINES[key]
    regime, tg, factors, evidence, secondary = estimate_tg(f, base)
    r = risks(f, base, regime)

    mult = REGIME_UNCERTAINTY_MULT.get(regime, 1.0)
    lo_ratio = base["lo"] / base["tg"]
    hi_ratio = base["hi"] / base["tg"]
    center_ratio = 1.0
    lo_ratio_widened = center_ratio - (center_ratio - lo_ratio) * mult
    hi_ratio_widened = center_ratio + (hi_ratio - center_ratio) * mult
    lo = round(max(0.0, tg * lo_ratio_widened), 2)
    hi = round(tg * hi_ratio_widened, 2)

    # Correctif #1 : "confidence", pas "probability" — ce n'est pas une
    # probabilite calibree statistiquement (V1 rule-based, <50 runs).
    confidence = 1.0 - max(r.values()) * 0.5
    if regime in ("CLEAN", "OPTIMAL"):
        confidence = max(confidence, 0.90)
    if not model_matched:
        confidence = min(confidence, 0.5)  # baisse la confiance si modele non reconnu
    # AJOUT 2026-09-06 : une baseline "theoretical_L3" n'est PAS une mesure —
    # ne jamais l'afficher avec la meme confiance qu'un baseline "measured",
    # meme si le regime detecte est CLEAN (ce qui remontait confidence a 0.90
    # ci-dessus independamment de la source).
    if base.get("source") == "theoretical_L3":
        confidence = min(confidence, 0.4)
    elif base.get("source") == "default":
        confidence = min(confidence, 0.5)

    baseline_age_days = None
    if base.get("measured_on"):
        try:
            t0 = time.mktime(time.strptime(base["measured_on"], "%Y-%m-%d"))
            baseline_age_days = round((time.time() - t0) / 86400.0, 1)
        except Exception:
            pass

    return {
        "model": key,
        "model_matched": model_matched,
        "model_arch": base.get("arch", "unknown"),
        "baseline_source": base.get("source", "measured"),
        "system_regime": regime,
        "system_regime_secondary": secondary,
        # alias retro-compatible pour les consommateurs qui lisent encore "regime"
        "regime": regime,
        "tg_expected": {"min": lo, "max": hi, "ref": round(tg, 2)},
        "confidence": round(min(0.99, confidence), 2),
        # alias retro-compatible
        "probability": round(min(0.99, confidence), 2),
        "risks": r,
        "bottleneck": {
            "DOZE_CAPPED": "CPU freq cappe par vraie veille Android (wakefulness != Awake)",
            "CPU_IDLE_THROTTLE": "frequences CPU basses mais ecran Awake — probablement "
                                 "gouverneur idle normal, pas un vrai bottleneck confirme",
            "MEMORY_THRASH": "swap/reclaim zram pendant decode",
            "THERMAL": "throttling thermique",
            "SYSTEM_LOAD": "contention AP / orchestration",
            "CLEAN": "aucun bottleneck systeme dominant (regime runtime HTP non evalue par ce predicteur V1)",
        }.get(regime, f"regime '{regime}' sans description (a completer)"),
        "recommendation": recommend(regime, f, base),
        "evidence": evidence,
        "factors": factors,
        "baseline_measured_on": base.get("measured_on"),
        "baseline_age_days": baseline_age_days,
        "baseline_needs_reverification": bool(base.get("needs_reverification")),
        "baseline_reverification_note": base.get("reverification_note"),
    }


def post_run(prediction, observed_tg, note=""):
    """Correctif #5 : boucle de retroaction prediction/observation, append-only.

    N'altere JAMAIS une prediction deja emise — enregistre un enregistrement
    separe pour permettre un calcul ulterieur de MAE/MAPE/couverture sur
    l'historique complet, sans jamais laisser le predicteur "prouver" ses
    propres hypotheses en se re-corrigeant silencieusement.
    """
    ref = prediction["tg_expected"]["ref"]
    lo = prediction["tg_expected"]["min"]
    hi = prediction["tg_expected"]["max"]
    abs_error = observed_tg - ref
    rel_error = (abs_error / ref) if ref else None
    inside_interval = lo <= observed_tg <= hi
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": prediction.get("model"),
        "system_regime": prediction.get("system_regime"),
        "predicted_tg_ref": ref, "predicted_tg_min": lo, "predicted_tg_max": hi,
        "observed_tg": observed_tg,
        "absolute_error": round(abs_error, 3),
        "relative_error_pct": round(rel_error * 100, 2) if rel_error is not None else None,
        "inside_prediction_interval": inside_interval,
        "note": note,
    }
    with open(OUTCOMES_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def summarize_outcomes(path=None):
    """MAE/MAPE/couverture sur l'historique post_run — a lancer apres 30-50
    enregistrements comme suggere par la revue architecturale."""
    path = path or OUTCOMES_LOG
    if not os.path.exists(path):
        return {"n": 0, "note": "aucun historique post_run"}
    errs, rel_errs, inside = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            errs.append(abs(r["absolute_error"]))
            if r.get("relative_error_pct") is not None:
                rel_errs.append(abs(r["relative_error_pct"]))
            inside.append(bool(r["inside_prediction_interval"]))
    n = len(errs)
    if n == 0:
        return {"n": 0, "note": "aucun enregistrement valide"}
    return {
        "n": n,
        "mae": round(sum(errs) / n, 3),
        "mape_pct": round(sum(rel_errs) / len(rel_errs), 2) if rel_errs else None,
        "coverage": round(sum(inside) / n, 3),
        "ready_for_v2_ml": n >= 50,
    }


def parse_meta(path):
    f = {"model": "gemma_mixed3", "wakefulness": "Awake",
         "caps_p0": MAX_P0, "caps_p6": MAX_P6}
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
        m = re.search(r"model=(\S+)", txt)
        if m:
            f["model"] = os.path.basename(m.group(1)).replace(".gguf", "")
        m = re.search(r"mWakefulness_before=(\w+)", txt)
        if m:
            f["wakefulness"] = m.group(1)
        m = re.search(r"scaling_max_freq_before=(\d+)/(\d+)", txt)
        if m:
            f["caps_p0"], f["caps_p6"] = int(m.group(1)), int(m.group(2))
        m = re.search(r"mem_available_kb=(\d+)", txt)
        if m:
            f["mem_available_kb"] = int(m.group(1))
        m = re.search(r"loadavg=([\d.]+)", txt)
        if m:
            f["loadavg"] = float(m.group(1))
        m = re.search(r"env_hexagon=.*?GGML_HEXAGON_OPPOLL=(\d+)", txt)
        if m:
            f["oppoll"] = int(m.group(1))
        # NOTE (robustesse, non corrige ici) : ces indices de zone thermique
        # (47/28/32) sont verifies exacts sur ce device au 2026-09-05 mais
        # restent un mapping par NUMERO, fragile aux changements firmware/
        # kernel. Un futur meta.txt qui logue le nom de zone permettrait un
        # mapping robuste par recherche de type. Voir AUDIT_PREDICTEUR... §.
        m = re.search(r"thermal_zones=.*?thermal_zone47:(\d+)", txt)
        if m:
            f["thermal_ddr_c"] = int(m.group(1)) / 1000.0
        m = re.search(r"thermal_zones=.*?thermal_zone28:(\d+)", txt)
        if m:
            f["thermal_hvx_c"] = int(m.group(1)) / 1000.0
        m = re.search(r"thermal_zones=.*?thermal_zone32:(\d+)", txt)
        if m:
            f["thermal_hmx_c"] = int(m.group(1)) / 1000.0
    except Exception as e:
        f["parse_error"] = str(e)
    return f


def main(argv=None):
    ap = argparse.ArgumentParser(prog="predictor_v1_corrige")
    ap.add_argument("--features", help="JSON dict de features systeme")
    ap.add_argument("--meta", help="chemin d'un meta.txt v3 (SNAPSHOT_BEFORE)")
    ap.add_argument("--post-run", nargs=2, metavar=("PREDICTION_JSON", "OBSERVED_TG"),
                    help="enregistre observed_tg vs une prediction JSON deja emise")
    ap.add_argument("--summarize-outcomes", action="store_true",
                    help="MAE/MAPE/couverture sur l'historique post_run")
    args = ap.parse_args(argv)

    if args.summarize_outcomes:
        print(json.dumps(summarize_outcomes(), ensure_ascii=False, indent=2))
        return
    if args.post_run:
        pred = json.loads(args.post_run[0])
        obs = float(args.post_run[1])
        print(json.dumps(post_run(pred, obs), ensure_ascii=False, indent=2))
        return
    if args.meta:
        f = parse_meta(args.meta)
    elif args.features:
        f = json.loads(args.features)
    else:
        f = {}
    print(json.dumps(predict(f), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
