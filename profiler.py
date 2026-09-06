#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""profiler.py — profiler_v3, version consolidee et nettoyee (2026-09-06) de
snapdragon_profiling/profile_model.py. Voir README.md du dossier profiler_v3
pour la vue d'ensemble (fichiers, workflow, ce qui est mesure vs simule).

Corrections apportees (2026-09-06), suite a l'audit
demande par l'utilisateur ("inspecte le dossier, audite les scripts, vois quels
angles il manque") + directive explicite : "on veut le max de qualite mais sans
perdre de debit et avec le moins de RAM possible, perdre un peu en precision
est moins grave que de OOM".

Angles corriges ici (2 des 8 identifies dans l'audit, les plus rentables) :

1. COMPATIBILITE QUANT <-> BACKEND (absente de l'original)
   `quant_plan()` original choisit le format le plus petit par famille SANS
   savoir que :
   - HTP (ggml-hexagon.cpp:ggml_hexagon_is_repack_type) n'accelere que
     F16/F32/Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4 — Q2_K/Q3_K/Q5_K/Q6_K/Q4_K tombent
     silencieusement en CPU (confirme RAPPORT_GEMMA_QUANT_MIXTE_20260903.md :
     +69% de debit juste en passant attn/MLP de Q2_K a Q4_0, sans changer la
     taille RAM de facon significative).
   - Le noyau MoE specialise d'OpenCL (dp4a, moe_router_reorder) dans ce fork
     n'accepte QUE Q4_0 pour les experts — confirme aujourd'hui (2026-09-06,
     TEST_GEMMA_NGL_EXPERTS_GPU_20260906.md) : GGML_ASSERT(0) systematique
     sur toute autre quantization, quel que soit le decoupage -ot tente.
   Le plan corrige choisit desormais, a taille egale ou quasi egale, le format
   qui reste sur l'accelerateur cible plutot qu'un format plus fin qui bascule
   silencieusement en CPU (bien plus lent) — le "max de qualite" doit se
   mesurer a debit egal, pas dans l'absolu.

2. MARGE DE SECURITE RAM ANTI-OOM (absente de l'original)
   `quant_plan()` original vise EXACTEMENT le budget fourni, sans marge. Deux
   incidents reels aujourd'hui (2026-09-06, INCIDENT_REBOOT_MTP_GPU_NPU_
   CONCURRENT et l'OOM-kill du sweep ngl Gemma, cf. TEST_GEMMA_NGL_EXPERTS_
   GPU_20260906.md) montrent que la marge reellement necessaire depasse la
   seule taille du fichier modele (buffers KV, activations, doublons
   temporaires au chargement, autres process). Le plan corrige soustrait une
   marge de securite (defaut 1.5 Go, parametrable) du budget AVANT toute
   allocation, et degrade la precision (jamais le contraire) tant que le
   budget effectif n'est pas respecte — conforme a la consigne explicite
   "perdre un peu en precision est moins grave que de OOM".

3. MODELE LOWER/UPPER BOUND (Hextimate-like, absent de l'original) — ajoute
   2026-09-06 suite a la confirmation OFFICIELLE (Qualcomm Cloud AI SDK,
   QAIRT Tools, verifie mot pour mot le 2026-09-06) que le backend Hextimate
   de Qualcomm produit exec_cycles_lower/exec_cycles_upper, pas un point
   unique. L'original `simulate_l3()` ne donne qu'un seul `lat_ms` par
   layer (deja une forme de "meilleur cas" : max(compute, memoire) plutot
   que somme). Nouvelle fonction `simulate_l3_bounds()` : reconstruit LOWER
   (chevauchement total, deja ce que fait l'original) ET UPPER (aucun
   chevauchement, somme complete compute+memoire+spill+dispatch+expert) a
   partir des memes lignes — sans recalibration, juste une lecture differente
   des donnees deja calculees. Le t/s reel mesure doit se situer entre les
   deux ; s'il est hors de la plage, c'est un signal fort qu'un terme du
   modele (pas seulement BW_EFF) est faux.

4. DECOMPOSITION BW_EFFECTIVE (absente de l'original) — le reverse
   engineering Datavorous (non officiel, a traiter comme hypothese
   d'ingenierie, PAS comme specification Qualcomm) donne :
     bandwidth = channels x width x efficiency x frequency
   Notre BW_EFF_GBS original est une seule constante calibree empiriquement
   (mesure de bout en bout), ce qui masque SI la difference vient de
   l'efficacite du kernel, de la contention (GPU+NPU concurrent, mesure
   aujourd'hui : NPU -20 a -30% selon l'ordre de lancement) ou du layout.
   Nouveaux facteurs separes (BW_EFFICIENCY, BW_CONTENTION, BW_LAYOUT,
   defaut 1.0 chacun = comportement identique a l'original) permettant
   d'isoler la cause d'un ecart mesure sans re-caler BW_EFF_GBS en bloc.

Usage (memes arguments que l'original, en plus) :
  py profiler.py modele.gguf --backend HTP --safety-margin-gb 1.5
  py profiler.py modele.gguf --backend OpenCL-MoE  # experts uniquement
  py profiler.py modele.gguf --bounds  # ajoute la section Hextimate-like lower/upper
  py profiler.py modele.gguf --live-trace trace.jsonl --emit-tensor-type-file out.txt
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_model as pm  # noqa: E402  (reutilise tout : lecture, analyse, L3)
import predictor as pred  # noqa: E402  (regime device reel, gouverneur)
import capability_db as cdb  # noqa: E402  (prediction de risque, avis seulement)

# ---------------------------------------------------------------------------
# 1. Compatibilite quant <-> backend (nouveau)
# ---------------------------------------------------------------------------
# Source : ggml-hexagon.cpp:212 ggml_hexagon_is_repack_type (fork JZ, verifie
# dans RAPPORT_GEMMA_QUANT_MIXTE_20260903.md section 1).
HTP_NATIVE_TYPES = {"F16", "Q8_0", "Q4_0"}  # + Q4_1/IQ4_NL/MXFP4 non listes
                                             # dans BPW de l'original -> ignores
                                             # ici par coherence avec profile_model.py
# Source : ggml-opencl.cpp, noyau kernel_gemm_moe_q4_0_q8_1_dp4a (fork,
# verifie empiriquement 2026-09-06, TEST_GEMMA_NGL_EXPERTS_GPU_20260906.md :
# GGML_ASSERT(0) sur tout type MoE autre que Q4_0, quel que soit le -ot).
OPENCL_MOE_NATIVE_TYPES = {"Q4_0"}

# Formats "fins" qui degradent silencieusement en CPU sur HTP si utilises
# pour attn/mlp/moe (mais restent valides pour norm/embed/lm_head, familles
# a faible poids relatif ou deja forcees F16/Q8_0 par FLOOR_WBITS).
HTP_OFFLOAD_FAMILIES = {"attn", "mlp", "moe", "ssm"}


def _best_htp_native_leq(fmt):
    """Le format HTP-natif le plus proche (<=) en bits du format demande,
    pour ne jamais monter en precision au-dela de ce qui a ete choisi par
    l'allocation de base — seulement re-brancher sur un format execute par
    l'accelerateur au lieu d'un fallback CPU silencieux."""
    order_by_bits = sorted(HTP_NATIVE_TYPES, key=lambda f: -pm.WBITS[f])
    for f in order_by_bits:
        if pm.WBITS[f] <= pm.WBITS[fmt]:
            return f
    return min(HTP_NATIVE_TYPES, key=lambda f: pm.WBITS[f])


def quant_plan_safe(a, budget_gb, backend="HTP", safety_margin_gb=1.5,
                     moe_target_backend="CPU"):
    """Plan de quant corrige : compatibilite backend + marge anti-OOM.

    backend : "HTP" (defaut) applique la regle de compatibilite HTP a
      attn/mlp/moe/ssm. "OpenCL-MoE" applique en plus la contrainte Q4_0
      stricte sur la famille "moe" (sinon les experts restent CPU, jamais
      GPU — cf. decouverte 2026-09-06).
    safety_margin_gb : soustrait du budget AVANT toute allocation. Le plan
      ne vise donc JAMAIS le budget exact — toujours une marge en dessous,
      quitte a degrader plus la precision (priorite explicite utilisateur :
      precision perdue < OOM).
    moe_target_backend : "CPU" (defaut, sur ce device les experts Q2_K/Q3_K
      restent CPU de toute facon) ou "OpenCL" (force Q4_0 sur la famille moe
      pour tenter un placement GPU — RAPPELER que cela reste EXPERIMENTAL et
      n'a jamais fonctionne au-dela de Q4_0 pur sur ce fork, cf. rapport
      2026-09-06 : le simple fait d'avoir Q4_0 ne suffit pas forcement selon
      la version du noyau OpenCL, a re-verifier avant deploiement).
    """
    effective_budget = max(budget_gb - safety_margin_gb, 0.5)
    if effective_budget < 0.5:
        print(f"  [safe] ATTENTION: budget {budget_gb} Go - marge "
              f"{safety_margin_gb} Go < 0.5 Go — marge reduite au plancher "
              f"0.5 Go, risque OOM residuel eleve.")

    base = pm.quant_plan(a, effective_budget)
    alloc = dict(base["alloc"])

    # ---- Correction anti-OOM absolue : franchir les floors si necessaire ----
    # BUG CORRIGE : pm.quant_plan() original respecte TOUJOURS FLOOR_WBITS
    # (ex. MoE jamais < 4 bits) meme si le total depasse quand meme le budget
    # -> il peut retourner "NE RENTRE PAS" alors qu'une degradation plus
    # agressive rentrerait. Consigne explicite utilisateur (2026-09-06) :
    # "perdre en precision est moins grave que de OOM" -> si le budget
    # effectif n'est toujours pas respecte apres degradation normale, on
    # continue a degrader EN DESSOUS du floor (avec avertissement fort),
    # famille la plus lourde d'abord, jusqu'a rentrer ou epuiser ORDER.
    below_floor_notes = []
    total_now = sum(a["by_family"][f][1] * pm.BPW[alloc[f]] / 2.0 for f in alloc)
    if total_now > effective_budget:
        fams_by_weight = sorted(alloc.keys(), key=lambda f: -a["by_family"][f][1])
        guard = 0
        while total_now > effective_budget and guard < 60:
            guard += 1
            moved = False
            for fam in fams_by_weight:
                idx = pm.ORDER.index(alloc[fam])
                if idx + 1 < len(pm.ORDER):
                    old = alloc[fam]
                    alloc[fam] = pm.ORDER[idx + 1]
                    below_floor_notes.append(
                        f"[SOUS FLOOR QUALITE] {fam}: {old} -> {alloc[fam]} "
                        f"(en dessous du plancher de securite qualite normal "
                        f"— accepte car budget effectif prioritaire sur la "
                        f"precision, conforme a la consigne anti-OOM)")
                    moved = True
                    break
            if not moved:
                break
            total_now = sum(a["by_family"][f][1] * pm.BPW[alloc[f]] / 2.0
                            for f in alloc)
        if total_now > effective_budget:
            below_floor_notes.append(
                f"[ECHEC] meme au format le plus bas disponible (IQ1_S), le "
                f"modele ({total_now:.1f} GiB) ne rentre pas dans le budget "
                f"effectif ({effective_budget:.1f} GiB) — ce modele est trop "
                f"gros pour ce device a cette marge de securite, aucune "
                f"quantization ne peut resoudre ca seule.")

        # AJOUT 2026-09-06 : prediction du risque de collapse qualitatif — pas
        # juste "moins precis", potentiellement un modele CASSE. Incident reel
        # confirme le meme jour : Qwen3.8-9B-Q2_K.gguf (dense, sans imatrix)
        # charge normalement mais genere un EOS immediat (0 token utile) —
        # fichier supprime suite a ce test. Le plancher normal (FLOOR_WBITS)
        # protege deja contre ca en temps normal ; ce garde-fou avertit
        # specifiquement quand la degradation anti-OOM ci-dessus force une
        # famille EN DESSOUS de Q3_K (indice ORDER >= 6), la ou le risque de
        # sortie degeneree a ete observe empiriquement, pas seulement suppose.
        collapse_risk_idx = pm.ORDER.index("Q3_K")
        at_risk = {fam: fmt for fam, fmt in alloc.items()
                  if pm.ORDER.index(fmt) >= collapse_risk_idx}
        if at_risk:
            fams_s = ", ".join(f"{f}={t}" for f, t in sorted(at_risk.items()))
            below_floor_notes.append(
                f"[RISQUE COLLAPSE QUALITE] familles <= Q3_K sans imatrix : {fams_s}. "
                f"Incident reel confirme 2026-09-06 : Qwen3.8-9B-Q2_K.gguf (dense) "
                f"chargeait normalement mais generait un EOS immediat (sortie vide/"
                f"inutilisable), pas juste 'moins precis'. Non garanti pour CE "
                f"modele, mais a TESTER avant deploiement, pas a supposer sans "
                f"danger sous pretexte que le fichier charge correctement.")

    # ---- Correction compatibilite backend ----
    notes = list(below_floor_notes)
    if backend == "HTP":
        for fam in HTP_OFFLOAD_FAMILIES:
            if fam not in alloc:
                continue
            fmt = alloc[fam]
            if fmt not in HTP_NATIVE_TYPES:
                native = _best_htp_native_leq(fmt)
                # Ne re-brancher que si le format natif ne fait pas
                # exploser le budget (sinon on reste sur le fallback CPU,
                # deja pris en compte par l'allocation gloutonne de base) :
                gib_native = a["by_family"][fam][1] * pm.BPW[native] / 2.0
                gib_current = a["by_family"][fam][1] * pm.BPW[fmt] / 2.0
                total_after = total_now - gib_current + gib_native
                if total_after <= effective_budget:
                    alloc[fam] = native
                    notes.append(
                        f"{fam}: {fmt} -> {native} (HTP natif, evite fallback "
                        f"CPU silencieux ; +{gib_native - gib_current:.2f} GiB, "
                        f"toujours <= budget effectif {effective_budget:.1f} GiB)")
                else:
                    notes.append(
                        f"{fam}: reste {fmt} (fallback CPU) — passer a {native} "
                        f"depasserait le budget effectif de "
                        f"{total_after - effective_budget:.2f} GiB ; a ce "
                        f"device/budget, {fmt}+CPU est le compromis retenu")

    total_now = sum(a["by_family"][f][1] * pm.BPW[alloc[f]] / 2.0 for f in alloc)
    if backend == "OpenCL-MoE" or moe_target_backend == "OpenCL":
        if "moe" in alloc and alloc["moe"] not in OPENCL_MOE_NATIVE_TYPES:
            gib_q40 = a["by_family"]["moe"][1] * pm.BPW["Q4_0"] / 2.0
            gib_current = a["by_family"]["moe"][1] * pm.BPW[alloc["moe"]] / 2.0
            total_after = total_now - gib_current + gib_q40
            mixed_still = False
            if total_after <= effective_budget:
                alloc["moe"] = "Q4_0"
                notes.append(
                    f"moe: -> Q4_0 (seul format accepte par le noyau MoE "
                    f"OpenCL dp4a de ce fork, cf. decouverte 2026-09-06) ; "
                    f"+{gib_q40 - gib_current:.2f} GiB")
            else:
                mixed_still = True
                notes.append(
                    "moe: placement OpenCL DEMANDE mais impossible sans "
                    f"depasser le budget effectif de "
                    f"{total_after - effective_budget:.2f} GiB — experts "
                    "resteront CPU malgre la demande (priorite RAM > "
                    "placement GPU, conforme a la consigne anti-OOM)")
            notes.append("[EXPERIMENTAL] Q4_0 seul necessaire mais PAS "
                          "confirme suffisant pour un vrai placement OpenCL "
                          "MoE sur ce fork — a revalider avant deploiement.")
            # AJOUT 2026-09-06 : avis de risque, PAS bloquant — la decision
            # d'aller quand meme en OpenCL reste a l'utilisateur, mais elle
            # doit voir ce signal AVANT de lancer un test device.
            risk = cdb.predict_crash_risk("Q4_0", "MUL_MAT (MoE experts)", "OpenCL",
                                          moe_mixed_format=mixed_still)
            notes.append(f"[{risk['risk']}] " + " ; ".join(risk["reasons"]))

    total = sum(a["by_family"][f][1] * pm.BPW[alloc[f]] / 2.0
                for f in alloc)
    return {
        "alloc": alloc,
        "total_gib": total,
        "budget_gb": budget_gb,
        "safety_margin_gb": safety_margin_gb,
        "effective_budget_gb": effective_budget,
        "fits_effective_budget": total <= effective_budget,
        "fits_nominal_budget": total <= budget_gb,
        "backend": backend,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# 3. Decomposition BW_effective (nouveau) — hypothese d'ingenierie inspiree
# du RE Datavorous (bandwidth = channels x width x efficiency x frequency),
# PAS une constante Qualcomm officielle. Defaut = 1.0 partout -> comportement
# strictement identique a l'original tant que non calibre.
# ---------------------------------------------------------------------------
BW_EFFICIENCY = 1.0    # perte due au kernel/layout du chemin memoire (<=1.0)
BW_CONTENTION = 1.0    # perte due a la concurrence GPU+NPU ou autres consommateurs
                       # DDR (mesure 2026-09-05/06 : NPU -20 a -30% selon ordre de
                       # lancement en co-execution GPU+NPU reelle -> ~0.70-0.80 si
                       # co-execution active, 1.0 si NPU seul)
BW_LAYOUT = 1.0        # perte due au layout tuile/flat/DDR (cf. by_path du live trace)


def bw_effective_decomposed(bw_physical, efficiency=None, contention=None, layout=None):
    """BW_effective = BW_physical x efficiency x contention x layout.
    Defaut = bw_physical inchange (retro-compatible avec l'original)."""
    e = BW_EFFICIENCY if efficiency is None else efficiency
    c = BW_CONTENTION if contention is None else contention
    l = BW_LAYOUT if layout is None else layout
    return bw_physical * e * c * l


# ---------------------------------------------------------------------------
# 4. Modele lower/upper bound Hextimate-like (nouveau)
# ---------------------------------------------------------------------------
def simulate_l3_bounds(a, ctx=4096, contention=None):
    """Reconstruit LOWER (chevauchement total, deja calcule par pm.simulate_l3
    via max(compute, memoire)) et UPPER (aucun chevauchement, somme complete)
    a partir des memes lignes de calcul — sans nouvelle calibration.

    contention : si fourni (ex. 0.75 pour une session GPU+NPU concurrente,
    cf. mesures 2026-09-05/06), applique un facteur multiplicatif a la partie
    memoire de CHAQUE couche avant de calculer les bornes — permet de
    modeliser l'effet mesure de la co-execution sans toucher a BW_EFF_GBS
    globalement (qui doit rester la calibration NPU-seul).
    """
    l3 = a["l3"]  # deja calcule par pm.analyze() -> a['l3']
    lower_total = 0.0
    upper_total = 0.0
    rows_out = []
    cf = BW_CONTENTION if contention is None else contention
    for r in l3["rows"]:
        mem = r["mem_ms_spill"] * (1.0 / cf if cf > 0 else 1.0)
        compute = r["compute_ms"]
        dispatch = r["dispatch_ms"]
        mmid = r.get("mmid_ms", 0.0) or 0.0
        lower = max(compute, mem) + dispatch + mmid
        upper = compute + mem + dispatch + mmid
        lower_total += lower
        upper_total += upper
        rows_out.append({**r, "mem_ms_contended": mem,
                         "lower_ms": lower, "upper_ms": upper})
    return {
        "rows": rows_out,
        "lower_total_ms": lower_total,
        "upper_total_ms": upper_total,
        "lower_tps": 1000.0 / upper_total if upper_total > 0 else 0.0,   # pire cas -> tps le PLUS BAS
        "upper_tps": 1000.0 / lower_total if lower_total > 0 else 0.0,   # meilleur cas -> tps le PLUS HAUT
        "contention_applied": cf,
    }


def render_bounds_section(bounds, measured_tps=None):
    L = ["", "## Modele Hextimate-like — bornes lower/upper (pas un point unique)", ""]
    L.append(f"- contention appliquee : x{1/bounds['contention_applied']:.2f} sur le terme "
             f"memoire (1.0 = NPU seul, cf. mesures co-execution GPU+NPU 2026-09-05/06)")
    L.append(f"- **t/s si AUCUN chevauchement (upper bound temps = pire cas) : "
             f"{bounds['lower_tps']:.1f} t/s**")
    L.append(f"- **t/s si chevauchement TOTAL (lower bound temps = meilleur cas) : "
             f"{bounds['upper_tps']:.1f} t/s**")
    L.append(f"- plage predite : [{bounds['lower_tps']:.1f}, {bounds['upper_tps']:.1f}] t/s")
    if measured_tps is not None:
        in_range = bounds['lower_tps'] <= measured_tps <= bounds['upper_tps']
        L.append(f"- t/s mesure : {measured_tps:.1f} — "
                 f"{'DANS la plage predite (modele coherent)' if in_range else '**HORS PLAGE** (un terme du modele est probablement faux, pas seulement BW_EFF)'}")
    L.append("")
    L.append("Lecture : contrairement a un point unique (l'original), une plage donne un test de "
             "coherence direct — un t/s mesure hors de [lower, upper] signale un terme manquant "
             "(ex. RPC/queue/synchronisation non modelise), pas juste une mauvaise valeur de BW_EFF.")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 4bis. Regime device REEL (nouveau) — integre governor/predictor.py
# ---------------------------------------------------------------------------
# PROBLEME CORRIGE : le "soutenu (thermique)" de l'original (a['l3']['thermal_tps'])
# applique une constante FIXE (THERMAL_TPS_FACTOR=0.65) IDENTIQUE quel que soit
# l'etat REEL du device au moment du profilage — jamais de regard sur le
# thermique/mem/load/doze effectifs. Le gouverneur (predictor.py,
# deja corrige le 2026-09-05 : regimes DOZE_CAPPED/MEMORY_THRASH/THERMAL/
# SYSTEM_LOAD/CLEAN, facteurs multiplicatifs par regime a partir de features
# device reelles) fait exactement ce que le L3 devrait faire ici — on le
# reutilise au lieu de dupliquer une nouvelle logique de regime.
def apply_device_regime(a, device_meta_path=None, device_features=None):
    """Remplace le facteur thermique FIXE de l'original par le facteur REEL
    du gouverneur (detect_regime), a partir d'un fichier meta (--device-meta,
    format texte cle=valeur attendu par predictor.parse_meta) ou
    d'un dict de features deja construit. Sans aucune des deux entrees,
    retourne None (pas de degradation silencieuse — le fixed 0.65 de
    l'original reste affiche tel quel, juste sans cette section en plus)."""
    f = device_features
    if f is None and device_meta_path:
        f = pred.parse_meta(device_meta_path)
    if f is None:
        return None
    regime, factors, evidence, secondary = pred.detect_regime(f)
    combined_factor = 1.0
    for v in factors.values():
        combined_factor *= v
    cold_tps = a["l3"]["cold_tps"]
    return {
        "regime": regime, "secondary": secondary, "evidence": evidence,
        "factors": factors, "combined_factor": combined_factor,
        "cold_tps": cold_tps,
        "regime_corrected_tps": cold_tps * combined_factor,
        "original_fixed_tps": a["l3"]["thermal_tps"],
        "original_fixed_factor": pm.THERMAL_TPS_FACTOR,
    }


def render_device_regime_section(r):
    if r is None:
        return ("\n## Regime device reel : NON APPLIQUE\n\n"
                "Aucun --device-meta fourni — le \"soutenu (thermique)\" ci-dessus reste "
                f"la constante FIXE de l'original (x{pm.THERMAL_TPS_FACTOR}), qui ne "
                "reflete PAS l'etat reel du device au moment du profilage. Fournir "
                "--device-meta <fichier> (meme format que celui lu par "
                "predictor.parse_meta : model=/mWakefulness_before=/"
                "scaling_max_freq_before=/mem_available_kb=/loadavg=) pour une estimation "
                "\"soutenu\" reflet de l'etat reel plutot que d'une moyenne generique.")
    L = ["", "## Regime device reel (gouverneur, remplace la constante thermique fixe)", ""]
    L.append(f"- regime detecte : **{r['regime']}**" +
             (f" (+ secondaire: {', '.join(r['secondary'])})" if r["secondary"] else ""))
    L.append(f"- evidence : {r['evidence']}")
    L.append(f"- facteur combine (caps x thermal x mem x load) : x{r['combined_factor']:.3f}")
    L.append(f"- t/s \"soutenu\" ORIGINAL (constante fixe x{r['original_fixed_factor']}) : "
             f"{r['original_fixed_tps']:.1f} t/s")
    L.append(f"- t/s \"soutenu\" CORRIGE (regime device reel) : "
             f"**{r['regime_corrected_tps']:.1f} t/s**")
    if r["regime"] not in ("CLEAN", "OPTIMAL", "CPU_IDLE_THROTTLE"):
        L.append(f"- [ATTENTION] regime dégradé détecté ({r['regime']}) — la mesure a "
                 "venir sur ce device sera probablement plus basse que le \"froid\" "
                 "L3, et l'ecart vs la constante fixe originale peut etre important.")
    L.append("")
    L.append("Limite : le facteur du gouverneur est une heuristique V1 (regles, pas "
             "calibration statistique) — voir REGIME_UNCERTAINTY_MULT dans "
             "predictor.py pour la largeur d'incertitude par regime.")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 5. Table par couche — DEPUIS 2026-09-06, alimentable par une VRAIE trace
# device (voir parse_hexagon_profile.py : l'instrumentation ggml-hexagon.cpp
# ggml_hexagon_dump_op_prof/ggml_hexagon_dump_batch_prof existe deja dans le
# runtime deploye, juste jamais activee (GGML_HEXAGON_PROFILE=1 + -lv 5) ni
# parsee avant ce jour). Le cout d'ARETE au sens strict (transition inter-
# couche : repack/layout/sync) reste hors de portee sans plus de travail —
# mais parse_hexagon_profile.py mesure deja un "overhead de batch" REEL
# (temps du batch FastRPC entier moins la somme de ses ops), qui EST une
# mesure directe du cout d'orchestration/dispatch manquant au modele L3
# (mesure 2026-09-06, Qwen3-0.9B : 485us/batch en moyenne vs DISPATCH_US=215us
# suppose par le L3 — 2.3x plus eleve). Ce n'est pas encore attribue a une
# PAIRE de couches precise (couche N -> couche N+1), mais c'est un vrai debut,
# pas une simulation.
# ---------------------------------------------------------------------------
def render_per_layer_table(live):
    """Table LAYER_COST a partir de live['per_layer'] (deja calcule par
    pm.aggregate_live_trace, jamais rendu en tableau dans l'original)."""
    pl = live.get("per_layer", {})
    if not pl:
        return ("\n## Table par couche\n\n(aucune donnee 'layer' dans cette trace — "
                "verifier que les evenements portent bien un champ 'layer')")
    L = ["", "## Table par couche (LAYER_COST, depuis la trace live)", ""]
    L.append("| layer | total us | MUL_MAT_ID us | ratio MMID | top kernel (2e) |")
    L.append("|---:|---:|---:|---:|---|")
    items = sorted(pl.items(), key=lambda kv: -kv[1]["total_us"])
    for lay, p in items:
        top2 = list(p.get("kernels_us", {}).items())[:2]
        top2_s = ", ".join(f"{k}={v:.0f}us" for k, v in top2)
        L.append(f"| {lay} | {p['total_us']:.0f} | {p['mmid_us']:.0f} | "
                 f"{p['mmid_ratio_layer']:.1%} | {top2_s} |")
    slowest = items[0] if items else None
    if slowest:
        L.append("")
        L.append(f"**Couche la plus couteuse : layer {slowest[0]} "
                 f"({slowest[1]['total_us']:.0f} us cumules).**")
    L.append("")
    L.append("[LIMITE] Ce tableau ne montre pas encore le cout de transition PAR PAIRE "
             "de couches (repack/layout/sync entre couche N et N+1 precisement) — mais "
             "voir parse_hexagon_profile.py --out pour l'overhead de dispatch/orchestration "
             "REEL par batch (mesure directe, pas une simulation), qui donne deja une "
             "premiere approximation de ce terme au niveau du batch entier.")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 2. OPPOLL — levier de dispatch AP-side manquant (nouveau, informationnel)
# ---------------------------------------------------------------------------
# Mesure 2026-09-05/06 : GGML_HEXAGON_OPPOLL=1 sur self-build-jz donne
# 32.47 t/s vs 21.4-22.3 t/s (defaut/OPPOLL=0) sur Marco-Nano — soit environ
# +50%. Un seul modele/config teste a ce jour : a traiter comme un defaut
# recommande, pas encore une constante calibree au meme niveau que BW_EFF_*.
RECOMMENDED_RUNTIME_ENV = {
    "GGML_HEXAGON_OPPOLL": {
        "value": "1",
        "gain_mesure": "+50% (32.47 vs 21.4-22.3 t/s, Marco-Nano, self-build-jz, 2026-09-05/06)",
        "validation": "1 modele, 1 build — a confirmer sur d'autres modeles/tailles avant de le considerer acquis partout",
        "risque": "aucun observe a ce jour (pas de correction degradee constatee)",
    },
}


# ---------------------------------------------------------------------------
# 7. Clarification section 3 vs section 6 (nouveau, suite a une confusion
# repetee 2026-09-06 : "orchestration : N ops/token x 215us" (section 3 de
# l'original) et le t/s du modele L3/Hextimate-like (section 6) semblent
# incoherents au premier regard (78.3ms/12.8t/s vs 23.6ms/42.3t/s sur Marco-
# Nano-V2) — CE N'EST PAS UN BUG : verifie dans le code source, ce sont deux
# modeles voulus differents. Section 3 compte CHAQUE micro-op individuellement
# (aucun regroupement, "borne fixe HTP" = garde-fou pathologique, jamais une
# prediction de t/s). Section 6 utilise ops=1 par couche, DEJA calibre par
# l'auteur original sur une mesure reelle Marco (29 dispatchs FastRPC mesures
# pour 28 couches -> le runtime batche les micro-ops par couche, cf.
# profile_model.py:635-637 "dispatch calibre : Marco reel = 29 ops/token / 28
# layers ~ 1 op/layer (batch HTP OPBATCH=512)"). Le t/s a retenir est TOUJOURS
# celui de la section 6 (L3), jamais 1000/orchestration_ms.
# ---------------------------------------------------------------------------
def render_orchestration_vs_l3_note(a):
    orch_ms = a["orchestration_ms"]
    orch_tps = 1000.0 / orch_ms if orch_ms > 0 else 0.0
    L = ["", "## Note : section 3 (orchestration) vs section 6 (L3) ne se comparent PAS", ""]
    L.append(f"- section 3 : {a['ops_per_token']} ops/token x 215us = {orch_ms:.1f} ms "
             f"(={orch_tps:.1f} t/s si lu a tort comme une prediction) — c'est une "
             f"borne PATHOLOGIQUE (aucun regroupement d'ops), PAS une prediction.")
    L.append(f"- section 6 (L3, a retenir) : {a['l3']['total_ms']:.1f} ms "
             f"(={1000.0/a['l3']['total_ms']:.1f} t/s) — calibree sur une mesure reelle "
             f"(29 dispatchs FastRPC/token mesures sur Marco, pas {a['ops_per_token']}).")
    L.append("Les deux chiffres NE DOIVENT JAMAIS etre compares directement — ils ne "
             "modelisent pas la meme granularite de dispatch. Le t/s a citer est "
             "toujours celui de la section 6 (L3) / --bounds, jamais 1000/orchestration_ms.")
    return "\n".join(L)


def render_safe_plan(plan, a):
    L = ["", "## Plan de quant SECURISE (anti-OOM + compatibilite backend)", ""]
    L.append(f"- budget nominal : {plan['budget_gb']:.1f} Go · marge securite : "
             f"{plan['safety_margin_gb']:.1f} Go · **budget effectif : "
             f"{plan['effective_budget_gb']:.1f} Go**")
    L.append(f"- backend cible : {plan['backend']}")
    L.append("| famille | format | GiB |")
    L.append("|---|---:|---:|")
    for fam, fmt in sorted(plan["alloc"].items(), key=lambda kv: -a["by_family"][kv[0]][1]):
        gib = a["by_family"][fam][1] * pm.BPW[fmt] / 2.0
        L.append(f"| {fam} | {fmt} | {gib:.2f} |")
    L.append(f"**Total : {plan['total_gib']:.2f} GiB — "
             f"{'OK (sous budget effectif)' if plan['fits_effective_budget'] else 'NE RENTRE PAS meme apres degradation'}**")
    if plan["notes"]:
        L.append("")
        L.append("Notes de compatibilite backend :")
        for n in plan["notes"]:
            L.append(f"- {n}")
    L.append("")
    L.append("### Leviers runtime recommandes (hors quantization)")
    for env, info in RECOMMENDED_RUNTIME_ENV.items():
        L.append(f"- `{env}={info['value']}` — gain mesure : {info['gain_mesure']} "
                 f"(validation : {info['validation']} ; risque : {info['risque']})")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 6. GGUF concret : rows par tenseur + fichier --tensor-type-file (nouveau)
# ---------------------------------------------------------------------------
# Ce que "ce qui est possible" recouvre reellement aujourd'hui (sans cout
# d'arete, cf. limite documentee dans render_per_layer_table) :
#   - le nom, la famille, le layer et le format GGUF de CHAQUE tenseur
#     (pm.classify() les calcule deja, mais pm.analyze() ne les renvoie pas —
#     seulement l'agrege par famille). build_tensor_rows() les recupere sans
#     toucher a l'original.
#   - le cout PAR COUCHE deja mesure par la trace live (per_layer, existant).
# En combinant les deux, on peut differencier le format PAR COUCHE au lieu du
# format uniforme par famille de quant_plan_safe : les couches mesurees les
# plus couteuses recoivent un format plus fin (dans la limite HTP-native, pour
# ne jamais retomber en fallback CPU), les couches les moins couteuses un
# format plus agressif — a taille totale sensiblement egale. C'est un vrai
# fichier --tensor-type-file pret pour llama-quantize (format confirme dans
# tools/quantize/quantize.cpp:317 parse_tensor_type, "nom=TYPE" par ligne,
# nom matche par regex complet contre le nom de tenseur reel, cf.
# src/llama-quant.cpp:188 tensor_type_patterns).
def build_tensor_rows(tensors, is_gguf):
    """Reconstruit les rows par tenseur (nom/famille/layer/format) que
    pm.analyze() calcule en interne mais ne renvoie pas — reutilise
    pm.classify(), aucune modification de l'original."""
    rows = []
    for name, info in tensors.items():
        r = pm.classify(name)
        r["elems"] = info.get("elems", 0)
        if is_gguf:
            r["bytes_real"] = info.get("bytes", 0)
            r["gguf_type"] = pm.V_TYPES.get(info.get("ttype", -1), "?")
        rows.append(r)
    return rows


def _layer_hot_cold(live, n_layer, top_fraction=0.25):
    """Classe les layers en 'hot'/'cold'/neutre a partir du total_us mesure
    dans la trace live. Retourne (hot_set, cold_set). Vide si pas de trace."""
    pl = (live or {}).get("per_layer", {})
    if not pl:
        return set(), set()
    ranked = sorted(pl.items(), key=lambda kv: -kv[1]["total_us"])
    n_edge = max(1, int(len(ranked) * top_fraction))
    hot = {int(k) for k, _ in ranked[:n_edge] if str(k).isdigit()}
    cold = {int(k) for k, _ in ranked[-n_edge:] if str(k).isdigit()}
    return hot, cold


def generate_tensor_type_file(rows, plan, live=None, n_layer=0, top_fraction=0.25,
                              out_path=None):
    """Genere les lignes 'tensor_name=TYPE' d'un --tensor-type-file llama-
    quantize, en partant du plan par famille (quant_plan_safe) et en
    differenciant par couche quand une trace live est disponible.

    Regle stricte : ne JAMAIS sortir de HTP_NATIVE_TYPES (evite tout fallback
    CPU silencieux, cf. correction 1 de ce fichier) — hot layer = format
    HTP-natif immediatement superieur au format de base de la famille ; cold
    layer = format HTP-natif immediatement inferieur. Neutre (pas de trace,
    ou layer ni hot ni cold) = format de base de la famille (aucune ligne
    emise, le --tensor-type-file ne doit lister QUE les divergences).
    """
    alloc = plan["alloc"]
    hot, cold = _layer_hot_cold(live, n_layer, top_fraction)
    order_by_bits = sorted(HTP_NATIVE_TYPES, key=lambda f: pm.WBITS[f])
    lines = []
    stats = {"hot_bumped": 0, "cold_demoted": 0, "unchanged": 0, "skipped_no_family": 0}
    for r in rows:
        fam = r["family"]
        base_fmt = alloc.get(fam)
        if base_fmt is None or fam not in HTP_OFFLOAD_FAMILIES:
            stats["skipped_no_family"] += 1
            continue
        target = base_fmt
        if r["layer"] is not None:
            idx = order_by_bits.index(base_fmt) if base_fmt in order_by_bits else None
            if idx is not None:
                if r["layer"] in hot and idx + 1 < len(order_by_bits):
                    target = order_by_bits[idx + 1]
                    stats["hot_bumped"] += 1
                elif r["layer"] in cold and idx - 1 >= 0:
                    target = order_by_bits[idx - 1]
                    stats["cold_demoted"] += 1
                else:
                    stats["unchanged"] += 1
            else:
                stats["unchanged"] += 1
        else:
            stats["unchanged"] += 1
        if target != base_fmt:
            # nom de tenseur utilise tel quel comme regex complet cote
            # llama-quant.cpp (std::regex_search sur le nom reel) — un nom
            # exact, meme non-echappe, matche son propre litteral (les seuls
            # metacaracteres presents dans les noms ggml usuels sont '.', qui
            # matche aussi lui-meme comme "n'importe quel caractere").
            lines.append(f"{r['name']}={target.lower()}")

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
    return {"lines": lines, "stats": stats, "n_hot_layers": len(hot),
            "n_cold_layers": len(cold), "out_path": out_path}


# ---------------------------------------------------------------------------
# 8. Prediction de quant PAR COUCHE avec impact inter-couches (nouveau,
# demande explicite : "predire la quantification de chaque layer avec les
# impacts sur les autres, la precision finale vs la quantite de RAM et le
# debit, et le raisonnement"). Allocation gloutonne sous budget PARTAGE entre
# couches — chaque Go depense sur une couche est un Go que les couches
# suivantes ne peuvent plus depenser : c'est exactement "l'impact sur les
# autres" demande, rendu explicite dans le rapport (section "skipped").
#
# LIMITE HONNETE SUR LE DEBIT (verifiee, pas supposee) : le precedent
# RAPPORT_DECISIF_ABC_Q8_Q4_MXFP4_MEMORYWALL_20260831.md (Qwen3.5-9B-D2-A,
# attention seule, protocole controle) montre que le debit NE SUIT PAS le
# nombre de bits de facon monotone : Q8->Q4 (-47% octets, MUL_MAT -17%, DSP
# -14%) a fait BAISSER le decode de 8% (pas augmenter) ; Q4->MXFP4 (octets/
# DSP quasi identiques) a donne +13%. Le debit depend du CHEMIN D'EXECUTION
# (tiling/kernel), pas seulement de la taille des poids. Ce module ne
# pretend donc PAS predire le debit par couche — seulement RAM et une
# "qualite proxy" ordinale (bits ponderes), explicitement pas une perplexite
# mesuree (aucune donnee imatrix/ppl disponible dans ce projet a ce jour).
# ---------------------------------------------------------------------------
def predict_layer_quant_impact(rows, a, budget_gb, live=None, safety_margin_gb=1.5):
    """Alloue le budget RAM couche par couche (allocation gloutonne, budget
    partage) : part de Q4_0 (floor HTP-natif) partout, puis monte les
    couches offrant le meilleur ratio qualite-proxy/Go tant que le budget
    effectif le permet. Retourne les decisions dans l'ordre + ce qui a ete
    SKIPPE faute de budget (l'impact concret sur les autres couches)."""
    effective_budget = max(budget_gb - safety_margin_gb, 0.5)
    order_by_bits = sorted(HTP_NATIVE_TYPES, key=lambda f: pm.WBITS[f])

    by_layer = {}
    for r in rows:
        if r["family"] not in HTP_OFFLOAD_FAMILIES or r["layer"] is None:
            continue
        by_layer.setdefault(r["layer"], []).append(r)

    def layer_gib(layer, fmt_):
        # BPW[fmt] est deja en octets/poids (ex Q4_0=18/32=0.5625, F16=2.0) —
        # bytes = elems * BPW[fmt] directement, PAS de /2.0 supplementaire
        # (bug corrige 2026-09-06 : la division en trop sous-estimait la RAM
        # de moitie, trouve en comparant au a['by_family'] deja valide).
        return sum(row["elems"] * pm.BPW[fmt_] for row in by_layer[layer]) / 2**30

    fmt = {layer: order_by_bits[0] for layer in by_layer}
    total_gib = sum(layer_gib(l, fmt[l]) for l in by_layer)

    pl = (live or {}).get("per_layer", {})
    tot_us = sum(p["total_us"] for p in pl.values()) if pl else 0.0
    # per_layer est keye par str (aggregate_live_trace des-jsonify les layers en
    # str), alors que by_layer (ci-dessus) est keye par int (pm.classify()) —
    # bug corrige ici : sans str(l) le lookup echouait silencieusement (0.0% partout).
    cost_share = {l: (pl.get(str(l), {}).get("total_us", 0.0) / tot_us if tot_us > 0 else None)
                 for l in by_layer}

    decisions = []
    guard = 0
    while guard < 10000:
        guard += 1
        best = None
        best_value = -1.0
        for l in by_layer:
            idx = order_by_bits.index(fmt[l])
            if idx + 1 >= len(order_by_bits):
                continue
            next_fmt = order_by_bits[idx + 1]
            delta_gib = layer_gib(l, next_fmt) - layer_gib(l, fmt[l])
            if delta_gib <= 0 or total_gib + delta_gib > effective_budget:
                continue
            delta_bits = pm.WBITS[next_fmt] - pm.WBITS[fmt[l]]
            value = delta_bits / delta_gib
            if value > best_value:
                best_value = value
                best = (l, next_fmt, delta_gib)
        if best is None:
            break
        l, next_fmt, delta_gib = best
        decisions.append({"layer": l, "from": fmt[l], "to": next_fmt,
                          "delta_gib": delta_gib, "cost_share": cost_share.get(l)})
        fmt[l] = next_fmt
        total_gib += delta_gib

    skipped = []
    for l in by_layer:
        idx = order_by_bits.index(fmt[l])
        if idx + 1 < len(order_by_bits):
            next_fmt = order_by_bits[idx + 1]
            delta_gib = layer_gib(l, next_fmt) - layer_gib(l, fmt[l])
            skipped.append({"layer": l, "would_be": next_fmt, "delta_gib": delta_gib})

    total_bytes_final = sum(layer_gib(l, fmt[l]) for l in by_layer)
    quality_proxy = (sum(pm.WBITS[fmt[l]] * layer_gib(l, fmt[l]) for l in by_layer) /
                     total_bytes_final) if total_bytes_final > 0 else 0.0

    for d in decisions:
        d["cumulative_gib"] = None  # rempli ci-dessous dans l'ordre chronologique
    running = sum(layer_gib(l, order_by_bits[0]) for l in by_layer)
    for d in decisions:
        running += d["delta_gib"]
        d["cumulative_gib"] = running

    return {"fmt": fmt, "decisions": decisions, "skipped": skipped,
            "total_gib": total_gib, "effective_budget_gb": effective_budget,
            "quality_proxy_bits": quality_proxy, "has_live_cost": bool(pl)}


def render_layer_quant_impact(result):
    L = ["", "## Prediction quant PAR COUCHE avec impact inter-couches (budget partage)", ""]
    L.append(f"- budget effectif : {result['effective_budget_gb']:.2f} Go")
    L.append(f"- total alloue (familles HTP-offloadables uniquement) : "
             f"{result['total_gib']:.2f} Go")
    L.append(f"- qualite proxy (bits ponderes par octet — PAS une perplexite mesuree, "
             f"aucune donnee imatrix/ppl disponible) : {result['quality_proxy_bits']:.2f} bits/poids")
    if not result["has_live_cost"]:
        L.append("- [SANS TRACE LIVE] la colonne 'cout mesure' ci-dessous est absente — "
                 "l'ordre de priorite ne s'appuie que sur le ratio qualite/Go, pas sur "
                 "l'importance reelle de la couche pour le debit")
    L.append("")
    L.append("| ordre | couche | avant | apres | +Go | Go cumules | part du temps mesure | raison |")
    L.append("|---:|---:|---|---|---:|---:|---:|---|")
    for i, d in enumerate(result["decisions"], 1):
        cost_s = f"{d['cost_share']*100:.1f}%" if d.get("cost_share") is not None else "—"
        L.append(f"| {i} | {d['layer']} | {d['from']} | {d['to']} | {d['delta_gib']:.3f} | "
                 f"{d['cumulative_gib']:.2f} | {cost_s} | meilleur ratio qualite-proxy/Go "
                 f"disponible a cet instant |")
    if result["skipped"]:
        L.append("")
        L.append("**Impact direct sur les autres couches (ce qui n'a PAS pu etre fait, "
                 "faute de budget deja consomme par les decisions ci-dessus) :**")
        for s in sorted(result["skipped"], key=lambda x: x["delta_gib"]):
            L.append(f"- couche {s['layer']} : aurait pu monter a {s['would_be']} "
                     f"pour +{s['delta_gib']:.3f} Go supplementaires")
    else:
        L.append("")
        L.append("Aucune couche skippee — le budget effectif suffit a monter TOUTES les "
                 "couches HTP-offloadables au format maximal (F16), pas de compromis a faire.")
    L.append("")
    L.append("### Debit (t/s) — avertissement explicite, PAS une prediction")
    L.append("Ce plan optimise RAM vs qualite-proxy (bits), **jamais le debit**. Precedent "
             "verifie (`RAPPORT_DECISIF_ABC_Q8_Q4_MXFP4_MEMORYWALL_20260831.md`, "
             "Qwen3.5-9B-D2-A, attention seule, protocole controle) : Q8_0->Q4_0 (-47% "
             "octets attn, MUL_MAT -17%, DSP total -14%) a fait **BAISSER** le decode de "
             "8% (5.86 vs 6.39 t/s) — pas augmenter comme on l'attendrait naivement. "
             "Q4_0->MXFP4 (octets/DSP quasi identiques, -1%/=) a donne **+13%** (6.64 t/s). "
             "Conclusion verifiee : le debit depend du CHEMIN D'EXECUTION (tiling/kernel "
             "dequant), pas seulement du nombre de bits des poids — ce module ne pretend "
             "donc pas predire un gain de t/s par couche, seulement RAM et qualite-proxy. "
             "Mesurer le t/s reel avant/apres tout changement de --tensor-type-file reste "
             "indispensable.")
    return "\n".join(L)


def render_tensor_type_section(result, plan):
    L = ["", "## Fichier --tensor-type-file genere (GGUF concret)", ""]
    s = result["stats"]
    if result["n_hot_layers"] == 0 and result["n_cold_layers"] == 0:
        L.append("[SANS TRACE LIVE] aucune trace --live-trace fournie -> aucune "
                 "differenciation par couche possible, le fichier genere est VIDE "
                 "(le plan par famille de quant_plan_safe suffit alors seul, "
                 "cf. --tensor-type-file inutile dans ce cas : passer directement "
                 "le format de chaque famille via des flags globaux equivalents).")
    else:
        L.append(f"- {result['n_hot_layers']} couche(s) 'hot' (mesurees les plus "
                 f"couteuses) -> format HTP-natif immediatement superieur "
                 f"({s['hot_bumped']} tenseurs bumpes)")
        L.append(f"- {result['n_cold_layers']} couche(s) 'cold' (mesurees les moins "
                 f"couteuses) -> format HTP-natif immediatement inferieur "
                 f"({s['cold_demoted']} tenseurs degrades)")
        L.append(f"- {s['unchanged']} tenseurs inchanges (format de base de la "
                 f"famille), {s['skipped_no_family']} hors familles HTP "
                 f"(norm/embed/lm_head/other — geres par le format de base, "
                 f"pas de surcharge par couche ici)")
    if result["out_path"]:
        L.append(f"- fichier ecrit : `{result['out_path']}` "
                 f"({len(result['lines'])} lignes)")
        L.append("")
        L.append("Commande llama-quantize recommandee (base = format le plus "
                 "represente du plan, cf. table ci-dessus ; le --tensor-type-file "
                 "ne fait que surcharger les couches hot/cold identifiees) :")
        base_type = plan["alloc"].get("attn", "Q4_0")
        L.append(f"```\nllama-quantize --tensor-type-file {result['out_path']} "
                 f"model.gguf model-optimized.gguf {base_type}\n```")
    L.append("")
    L.append("[LIMITE] Cette differenciation par couche s'appuie UNIQUEMENT sur le "
             "cout mesure PAR COUCHE (temps cumule des kernels de cette couche) — "
             "PAS sur un cout d'arete/transition inter-couche (repack/layout/sync), "
             "qui reste hors de portee sans nouvelle instrumentation C++ device "
             "(cf. limite deja documentee dans render_per_layer_table). C'est "
             "'ce qui est possible' avec les donnees actuelles, pas le modele "
             "complet de cout par tenseur avec aretes vise a terme.")
    L.append("")
    L.append("[ANGLE MORT CONFIRME 2026-09-06] L'hypothese \"format HTP-natif = "
             "jamais de fallback CPU\" n'est validee dans nos rapports QUE pour "
             "attn/MLP denses (RAPPORT_GEMMA_QUANT_MIXTE_20260903.md). Aucun rapport "
             "existant ne confirme ni n'infirme ce comportement pour la famille "
             "'moe' (ffn_*_exps) — cette generation de fichier applique la meme "
             "regle a moe par extrapolation, PAS par verification directe. A "
             "revalider avant un premier deploiement sur un modele MoE (Gemma-4-26B "
             "notamment). Par ailleurs, norm/embed/lm_head restent hors de cette "
             "surcharge par couche par construction (HTP_OFFLOAD_FAMILIES) car les "
             "rapports montrent qu'ils tombent en CPU/F32 independamment du format "
             "cible choisi (role/shape, pas quant) — corrige ici en les excluant "
             "plutot qu'en pretendant les optimiser.")
    return "\n".join(L)


def _run_one(path, args):
    """Profile un seul modele (gguf/safetensors/HF) — factorise pour --scan.
    Reprend les chemins de chargement de l'original profile_model.py::_run_one
    (HF headers-only, --shards override) qui n'existaient pas dans une
    premiere version de ce fichier (corrige suite a l'audit 2026-09-06 :
    "est-ce qu'il utilise tous les parametres de mon dossier de profilage ?")."""
    shards = []
    if getattr(args, "hf", None):
        print(f"[load] HF {args.hf} (headers only)...")
        cfg, tensors, shards = pm.load_hf(args.hf)
        model_name = args.hf.split("/")[-1]
        is_gguf = False
        meta = None
    elif path and path.lower().endswith(".gguf"):
        meta, tlist = pm.read_gguf_header(path)
        tensors = {}
        for name, dims, ttype, nb in tlist:
            ne = 1
            for d in dims:
                ne *= d
            tensors[name] = {"dtype": pm.V_TYPES.get(ttype, "?"), "shape": dims,
                             "bytes": nb, "elems": ne, "ttype": ttype}
        cfg = {}
        model_name = meta.get("general.name") or os.path.basename(path)
        is_gguf = True
    elif path and os.path.isdir(path):
        cfg, tensors, shards = pm.load_safetensors_local(
            path, args.shards.split(",") if getattr(args, "shards", None) else None)
        model_name = cfg.get("_name_or_path", os.path.basename(path.rstrip("/\\")))
        meta = None
        is_gguf = False
    else:
        sys.exit(f"chemin invalide : {path!r} (attendu : .gguf, dossier "
                 "safetensors, ou --hf REPO)")

    a = pm.analyze(cfg, tensors, gguf_meta=meta if is_gguf else None,
                   shard_names=shards, is_gguf=is_gguf, ctx=args.ctx)

    base_report = pm.render(a, model_name, args.budget_gb, args.ctx)
    plan = quant_plan_safe(a, args.budget_gb, backend=args.backend,
                            safety_margin_gb=args.safety_margin_gb)
    out_chunks = [base_report, render_orchestration_vs_l3_note(a),
                 render_safe_plan(plan, a)]

    if args.bounds:
        bounds = simulate_l3_bounds(a, ctx=args.ctx, contention=args.contention)
        out_chunks.append(render_bounds_section(bounds, measured_tps=args.measured_tps))

    regime = apply_device_regime(a, device_meta_path=args.device_meta)
    out_chunks.append(render_device_regime_section(regime))

    live = None
    if args.live_trace:
        events = pm.load_live_trace(args.live_trace)
        live = pm.aggregate_live_trace(events)
        out_chunks.append(render_per_layer_table(live))
        # Comparaison originale live-vs-L3 + plan adaptatif (existants dans
        # profile_model.py, jamais repris ici avant cette correction) :
        cmp_ = pm.compare_live_vs_l3(a, live, measured_tps_wall=args.measured_tps)
        out_chunks.append(pm.render_live_section(live, cmp_))
        adapt = pm.quant_plan_adaptive(a, args.budget_gb, live=live,
                                       max_spill_freq=args.max_spill_freq)
        out_chunks.append(f"\n**Plan de quant adaptatif** ({adapt['adaptive_note']}) : "
                          f"{adapt['alloc']} — total {adapt['total_gib']:.1f} GiB")

    if args.emit_tensor_type_file:
        rows = build_tensor_rows(tensors, is_gguf=is_gguf)
        ttf = generate_tensor_type_file(rows, plan, live=live,
                                        n_layer=a["arch"]["n_layer"],
                                        top_fraction=args.hot_fraction,
                                        out_path=args.emit_tensor_type_file)
        out_chunks.append(render_tensor_type_section(ttf, plan))

    if args.layer_quant_impact:
        rows = build_tensor_rows(tensors, is_gguf=is_gguf)
        impact = predict_layer_quant_impact(rows, a, args.budget_gb, live=live,
                                            safety_margin_gb=args.safety_margin_gb)
        out_chunks.append(render_layer_quant_impact(impact))

    report = "\n".join(out_chunks)
    print(report)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        md = os.path.join(args.out, f"PROFILE_{model_name.replace('/', '_')}.md")
        with open(md, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"[out] {md}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="fichier .gguf OU dossier safetensors")
    ap.add_argument("--hf", help="repo HuggingFace (headers only, ex google/gemma-4-26B-A4B) "
                    "— repris de l'original, absent de la premiere version de ce fichier")
    ap.add_argument("--shards", help="liste de shards safetensors separes par des "
                    "virgules (override) — repris de l'original")
    ap.add_argument("--scan", metavar="DIR", help="scanne DIR : profile TOUS les "
                    ".gguf + dossiers safetensors trouves — repris de l'original")
    ap.add_argument("--out", help="dossier de sortie (rapport .md par modele) — "
                    "repris de l'original")
    ap.add_argument("--device-config", help="JSON de constantes device calibrees "
                    "(voir --write-default-device-config) — repris de l'original, "
                    "SANS CA la calibration SM8850 codee en dur est TOUJOURS utilisee, "
                    "impossible a surcharger sans modifier profile_model.py directement")
    ap.add_argument("--write-default-device-config", metavar="PATH",
                    help="ecrit les constantes SM8850 par defaut dans PATH et quitte")
    ap.add_argument("--budget-gb", type=float, default=pm.DEFAULT_RAM_BUDGET_GB)
    ap.add_argument("--safety-margin-gb", type=float, default=1.5,
                    help="marge soustraite du budget avant allocation (defaut 1.5 Go)")
    ap.add_argument("--backend", choices=["HTP", "OpenCL-MoE", "none"], default="HTP")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--bounds", action="store_true",
                    help="ajoute la section Hextimate-like lower/upper bound")
    ap.add_argument("--contention", type=float, default=None,
                    help="facteur de contention memoire (ex. 0.75 pour co-execution "
                         "GPU+NPU mesuree) applique aux bornes --bounds")
    ap.add_argument("--device-meta", metavar="PATH",
                    help="fichier meta device (meme format que predictor."
                         "parse_meta) — remplace la constante thermique fixe du 'soutenu' "
                         "par le regime REEL detecte par le gouverneur")
    ap.add_argument("--measured-tps", type=float, default=None,
                    help="t/s wall mesure, compare a la plage --bounds ET a la "
                         "comparaison live-vs-L3 de l'original")
    ap.add_argument("--max-spill-freq", type=float, default=0.15,
                    help="seuil spill_freq pour le plan de quant adaptatif original "
                         "(quant_plan_adaptive) — repris de l'original")
    ap.add_argument("--live-trace", help="JSONL d'evenements mesures : active la "
                    "table par couche (LAYER_COST) + la comparaison live-vs-L3 originale")
    ap.add_argument("--emit-tensor-type-file", metavar="PATH",
                    help="genere un --tensor-type-file llama-quantize (surcharge "
                         "par couche hot/cold, necessite --live-trace pour etre "
                         "non-vide) pret a l'emploi")
    ap.add_argument("--hot-fraction", type=float, default=0.25,
                    help="fraction des couches (mesurees) traitees comme "
                         "hot/cold pour --emit-tensor-type-file (defaut 0.25)")
    ap.add_argument("--layer-quant-impact", action="store_true",
                    help="predit une allocation de quant PAR COUCHE sous budget "
                         "partage (impact explicite sur les autres couches) + "
                         "qualite-proxy. NE predit PAS le debit (cf. section "
                         "dediee dans la sortie) — RAM et qualite-proxy uniquement")
    args = ap.parse_args()

    if args.write_default_device_config:
        with open(args.write_default_device_config, "w", encoding="utf-8") as f:
            json.dump({"device_name": "SM8850", **pm.DEVICE_DEFAULTS}, f, indent=2)
        print(f"[device-config] gabarit ecrit : {args.write_default_device_config}")
        return

    if args.device_config:
        applied = pm.load_device_profile(args.device_config)
        print(f"[device-config] {pm.DEVICE_NAME} charge : {applied}")

    if args.scan:
        found = []
        for dirpath, dirnames, filenames in os.walk(args.scan):
            dirnames[:] = [d for d in dirnames if d not in
                          ("node_modules", ".git", "__pycache__", "_toolchains")]
            for fn in filenames:
                if fn.lower().endswith(".gguf"):
                    found.append(os.path.join(dirpath, fn))
                elif fn == "config.json" and any(
                        x.endswith(".safetensors") or x.endswith("_head.bin")
                        for x in filenames):
                    found.append(dirpath)
        uniq = sorted(set(os.path.normcase(f) for f in found))
        print(f"[scan] {len(uniq)} modele(s) trouve(s) dans {args.scan}")
        ok = 0
        for m in uniq:
            try:
                print(f"\n===== {m} =====")
                _run_one(m, args)
                ok += 1
            except Exception as e:
                print(f"[scan] ECHEC {m}: {e}")
        print(f"\n[scan] TERMINE : {ok}/{len(uniq)} profiles")
        return

    if not args.path and not args.hf:
        sys.exit("usage : profiler.py <modele.gguf|dossier> [--hf REPO] [--scan DIR] ...")
    _run_one(args.path, args)


if __name__ == "__main__":
    main()
