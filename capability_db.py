#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""capability_db.py — schema + base de capacite quant x operateur x backend
(2026-09-06). Cree suite a la discussion sur le "Layer/Quantization Capability
Profiler" (architecture profileur -> features -> predicteur -> scheduler).

STATUT EXPLICITE DEMANDE PAR L'UTILISATEUR : "cree tout mais active pas tous".
Ce module est CREE (schema complet, entrees reelles) mais INERTE — rien dans
profiler.py/parse_hexagon_profile.py ne le consulte automatiquement pour
changer un plan de quant ou un placement. C'est une base de reference
consultable a la demande (import + query), pas un moteur de decision actif.

Pourquoi inerte par defaut : cartographier vraiment la matrice quant x op x
shape x backend necessite de pousser des combinaisons JUSQU'AU CRASH sur
device (verifie 2026-09-06 : HTP_STATUS_NO_SUPPORT/VTCM_TOO_SMALL font avorter
TOUT le batch, ggml-hexagon/htp/entry.c:2230-2239 — pas un fallback silencieux
observable passivement). C'est un chantier de test device deliberement risque,
distinct du profilage passif deja construit — a activer seulement sur decision
explicite, jamais par defaut.

Discipline de source (recommandee par le red-team de la discussion, appliquee
ici) : chaque entree porte un champ `source` parmi :
  - "verified_in_code"  : confirme en lisant le source C++ du fork (cite avec
                          chemin:ligne).
  - "observed_trace"    : confirme dans une trace device reelle capturee.
  - "incident"          : confirme par un echec/comportement reel reproduit
                          aujourd'hui (ex. Q2_K -> EOS immediat).
  - "hypothesis"        : plausible mais NON verifie — ne jamais traiter comme
                          un fait tant que non confirme.
Ne jamais promouvoir une entree "hypothesis" en fait sans verification directe.
"""

# ---------------------------------------------------------------------------
# 1. Registre des formats de quantification — 3 niveaux distincts (demande
# explicite : ne pas confondre type logique / type physique HTP / type
# calcule par le kernel).
# ---------------------------------------------------------------------------
QUANT_FORMATS = {
    "Q4_0": {
        "logical": "Q4_0", "bitwidth": 4.5,  # cf. pm.BPW["Q4_0"]=18/32 octets/poids
        "physical_htp": "Q4_0_TILED",
        "physical_htp_source": "verified_in_code",
        "tile_qk": 256,  # 32x32 tiled layout, confirme
        "verified_ref": "htp/htp-ops.h:31,40 (HTP_TYPE_Q4_0_TILED, QK_Q4_0_TILED=256)",
    },
    "Q4_1": {"logical": "Q4_1", "bitwidth": 5.0, "physical_htp": None,
             "physical_htp_source": "hypothesis", "tile_qk": None},
    "Q8_0": {
        "logical": "Q8_0", "bitwidth": 8.5,
        "physical_htp": "Q8_0_TILED",
        "physical_htp_source": "verified_in_code",
        # QK_Q8_0_TILED confirme textuellement (2026-09-06) :
        "tile_qk": 128,
        "verified_ref": "ggml-hexagon-fastrpc.cpp:3031,3126 (QK_Q8_0_TILED=128)",
    },
    "MXFP4": {"logical": "MXFP4", "bitwidth": None, "physical_htp": "MXFP4_TILED",
              "physical_htp_source": "verified_in_code", "tile_qk": 256,
              "verified_ref": "htp/htp-ops.h:34,42 (HTP_TYPE_MXFP4_TILED, QK_MXFP4_TILED=256)"},
    "F16": {"logical": "F16", "bitwidth": 16.0, "physical_htp": None,
            "physical_htp_source": "n/a (deja natif, pas de repack)", "tile_qk": None},
    "F32": {"logical": "F32", "bitwidth": 32.0, "physical_htp": None,
            "physical_htp_source": "n/a", "tile_qk": None},
    "IQ4_NL": {"logical": "IQ4_NL", "bitwidth": None, "physical_htp": None,
               "physical_htp_source": "hypothesis (dit reutiliser layout Q4_0 "
                                      "dans l'analyse relayee, non verifie ici)",
               "tile_qk": None},
}

# ---------------------------------------------------------------------------
# 2. Statuts HTP reels (confirmes dans le code, PAS une supposition) —
# ggml-hexagon/htp/entry.c:2230-2239, ggml-hexagon.cpp:130-134.
# ---------------------------------------------------------------------------
HTP_STATUS_CODES = {
    "HTP_STATUS_OK": {"meaning": "op executee normalement",
                      "source": "verified_in_code"},
    "HTP_STATUS_NO_SUPPORT": {"meaning": "operateur/format non supporte par ce "
                              "kernel HTP — FAIT AVORTER TOUT LE BATCH "
                              "(pas un fallback silencieux)",
                              "source": "verified_in_code",
                              "ref": "ggml-hexagon/htp/entry.c:2233,2239"},
    "HTP_STATUS_INVAL_PARAMS": {"meaning": "parametres invalides pour cette op "
                                "(shape/stride incompatible)",
                                "source": "verified_in_code",
                                "ref": "ggml-hexagon/htp/entry.c:2234"},
    "HTP_STATUS_VTCM_TOO_SMALL": {"meaning": "VTCM insuffisant pour cette op a "
                                  "cette geometrie — FAIT AVORTER TOUT LE BATCH",
                                  "source": "verified_in_code",
                                  "ref": "ggml-hexagon/htp/entry.c:2235,2239 ; "
                                         "aussi binary-ops.c:772,779,800, "
                                         "argsort-ops.c:471, concat-ops.c:260"},
    "HTP_STATUS_INTERNAL_ERR": {"meaning": "erreur interne generique",
                               "source": "verified_in_code"},
}

# ---------------------------------------------------------------------------
# 3. Base de capacite empirique — SEULEMENT les incidents reellement vecus
# aujourd'hui (2026-09-06), aucune entree theorique/devinee. C'est le debut
# concret de la "EMPIRICAL_CAPABILITY_DB" proposee dans la discussion, pas la
# matrice complete (qui necessiterait des tests deliberes jusqu'au crash,
# explicitement NON lances ici).
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# BLAST_RADIUS — champ ajoute 2026-09-06 suite a la demande explicite de
# limiter/neutraliser le risque de crash. Distinction cruciale trouvee en
# relisant le code + verifiee par recherche web (issue upstream #19617,
# multi-session HTP = zone de fragilite connue, PAS specifique a notre fork) :
#   "process"       : le PROCESS llama-cli/server plante ou s'arrete
#                      proprement (SIGSEGV, AEE_EFAILED, GGML_ASSERT) — le
#                      DEVICE ne redemarre pas, aucune perte au-dela de ce
#                      seul process. Risque FAIBLE : verifier device libre
#                      avant/apres suffit, testable deliberement sans
#                      precaution particuliere.
#   "device_reboot" : le DEVICE ENTIER redemarre (uptime remis a zero) —
#                      confirme UNE SEULE FOIS a ce jour (MTP actif sur les
#                      DEUX backends simultanement + modele hybride Mamba/GDN
#                      deja fragile, cf. INCIDENT_REBOOT_MTP_GPU_NPU_
#                      CONCURRENT_20260906.md). Risque ELEVE : ne jamais
#                      retenter sans reduction de charge (-c/-n plus petits,
#                      surveillance RAM en continu) et accord explicite.
# 4 des 5 incidents ci-dessous sont "process" (deja verifie en relisant
# chaque rapport source) — SEUL le combo MTP+double-backend est
# "device_reboot". La plupart du "risque de crash" catalogue ici est donc
# CONTENU, pas catastrophique — a ne pas confondre.
# ---------------------------------------------------------------------------
CAPABILITY_ENTRIES = [
    {
        "quant": "Q2_K", "op": "MUL_MAT (MoE experts)", "backend": "OpenCL",
        "shape_constraint": "n'importe quelle geometrie testee",
        "status": "UNSUPPORTED", "source": "incident", "blast_radius": "process",
        "note": ("kernel_gemm_moe_q4_0_q8_1_dp4a n'accepte QUE Q4_0 pur pour "
                "les experts MoE — GGML_ASSERT(0)/status -38 systematique sur "
                "Q2_K, quel que soit le decoupage --override-tensor tente"),
        "ref": "TEST_GEMMA_NGL_EXPERTS_GPU_20260906.md",
        "date": "2026-09-06",
    },
    {
        "quant": "Q4_0 (partiel, reste MoE en Q2_K)", "op": "MUL_MAT (MoE experts)",
        "backend": "OpenCL", "shape_constraint": "melange de types dans la meme couche MoE",
        "status": "UNSUPPORTED", "source": "incident", "blast_radius": "process",
        "note": ("le noyau MoE OpenCL exige TOUS les tenseurs de la couche "
                "(routing+gate_up+down) dans le MEME format Q4_0 — un melange "
                "partiel echoue aussi, pas seulement le format non-Q4_0 pur"),
        "ref": "TEST_GEMMA_NGL_EXPERTS_GPU_20260906.md",
        "date": "2026-09-06",
    },
    {
        "quant": "Q2_K", "op": "graphe complet (dense, sans imatrix)", "backend": "HTP",
        "shape_constraint": "Qwen3.8-9B dense, aucune calibration imatrix",
        "status": "SUPPORTED_BUT_DEGENERATE", "source": "incident",
        "blast_radius": "none",  # pas de crash du tout, juste sortie inutilisable
        "note": ("charge normalement (pas de crash), mais genere un EOS "
                "immediat (0 token utile) — collapse qualitatif, pas un "
                "probleme de compatibilite backend. Fichier supprime apres ce "
                "test (n'a jamais rien produit d'exploitable)"),
        "ref": "conversation 2026-09-06 (supprime : Qwen3.8-9B-Q2_K.gguf)",
        "date": "2026-09-06",
    },
    {
        "quant": "Q4_0", "op": "graphe MTP complet (draft-mtp, spec-draft-n-max 1)",
        "backend": "HTP", "shape_constraint": "Qwen3.8-9B-Cyber-Exploit-Agent-v3, "
        "couches MTP (mtp_*) fusionnees dans le meme graphe que le modele principal",
        "status": "SUPPORTED_WITH_WARNING", "source": "incident",
        "blast_radius": "none",  # aucun crash, juste diagnostic + re-allocation
        "note": ("le bug gallocr #28448 (deja prouve formellement par test "
                "synthetique, tests/test-alloc.cpp::test_adversarial_topology_"
                "change) se manifeste REELLEMENT ici : des dizaines de "
                "'DIAGNOSTIC plan/identity mismatch' par token sur les noeuds "
                "mtp_*. Pas de crash (le runtime detecte et re-alloue, ec=0 sur "
                "tous les splits), sortie coherente produite (+21% t/s avec MTP, "
                "9.3->11.3), mais c'est un signal de fragilite reel, pas "
                "seulement theorique — le graphe MTP change l'identite des "
                "tenseurs a position fixe d'une facon qui declenche ce bug connu"),
        "ref": "conversation 2026-09-06 (test MTP reel sur device)",
        "date": "2026-09-06",
    },
    {
        "quant": "n/a (poids overrides via -ot)", "op": "toute op sur tenseur "
        "override vers OpenCL", "backend": "mempool (variante ggml-hexagon)",
        "shape_constraint": "n'importe quelle couche (blk.6, blk.35 identiques) — "
        "pas dependant de la couche",
        "status": "UNSUPPORTED", "source": "incident", "blast_radius": "process",
        "note": ("SIGSEGV rc=139 systematique (6/6 runs) sur toute "
                "'-ot \"blk.N=OpenCL\"' avec -dev HTP0 sur la voie mempool. "
                "Root cause (lecture de code, pas juste observation) : "
                "tensor_buft_overrides route le poids vers le buffer OpenCL "
                "au chargement, qui utilise un FAUX pointeur host (data = "
                "base+offset, vraies donnees via cl_mem/extra) ; hexagon "
                "rejette l'op (supports_op) -> fallback CPU -> dereference le "
                "faux pointeur -> SIGSEGV. La voie dspqueue n'a pas ce bug "
                "(elle co-ordonne OpenCL directement, ses ops -ot restent sur "
                "OpenCL). Fix identifie mais NON committe a ce jour."),
        "ref": "BILAN_MEMPOOL_DSPQUEUE_FINAL_20260906.md §1.4 (autre session), "
              "RAPPORT_OT_PP512_MEMPOOL_BLOQUE_20260906.md",
        "date": "2026-09-06",
    },
    {
        "quant": "Q4_0", "op": "MTP actif sur 2 backends simultanes (GPUOpenCL + "
        "HTP0, 2x llama-server independants, mmap partage)",
        "backend": "GPUOpenCL+HTP0 concurrent", "shape_constraint": "Qwen3.5-9B-D2-A "
        "(hybride GATED_DELTA_NET/SSM), --spec-type draft-mtp --spec-draft-n-max 1 "
        "sur LES DEUX serveurs en meme temps",
        "status": "UNSUPPORTED", "source": "incident", "blast_radius": "device_reboot",
        "note": ("SEUL incident CONFIRME provoquant un reboot COMPLET du device "
                "(uptime 15964s -> 56s) parmi tous les incidents catalogues ici — "
                "tous les autres sont blast_radius=process ou none. Root cause "
                "(diagnostic, pas juste observation) : (1) MTP par-session ajoute "
                "un etat/buffers de verification NON partages par le mmap des "
                "poids -> double la pression memoire malgre poids partages ; "
                "(2) modele hybride Mamba/GDN deja identifie fragile sous file "
                "d'ops-in-flight HTP saturee (RAPPORT_9B_D2_SINGLE_HTP0_"
                "20260831.md) ; (3) confirme par recherche web (2026-09-06, "
                "issue upstream ggml-org/llama.cpp#19617) : les sessions HTP "
                "multiples simultanees sont une zone de fragilite CONNUE du "
                "projet en amont, pas specifique a notre fork. GPU+NPU SANS MTP "
                "= valide et sur (teste 2x). MTP SEUL (1 backend) = valide et "
                "sur. C'est la COMBINAISON des 3 facteurs qui casse, pas chacun "
                "isolement."),
        "mitigation": ("Ne JAMAIS retenter MTP+double-backend simultane sur ce "
                      "modele sans reduction de charge (-c 512 au lieu de 2048, "
                      "-n 8, surveillance RAM en continu, tuer si MemAvailable "
                      "< 2 Go) ET accord explicite prealable. Si donnee "
                      "necessaire : tester d'abord sur un modele MoE petit/"
                      "dense (Qwen3-0.9B, Huihui, EuroMoE — architecture non-"
                      "Mamba, deja testee individuellement sans probleme) avant "
                      "de reapprocher un modele hybride connu fragile."),
        "ref": "INCIDENT_REBOOT_MTP_GPU_NPU_CONCURRENT_20260906.md",
        "date": "2026-09-06",
    },
]

# ---------------------------------------------------------------------------
# 6. Choix de build mempool vs dspqueue — mesure REELLE de l'autre session du
# jour (BILAN_MEMPOOL_DSPQUEUE_FINAL_20260906.md, ancree a des preuves md5/
# fichier:ligne, PAS une simple affirmation). A froid, caps stables :
# ---------------------------------------------------------------------------
BUILD_VARIANT_COMPARISON = [
    {"test": "pp16", "mempool": 250.09, "mempool_std": 0.14,
     "dspqueue": 188.07, "dspqueue_std": 5.50, "winner": "mempool", "delta_pct": 33},
    {"test": "tg32", "mempool": 23.39, "mempool_std": 0.24,
     "dspqueue": 19.32, "dspqueue_std": 0.80, "winner": "mempool", "delta_pct": 21},
    {"test": "tg128", "mempool": 23.34, "mempool_std": 0.04,
     "dspqueue": 16.89, "dspqueue_std": 1.19, "winner": "mempool", "delta_pct": 38},
    {"test": "pp512", "mempool": 1205.07, "mempool_std": 17.09,
     "dspqueue": 1332.56, "dspqueue_std": 1.56, "winner": "dspqueue", "delta_pct": 10.6},
]
# Recommandation de l'autre session (coherente avec les chiffres) : mempool
# pour l'usage interactif (latence, insensible thermique <1%), dspqueue pour
# le prefill long batche.
# VERIFIE 2026-09-06 (md5sum reel sur device, lecture seule, pas de nouveau
# test) : /data/local/tmp/rt_clean/bin/libggml-hexagon.so = 7f9080d3... —
# NE CORRESPOND A AUCUN des deux md5 de cette table (mempool 09a6b2e5,
# dspqueue 8cfa206a). rt_clean est un TROISIEME binaire (notre propre build
# d'integration du jour, base self-build-jz, anterieur aux fixs bug3/bug4 de
# l'autre session) — TOUTES les mesures de cette conversation (Marco-Nano,
# Qwen3-0.9B, Huihui, routing MoE x3.1, comparaison HTP/GPU/CPU) viennent de
# ce troisieme binaire, pas directement comparables ligne a ligne a cette
# table mempool/dspqueue sans le confirmer separement.

# ---------------------------------------------------------------------------
# 5. Comparaison multi-backend REELLE (nouveau, 2026-09-06) — ferme partiellement
# le gap "aucune comparaison HTP/GPU/CPU". Mesures directes, meme modele, meme
# prompt, sur ce device (SM8850, Adreno 840). PAS une matrice complete (1
# modele, 1 taille) mais une premiere donnee reelle, remplace la case vide.
# ---------------------------------------------------------------------------
BACKEND_COMPARISON = [
    {"model": "Qwen3-0.9B-A0.6B (Q4_0, MoE 2 experts/top-1)",
     "backend": "HTP0", "tps": 62.6, "n_runs": 3,
     "note": "moyenne sur 3 runs (56.5-69.0)"},
    {"model": "Qwen3-0.9B-A0.6B (Q4_0, MoE 2 experts/top-1)",
     "backend": "CPU", "tps": 60.7, "n_runs": 1,
     "note": "quasi identique au HTP pour ce petit modele"},
    {"model": "Qwen3-0.9B-A0.6B (Q4_0, MoE 2 experts/top-1)",
     "backend": "GPUOpenCL (Adreno 840)", "tps": 26.25, "n_runs": 1,
     "note": "LE PLUS LENT des 3 sur ce modele — contre-intuitif, probablement "
            "overhead de dispatch OpenCL dominant sur un si petit graphe"},
]

# Perplexite REELLE (llama-perplexity, CPU-only, corpus court ~223 tokens,
# ctx=64) — remplace le "proxy qualite ordinal" par une vraie mesure pour au
# moins un modele. Echantillon petit (haute variance +/-1.4-1.6) : indicatif,
# pas une PPL de reference publiable, mais REEL, pas invente.
PERPLEXITY_MEASURED = [
    {"model": "Marco-Nano-Instruct", "format": "Q4_0", "ppl": 5.7080,
     "ppl_stderr": 1.55022, "corpus": "223 tokens, texte prose generaliste",
     "date": "2026-09-06", "file_gib": 4.57, "tps_htp": 26.0},
    {"model": "Marco-Nano-Instruct", "format": "Q8_0", "ppl": 5.0694,
     "ppl_stderr": 1.36120, "corpus": "223 tokens, texte prose generaliste",
     "date": "2026-09-06", "file_gib": 8.53, "tps_htp": 23.4},
]
# Q8_0 mesure ~11% meilleur (PPL plus bas = meilleur) que Q4_0 sur ce corpus —
# coherent avec l'attente theorique (plus de bits = moins de perte), mais
# maintenant CHIFFRE, pas suppose. file_gib et tps_htp AJOUTES 2026-09-06
# (meme modele, meme binaire rt_clean, meme prompt) pour permettre le calcul
# COMPLET cout/benefice ci-dessous (RAM x qualite x debit), pas juste un axe
# isole — exactement ce qui manquait avant : "Q8 est 11% meilleur" ne dit rien
# sans dire aussi combien ca coute en RAM et en debit.


def compare_format_tradeoff(model_name, fmt_a, fmt_b):
    """Calcule le VRAI compromis cout/benefice entre deux formats du MEME
    modele, a partir de mesures reelles deja enregistrees (PERPLEXITY_MEASURED
    et/ou BACKEND_COMPARISON) — jamais une estimation theorique. Retourne None
    si les deux formats n'ont pas ete mesures pour ce modele (pas de valeur
    inventee pour combler un trou)."""
    entries = {e["format"]: e for e in PERPLEXITY_MEASURED if e["model"] == model_name}
    a, b = entries.get(fmt_a), entries.get(fmt_b)
    if a is None or b is None:
        missing = [f for f in (fmt_a, fmt_b) if f not in entries]
        return {"error": f"pas de mesure reelle pour {model_name}/{missing} — "
                         "aucun calcul possible sans donnee (pas d'estimation)"}

    ram_delta_pct = (b["file_gib"] / a["file_gib"] - 1.0) * 100
    ppl_delta_pct = (b["ppl"] / a["ppl"] - 1.0) * 100  # negatif = b meilleur (PPL plus bas)
    tps_delta_pct = None
    if a.get("tps_htp") and b.get("tps_htp"):
        tps_delta_pct = (b["tps_htp"] / a["tps_htp"] - 1.0) * 100

    # Verdict automatique — PAS juste 3 chiffres a interpreter a l'oeil.
    # Heuristique explicite (pas une IA de decision) : le gain qualite (valeur
    # absolue de ppl_delta_pct) doit depasser le cout RAM ET ne pas degrader
    # le debit de plus de 15% pour etre juge "rentable". Seuils arbitraires
    # mais EXPLICITES et modifiables — pas caches dans un score composite opaque.
    quality_gain = -ppl_delta_pct  # positif = mieux
    verdict = "RENTABLE"
    reasons = []
    if ram_delta_pct > quality_gain * 3:
        verdict = "PAS RENTABLE"
        reasons.append(f"cout RAM ({ram_delta_pct:+.0f}%) largement disproportionne "
                       f"vs gain qualite ({quality_gain:+.0f}%)")
    if tps_delta_pct is not None and tps_delta_pct < -15:
        verdict = "PAS RENTABLE"
        reasons.append(f"perte de debit ({tps_delta_pct:+.0f}%) trop importante")
    if not reasons:
        reasons.append(f"gain qualite ({quality_gain:+.0f}%) justifie le cout "
                       f"RAM ({ram_delta_pct:+.0f}%) et debit "
                       f"({tps_delta_pct:+.0f}% si mesure)")

    return {
        "model": model_name, "from": fmt_a, "to": fmt_b,
        "ram_gib": {"from": a["file_gib"], "to": b["file_gib"],
                   "delta_pct": round(ram_delta_pct, 1)},
        "ppl": {"from": a["ppl"], "to": b["ppl"], "delta_pct": round(ppl_delta_pct, 1)},
        "tps_htp": ({"from": a["tps_htp"], "to": b["tps_htp"],
                     "delta_pct": round(tps_delta_pct, 1)}
                    if tps_delta_pct is not None else "non mesure pour l'un des deux"),
        "verdict": verdict, "verdict_reasons": reasons,
    }


def render_format_tradeoff(t):
    if "error" in t:
        return f"[tradeoff] {t['error']}"
    L = [f"## Compromis reel {t['from']} -> {t['to']} ({t['model']})", ""]
    L.append(f"- RAM : {t['ram_gib']['from']:.2f} -> {t['ram_gib']['to']:.2f} GiB "
             f"({t['ram_gib']['delta_pct']:+.1f}%)")
    L.append(f"- Qualite (PPL, plus bas = mieux) : {t['ppl']['from']:.3f} -> "
             f"{t['ppl']['to']:.3f} ({t['ppl']['delta_pct']:+.1f}%)")
    if isinstance(t["tps_htp"], dict):
        L.append(f"- Debit HTP mesure : {t['tps_htp']['from']:.1f} -> "
                 f"{t['tps_htp']['to']:.1f} t/s ({t['tps_htp']['delta_pct']:+.1f}%)")
    else:
        L.append(f"- Debit : {t['tps_htp']}")
    L.append("")
    L.append(f"**VERDICT : {t['verdict']}**")
    for r in t["verdict_reasons"]:
        L.append(f"  - {r}")
    return "\n".join(L)


def query_capability(quant=None, op=None, backend=None):
    """Recherche dans CAPABILITY_ENTRIES (3 entrees a ce jour — ne PAS
    presenter un resultat vide comme "SUPPORTED implicite", toujours dire
    explicitement qu'aucune donnee n'existe pour cette combinaison."""
    hits = [e for e in CAPABILITY_ENTRIES
            if (quant is None or quant.lower() in e["quant"].lower())
            and (op is None or op.lower() in e["op"].lower())
            and (backend is None or backend.lower() == e["backend"].lower())]
    return hits


# ---------------------------------------------------------------------------
# 4. Prediction de risque de crash — PAS de nouveau test device, seulement
# une combinaison de signaux DEJA connus (incidents reels + registre de
# formats). Repond a la demande explicite "vois comment predire les crash"
# SANS provoquer de nouveau crash : c'est un heuristique statique, pas un
# oracle — un score "aucun signal connu" ne veut jamais dire "garanti sur".
# ---------------------------------------------------------------------------
def predict_crash_risk(quant, op, backend, moe_mixed_format=False, uses_mtp_graph=False):
    """Retourne {"risk": "HIGH"|"MEDIUM"|"UNKNOWN"|"NO_KNOWN_SIGNAL",
    "reasons": [...]}, base UNIQUEMENT sur des signaux deja verifies :
    1. correspondance exacte dans CAPABILITY_ENTRIES (incident reel deja vecu)
    2. format absent du registre QUANT_FORMATS ou physical_htp non confirme
       (verified_in_code) -> zone grise jamais testee
    3. regle specifique MoE format-mixte-dans-la-meme-couche (incident reel :
       OpenCL exige un format UNIQUE pour toute la couche MoE)
    4. regle specifique graphe MTP fusionne (incident reel : declenche le bug
       gallocr #28448 connu, pas un crash mais un signal de fragilite)
    """
    RANK = {"NO_KNOWN_SIGNAL": 0, "UNKNOWN": 1, "MEDIUM": 2, "HIGH": 3}
    reasons = []
    risk = "NO_KNOWN_SIGNAL"

    def _bump(new_risk):
        nonlocal risk
        if RANK[new_risk] > RANK[risk]:
            risk = new_risk

    hits = query_capability(quant=quant, op=op, backend=backend)
    max_blast_radius = "none"
    blast_rank = {"none": 0, "process": 1, "device_reboot": 2}
    for h in hits:
        br = h.get("blast_radius", "unknown")
        if blast_rank.get(br, 1) > blast_rank.get(max_blast_radius, 0):
            max_blast_radius = br
        if h["status"] in ("UNSUPPORTED",):
            _bump("HIGH")
            reasons.append(f"incident reel exact : {h['status']} "
                          f"[blast_radius={br}] ({h['ref']})")
        elif h["status"] in ("SUPPORTED_BUT_DEGENERATE", "SUPPORTED_WITH_WARNING"):
            _bump("MEDIUM")
            reasons.append(f"incident reel exact (pas un crash mais degrade) : "
                          f"{h['status']} [blast_radius={br}] ({h['ref']})")

    if moe_mixed_format:
        _bump("HIGH")
        reasons.append("format MoE non-uniforme dans la meme couche — incident "
                       "reel confirme (noyau OpenCL exige un format UNIQUE pour "
                       "toute la couche routing+gate_up+down)")

    if uses_mtp_graph:
        _bump("MEDIUM")
        reasons.append("graphe MTP fusionne — declenche le bug gallocr #28448 "
                       "connu (diagnostic plan/identity mismatch a chaque token, "
                       "pas un crash mais un signal de fragilite reel confirme)")

    fmt_entry = QUANT_FORMATS.get(quant)
    if fmt_entry is None:
        _bump("UNKNOWN")
        reasons.append(f"format '{quant}' absent du registre QUANT_FORMATS — "
                       "zone grise, aucune donnee (verifiee ou empirique)")
    elif fmt_entry.get("physical_htp_source") == "hypothesis":
        reasons.append(f"representation physique HTP de '{quant}' non confirmee "
                       "dans le code (hypothesis, pas verified_in_code) — risque "
                       "de compatibilite non evalue")

    if not reasons:
        reasons.append("aucun signal connu pour cette combinaison — absence de "
                       "signal != garantie de fonctionnement, juste absence de "
                       "donnee dans une base qui ne contient que 4 incidents a ce jour")

    return {"quant": quant, "op": op, "backend": backend, "risk": risk,
            "reasons": reasons, "max_blast_radius": max_blast_radius}


def render_crash_risk(result):
    br = result.get("max_blast_radius", "none")
    br_note = {
        "none": "aucun crash connu",
        "process": "CONTENU — le process plante/s'arrete, le device ne redemarre pas",
        "device_reboot": "**ELEVE — peut provoquer un REDEMARRAGE COMPLET du device**",
    }.get(br, br)
    L = [f"[crash_risk] {result['quant']} / {result['op']} / {result['backend']} "
        f"-> RISQUE {result['risk']} (portee : {br_note})"]
    for r in result["reasons"]:
        L.append(f"  - {r}")
    return "\n".join(L)


def format_capability_report(hits, query_desc=""):
    if not hits:
        return (f"[capability_db] AUCUNE donnee pour {query_desc or 'cette requete'} "
                "— absence de donnee != SUPPORTED. Cette base ne contient que "
                f"{len(CAPABILITY_ENTRIES)} incidents reels a ce jour, pas une "
                "matrice complete. Ne jamais deduire un support par defaut.")
    lines = [f"[capability_db] {len(hits)} entree(s) trouvee(s) :"]
    for e in hits:
        lines.append(f"  - {e['quant']} / {e['op']} / {e['backend']} -> "
                     f"{e['status']} (source: {e['source']}, {e['date']})")
        lines.append(f"    {e['note']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    quant = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"Registre formats : {len(QUANT_FORMATS)} formats connus")
    print(f"Statuts HTP verifies dans le code : {len(HTP_STATUS_CODES)}")
    print(f"Base de capacite empirique : {len(CAPABILITY_ENTRIES)} incidents reels")
    print()
    hits = query_capability(quant=quant)
    print(format_capability_report(hits, quant or "toutes entrees"))
