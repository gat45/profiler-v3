# profiler_v3 — profilage NPU/LLM OnePlus 15 (SM8850, Hexagon HTP v81)

Version consolidée du profiler (créée 2026-09-06, mise à jour en continu le
même jour). Remplace la dispersion `snapdragon_profiling/` (~150 scripts,
beaucoup morts ou redondants) + les copies `bench_results/corrections/*_corrige.py`
par un jeu de fichiers cohérent — **ceci EST la version corrigée**, pas une
n-ième copie. `snapdragon_profiling/` original reste intact, rien n'y a été
supprimé ni modifié.

## Arborescence

```
profiler_v3/
├── README.md                      # ce fichier — vue d'ensemble, découvertes, usage
├── FONCTIONS.md                   # detail de CHAQUE fonction, organise par usage
│                                   # (profiler / predire HF / mesurer temps reel)
├── profile_model.py                # ORIGINAL (snapdragon_profiling/) — ne pas modifier
├── profiler.py                     # point d'entree principal (CLI)
├── predictor.py                    # regime device reel + baselines mesurees
├── predict_from_hf.py              # predire un modele HF SANS le telecharger
├── capability_db.py                # formats HTP, incidents de crash, tradeoff RAM/qualite/debit
├── adaptive_lever.py                # recommandation de leviers runtime (GGML_HEXAGON_OPPOLL...)
├── parse_hexagon_profile.py         # trace reelle NPU/HTP (GGML_HEXAGON_PROFILE)
├── parse_opencl_profile.py          # trace reelle GPU (OpenCL) + CPU (eval_callback)
├── parse_backend_copy_profile.py    # ping-pong memoire CPU<->GPU<->NPU (ggml-backend.cpp patch)
├── live_monitor.py                  # monitoring temps reel (pendant que le modele tourne)
├── android_telemetry.py             # lecture Android/kernel directe (RAM, GPU par-process)
├── telemetry_dashboard.py           # affichage PC live de la telemetrie (via adb forward)
├── predictor_v1_outcomes.jsonl      # historique predictions vs mesures reelles
└── device_logs/                     # captures reelles + traces deja parsees
    ├── .gitignore                   # exclut les logs bruts >2 Mo (gardes en local)
    ├── PROFILE_CPU_Qwen17B.md       # rapport genere : profil CPU reel Qwen3-1.7B
    ├── PROFILE_GPU_Qwen17B.md       # rapport genere : profil GPU reel Qwen3-1.7B
    ├── PROFILE_PINGPONG_Qwen17B.md  # rapport genere : ping-pong CPU/GPU mesure
    ├── cl_profiling_qwen17b.csv      # trace brute GPU (cl_profiling.csv pulled)
    ├── cpu_profiling_qwen17b.csv     # trace brute CPU (cpu_profiling.csv pulled)
    ├── backend_copy_qwen17b_ngl10.csv# trace brute ping-pong (backend_copy_profile.csv pulled)
    └── *.jsonl / *.txt               # traces live deja parsees, prompts de test

android/hw_monitor/                  # app Android compagnon (Kotlin, Gradle)
├── app/src/main/java/com/geniex/hwmonitor/
│   ├── HwMonitor.kt                 # etat GLOBAL (CPU/GPU/NPU/RAM/batt), via su/root
│   ├── ProcessMemoryMonitor.kt      # RAM + memoire GPU PAR PROCESSUS (direct + fallback su)
│   ├── TelemetryServer.kt           # serveur HTTP local (127.0.0.1:8082), JSON, /run distant
│   └── MainActivity.kt              # UI + demarrage auto du serveur au lancement
└── app/src/main/res/layout/activity_main.xml
```

## Vue d'ensemble du pipeline

```
                     ┌─────────────────────┐
   HF repo (remote)  │ predict_from_hf.py   │  ← lit juste l'en-tête GGUF
   ────────────────► │ (sans téléchargement)│    (HTTP Range, quelques Mo)
                     └──────────┬───────────┘
                                │ appelle
                                ▼
   GGUF local         ┌──────────────────┐        ┌──────────────────┐
   ─────────────────► │  profile_model.py │◄──────►│   profiler.py     │
   (analyse RAM/L3)   │  (original,       │ étend  │  (point d'entrée  │
                       │   ne pas modifier)│        │   principal)      │
                       └──────────────────┘        └────────┬─────────┘
                                                             │ consulte
                                              ┌──────────────┼──────────────┐
                                              ▼              ▼              ▼
                                    predictor.py   capability_db.py   parse_hexagon_
                                    (régime device,  (risque crash,    profile.py
                                     baselines        formats HTP,     (trace réelle
                                     mesurées)        tradeoff RAM/    par couche,
                                                       qualité/débit)  device réel)
```

## Fichiers

| Fichier | Rôle | Statut |
|---|---|---|
| `profile_model.py` | Original, copié tel quel (`snapdragon_profiling/profile_model.py`) — lecture GGUF, analyse par famille, plan de quant glouton, simulation L3/roofline | Dépendance, **ne jamais modifier** |
| `profiler.py` | Corrections + ajouts du 2026-09-06 (compat quant↔backend, marge anti-OOM, bornes lower/upper, régime device réel, plan de quant par couche avec budget, warnings collapse qualité) | **Point d'entrée principal** (CLI) |
| `predictor.py` | Régime device réel (thermique/doze/mémoire/charge/idle-throttle), baselines par modèle avec distinction mesuré/théorique, log d'outcomes pour calibration future | Gouverneur, complémentaire à `profiler.py` |
| `capability_db.py` | Registre des formats de quant supportés par HTP (avec ref exacte fichier:ligne), codes de statut HTP, incidents de crash catalogués (avec `blast_radius`), calculateur de tradeoff RAM/qualité/débit avec verdict automatique | Base de connaissance, alimentée par incidents réels |
| `parse_hexagon_profile.py` | Parse la trace réelle DSP par op/par couche (HVX/HMX/DMA), reconstruction des compteurs de cycles 64-bit tronqués | Nouveau, validé sur device réel (56% de mapping trace-evt→couche) |
| `predict_from_hf.py` | Pipeline complet "prédire sans télécharger" : en-tête distant → analyse RAM/L3 → correction routing MoE / dense → risque de crash → plan de quant par couche | Nouveau, testé de bout en bout |
| `adaptive_lever.py` | Recommandation de leviers runtime (`GGML_HEXAGON_OPPOLL`, etc.) selon la classe de goulot détectée | Corrections mineures |
| `live_monitor.py` | Monitoring VRAIMENT temps réel — parse les ops pendant que le modèle tourne encore sur le device (pas après coup) | Nouveau, testé device réel |
| `parse_opencl_profile.py` | Parseur générique GPU (OpenCL) + CPU (`ggml_backend_sched_eval_callback`) — profil par couche, RAM (RSS), débit (octets/op), timeline | Nouveau, testé device réel (rebuild requis) |
| `parse_backend_copy_profile.py` | Le "ping-pong" mémoire CPU↔GPU↔NPU — copies inter-backend réelles (patch `ggml-backend.cpp`) | Nouveau, testé device réel (rebuild requis) |
| `android_telemetry.py` | Lecture Android/kernel directe SANS ggml ni root : RAM globale, RSS par PID, mémoire GPU par processus (`dumpsys gpu`), charge GPU (`kgsl gpubusy`) | Nouveau, testé device réel, sans rebuild |
| `device_logs/` | Logs de capture réels + traces JSONL déjà parsées (exemples de référence) | Données, pas du code |
| `predictor_v1_outcomes.jsonl` | Journal des prédictions vs mesures réelles (12 entrées au 2026-09-06) — sert à décider quand passer à un modèle ML (`ready_for_v2_ml`) | Données de calibration |

## Ce qui a changé vs l'original (`snapdragon_profiling/profile_model.py`)

1. **Compatibilité quant↔backend** — évite le fallback CPU silencieux (HTP
   n'accélère que F16/F32/Q8_0/Q4_0/Q4_1/IQ4_NL/MXFP4 ; OpenCL-MoE n'accepte
   QUE Q4_0 pour les experts, confirmé par `GGML_ASSERT(0)` systématique sur
   tout autre format).
2. **Marge de sécurité RAM anti-OOM** — dégrade la précision plutôt que
   risquer un OOM (consigne explicite : précision perdue < OOM). Deux
   incidents réels (`INCIDENT_REBOOT_MTP_GPU_NPU_CONCURRENT`, OOM-kill sweep
   Gemma) ont montré que viser exactement le budget fourni est insuffisant.
3. **Bornes lower/upper (Hextimate-like)** au lieu d'un point unique — permet
   un test de cohérence direct (`--bounds --measured-tps X`) : le t/s réel
   mesuré doit tomber entre les deux, sinon un terme du modèle est faux.
4. **Décomposition BW_EFFECTIVE** en facteurs séparés (efficacité kernel,
   contention GPU+NPU, layout) au lieu d'une seule constante calibrée en bloc
   — permet d'isoler la cause d'un écart mesuré.
5. **Régime device réel** (`--device-meta`, via `predictor.py`) remplace la
   constante thermique fixe (`x0.65`) par le facteur réellement détecté.
6. **Génération de `--tensor-type-file`** pour `llama-quantize`, avec
   différenciation par couche hot/cold quand une trace live est disponible.
7. **Plan de quant par couche avec budget** (`--layer-quant-impact`) —
   allocation gloutonne (Q4_0 plancher, upgrade des couches au meilleur ratio
   qualité/Go d'abord), avec impact documenté sur les autres couches.
8. **`parse_hexagon_profile.py`** (le plus important) — ferme le gap
   identifié le 2026-09-06 : aucun producteur réel de coût par couche
   n'existait nulle part dans le projet (confirmé après audit complet). La
   solution n'était PAS d'écrire une nouvelle instrumentation C++ : elle
   existait déjà, compilée dans le runtime déployé
   (`ggml_hexagon_dump_op_prof`/`ggml_hexagon_dump_batch_prof`), juste jamais
   activée (`GGML_HEXAGON_PROFILE=1/3`) ni parsée.
9. **`capability_db.py`** (nouveau) — catalogue les incidents de crash réels
   avec sévérité (`blast_radius`), pour distinguer un crash process contenu
   (llama-cli plante, device intact) d'un vrai risque de reboot.
10. **`predict_from_hf.py`** (nouveau) — étend tout ce qui précède à un modèle
    jamais téléchargé, en lisant uniquement l'en-tête GGUF distant.

## Découvertes majeures du 2026-09-06

### 1. Coût de routing MoE — réel et massif, absent du modèle L3 original

Mesuré ×3.1 à ×3.4 selon la paire comparée, entre Marco-Nano (232 experts/
top-8, 18.4-20.8 t/s) et Qwen3-0.9B-A0.6B (2 experts/top-1, 56.5-69.0 t/s),
alors que le modèle L3 original prédit 42.5 t/s **identique** pour les deux
(son terme de coût MoE ne dépend que de `hidden`, jamais de `n_experts`/
`top_k` — `MMID_N_OPS_DECODE=1` fixé en dur). Calibré sur 4 points réels
(voir `ROUTING_CALIBRATION` dans `predict_from_hf.py`) :

| n_experts | top_k | t/s mesuré | facteur vs L3 | n_runs |
|---:|---:|---:|---:|---:|
| 2 | 1 | 62.6 | ×1.47 | 1 |
| 3 | 1 | 54.1 | ×1.27 | 1 |
| 64 | 8 | 62.2 (moy) | ×0.87 | 3 (59.4/62.7/64.4) |
| 232 | 8 | 19.9 | ×0.47 | 1 |

### 2. Calibration dense — même écart, mais direction inverse

Contrairement au routing MoE (facteur très variable, 0.47-1.47), les modèles
denses testés montrent un facteur **stable autour de ×1.3** (le modèle L3
original SOUS-estime systématiquement les modèles denses testés) :

| Modèle | bytes/token (proxy taille) | t/s mesuré | t/s L3 froid | facteur |
|---|---:|---:|---:|---:|
| Qwen3-1.7B-Q4_0 | 1.472 Go | 38.2 | 28.6 | ×1.336 |
| Qwen3-4B-Q4_0 | 2.935 Go | 20.0 | 15.3 | ×1.307 |

Seulement 2 points, proches l'un de l'autre — extrapolation hors de
`[1.47, 2.94]` Go/token marquée explicitement `extrapolated_unreliable` dans
`predict_dense_factor()`. Intégré dans `predict_from_hf.py::predict_no_download`
(remplace le facteur neutre `1.0` qui était appliqué à tout modèle dense
jusqu'à cette correction).

### 3. Overhead de dispatch : prefill vs decode, facteur ×5

Mesuré sur Marco-Nano-V2 via `parse_hexagon_profile.py::render_batch_overhead_report`
: **2.3%** d'overhead en prefill (long prompt, traité par batch) contre
**11.6%** en decode (token par token) — le coût de dispatch fixe par batch
pèse proportionnellement bien plus quand chaque batch ne traite qu'un seul
token.

### 4. Format ne prédit pas le débit de façon monotone

Q8_0→Q4_0 a fait **régresser** le débit de 8% malgré -17% sur `MUL_MAT`
(memory-bound, pas compute-bound — moins d'octets ≠ toujours plus rapide côté
HTP) ; Q4_0→MXFP4 a donné +13% à taille quasi identique. Voir
`RAPPORT_DECISIF_ABC_Q8_Q4_MXFP4_MEMORYWALL_20260831.md`. Conséquence directe
dans `capability_db.py::compare_format_tradeoff()` : le verdict de rentabilité
se base sur 3 mesures réelles (RAM, PPL, débit), jamais sur la taille de
fichier seule.

### 5. Reconstruction des compteurs de cycles 64-bit (`CycleUnwrapper`)

Le format `trace-evt` (HVX/HMX/DMA) tronque les cycles sur 32 bits alors que
le compteur réel est 64-bit. Un premier essai de correspondance naïve
(bisection directe entre cycles OPBATCH ~6.5e11 et cycles trace-evt ~1e9) a
été **infirmé empiriquement** — ce sont bien le même compteur sous-jacent,
mais pas comparables sans dé-troncature. `CycleUnwrapper` (identique à
l'implémentation officielle upstream `llama.cpp/scripts/snapdragon/
ggml-hexagon-profile.py`) reseed à chaque OPBATCH et permet un mapping
trace→couche réel, validé à **56% de succès** (264182/471892 events) sur une
trace réelle Qwen3-0.9B.

## Risque de crash — `capability_db.py`

6 incidents catalogués (formats, backends, ops), chacun avec un
`blast_radius` explicite :

- **`process`** (5/6 incidents) — llama-cli/server plante proprement,
  device intact. Ex : `-ot`→OpenCL → SIGSEGV (faux pointeur host après
  fallback CPU) ; formats non supportés → `GGML_ASSERT(0)` ou
  `AEE_EFAILED` (voir `ggml-hexagon/htp/entry.c:2230-2239`).
- **`none`** — pas de crash, juste dégradation silencieuse (ex : collapse
  qualité Q2_K dense → EOS immédiat, mais le process tourne normalement).
- **`device_reboot`** (1/6, le seul) — MTP sur 2 backends simultanés
  (GPUOpenCL+HTP0, Qwen3.5-9B-D2 hybrid Mamba) → redémarrage complet du
  device. Root-causé dans `INCIDENT_REBOOT_MTP_GPU_NPU_CONCURRENT_20260906.md`,
  et **corroboré indépendamment par la recherche web** (issue GitHub
  ggml-org/llama.cpp#19617 : fragilité multi-session HTP connue en amont,
  pas spécifique à ce fork). `capability_db.py` documente la mitigation
  concrète (ne jamais lancer 2 backends actifs simultanément sur le même
  modèle hybride Mamba).

Conclusion pratique : étendre `capability_db.py` à d'autres combinaisons
format/op/backend est **globalement à faible risque** (crash process contenu
dans 5 cas sur 6 catalogués), le vrai risque de reboot étant isolé à un
scénario précis et déjà évité par construction.

## Usage complet (workflow réel, testé de bout en bout)

```bash
# 1. Capturer une vraie trace sur device (instrumentation deja presente,
#    juste jamais activee avant le 2026-09-06) :
adb shell "cd /data/local/tmp/rt_clean/bin && \
    LD_LIBRARY_PATH=/data/local/tmp/rt_clean/bin GGML_HEXAGON_PROFILE=1 \
    GGML_HEXAGON_OPPOLL=1 ./llama-cli --single-turn -m ../models/X.gguf \
    -ngl 99 -lv 5 -p 'prompt' -n 64 > /data/local/tmp/prof.log 2>&1"
adb pull /data/local/tmp/prof.log device_logs/

# 2. Parser vers une trace live exploitable :
python3 parse_hexagon_profile.py device_logs/prof.log --out device_logs/trace.jsonl

# 3. Profiler avec la VRAIE trace (table par couche + overrides GGUF) :
python3 profiler.py model.gguf --budget-gb 9.0 --backend HTP --bounds \
    --live-trace device_logs/trace.jsonl \
    --emit-tensor-type-file overrides.txt --measured-tps <t/s reel>

# 4. Plan de quant par couche avec budget explicite :
python3 profiler.py model.gguf --budget-gb 9.0 --layer-quant-impact

# 5. Prédire un modèle HuggingFace SANS le télécharger :
python3 predict_from_hf.py mradermacher/Some-Model-GGUF          # liste les fichiers
python3 predict_from_hf.py mradermacher/Some-Model-GGUF --file Some-Model.Q4_0.gguf
```

## Couverture des paramètres de l'original (vérifiée 2026-09-06)

Une première version de `profiler.py` n'exposait pas tous les flags de
`profile_model.py` original — corrigé après vérification explicite. Repris
depuis l'original : `--hf` (profil HF headers-only), `--shards`, `--scan`
(dossier entier), `--out` (écriture .md), `--device-config` /
`--write-default-device-config` (calibration device surchargeable — sans ça,
la calibration SM8850 codée en dur était la SEULE utilisable), `--max-spill-freq`
et la comparaison `compare_live_vs_l3`/`quant_plan_adaptive` originale (qui a
sa propre suggestion de recalibration — testée, elle retrouve indépendamment
le même facteur ×1.47 que la mesure directe du jour sur Qwen3-0.9B).

## Bugs réels trouvés et corrigés dans ce dossier (2026-09-06)

1. **`layer_gib()`** (`profiler.py`) — division erronée par 2.0 en trop
   (BPW déjà en octets/poids). Détecté par comparaison avec `a['by_family']`.
2. **`predict_layer_quant_impact`** — lookup par clé string vs int dans
   `per_layer` (JSON désérialisé en clés string), retournait silencieusement
   0.0% pour toutes les couches.
3. **`DOZE_CAPPED` faux positif** (`predictor.py`) — condition `OR` confondait
   l'idle CPU normal (`caps_ratio` bas, `wakefulness="Awake"`) avec un vrai
   Doze Android. Détecté avec un vrai fichier de méta device (prédisait 17.2
   t/s vs 62-69 t/s réellement mesurés). Corrigé en séparant `DOZE_CAPPED`
   (nécessite `wake != "Awake"`) et un nouveau régime `CPU_IDLE_THROTTLE`
   (aucune pénalité).
4. **Code mort** dans `render_tensor_type_section` — `max()` sur une clé
   booléenne, jamais utilisé.
5. **Import inutilisé** `sys` dans `parse_hexagon_profile.py` (trouvé via
   pyflakes).
6. **`predict_routing_factor`** — points de calibration aux limites
   mal étiquetés `extrapolated_unreliable` à cause de comparaisons `<=`/`>=`
   au lieu d'un test d'égalité exacte en premier.
7. **Collision de nom** `import profiler as p` dans `predict_from_hf.py` —
   masquait silencieusement le paramètre `p` utilisé dans toutes les
   fonctions de rendu. Renommé en `pv3`.
8. **Mapping trace-evt→batch** — première tentative (bisection naïve entre
   deux domaines de cycles non comparables) abandonnée puis résolue par
   `CycleUnwrapper` (voir Découverte #5 ci-dessus).

## Ce qui reste une simulation, pas une mesure (honnêteté explicite)

- Le coût d'arête **précis par paire de couches** (couche N → N+1 : repack,
  layout, synchronisation) — `parse_hexagon_profile.py` mesure un overhead de
  dispatch **par batch entier** (une bonne approximation, ~1 batch ≈ 1
  couche), pas encore une attribution exacte à une transition N→N+1
  spécifique. Mapping trace-evt→couche à 56% de succès, pas 100%.
- Le "soutenu (thermique)" reste la constante fixe de l'original tant que
  `--device-meta` n'est pas fourni avec un vrai log device.
- L'hypothèse "HTP-natif = jamais de fallback CPU" n'est validée que pour
  attn/MLP denses, pas pour la famille MoE (extrapolée, pas vérifiée).
- La calibration dense (`DENSE_CALIBRATION`) ne repose que sur 2 points très
  proches (1.47 et 2.94 Go/token) — fiable dans cette plage, extrapolation
  hors plage explicitement marquée peu fiable.
- Le test batch/ubatch (default=48.1, ub512=59.7, ub32=58.3 t/s sur
  Qwen3-0.9B) est **inconclusif** — dans la variance run-à-run déjà connue du
  modèle (~20%, 56.5-69.0 t/s vu ailleurs à config identique), pas un effet
  isolé propre.

## Baselines mesurées (`predictor.py::BASELINES`)

| Baseline | t/s | Plage (lo-hi) | n_runs | Statut |
|---|---:|---|---:|---|
| `marco_nano_v2` | 19.9 | 18.4-20.8 | 3 | mesuré |
| `qwen3_0_9b_a0_6b` | 62.6 | 56.5-69.0 | 3 | mesuré |
| `huihui_moe_1_2b` | 54.1 | 50.7-57.1 | 3 | mesuré |
| `euromoe_2_6b_a0_6b` | 62.2 | 59.4-64.4 | 3 | mesuré |
| `marco_nano` | 29.3 | — | — | **stale, needs_reverification** |
| `gemma_mixed3` / `gemma_mixed` / `d2a` | — | — | — | non retesté n≥3 (historique d'incidents documenté : OOM-kill, spike thermique, reboot MTP — pas de consigne explicite pour retester malgré le risque) |
| Marco-Nano V1 Q4_0 | 23.1 (moy) | 20.8-26.0 | 3 | mesuré 2026-09-06, [26.0, 22.6, 20.8] — **pas encore écrit dans une table/rapport dédié** |

`predictor_v1_outcomes.jsonl` : 12 prédictions vs mesures loguées,
`mae=1.892`, `mape=3.49%`, `coverage=1.0`, `ready_for_v2_ml=False` (pas encore
assez de données pour justifier un modèle ML au lieu de règles explicites).

## Modèles disponibles pour test (état réel du device au 2026-09-06)

- Sur device (`/data/local/tmp/`) : Qwen3-1.7B-Q4_0, Qwen3-4B-Q4_0,
  `gemma-4-26B-A4B.Q2_K.gguf`, `mxq/gemma_mixed3.gguf`, plus un historique
  massif de logs de bench déjà réalisés sur Gemma (`bench_out/gemma_*`,
  cold/warm/oppoll/nsp/cdsp/swap/purge).
- **Pas de safetensors Gemma réels** sur E:\ ou D:\ — seulement du code
  source llama.cpp (fichiers de conversion), des rapports d'estimation
  théorique (`estimate_gemma4_26b.py`), et des GGUF de vocabulaire seuls
  (`ggml-vocab-gemma-4.gguf`, pour tester le tokenizer, pas le modèle).
- `D:\saftensore 9b\` — nom trompeur : contient en réalité
  **Qwen3.5-9B-heretic-v2** (safetensors complets, 4 shards), pas Gemma.
  C'est le modèle "d2a" impliqué dans l'incident de reboot MTP.

## Découverte externe intégrée (autre session, même jour)

`bench_results/BILAN_MEMPOOL_DSPQUEUE_FINAL_20260906.md` — campagne parallèle
ayant corrigé de vrais bugs dans `ggml-hexagon` (fork `ab-wt`) et mesuré (à
froid, preuves md5-ancrées) : **mempool +21-38%** sur pp16/tg32/tg128 vs
dspqueue, **dspqueue +10.6%** sur pp512. Un crash réel root-causé (`-ot`
vers OpenCL → SIGSEGV, faux pointeurs host) ajouté à `capability_db.py`.
**Vérifié** (md5sum réel, lecture seule) : `rt_clean` (utilisé pour toutes les
mesures de cette conversation) est un **troisième binaire**, ni mempool ni
dspqueue de cette campagne — nos résultats du jour (routing MoE, dense,
HTP/GPU/CPU, MTP) ne sont donc pas directement comparables ligne à ligne à
cette table sans le vérifier séparément.

## Environnement d'exécution (note pratique)

Tout ce dossier a été développé et testé via WSL (Ubuntu) pour l'accès `adb`
et l'exécution des scripts Python. Le 2026-09-06, WSL est tombé en panne au
niveau service (`WSLService` bloqué en `START_PENDING`/`NOT_STOPPABLE`,
non réparable sans droits admin) — la calibration dense a été complétée en
contournant WSL via un `adb.exe` Windows natif
(`E:\oneplus\geniex_harness\tools\platform-tools\adb.exe`) et un interpréteur
Python natif Windows, prouvant que rien dans ce projet ne dépend réellement
de WSL en soi (juste de `adb` + `python3`, disponibles nativement sous
Windows).

## Fichiers volontairement non repris de `snapdragon_profiling/`

Tout `governor/`, `apk_profiler/` (sauf ce qui a été audité et jugé sans
valeur ajoutée réelle — cf.
`AUDIT_COMPLET_PROFILAGE_PAR_COUCHE_INEXISTANT_20260906.md`), et la
quarantaine de scripts `tools/*.sh`/`tools/*.py` — la plupart sont soit des
bench ad hoc jetables, soit des doublons de fonctionnalités maintenant
couvertes par `profiler.py`. Rien n'est supprimé de `snapdragon_profiling/`
(dossier original intact), ce dossier est un nouveau départ propre, pas un
remplacement destructif.

## Relation avec `snapdragon-d2-planner` (GitHub, projet antérieur de l'utilisateur)

`snapdragon-d2-planner` (GaTmaNnes/snapdragon-d2-planner) est un projet
**antérieur** de l'utilisateur : architecture en couches (profiler → predictor
→ scheduler) similaire dans l'esprit, mais ses constantes (simulateur
Hextimate, modèle de dispatch, modèle thermique) sont **dérivées de la
littérature/documentation publique, jamais mesurées sur device réel**
(confiance auto-déclarée "70% Phase 2 theory"). `profiler_v3` reprend l'idée
d'architecture en couches mais la fonde sur des mesures réelles day-to-day
(routing MoE, dense, HTP/GPU/CPU, PPL, crash risk) qui dépassent déjà son
niveau de validation empirique. Une correction de bug mineure a été apportée
localement dans la copie extraite (`core/analysis/__init__.py`, mauvais
import de `compute_layer_profiles_from_safetensors`), mais ce projet n'est
**pas** la priorité de développement actuelle — il ne fait pas partie de
`profiler_v3` et n'est pas maintenu ici.
