# profiler_v3 — détail de toutes les fonctions, organisées par usage

Ce document liste **toutes les fonctions réelles du code** (pas un résumé
marketing), organisées en 3 usages tels que demandés :

1. **Profiler un modèle qu'on a déjà** (fichier local ou sur device)
2. **Prédire le profilage d'un modèle HuggingFace SANS le télécharger**
3. **Mesurer et comprendre ce qui se passe sur le téléphone en temps réel**

Pour chaque fonction : signature, ce qu'elle fait réellement, mesuré vs
simulé quand c'est pertinent.

---

## 1. PROFILER UN MODÈLE (fichier local .gguf ou safetensors déjà en main)

### `profile_model.py` — moteur de base (original, ne pas modifier)

Lecture / chargement :
- **`read_gguf_header(path)`** — ouvre un fichier local et lit l'en-tête GGUF
  (métadonnées + liste des tenseurs), sans lire les poids.
- **`_read_gguf_header_inner(f)`** — parseur bas niveau du format GGUF
  (magic/version/n_tensors/n_kv, puis chaque paire clé-valeur et chaque
  descripteur de tenseur). Utilisée à la fois en local (`read_gguf_header`) et
  à distance (`predict_from_hf.py::fetch_gguf_header_remote`) — même code, la
  seule différence est la source du buffer (fichier disque vs réponse HTTP).
- **`parse_safetensors_header(data)`** — parse l'en-tête JSON d'un fichier
  `.safetensors` (offset 8 octets + JSON de longueur variable).
- **`load_safetensors_local(directory, shards_override=None)`** — charge
  `config.json` + tous les shards `.safetensors` d'un dossier local.
- **`load_hf(repo_id)`** — charge un repo HuggingFace **format transformers**
  (attend un `config.json` + des safetensors) ; échoue en 404 sur un repo GGUF
  pur type `mradermacher/*` (c'est pour ça que `predict_from_hf.py` existe à
  côté, avec sa propre lecture directe du `.gguf`).
- **`_http_get(url, binary=True)`** — GET HTTP générique utilisé par `load_hf`.

Classification des tenseurs :
- **`ggml_type_size(t)`** — bits/poids pour un type ggml donné (table
  `TYPE_SIZES`).
- **`family_of(name)`** — classe un nom de tenseur en famille (`attn`, `mlp`,
  `embed`, `norm`, ...) par correspondance de motifs (`FAMILY_RULES`).
- **`classify(name)`** — classification plus fine incluant le numéro de
  couche (regex `blk\.(\d+)\.`) et le rôle exact du tenseur.

Analyse et simulation (le cœur du profilage) :
- **`analyze(cfg, tensors, gguf_meta=None, shard_names=None, is_gguf=False, ctx=4096)`**
  — fonction centrale. À partir de la config (HF ou GGUF) et de la liste des
  tenseurs : détermine l'architecture (hidden_size, n_layers, n_experts,
  top_k, ...), calcule la RAM par famille (`by_family`), et appelle
  `simulate_l3()` pour la prédiction de débit. Retourne le dict `a` utilisé
  par TOUT le reste du pipeline (`a["arch"]`, `a["l3"]`, `a["by_family"]`).
- **`simulate_l3(a, ctx)`** — modèle "L3" (roofline par couche) : pour chaque
  couche, calcule poids Q4 streamés depuis la DDR + KV cache + activations
  (avec `spill_fill_factor` si elles dépassent le VTCM) + coût MoE + coût de
  dispatch (calibré sur 29 dispatchs FastRPC/token mesurés sur Marco). Retourne
  `cold_tps` (débit "à froid", sans throttle thermique) et `sustained_tps`
  (avec le facteur thermique — fixe par défaut, réel si `--device-meta`).
  **C'est une SIMULATION**, calibrée sur un point de mesure historique
  (Marco), pas une mesure directe du modèle en cours d'analyse.
- **`spill_fill_factor(working_set_mb)`** — pénalité (×1.0 à ×2.0+) quand les
  activations dépassent la capacité du scratchpad VTCM et doivent déborder en
  DDR (équation empirique dite "datavorous", non officielle Qualcomm).
- **`load_device_profile(path)`** — charge un JSON de constantes device
  (bande passante, latences, VTCM) et écrase les valeurs par défaut SM8850
  codées en dur — permet de recalibrer pour un autre chipset.

Plan de quantization :
- **`quant_plan(a, budget_gb)`** — plan de base : choisit le format le plus
  petit par famille qui tient dans le budget RAM donné. Ne connaît PAS la
  compatibilité backend (peut choisir un format qui tombe silencieusement en
  CPU) — c'est ce que `profiler.py::quant_plan_safe` corrige.
- **`quant_plan_adaptive(a, budget_gb, live=None, max_spill_freq=0.15)`** —
  comme `quant_plan`, mais dégrade en plus la famille la plus grosse si une
  trace live montre trop de fréquence de spill mesurée.

Trace live (déjà existant dans l'original, avant même `parse_hexagon_profile.py`) :
- **`load_live_trace(path)`** — charge un fichier JSONL d'événements déjà
  parsés (le format produit par `parse_hexagon_profile.py --live-trace` ou
  par `adaptive_lever.py`).
- **`aggregate_live_trace(events)`** — agrège les événements par kernel et par
  couche (latence totale, p50/p95 via `_percentile`, pics VTCM).
- **`classify_bottleneck(live, cmp_=None)`** — classe le goulot dominant
  (compute/memory/dispatch/spill) à partir de la trace agrégée, selon une
  matrice modèle→goulot validée en gouvernance le 2026-09-03.
- **`render_moe_layers(live)`** — tableau par couche du coût `MUL_MAT_ID`
  (l'op de routing MoE) mesuré réellement.
- **`compare_live_vs_l3(a, live, measured_tps_wall=None)`** — compare le débit
  RÉEL mesuré au débit prédit par `simulate_l3`, et suggère un facteur de
  recalibration simple (ratio, pas un fit statistique — un seul point de
  mesure ne justifie pas mieux). C'est cette fonction qui a retrouvé
  indépendamment le facteur ×1.47 sur Qwen3-0.9B, confirmant la découverte du
  coût de routing MoE.

Rendu :
- **`render(a, model_name, budget_gb, ctx, out_path=None)`** — assemble le
  rapport Markdown complet (RAM par famille, débit L3, plan de quant, trace
  live si disponible), écrit sur disque si `out_path` est fourni.
- **`main()` / `_run_one(path, args)`** — CLI de l'original (`--hf`,
  `--shards`, `--scan`, `--out`, `--live-trace`, `--device-config`).

### `profiler.py` — point d'entrée principal (corrections + ajouts 2026-09-06)

Compatibilité backend et sécurité RAM :
- **`_best_htp_native_leq(fmt)`** — retrouve le format HTP-natif le plus
  proche (≤ en bits) du format demandé, pour ne jamais choisir un format qui
  tombe silencieusement en CPU.
- **`quant_plan_safe(a, budget_gb, backend="HTP", safety_margin_gb=1.5, ...)`**
  — remplace `quant_plan()` : soustrait la marge de sécurité AVANT
  allocation, ne choisit que des formats compatibles avec le backend cible,
  et ajoute un warning explicite si une famille tombe à ≤Q3_K (risque de
  collapse qualité, cf. incident Qwen3.8-9B).

Modèle de bornes (Hextimate-like) :
- **`bw_effective_decomposed(bw_physical, efficiency=None, contention=None, layout=None)`**
  — décompose la bande passante effective en 3 facteurs indépendants
  (efficacité kernel / contention GPU+NPU / layout mémoire), au lieu d'une
  seule constante calibrée en bloc.
- **`simulate_l3_bounds(a, ctx=4096, contention=None)`** — reconstruit une
  borne LOWER (chevauchement total — déjà ce que fait `simulate_l3` via
  `max(compute, mémoire)`) et une borne UPPER (aucun chevauchement, somme
  complète compute+mémoire+spill+dispatch+expert) à partir des mêmes lignes
  de calcul, sans recalibration. Le t/s réel doit tomber entre les deux.
- **`render_bounds_section(bounds, measured_tps=None)`** — affiche les deux
  bornes et, si `--measured-tps` est fourni, dit explicitement si la mesure
  réelle est dans la plage ou pas (signal de modèle faux si hors plage).

Régime device réel :
- **`apply_device_regime(a, device_meta_path=None, device_features=None)`** —
  remplace la constante thermique fixe (×0.65) par le facteur réel détecté
  via `predictor.detect_regime()`, à partir d'un fichier `--device-meta`.
- **`render_device_regime_section(r)`** — affiche le régime détecté (ou dit
  explicitement qu'aucun `--device-meta` n'a été fourni et que la constante
  fixe reste utilisée).

Rendu par couche et plan par couche avec budget :
- **`render_per_layer_table(live)`** — table `LAYER_COST` à partir de la
  trace live agrégée (jamais rendue en tableau dans l'original).
- **`render_orchestration_vs_l3_note(a)`** — avertissement explicite : le
  nombre brut d'ops/token (`orchestration_ms`, borne pathologique sans
  regroupement) et le débit L3 (calibré, à citer) NE se comparent PAS
  directement.
- **`render_safe_plan(plan, a)`** — rendu du plan de `quant_plan_safe`.
- **`build_tensor_rows(tensors, is_gguf)`** — reconstruit les lignes par
  tenseur (nom/famille/couche/format) que `pm.analyze()` calcule en interne
  mais ne renvoie pas.
- **`_layer_hot_cold(live, n_layer, top_fraction=0.25)`** — classe les
  couches en "hot"/"cold" à partir du `total_us` mesuré dans la trace live
  (les 25% les plus chaudes vs le reste).
- **`generate_tensor_type_file(rows, plan, live=None, n_layer=0, top_fraction=0.25, ...)`**
  — génère le fichier `--tensor-type-file` consommable par `llama-quantize`,
  avec une précision différenciée par couche hot/cold si une trace live est
  disponible (les couches chaudes gardent plus de précision).
- **`predict_layer_quant_impact(rows, a, budget_gb, live=None, safety_margin_gb=1.5)`**
  — **la fonction demandée explicitement par l'utilisateur** ("connaître à
  l'avance comment le modèle serait quantifié par couche, pourquoi, et la
  taille sur la RAM"). Allocation gloutonne partagée : démarre à Q4_0 (floor
  HTP-natif) sur toutes les couches, puis monte en précision les couches
  offrant le meilleur ratio qualité-proxy/Go jusqu'à épuisement du budget.
  Retourne les décisions par couche + les couches sacrifiées (impact
  documenté) + un proxy de qualité globale en bits.
- **`render_layer_quant_impact(result)`** / **`render_tensor_type_section(result, plan)`**
  — rendu Markdown des deux fonctions ci-dessus.

CLI :
- **`_run_one(path, args)`** — profile un seul modèle (gguf/safetensors/HF),
  factorisé pour être appelable en boucle par `--scan`.
- **`main()`** — CLI complète : `--hf`, `--shards`, `--scan`, `--out`,
  `--device-config`, `--write-default-device-config`, `--budget-gb`,
  `--safety-margin-gb`, `--backend`, `--ctx`, `--bounds`, `--contention`,
  `--device-meta`, `--measured-tps`, `--max-spill-freq`, `--live-trace`,
  `--emit-tensor-type-file`, `--hot-fraction`, `--layer-quant-impact`.

**Commande type** (profiler un modèle déjà téléchargé) :
```bash
python3 profiler.py mon_modele.gguf --budget-gb 9.0 --backend HTP \
    --bounds --layer-quant-impact --out rapport.md
```

---

## 2. PRÉDIRE LE PROFILAGE VIA HUGGINGFACE (sans télécharger le modèle)

### `predict_from_hf.py` — pipeline complet

Lecture distante (aucun poids téléchargé) :
- **`list_hf_gguf_files(repo_id)`** — liste les fichiers `.gguf` d'un repo HF
  via l'API `/api/models/{repo_id}` (metadata seule, pas de contenu).
- **`fetch_gguf_header_remote(repo_id, filename, max_header_bytes=8*1024*1024)`**
  — **la fonction clé** : une seule requête HTTP Range
  (`bytes=0-{max_header_bytes-1}`) sur l'URL `resolve/main/{filename}`, puis
  réutilise `pm._read_gguf_header_inner()` (le même parseur que le chemin
  local) sur les octets reçus. 8 Mo par défaut suffit largement (le plus gros
  en-tête vu à ce jour : ~200 Ko pour 775 tenseurs).

Correction empirique (le modèle L3 seul se trompe systématiquement) :
- **`predict_routing_factor(n_experts, top_k)`** — corrige le débit MoE.
  Calibré sur 4 points réels (`ROUTING_CALIBRATION`, 2 à 232 experts).
  Retourne `(facteur, confiance, note)` — confiance =
  `"measured_exact"` (point calibré exact) / `"interpolated"` (entre deux
  points, log-space) / `"extrapolated_unreliable"` (hors plage calibrée,
  explicitement signalé peu fiable).
- **`predict_dense_factor(bytes_per_token_gb)`** — équivalent pour les
  modèles denses (pas de MoE). Calibré sur 2 points réels
  (`DENSE_CALIBRATION` : Qwen3-1.7B et Qwen3-4B, facteur ≈×1.3 les deux).
  Même logique de confiance qu'au-dessus, plage fiable `[1.47, 2.94]`
  Go/token.

Pipeline complet :
- **`predict_no_download(repo_id, filename, budget_gb=9.0, safety_margin_gb=1.5)`**
  — enchaîne : en-tête distant → `pm.analyze()` (RAM + débit L3 brut) →
  choix automatique MoE vs dense → `predict_routing_factor` OU
  `predict_dense_factor` → débit corrigé → `cdb.predict_crash_risk()`
  (risque de crash connu) → `pv3.quant_plan_safe()` (plan de quant) →
  `pv3.predict_layer_quant_impact()` (plan par couche avec budget). Retourne
  un seul dict avec tout, prêt pour `render_prediction`.
- **`render_prediction(p)`** — assemble le rapport Markdown final :
  architecture, RAM par famille, débit L3, débit corrigé (avec la note de
  confiance), risque de crash, plan de quant, impact par couche, et un
  disclaimer explicite sur les limites (2-4 points de calibration seulement).
- **`main()`** — CLI (`python3 predict_from_hf.py <repo> [--file <nom.gguf>]`).

**Commandes types** :
```bash
# Lister les fichiers GGUF disponibles d'un repo :
python3 predict_from_hf.py mradermacher/Some-Model-GGUF

# Prédire un fichier précis SANS le télécharger :
python3 predict_from_hf.py mradermacher/Some-Model-GGUF --file Some-Model.Q4_0.gguf
```

**Ce que ça donne concrètement** (répond à la demande explicite "connaître à
l'avance comment le modèle serait quantifié par couche pourquoi, la taille
sur la RAM, les token/s") :
- Architecture détectée (n_layers, hidden_size, n_experts/top_k si MoE)
- RAM par famille de tenseurs (attn/mlp/embed/norm) au format quant choisi
- Débit t/s prédit, avec le facteur de correction appliqué et sa fiabilité
- Risque de crash connu pour cette combinaison format/backend
- Plan de quant par couche avec la RAISON de chaque choix (ratio
  qualité/Go) et les couches sacrifiées si le budget est serré

**Limite assumée** : la correction routing/dense est une INTERPOLATION entre
quelques points réels, pas une mesure du modèle demandé — toujours présentée
avec son niveau de confiance, jamais comme une certitude.

---

## 3. MESURER ET COMPRENDRE CE QUI SE PASSE SUR LE TÉLÉPHONE EN TEMPS RÉEL

Deux niveaux : (a) l'état système du device au moment du run (thermique,
CPU, doze, mémoire) via `predictor.py`, et (b) ce qui se passe DANS le
runtime d'inférence lui-même, op par op, via `parse_hexagon_profile.py`.

### (a) `predictor.py` — état système réel du device

- **`parse_meta(path)`** — lit un fichier `meta.txt` capturé sur le device
  (format clé=valeur : `model=`, `mWakefulness_before=`,
  `scaling_max_freq_before=P0/P6`, `mem_available_kb=`, `loadavg=`,
  `thermal_zones=...thermal_zone47/28/32:...` pour DDR/HVX/HMX). C'est la
  fonction qui transforme un snapshot device brut en features exploitables.
- **`detect_regime(f)`** — classe l'état du device en régime : `DOZE_CAPPED`
  (vrai Android Doze, `wakefulness != "Awake"`), `CPU_IDLE_THROTTLE`
  (idle CPU normal, caps bas mais Awake — PAS une pénalité, corrigé le
  2026-09-06 après un faux positif réel), `THERMAL_THROTTLE`, ou `NOMINAL`.
- **`estimate_tg(f, base)`** — applique le facteur du régime détecté à une
  baseline mesurée pour estimer le débit attendu dans les conditions
  actuelles du device.
- **`risks(f, base, regime)`** — liste les risques connus pour ce régime
  (ex : proximité d'un OOM si `mem_available_kb` est bas).
- **`recommend(regime, f, base)`** — recommandation actionnable (ex :
  "attendre la fin du throttle thermique avant de lancer un bench").
- **`predict(f)`** — assemble tout ce qui précède en une prédiction complète
  avec niveau de confiance (plafonné selon la source de la baseline —
  mesuré/théorique/défaut).
- **`post_run(prediction, observed_tg, note="")`** — enregistre l'écart
  réel entre la prédiction et le débit observé après coup, dans
  `predictor_v1_outcomes.jsonl`.
- **`summarize_outcomes(path=None)`** — statistiques sur tout l'historique
  de prédictions (MAE, MAPE, couverture) — sert à décider si assez de
  données existent pour passer à un modèle ML (`ready_for_v2_ml`).

**Commande type** (capturer l'état device AVANT un bench, pour contextualiser
le résultat) :
```bash
adb shell "echo model=... ; dumpsys power | grep mWakefulness ; \
    cat /sys/class/thermal/thermal_zone{28,32,47}/temp ; ..." > meta.txt
python3 predictor.py --meta meta.txt
```

### (b) `parse_hexagon_profile.py` — ce qui se passe DANS le runtime, op par op

C'est le seul producteur RÉEL (pas simulé) de coût par opération/par couche
dans tout le projet — il active et parse une instrumentation qui existait
déjà, compilée dans le runtime (`ggml_hexagon_dump_op_prof`/
`ggml_hexagon_dump_batch_prof`), jamais utilisée avant le 2026-09-06.

Parsing bas niveau :
- **`parse_op_line(rest)`** — extrait d'une ligne `profile-op` : nom du
  kernel, couche (regex `blk\.(\d+)\.`), latence en µs, cycles, position de
  départ des cycles, chemin d'exécution (repack/tiled/row-block), octets
  VTCM utilisés, dimensions, et les types RÉELS des opérandes (poids/
  activations/sortie) — lus directement dans l'appel kernel, PAS déduits du
  nom du fichier GGUF (vérifié empiriquement : les activations restent
  f32/f16, jamais "Q8_0_TILED" comme une hypothèse relayée l'affirmait à
  tort).
- **`class CycleUnwrapper`** — reconstruit le compteur de cycles 64-bit à
  partir des valeurs 32-bit tronquées du format `trace-evt`, en se reseedant
  à chaque OPBATCH sur son cycles_start (large). Implémentation identique à
  celle utilisée officiellement en amont
  (`llama.cpp/scripts/snapdragon/ggml-hexagon-profile.py`).
- **`parse_log(path)`** — parseur principal : maintient un `CycleUnwrapper`
  par device et par thread, unwrap chaque op ET chaque event `trace-evt`
  (HVX/HMX/DMA/L2FLUSH), puis fait correspondre chaque event à la couche
  dont la fenêtre de cycles le contient. Retourne `(events, batches, stats)`
  avec `stats["per_layer_engine"]` (compteurs HVX/HMX/DMA par couche).
  **AMÉLIORÉ 2026-09-08** : la fenêtre de chaque op est étendue de
  `[start,end]` à `[start, début_op_suivant)`, éliminant le trou inter-op
  (dispatch/sync/DMA) où les événements tombaient et étaient silencieusement
  perdus → mapping passé de **56.0%** à **99.99%** (471837/471892) sur la
  même trace réelle Qwen3-0.9B ; le reliquat (2 avant le premier op, 53
  après le dernier) est désormais explicitement compté
  (`stats["n_trace_before_first_op"]` / `["n_trace_after_last_op"]`) au lieu
  d'être perdu sans trace. Capture aussi `stats["vtcm_budget_bytes"]`
  (ligne `hwinfo ... vtcm N MB`) et `stats["vtcm_spills"]` (messages
  `HEX_VERBOSE` de spill/fallback — nécessitent `GGML_HEXAGON_VERBOSE=1`,
  une variable distincte de `GGML_HEXAGON_PROFILE`, sans quoi ce signal ne
  peut structurellement pas apparaître).
- **`build_per_layer_vtcm(events)`** / **`render_per_layer_vtcm_report(...)`**
  (nouveau 2026-09-08) — corrèlent le mapping par couche ci-dessus avec le
  budget/usage VTCM réel déjà loggé par op (`path`+`vtcm_bytes`, déjà parsés
  par `parse_op_line` mais jamais agrégés par couche avant) : vtcm moyen/max,
  %budget, latence totale, chemin kernel dominant par couche. Vérifié sur
  device réel (OnePlus 15, `GGML_HEXAGON_PROFILE=2` + `GGML_HEXAGON_VERBOSE=1`
  ensemble) : aucun spill/fallback réel, couches proches du plafond (~8 Mo)
  sans le dépasser.

Analyse par couche (le "fingerprint" temps réel demandé) :
- **`build_layer_fingerprint(events, layer)`** — agrège TOUS les événements
  réels d'une couche donnée : nombre d'ops, latence totale (somme réelle),
  types de poids utilisés, types d'activation réels, chemins d'exécution
  observés, VTCM cumulé, et l'opération la plus lente avec ses dimensions.
- **`render_layer_fingerprint_card(fp)`** — affiche la fiche d'une couche.
  Indique explicitement `HTP FITNESS` = mesuré (seul backend tracé),
  `GPU FITNESS` / `CPU FITNESS` = **NON MESURABLE** (aucune instrumentation
  équivalente n'existe pour OpenCL ou CPU dans ce projet — refuse
  explicitement d'inventer un chiffre sans trace réelle, point soulevé par
  le red-team de la discussion du jour).

Rapports agrégés :
- **`render_trace_evt_report(counts)`** — agrégat global HVX/HMX/DMA sur
  toute la trace (toutes couches confondues).
- **`render_per_layer_engine_report(per_layer_engine, n_total_trace_evt,
  n_before_first=0, n_after_last=0)`** — même chose mais PAR couche
  (remplace une première tentative abandonnée faute de mapping cycles
  fiable, résolue par `CycleUnwrapper`). Affiche depuis 2026-09-08 le
  compte explicite des événements structurellement hors mapping (avant le
  premier op / après le dernier).
- **`render_batch_overhead_report(batches)`** — **c'est ici qu'est mesuré le
  "coût d'arête"** : pas entre deux couches précises, mais le coût de
  dispatch par batch entier (~1 batch ≈ 1 couche). A produit la découverte
  prefill (2.3%) vs decode (11.6%) — facteur ×5.
- **`main()`** — CLI (`python3 parse_hexagon_profile.py log.txt --out trace.jsonl`).

**Commande type** (mesurer en temps réel ce qui se passe sur le téléphone
pendant une inférence) :
```bash
# Activer l'instrumentation déjà présente dans le runtime et lancer :
adb shell "cd /data/local/tmp/rt_clean/bin && \
    GGML_HEXAGON_PROFILE=3 ./llama-cli -m model.gguf -ngl 99 -lv 5 \
    -p 'prompt' -n 64 > /data/local/tmp/prof.log 2>&1"
adb pull /data/local/tmp/prof.log .

# Parser et obtenir la fiche de chaque couche :
python3 parse_hexagon_profile.py prof.log --out trace.jsonl
```

### `adaptive_lever.py` — réagir en temps réel à ce qui est mesuré

- **`recommend_oppoll()`** — recommande d'activer `GGML_HEXAGON_OPPOLL=1`
  (gain mesuré +50% sur Marco-Nano, à confirmer sur d'autres modèles).
- **`recommend_lever(live=None, cls=None, evidence=None, measured_tps=None)`**
  — recommandation de levier runtime (pas de quantization) selon la classe de
  goulot détectée par `classify_bottleneck` — ex : dispatch-bound → OPPOLL,
  memory-bound → pas de levier runtime utile, il faut baisser la RAM.
- **`_classify_from_trace(path)`** — charge une trace live et appelle
  `classify_bottleneck` pour en tirer la classe.
- **`record_outcome(cls, env_applied, baseline_tps, measured_tps, note="")`**
  — enregistre si un levier appliqué a réellement amélioré le débit, pour
  bâtir un historique de leviers validés vs leviers théoriques.
- **`main()`** — CLI.

---

### `live_monitor.py` (nouveau, 2026-09-06) — monitoring VRAIMENT en temps réel

Contrairement à tout ce qui précède (capture → pull → parse → rapport,
toujours APRÈS la fin du run), `live_monitor.py` lance le run sur le device
via `adb shell` en gardant le pipe stdout **ouvert**, et parse chaque ligne
`profile-op` au fur et à mesure qu'elle arrive — donc pendant que le modèle
tourne encore. Zéro nouveau C++, zéro nouvelle instrumentation device :
réutilise `parse_op_line` de `parse_hexagon_profile.py`, la seule différence
est qu'on ne referme pas le fichier avant de le lire.

- **`find_adb()`** — auto-détecte `adb`/`adb.exe` (PATH, ou fallback connu du
  projet), pour marcher aussi bien sous WSL que via `adb.exe` natif Windows
  (contournement utilisé après la panne WSL du 2026-09-06).
- **`build_remote_cmd(bin_dir, model, prompt, n_predict, extra_env)`** —
  construit la commande `adb shell` (active `GGML_HEXAGON_PROFILE=1`,
  suffisant pour un affichage live lisible — `=3` est trop volumineux, ~10x
  plus de lignes, réservé au mode post-hoc de `parse_hexagon_profile.py`).
- **`class LiveState`** — état accumulé pendant le run : `per_layer` (n_ops,
  latence cumulée, types de poids vus, dernier kernel), compteur de tokens
  (incrémenté à chaque `OPBATCH`), t/s estimé en continu.
  - **`.ingest(kind, info)`** — absorbe un événement déjà parsé.
  - **`.render()`** — affiche un tableau ANSI (clear + refresh) mis à jour
    toutes les `REFRESH_EVERY_N_OPS` (40) opérations.
- **`run_live(cmd_argv, save_trace_path=None)`** — boucle principale :
  `subprocess.Popen` avec stdout en pipe, lecture ligne par ligne en direct,
  Ctrl+C pour arrêter proprement. Sauvegarde optionnelle en JSONL au format
  exact attendu par `profile_model.py::aggregate_live_trace`
  (`{"layer","kernel","latency_us"}`) — la session live devient directement
  exploitable ensuite par tout le reste du pipeline (`--live-trace`).
- **`main()`** — CLI (`--bin`, `--model`, `--prompt`, `-n`, `--oppoll`, `--adb`,
  `--save-trace`).

**Testé réel** (2026-09-06, Qwen3-1.7B-Q4_0, prompt court, n=24) : les 28
couches se remplissent visiblement en direct pendant l'exécution (~11900 ops
vus en 3.1s), t/s affiché en continu (~8.3 t/s brut — inclut le coût de
démarrage du process, normal sur un run aussi court ; le débit "de croisière"
mesuré sur ce modèle est 38.2 t/s, voir `DENSE_CALIBRATION`).

**Limite honnête** : `GGML_HEXAGON_PROFILE=1` donne latence/type/chemin par
op, mais pas les compteurs HVX/HMX/DMA fins (`trace-evt`, mode `=3`) — ceux-là
restent réservés au mode post-hoc pour l'instant, le volume de log serait
trop dense pour un affichage live lisible sans agrégation supplémentaire (à
construire si le besoin se confirme).

---

### `parse_opencl_profile.py` (nouveau, 2026-09-06) — équivalent GPU réel, FAIT

Ce qui était listé "n'existe pas encore" dans une version précédente de ce
document a été construit le jour même. `ggml-opencl.cpp`
(`ggml-opencl/ggml-opencl.cpp:928-1124`) a un vrai profileur par-kernel côté
GPU (`struct ProfilingInfo`, `CL_QUEUE_PROFILING_ENABLE`), gardé derrière un
flag de COMPILATION CMake (`-DGGML_OPENCL_PROFILING=ON`, confirmé actif et
maintenu en amont via [PR #12442](https://github.com/ggml-org/llama.cpp/pull/12442)),
jamais activé ni parsé dans ce projet avant ce jour.

**Rebuild effectué** (`E:/oneplus/ab-build-opencl-prof/`, source
`xdna2-forensics/downloads/sources/llama-upstream`) : cross-compile Android
arm64 via NDK r27c, `-DGGML_HEXAGON=OFF -DGGML_OPENCL=ON
-DGGML_OPENCL_PROFILING=ON -DGGML_OPENCL_USE_ADRENO_KERNELS=ON`. Deux
obstacles réels rencontrés et résolus (pas des murs, juste de la plomberie) :
1. Headers/lib OpenCL absents pour la cross-compilation Android → copiés
   depuis `qairt/2.49.0.260730/.../CL/*.h` et un `libOpenCL.so` arm64 déjà
   présent dans `marco_moe_htp_test/assets/npu/`.
2. L'outil hôte `llama-ui-embed-host` (génère l'UI web embarquée, aucun
   rapport avec OpenCL) ne compile pas avec le clang++ du NDK utilisé comme
   "host compiler" sur cette machine (pas de sysroot Windows correct,
   `inttypes.h` introuvable) → contourné avec un wrapper batch qui traduit
   les flags GCC-style vers MSVC (`cl.exe`, déjà installé).

**Testé réel** (Qwen3-1.7B-Q4_0, `-ngl 99`, 8 tokens) : `cl_profiling.csv`
(736 Ko, 6023 événements réels) et `cl_trace.json` (Chrome trace, 4.7 Mo)
générés sur le device. Confirmé : le suffixe `-N` sur le nom d'op EST le
numéro de couche (0 à 27, cohérent avec les 28 couches du modèle) — pas
besoin de regex sur des noms `blk.N.` comme côté HTP.

- **`parse_csv(path)`** — parse `cl_profiling.csv`, extrait le numéro de
  couche depuis le suffixe de l'op (`OP_LAYER_RE`).
- **`build_layer_fingerprint(rows, layer)`** / **`render_layer_fingerprint_card(fp)`**
  — symétriques exacts des fonctions HTP du même nom : n_ops, ms cumulés
  (mesure réelle `CL_QUEUE_PROFILING`), kernels OpenCL réels utilisés (ex.
  `kernel_gemm_noshuffle_q4_0_f32` = GEMM natif Q4_0 sur Adreno), op la plus
  lente. **Directement comparable** à la fiche HTP produite par
  `parse_hexagon_profile.py` sur le même modèle/prompt.
- **`render_summary(rows)`** — classement des kernels les plus coûteux sur
  toute la trace + échantillon de fiches par couche.

**Résultat déjà informatif** sur Qwen3-1.7B-Q4_0 : `kernel_gemm_noshuffle_q4_0_f32`
domine largement (570 appels, 234 ms cumulés sur 446 ms totaux, ~52%) —
suivi de `kernel_gemv_noshuffle_q4_0_f32` (94 ms) et, fait notable,
`kernel_mul_mv_q6_K_f32_flat` : seulement 10 appels mais 64.6 ms cumulés
(6.46 ms/appel en moyenne, de très loin le kernel le plus lent par appel) —
signal cohérent avec le "TODO: implement Q4_0/Q8_0/... support" laissé dans
le code source (`ggml-opencl.cpp:7558`, confirmé lu directement) : certaines
opérations tombent sur un chemin générique non optimisé pour ce format.

### `common/cpu_profile.cpp` (nouveau patch C++, 2026-09-06) — équivalent CPU réel, FAIT

Patch minimal dans le fork `xdna2-forensics/downloads/sources/llama-upstream`
(2 fichiers neufs + 3 lignes dans `common.cpp`) : réutilise le hook officiel
`ggml_backend_sched_set_eval_callback` (`ggml-backend.h:316`, déjà câblé de
bout en bout via `common_params.cb_eval` → utilisé par `imatrix`/`debug.cpp`
pour de la collecte d'activations, jamais pour du chronométrage avant ce
jour). Activé par `GGML_CPU_PROFILE=1` (miroir de `GGML_HEXAGON_PROFILE`),
écrit `cpu_profiling.csv` avec le même schéma de colonnes que
`cl_profiling.csv` **plus 3 colonnes ajoutées suite à la demande explicite
"il manque la ram, débit, et qui consomme quoi et quand"** :
- `t_since_start_ms` — QUAND dans le run (position sur la frise globale, pas
  juste la durée isolée de l'op).
- `tensor_bytes` (`ggml_nbytes(t)`) — QUI consomme, taille réelle de la
  sortie de ce tenseur précis (proxy bande passante DDR par op).
- `rss_kb` (lu depuis `/proc/self/statm`, ~1 syscall négligeable) — COMBIEN
  au total, RSS du process entier au moment où cette op finit.

**Testé réel** (Qwen3-1.7B-Q4_0, `-ngl 0`, 8 tokens, `/data/local/tmp/opencl_prof/`)
: 9860 événements, RSS process 5560.1 Mo → 5565.0 Mo (delta +5.0 Mo sur 8
tokens — croissance KV cache cohérente, pas une fuite). `MUL_MAT` domine
(1970 appels, 634.6 ms cumulés, 58% du temps CPU total).

**Limite honnête documentée dans le rapport lui-même** : `tensor_bytes` pour
un `VIEW`/`SET_ROWS` sur le cache KV (`cache_k_lN`/`cache_v_lN`) reflète la
taille DÉCLARÉE du tenseur vue (le buffer KV entier pré-alloué à `n_ctx`),
pas les octets réellement écrits par cet appel précis (un update KV par
token n'écrit qu'une seule ligne) — signalé explicitement dans
`render_memory_timeline()` pour ne pas laisser croire que "chaque token
déplace 844 Mo".

`parse_opencl_profile.py` a été rendu générique (lit les colonnes par NOM
d'en-tête, pas par position fixe) pour parser `cl_profiling.csv` (GPU, 6
colonnes) ET `cpu_profiling.csv` (CPU, 9 colonnes) avec le même code —
`render_summary`/`render_layer_fingerprint_card` détectent automatiquement
le backend (présence de `rss_kb`) et adaptent le libellé + le vocabulaire
("kernel OpenCL" vs "op ggml") en conséquence. `render_memory_timeline(rows)`
— nouvelle fonction, seulement active sur une trace CPU (dit explicitement
pourquoi si appelée sur une trace GPU plutôt que d'inventer un chiffre) :
RSS début/fin/pic, et classement des ops par octets cumulés (proxy bande
passante, "qui consomme le plus").

**Avertissement (même compromis que côté HTP/GPU)** : activer ce callback
désactive le regroupement de nœuds du scheduler (chaque op dispatchée seule)
— c'est un OBSERVATEUR qui change le comportement mesuré. Ne jamais comparer
un t/s mesuré AVEC ce callback actif à un t/s mesuré sans.
### `parse_backend_copy_profile.py` (nouveau, 2026-09-06) — le "ping-pong" CPU/GPU/NPU, FAIT

Question explicite de l'utilisateur après avoir vu les 3 profileurs par
backend : *"du coup en verra les ping pong de la memoire entre cpu/npu/gpu ?"*
Réponse honnête au moment de la question : **NON** — aucun des 3 profileurs
(HTP, GPU, CPU) ne voit cette étape, chacun ne voyant que le calcul DANS son
propre backend, jamais le TRANSPORT entre backends. Fermé le jour même.

**Patch** : `ggml-backend.cpp::ggml_backend_sched_compute_splits` est le
point de passage UNIQUE où TOUTES les copies inter-backend transitent (déjà
repéré en lisant ce fichier pour le patch CPU — lignes ~1676-1740,
`ggml_backend_tensor_copy(input, input_cpy)` × 2 + le chemin async
`cpy_tensor_async`). Ajout d'un petit RAII (`ggml_backend_copy_timer`,
constructeur/destructeur) autour de ces deux points, gardé derrière
`GGML_BACKEND_COPY_PROFILE=1`, écrit `backend_copy_profile.csv` (tensor,
backend source, backend destination, octets, durée ms).

**Testé réel** (Qwen3-1.7B-Q4_0, `-ngl 10` — offload PARTIEL pour forcer un
run mixte CPU+GPU, 8 tokens) : 70 copies détectées, frontière de split
identifiée exactement sur la couche 18 (là où `-ngl 10` arrête l'offload),
sens CPU→OpenCL, débit effectif mesuré ~1.9 Go/s. `ffn_out`/`ffn_inp` de la
couche frontière sont les tenseurs les plus copiés.

**Limite honnête, documentée dans le script lui-même** : un run avec `-ngl 99`
(tout sur un seul backend) ou `-ngl 0` (tout CPU) ne montrera quasiment rien
ici — ce n'est pas un bug, juste la conséquence logique d'un seul backend
actif. Ce script n'a de sens que pour un run explicitement MIXTE. Le chemin
de copie spécifique aux experts MoE (`copy_experts`, plusieurs petites copies
groupées) n'est pas encore instrumenté — seul le chemin d'entrée de split
générique l'est.

## Ce qui reste hors de portée (limites honnêtes)

- Le mapping trace-evt→couche (mode post-hoc `=3`) est à **99.99%** depuis
  le 2026-09-08 (était 56% avant l'extension de fenêtre inter-op) — le
  reliquat (2 avant le premier op, 53 après le dernier sur la trace de
  référence) est structurel et désormais explicitement compté.
- La détection de spill/fallback VTCM nécessite `GGML_HEXAGON_VERBOSE=1`
  EN PLUS de `GGML_HEXAGON_PROFILE` — sans ce flag distinct, l'absence de
  signal ne prouve rien. Vérifié avec les deux flags actifs sur device réel
  (2026-09-08) : aucun spill sur le run testé (Qwen3-0.6B, 16 tokens),
  couches proches du plafond VTCM (~8 Mo) sans le dépasser — mais un seul
  run/modèle testé, pas encore généralisé à un modèle plus gros (MoE)
  où un vrai spill est plus probable.
- Le live monitor (`live_monitor.py`) affiche un tableau texte simple, pas un
  dashboard graphique — suffisant pour comprendre ce qui se passe en direct,
  mais pas pensé pour un affichage long-terme/historique multi-runs.
- Le ping-pong CPU/GPU/NPU (`parse_backend_copy_profile.py`) ne couvre pas
  encore le chemin de copie spécifique aux experts MoE.
