# Patches nécessaires — quel parseur marche sur quel binaire

Réponse directe à une critique reçue (2026-09-08) : deux des parseurs de ce
dépôt ne fonctionnent PAS sur un binaire llama.cpp standard, sans que ce
soit dit clairement avant. Ce fichier liste précisément lequel a besoin de
quoi, pour ne pas perdre de temps à chercher pourquoi un parseur ne sort
rien.

## Résumé

| Parseur | Nécessite un rebuild patché ? | Quoi exactement |
|---|---|---|
| `parse_hexagon_profile.py` | **Non** | Instrumentation déjà présente dans tout binaire `ggml-hexagon` (`ggml_hexagon_dump_op_prof`), juste jamais activée par défaut — active-la avec `GGML_HEXAGON_PROFILE=1/2/3` (voir `REPRODUCIBILITY.md`) |
| `android_telemetry.py` | **Non** | Lit `/proc`, `dumpsys gpu`, `kgsl gpubusy` directement — aucune dépendance à ggml |
| `live_monitor.py` | **Non** | Consomme le même flux `GGML_HEXAGON_PROFILE` que `parse_hexagon_profile.py` |
| `parse_opencl_profile.py` | **Oui — flag de compilation** | `-DGGML_OPENCL=ON -DGGML_OPENCL_PROFILING=ON` (le mécanisme `ProfilingInfo`/`CL_QUEUE_PROFILING_ENABLE` existe dans `ggml-opencl.cpp` upstream, mais est gardé derrière ce flag, jamais activé par défaut) |
| `parse_backend_copy_profile.py` | **Oui — patch source manuel** | Ajout de logs dans `ggml_backend_sched_compute_splits` (`ggml-backend.cpp`), déclenché par `GGML_BACKEND_COPY_PROFILE=1` — voir détail ci-dessous |

## `parse_backend_copy_profile.py` — patch exact

**État actuel (2026-09-08) : ce patch n'est PAS présent dans le worktree de
build courant (`ab-wt`)** — il a été appliqué une fois, le 2026-09-06, sur un
arbre de build qui n'a pas été conservé tel quel. Ceci est documenté
honnêtement plutôt que de fournir un diff qui ne correspondrait plus à rien.

Ce qui est garanti stable (interface, pas implémentation) :
- **Point d'accroche** : `ggml_backend_sched_compute_splits`, juste avant le
  calcul de chaque split — c'est le seul endroit où TOUTES les copies
  inter-backend (CPU↔GPU↔NPU) transitent.
- **Activation** : variable d'environnement `GGML_BACKEND_COPY_PROFILE=1`.
- **Sortie attendue** : `backend_copy_profile.csv` avec les colonnes
  `tensor,src_backend,dst_backend,bytes,duration_ms`.
- **Condition d'observation** : un run explicitement MIXTE (`-ngl` partiel,
  au moins 2 backends actifs). Un run `-ngl 99` ou `-ngl 0` ne produit
  quasiment rien — ce n'est pas un bug du patch, juste la conséquence d'un
  seul backend actif.

**Mode dégradé (2026-09-08, ajout suite à la critique)** : si le CSV attendu
est absent ou ne contient pas les colonnes ci-dessus, `parse_backend_copy_profile.py`
doit afficher explicitement `"runtime non instrumenté — données de copie
inter-backend indisponibles (voir PATCHES.md)"` au lieu d'un rapport vide ou
d'un traceback — voir la TODO correspondante dans le fichier.

## `parse_opencl_profile.py` — build exact utilisé une fois (2026-09-06)

- Source : `xdna2-forensics/downloads/sources/llama-upstream`.
- Flags : `GGML_HEXAGON=OFF -DGGML_OPENCL=ON -DGGML_OPENCL_PROFILING=ON`.
- Cross-compile Android arm64 via NDK, headers/lib OpenCL copiés depuis
  `qairt/2.49.0.260730` (`CL/*.h`) + un `libOpenCL.so` arm64 emprunté à un
  autre sous-projet (`marco_moe_htp_test/assets/npu`) — aucune de ces deux
  dépendances n'est incluse dans ce dépôt (à récupérer séparément, taille
  et provenance externe).
- Build fait dans `E:/oneplus/ab-build-opencl-prof/`, non versionné ici.

## Pourquoi ne pas juste patcher upstream et proposer une PR ?

C'est la vraie amélioration à moyen terme (item évoqué : "explorer une
instrumentation plus légère via les hooks déjà présents dans le backend
Hexagon plutôt que de patcher le scheduler central"). `parse_hexagon_profile.py`
le fait déjà (zéro patch requis). Porter le même principe à OpenCL/ping-pong
demanderait de trouver un point d'instrumentation équivalent déjà présent
dans `ggml-backend.cpp` upstream — pas fait à ce jour, piste ouverte.
