# Reproductibilité — comment refaire exactement les mesures de ce dépôt

Réponse directe à une critique reçue (2026-09-08) : les chiffres cités ici
("99.99%", "100%", "×3.1-×3.4", overhead decode ×5...) ne sont vérifiables
par un tiers que si la chaîne complète peut être reproduite. Ce document
liste tout ce qu'il faut, sans rien cacher.

## 1. Device et SoC (portée des calibrations)

- **OnePlus 15**, SoC **Snapdragon SM8850**, DSP **Hexagon HTP v81**.
- Toute constante de ce dépôt (`GGML_BW=30.0`, budget VTCM ~8 Mo, facteurs
  de correction MoE, seuils de mapping) n'a de validité démontrée que sur
  **ce SoC précis**. Aucune mesure sur v75/v79/autre SM88xx à ce jour.

## 2. Runtime : quel binaire exact produit ces logs

- Fork source : [zhouwg/ggml-hexagon](https://github.com/zhouwg/ggml-hexagon),
  branche `self-build-jz` (voir `tools/BUILD_JZ_FASTRPC.md` dans
  `geniex_harness` pour le detail des 3 bugs de build rencontrés et leurs
  fix).
- Flags de compilation : `GGML_HEXAGON=ON`, `GGML_HEXAGON_USE_MEMPOOL=ON`.
- Toolchain : Android NDK r29, Hexagon SDK 6.6.0.0.
- **Ceci n'est PAS un binaire llama.cpp upstream standard** — les parseurs
  `parse_backend_copy_profile.py` et `parse_opencl_profile.py` nécessitent
  en plus un patch manuel de `ggml-backend.cpp` (voir `PATCHES.md`) ; sans
  ce patch, leurs colonnes de sortie n'existent pas dans le log source.
  `parse_hexagon_profile.py`, lui, ne nécessite PAS de patch — il active
  une instrumentation déjà présente dans le runtime déployé
  (`ggml_hexagon_dump_op_prof`/`ggml_hexagon_dump_batch_prof`), juste
  jamais utilisée avant le 2026-09-06.

## 3. Variables d'environnement — laquelle active quoi (souvent confondues)

| Variable | Effet | Nécessaire pour |
|---|---|---|
| `GGML_HEXAGON_PROFILE=1` | logs `profile-op` par op (latence, cycles, path/vtcm) | `parse_hexagon_profile.py` de base |
| `GGML_HEXAGON_PROFILE=2` | + compteurs PMU | idem, avec PMU |
| `GGML_HEXAGON_PROFILE=3` | + `trace-evt` (HVX/HMX/DMA bas niveau) | mapping trace→couche (§ README) |
| `GGML_HEXAGON_VERBOSE=1` | logs `supports-op`/`execute-op` + messages `HEX_VERBOSE` (skip/fallback VTCM) | **détection de spill VTCM réel** — variable DISTINCTE de `PROFILE`, sans elle le signal de spill ne peut structurellement pas apparaître (confondu une fois dans ce dépôt, corrigé le 2026-09-08) |
| `GGML_HEXAGON_NDEV` | nombre de devices HTP virtuels | multi-session (non concluant à ce jour, cf. README) |
| `GGML_HEXAGON_MBUF` | taille buffer mempool | perf, valeur `3200` utilisée dans toutes les captures listées ci-dessous |
| `GGML_HEXAGON_USE_HMX` | active/désactive l'accélérateur matriciel | **NE JAMAIS mettre à 0** en test de contournement — crash confirmé 2× sur ce device |

## 4. Commande de capture exacte (root shell, recette utilisée pour toutes les traces `device_logs/`)

```bash
adb connect <ip>:<port>   # wifi, ou USB direct sans adb connect
adb -s <serial> shell "su -c '
cd /data/local/tmp/npu
export LD_LIBRARY_PATH=/data/local/tmp/npu
export ADSP_LIBRARY_PATH=/data/local/tmp/npu
export GGML_HEXAGON_NDEV=1
export GGML_HEXAGON_MBUF=3200
export GGML_HEXAGON_USE_HMX=1
export GGML_HEXAGON_PROFILE=3
export GGML_HEXAGON_VERBOSE=1
./llama-server -m <model.gguf> -ngl 99 -dev HTP0 -t 6 -c 2048 \
    --host 127.0.0.1 --port 8080 -lv 5 > /data/local/tmp/trace.log 2>&1
'"
adb -s <serial> pull /data/local/tmp/trace.log .
py parse_hexagon_profile.py trace.log --out trace.jsonl
```

## 5. Conditions thermiques / charge — ce qui N'EST PAS contrôlé

Aucune des captures listées dans `device_logs/` n'a de mesure de température
de départ enregistrée systématiquement (contrairement aux campagnes dédiées
`bench_results/RAPPORT_MTP_2X5_CAUSAL_20260831.md` dans `geniex_harness`, qui
elles contrôlent explicitement ce facteur). Prendre les chiffres de latence
absolue (pas les ratios structurels comme le %mapping) avec cette réserve.

## 6. Traces de référence utilisées pour les chiffres cités dans ce README

| Trace | Modèle | Événements trace-evt | Mapping obtenu | Où |
|---|---|---:|---:|---|
| `prof3_qwen09b.log` | Qwen3-0.9B-A0.6B | 471 892 | 99.99% | `device_logs/` |
| capture 2026-09-08 (non versionnée, 220 Mo) | Qwen3-1.7B | 1 332 495 | 100.0% | reproductible via §4 avec `-m Qwen3-1.7B-Q4_0.gguf` |

Ces deux traces sont **indépendantes** (modèles différents, captures à 2
jours d'écart) — voir README pour pourquoi c'est important (réponse directe
au risque de sur-ajustement sur une seule mesure).
