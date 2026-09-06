# Profil CPU (ggml_backend_sched_eval_callback) — mesure reelle

- 9860 evenements kernel, 1095.77 ms cumules, 28 couches distinctes identifiees, 3420 evenements hors-couche (embedding/lm_head/norm final/...)

## Kernels les plus couteux (cumule sur toute la trace)

| kernel | n appels | ms cumules | ms moyen |
|---|---:|---:|---:|
| MUL_MAT | 1970 | 634.56 | 0.3221 |
| RMS_NORM | 1130 | 159.81 | 0.1414 |
| FLASH_ATTN_EXT | 280 | 102.40 | 0.3657 |
| MUL | 1130 | 87.77 | 0.0777 |
| ROPE | 560 | 42.74 | 0.0763 |
| ADD | 560 | 25.58 | 0.0457 |
| SET_ROWS | 560 | 13.69 | 0.0244 |
| PERMUTE | 840 | 11.56 | 0.0138 |
| GLU | 280 | 10.16 | 0.0363 |
| GET_ROWS | 30 | 4.95 | 0.1650 |
| VIEW | 1400 | 2.14 | 0.0015 |
| RESHAPE | 1120 | 0.40 | 0.0004 |

## RAM / débit dans le temps (mesuré, trace CPU)

- RSS process : 5560.1 Mo au debut -> 5565.0 Mo a la fin (delta +5.0 Mo sur 9860 evenements) — croissance attendue = KV cache qui grandit a chaque token genere, PAS une fuite si le delta reste proportionnel au nombre de tokens.
- Pic RSS : 5565.0 Mo, atteint sur l'op `embd` (GET_ROWS) a t=1225.5 ms depuis le debut du run.

### QUI consomme le plus (octets cumulés par type d'op, proxy bande passante DDR)

**Limite honnête** : `tensor_bytes` = `ggml_nbytes()` du tenseur DESTINATION de l'op — pour un `VIEW`/`SET_ROWS` sur le cache KV (`cache_k_lN`/`cache_v_lN`), cela reflete la taille DECLAREE du tenseur vue (souvent le buffer KV entier pre-alloue a n_ctx), PAS le nombre d'octets reellement ecrits par CET appel (un update KV par token n'ecrit qu'une seule ligne). Ne pas lire ces lignes comme "chaque token deplace 844 Mo" — c'est la taille du buffer, pas le trafic reel de cet appel. Fiable en revanche pour les MUL_MAT/gros tenseurs calcules en entier a chaque appel.

| op (sans suffixe couche) | octets cumulés (déclarés) | Mo |
|---|---:|---:|
| cache_k_l0 (view) | 844103680 | 844.10 |
| cache_v_l0 (view) | 844103680 | 844.10 |
| cache_k_l1 (view) | 844103680 | 844.10 |
| cache_v_l1 (view) | 844103680 | 844.10 |
| cache_k_l2 (view) | 844103680 | 844.10 |
| cache_v_l2 (view) | 844103680 | 844.10 |
| cache_k_l3 (view) | 844103680 | 844.10 |
| cache_v_l3 (view) | 844103680 | 844.10 |
| cache_k_l4 (view) | 844103680 | 844.10 |
| cache_v_l4 (view) | 844103680 | 844.10 |
| cache_k_l5 (view) | 844103680 | 844.10 |
| cache_v_l5 (view) | 844103680 | 844.10 |

## Fiche par couche (echantillon — 3 premieres + 3 dernieres)

LAYER 0 (CPU)
────────────────────────────────────────
N OPS           230 (mesure)
TOTAL MS        272.766 (mesure reelle, std::chrono (wall-clock, hook ggml_backend_sched_eval_callback))
KERNELS         ADD, GLU, MUL, MUL_MAT, RESHAPE, RMS_NORM, ROPE (mesure — ops ggml reelles (ex MUL_MAT, RMS_NORM) — pas de notion de kernel GPU cote CPU)
OP LA PLUS LENTE norm via RMS_NORM (16.562 ms, output=2048x2x1x1)

CPU FITNESS      mesure directement ci-dessus (comparable a la fiche produite par le meme pipeline sur un AUTRE backend, MEME modele/prompt, pour une comparaison directe par couche)

LAYER 1 (CPU)
────────────────────────────────────────
N OPS           230 (mesure)
TOTAL MS        65.468 (mesure reelle, std::chrono (wall-clock, hook ggml_backend_sched_eval_callback))
KERNELS         ADD, GLU, MUL, MUL_MAT, RESHAPE, RMS_NORM, ROPE (mesure — ops ggml reelles (ex MUL_MAT, RMS_NORM) — pas de notion de kernel GPU cote CPU)
OP LA PLUS LENTE Vcur via MUL_MAT (12.032 ms, output=1024x9x1x1)

CPU FITNESS      mesure directement ci-dessus (comparable a la fiche produite par le meme pipeline sur un AUTRE backend, MEME modele/prompt, pour une comparaison directe par couche)

LAYER 2 (CPU)
────────────────────────────────────────
N OPS           230 (mesure)
TOTAL MS        14.188 (mesure reelle, std::chrono (wall-clock, hook ggml_backend_sched_eval_callback))
KERNELS         ADD, GLU, MUL, MUL_MAT, RESHAPE, RMS_NORM, ROPE (mesure — ops ggml reelles (ex MUL_MAT, RMS_NORM) — pas de notion de kernel GPU cote CPU)
OP LA PLUS LENTE ffn_out via MUL_MAT (1.189 ms, output=2048x9x1x1)

CPU FITNESS      mesure directement ci-dessus (comparable a la fiche produite par le meme pipeline sur un AUTRE backend, MEME modele/prompt, pour une comparaison directe par couche)

LAYER 25 (CPU)
────────────────────────────────────────
N OPS           230 (mesure)
TOTAL MS        34.366 (mesure reelle, std::chrono (wall-clock, hook ggml_backend_sched_eval_callback))
KERNELS         ADD, GLU, MUL, MUL_MAT, RESHAPE, RMS_NORM, ROPE (mesure — ops ggml reelles (ex MUL_MAT, RMS_NORM) — pas de notion de kernel GPU cote CPU)
OP LA PLUS LENTE Kcur_normed via MUL (6.037 ms, output=128x8x2x1)

CPU FITNESS      mesure directement ci-dessus (comparable a la fiche produite par le meme pipeline sur un AUTRE backend, MEME modele/prompt, pour une comparaison directe par couche)

LAYER 26 (CPU)
────────────────────────────────────────
N OPS           230 (mesure)
TOTAL MS        14.726 (mesure reelle, std::chrono (wall-clock, hook ggml_backend_sched_eval_callback))
KERNELS         ADD, GLU, MUL, MUL_MAT, RESHAPE, RMS_NORM, ROPE (mesure — ops ggml reelles (ex MUL_MAT, RMS_NORM) — pas de notion de kernel GPU cote CPU)
OP LA PLUS LENTE ffn_up via MUL_MAT (1.192 ms, output=6144x9x1x1)

CPU FITNESS      mesure directement ci-dessus (comparable a la fiche produite par le meme pipeline sur un AUTRE backend, MEME modele/prompt, pour une comparaison directe par couche)

LAYER 27 (CPU)
────────────────────────────────────────
N OPS           230 (mesure)
TOTAL MS        10.174 (mesure reelle, std::chrono (wall-clock, hook ggml_backend_sched_eval_callback))
KERNELS         ADD, GLU, MUL, MUL_MAT, RESHAPE, RMS_NORM, ROPE (mesure — ops ggml reelles (ex MUL_MAT, RMS_NORM) — pas de notion de kernel GPU cote CPU)
OP LA PLUS LENTE Qcur via MUL_MAT (0.403 ms, output=2048x9x1x1)

CPU FITNESS      mesure directement ci-dessus (comparable a la fiche produite par le meme pipeline sur un AUTRE backend, MEME modele/prompt, pour une comparaison directe par couche)