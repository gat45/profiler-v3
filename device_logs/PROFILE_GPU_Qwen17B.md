# Profil GPU (OpenCL/Adreno) — mesure reelle

- 6023 evenements kernel, 446.23 ms cumules, 28 couches distinctes identifiees, 1328 evenements hors-couche (embedding/lm_head/norm final/...)

## Kernels les plus couteux (cumule sur toute la trace)

| kernel | n appels | ms cumules | ms moyen |
|---|---:|---:|---:|
| kernel_gemm_noshuffle_q4_0_f32 | 570 | 233.98 | 0.4105 |
| kernel_gemv_noshuffle_q4_0_f32 | 1360 | 93.57 | 0.0688 |
| kernel_mul_mv_q6_K_f32_flat | 10 | 64.59 | 6.4586 |
| flash_attn_f32_f16_q1_vec | 196 | 17.72 | 0.0904 |
| kernel_gemm_noshuffle_q4_1_f32 | 9 | 10.37 | 1.1524 |
| kernel_rms_norm_mul | 1130 | 5.46 | 0.0048 |
| flash_attn_f32_f16 | 84 | 5.22 | 0.0621 |
| kernel_transpose_32_16 | 579 | 3.34 | 0.0058 |
| kernel_gemv_noshuffle_q4_1_f32 | 21 | 2.68 | 0.1277 |
| kernel_add | 162 | 2.18 | 0.0135 |
| kernel_swiglu | 280 | 1.79 | 0.0064 |
| kernel_rope_neox_f32 | 560 | 1.67 | 0.0030 |
| kernel_set_rows_f16_i64 | 560 | 1.65 | 0.0029 |
| flash_attn_blk_f16 | 84 | 1.08 | 0.0129 |
| kernel_add_row | 398 | 0.86 | 0.0022 |

## RAM / debit dans le temps

Non disponible sur cette trace — colonnes rss_kb/t_since_start_ms/tensor_bytes absentes (trace GPU cl_profiling.csv : CL_QUEUE_PROFILING_ENABLE ne donne pas acces au RSS process, seulement au temps d'execution kernel). Utiliser cpu_profiling.csv (GGML_CPU_PROFILE=1) pour ce rapport.

## Fiche par couche (echantillon — 3 premieres + 3 dernieres)

LAYER 0 (GPU)
────────────────────────────────────────
N OPS           168 (mesure)
TOTAL MS        12.166 (mesure reelle, CL_QUEUE_PROFILING)
KERNELS         kernel_add, kernel_add_row, kernel_gemm_noshuffle_q4_0_f32, kernel_gemm_noshuffle_q4_1_f32, kernel_gemv_noshuffle_q4_0_f32, kernel_gemv_noshuffle_q4_1_f32, kernel_rms_norm_mul, kernel_rope_neox_f32, kernel_swiglu, kernel_transpose_32_16 (mesure — kernels OpenCL reels, ex kernel_gemm_noshuffle_q4_0_f32 = GEMM natif Q4_0 sur Adreno)
OP LA PLUS LENTE ffn_out via kernel_gemm_noshuffle_q4_1_f32 (1.184 ms, output=2048x2x1x1)

GPU FITNESS      mesure directement ci-dessus (comparable a la fiche produite par le meme pipeline sur un AUTRE backend, MEME modele/prompt, pour une comparaison directe par couche)

LAYER 1 (GPU)
────────────────────────────────────────
N OPS           168 (mesure)
TOTAL MS        12.114 (mesure reelle, CL_QUEUE_PROFILING)
KERNELS         kernel_add, kernel_add_row, kernel_gemm_noshuffle_q4_0_f32, kernel_gemm_noshuffle_q4_1_f32, kernel_gemv_noshuffle_q4_0_f32, kernel_gemv_noshuffle_q4_1_f32, kernel_rms_norm_mul, kernel_rope_neox_f32, kernel_swiglu, kernel_transpose_32_16 (mesure — kernels OpenCL reels, ex kernel_gemm_noshuffle_q4_0_f32 = GEMM natif Q4_0 sur Adreno)
OP LA PLUS LENTE ffn_out via kernel_gemm_noshuffle_q4_1_f32 (1.159 ms, output=2048x9x1x1)

GPU FITNESS      mesure directement ci-dessus (comparable a la fiche produite par le meme pipeline sur un AUTRE backend, MEME modele/prompt, pour une comparaison directe par couche)

LAYER 2 (GPU)
────────────────────────────────────────
N OPS           168 (mesure)
TOTAL MS        12.076 (mesure reelle, CL_QUEUE_PROFILING)
KERNELS         kernel_add, kernel_add_row, kernel_gemm_noshuffle_q4_0_f32, kernel_gemm_noshuffle_q4_1_f32, kernel_gemv_noshuffle_q4_0_f32, kernel_gemv_noshuffle_q4_1_f32, kernel_rms_norm_mul, kernel_rope_neox_f32, kernel_swiglu, kernel_transpose_32_16 (mesure — kernels OpenCL reels, ex kernel_gemm_noshuffle_q4_0_f32 = GEMM natif Q4_0 sur Adreno)
OP LA PLUS LENTE ffn_out via kernel_gemm_noshuffle_q4_1_f32 (1.178 ms, output=2048x9x1x1)

GPU FITNESS      mesure directement ci-dessus (comparable a la fiche produite par le meme pipeline sur un AUTRE backend, MEME modele/prompt, pour une comparaison directe par couche)

LAYER 25 (GPU)
────────────────────────────────────────
N OPS           168 (mesure)
TOTAL MS        11.447 (mesure reelle, CL_QUEUE_PROFILING)
KERNELS         kernel_add, kernel_add_row, kernel_gemm_noshuffle_q4_0_f32, kernel_gemv_noshuffle_q4_0_f32, kernel_rms_norm_mul, kernel_rope_neox_f32, kernel_swiglu, kernel_transpose_32_16 (mesure — kernels OpenCL reels, ex kernel_gemm_noshuffle_q4_0_f32 = GEMM natif Q4_0 sur Adreno)
OP LA PLUS LENTE ffn_out via kernel_gemm_noshuffle_q4_0_f32 (0.976 ms, output=2048x2x1x1)

GPU FITNESS      mesure directement ci-dessus (comparable a la fiche produite par le meme pipeline sur un AUTRE backend, MEME modele/prompt, pour une comparaison directe par couche)

LAYER 26 (GPU)
────────────────────────────────────────
N OPS           168 (mesure)
TOTAL MS        11.438 (mesure reelle, CL_QUEUE_PROFILING)
KERNELS         kernel_add, kernel_add_row, kernel_gemm_noshuffle_q4_0_f32, kernel_gemv_noshuffle_q4_0_f32, kernel_rms_norm_mul, kernel_rope_neox_f32, kernel_swiglu, kernel_transpose_32_16 (mesure — kernels OpenCL reels, ex kernel_gemm_noshuffle_q4_0_f32 = GEMM natif Q4_0 sur Adreno)
OP LA PLUS LENTE ffn_out via kernel_gemm_noshuffle_q4_0_f32 (1.009 ms, output=2048x2x1x1)

GPU FITNESS      mesure directement ci-dessus (comparable a la fiche produite par le meme pipeline sur un AUTRE backend, MEME modele/prompt, pour une comparaison directe par couche)

LAYER 27 (GPU)
────────────────────────────────────────
N OPS           159 (mesure)
TOTAL MS        7.426 (mesure reelle, CL_QUEUE_PROFILING)
KERNELS         kernel_add_row, kernel_gemm_noshuffle_q4_0_f32, kernel_gemv_noshuffle_q4_0_f32, kernel_rms_norm_mul, kernel_rope_neox_f32, kernel_swiglu, kernel_transpose_32_16 (mesure — kernels OpenCL reels, ex kernel_gemm_noshuffle_q4_0_f32 = GEMM natif Q4_0 sur Adreno)
OP LA PLUS LENTE Qcur via kernel_gemm_noshuffle_q4_0_f32 (0.340 ms, output=2048x2x1x1)

GPU FITNESS      mesure directement ci-dessus (comparable a la fiche produite par le meme pipeline sur un AUTRE backend, MEME modele/prompt, pour une comparaison directe par couche)