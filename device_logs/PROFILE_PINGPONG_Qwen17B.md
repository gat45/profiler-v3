# Ping-pong mémoire inter-backend — mesuré réel

- 70 copies inter-backend, 0.34 Mo cumulés, 0.18 ms cumulées
- Frontière(s) de split détectée(s) sur les couches : [18]

## Par paire de backends (src → dst)

| src | dst | n copies | Mo cumulés | ms cumulées | Mo/s effectif |
|---|---|---:|---:|---:|---:|
| CPU | OpenCL | 70 | 0.338 | 0.176 | 1922.6 |

## Tenseurs les plus copiés (cumulé)

| tenseur (sans suffixe couche) | n | Mo cumulés | ms cumulées |
|---|---:|---:|---:|
| ffn_out | 10 | 0.164 | 0.074 |
| ffn_inp | 10 | 0.164 | 0.050 |
| attn_inp_kq_mask | 10 | 0.010 | 0.014 |
| leaf_10 | 10 | 0.000 | 0.007 |
| leaf_12 | 10 | 0.000 | 0.007 |
| leaf_6 | 10 | 0.000 | 0.012 |
| leaf_370 | 10 | 0.000 | 0.012 |