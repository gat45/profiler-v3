#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""parse_backend_copy_profile.py — le "ping-pong" CPU/GPU/NPU (2026-09-06).

Reponse a la question explicite de l'utilisateur : "on verra les ping pong
de la memoire entre cpu/npu/gpu ?" — reponse au moment de la question :
NON, aucun des 3 profileurs construits ce jour (HTP, GPU, CPU) ne voit cette
etape, chacun ne voyant que le calcul DANS son propre backend, jamais le
TRANSPORT entre backends.

Ferme ce gap via un patch dans ggml-backend.cpp (le point de passage UNIQUE
ou toutes les copies inter-backend transitent, dans
ggml_backend_sched_compute_splits, juste avant chaque calcul de split) —
voir les commentaires `GGML_BACKEND_COPY_PROFILE` dans ce fichier pour le
detail. Active par GGML_BACKEND_COPY_PROFILE=1, ecrit
backend_copy_profile.csv (tensor, src_backend, dst_backend, bytes,
duration_ms).

IMPORTANT — pour observer du VRAI ping-pong il faut un run MIXTE (au moins
2 backends actifs, ex -ngl PARTIEL qui laisse une partie des couches sur
CPU et le reste sur GPU/NPU). Un run avec -ngl 99 (tout sur un seul
backend) ou -ngl 0 (tout CPU) ne montrera QUASIMENT rien ici — ce n'est PAS
un bug, juste la consequence logique d'un seul backend actif.

Usage :
  adb shell "cd .../bin && GGML_BACKEND_COPY_PROFILE=1 LD_LIBRARY_PATH=. \\
      ./llama cli -m model.gguf -ngl <N_PARTIEL> -p '...' -n 32 --single-turn"
  adb pull .../backend_copy_profile.csv .
  python3 parse_backend_copy_profile.py backend_copy_profile.csv
"""
import argparse
import csv
import re
from collections import defaultdict

LAYER_RE = re.compile(r"-(\d+)$")


def parse_csv(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for line in reader:
            try:
                bytes_ = int(line["bytes"])
                ms = float(line["duration_ms"])
            except (KeyError, ValueError):
                continue
            m = LAYER_RE.search(line["tensor"])
            rows.append({"tensor": line["tensor"],
                        "layer": int(m.group(1)) if m else None,
                        "src": line["src_backend"], "dst": line["dst_backend"],
                        "bytes": bytes_, "ms": ms})
    return rows


def render_report(rows):
    if not rows:
        return ("Aucun evenement de copie inter-backend dans cette trace — "
                "soit un seul backend etait actif (run non-mixte, ex -ngl 99 "
                "ou -ngl 0 : c'est attendu, pas une erreur), soit "
                "GGML_BACKEND_COPY_PROFILE n'etait pas active.")

    pairs = defaultdict(lambda: {"n": 0, "bytes": 0, "ms": 0.0})
    for r in rows:
        k = (r["src"], r["dst"])
        pairs[k]["n"] += 1
        pairs[k]["bytes"] += r["bytes"]
        pairs[k]["ms"] += r["ms"]

    total_ms = sum(r["ms"] for r in rows)
    total_bytes = sum(r["bytes"] for r in rows)
    layers_crossing = sorted(set(r["layer"] for r in rows if r["layer"] is not None))

    L = ["# Ping-pong mémoire inter-backend — mesuré réel", "",
        f"- {len(rows)} copies inter-backend, {total_bytes/1e6:.2f} Mo cumulés, "
        f"{total_ms:.2f} ms cumulées",
        f"- Frontière(s) de split détectée(s) sur les couches : "
        f"{layers_crossing if layers_crossing else '(aucune couche identifiable dans les noms de tenseur copiés)'}",
        "", "## Par paire de backends (src → dst)", "",
        "| src | dst | n copies | Mo cumulés | ms cumulées | Mo/s effectif |",
        "|---|---|---:|---:|---:|---:|"]
    for (src, dst), agg in sorted(pairs.items(), key=lambda kv: -kv[1]["bytes"]):
        mbps = (agg["bytes"] / 1e6) / (agg["ms"] / 1000.0) if agg["ms"] > 0 else 0.0
        L.append(f"| {src} | {dst} | {agg['n']} | {agg['bytes']/1e6:.3f} | "
                 f"{agg['ms']:.3f} | {mbps:.1f} |")

    L.append("")
    L.append("## Tenseurs les plus copiés (cumulé)")
    L.append("")
    by_tensor_base = defaultdict(lambda: {"n": 0, "bytes": 0, "ms": 0.0})
    for r in rows:
        base = LAYER_RE.sub("", r["tensor"])
        by_tensor_base[base]["n"] += 1
        by_tensor_base[base]["bytes"] += r["bytes"]
        by_tensor_base[base]["ms"] += r["ms"]
    L.append("| tenseur (sans suffixe couche) | n | Mo cumulés | ms cumulées |")
    L.append("|---|---:|---:|---:|")
    for name, agg in sorted(by_tensor_base.items(), key=lambda kv: -kv[1]["bytes"])[:10]:
        L.append(f"| {name} | {agg['n']} | {agg['bytes']/1e6:.3f} | {agg['ms']:.3f} |")

    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path")
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = parse_csv(args.csv_path)
    report = render_report(rows)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[out] {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
