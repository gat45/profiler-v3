#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""parse_opencl_profile.py — equivalent GPU (OpenCL/Adreno) de
parse_hexagon_profile.py. Ferme le gap explicitement signale dans
FONCTIONS.md ("GPU FITNESS NON MESURABLE") : le mecanisme existait deja,
compile dans ggml-opencl.cpp (struct ProfilingInfo, CL_QUEUE_PROFILING_ENABLE),
gardé derriere un flag de COMPILATION (-DGGML_OPENCL_PROFILING=ON), jamais
active ni parse dans ce projet avant le 2026-09-06.

DECOUVERTE (2026-09-06) : contrairement a l'instrumentation HTP (juste une
variable d'env), celle-ci demande un REBUILD (cross-compile Android arm64 via
NDK). Rebuild fait dans E:/oneplus/ab-build-opencl-prof/ (source :
xdna2-forensics/downloads/sources/llama-upstream, GGML_HEXAGON=OFF,
GGML_OPENCL=ON, GGML_OPENCL_PROFILING=ON). Deux obstacles reels rencontres et
resolus :
  1. Headers/lib OpenCL absents pour la cross-compilation Android -> copies
     depuis qairt/2.49.0.260730 (headers CL/*.h) et un libOpenCL.so arm64
     deja present dans un autre sous-projet (marco_moe_htp_test/assets/npu).
  2. L'outil hote llama-ui-embed-host (genere l'UI web embarquee, AUCUN
     rapport avec OpenCL) ne compile pas avec le clang++ du NDK utilise en
     "host compiler" (inttypes.h introuvable, absence de sysroot Windows
     correct) -> contourne via un wrapper qui traduit les flags GCC-style
     vers MSVC (cl.exe, deja installe sur cette machine).

Format de ligne source (confirme empiriquement, Qwen3-1.7B-Q4_0, -ngl 99,
8 tokens, /data/local/tmp/opencl_prof/) :
  "<op name>-<layer>,<kernel name>,<exec duration ms>,<global size>,
   <local size>,<output size>"
Le SUFFIXE "-<layer>" sur le nom de l'op EST le numero de couche (verifie :
va bien de -0 a -27 sur un modele a 28 couches) — pas besoin de regex sur des
noms de tenseurs blk.N. comme cote HTP, c'est deja dans op name.

Usage :
  # 1. Build avec GGML_OPENCL_PROFILING=ON (voir README.md pour la commande
  #    cmake complete), pousser le binaire + libggml-opencl.so sur device.
  # 2. Lancer un run avec -ngl 99 (offload GPU) — cl_profiling.csv et
  #    cl_trace.json sont ecrits dans le cwd du process sur le device :
  adb shell "cd /data/local/tmp/opencl_prof/bin && \\
      LD_LIBRARY_PATH=. ./llama cli -m model.gguf -ngl 99 -p '...' -n 64 \\
      --single-turn"
  adb pull /data/local/tmp/opencl_prof/bin/cl_profiling.csv .
  # 3. Parser :
  python3 parse_opencl_profile.py cl_profiling.csv
"""
import argparse
import csv
import re
import sys
from collections import defaultdict

OP_LAYER_RE = re.compile(r"^(.*)-(\d+)$")


def parse_csv(path):
    """Retourne une liste de dicts. Colonnes de base (toujours presentes,
    format cl_profiling.csv d'origine) : op/layer/kernel/ms/global_size/
    local_size/output_size. Colonnes optionnelles (presentes seulement dans
    cpu_profiling.csv, produit par common/cpu_profile.cpp) : t_since_start_ms
    (QUAND dans le run), tensor_bytes (QUI consomme, taille reelle de la
    sortie de ce tenseur), rss_kb (COMBIEN au total, RSS process a cet
    instant) — lues par NOM d'entete, pas par position fixe, pour rester
    compatible avec les deux schemas de CSV sans dupliquer le parseur.
    layer=None si le nom d'op ne porte pas de suffixe -N (ops hors-couche :
    embedding, lm_head, norm final...)."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in (next(reader, None) or [])]
        idx = {name: i for i, name in enumerate(header)}

        def col(line, name, cast=str, default=None):
            i = idx.get(name)
            if i is None or i >= len(line):
                return default
            try:
                return cast(line[i].strip())
            except ValueError:
                return default

        for line in reader:
            if len(line) < 3:
                continue
            op_raw = col(line, "op name", str, "")
            ms = col(line, "exec duration (ms)", float)
            if not op_raw or ms is None:
                continue
            m = OP_LAYER_RE.match(op_raw)
            op, layer = (m.group(1), int(m.group(2))) if m else (op_raw, None)
            rows.append({
                "op": op, "layer": layer,
                "kernel": col(line, "kernel name", str, "?"),
                "ms": ms,
                "global_size": col(line, "global size", str, "?"),
                "local_size": col(line, "local size", str, "?"),
                "output_size": col(line, "output size", str, "?"),
                # None si la colonne n'existe pas dans ce CSV (cote GPU) —
                # jamais une valeur inventee.
                "t_since_start_ms": col(line, "t_since_start_ms", float),
                "tensor_bytes": col(line, "tensor_bytes", int),
                "rss_kb": col(line, "rss_kb", int),
            })
    return rows


def build_layer_fingerprint(rows, layer):
    """Symetrique de parse_hexagon_profile.py::build_layer_fingerprint."""
    ops = [r for r in rows if r["layer"] == layer]
    if not ops:
        return None
    total_ms = sum(r["ms"] for r in ops)
    kernels = sorted(set(r["kernel"] for r in ops))
    slowest = max(ops, key=lambda r: r["ms"])
    return {"layer": layer, "n_ops": len(ops), "total_ms": total_ms,
            "kernels": kernels, "slowest_op": slowest["op"],
            "slowest_kernel": slowest["kernel"], "slowest_ms": slowest["ms"],
            "slowest_output_size": slowest["output_size"]}


def render_layer_fingerprint_card(fp, backend="GPU"):
    if fp is None:
        return "(aucune donnee pour cette couche dans cette trace)"
    is_gpu = backend == "GPU"
    timer_note = "CL_QUEUE_PROFILING" if is_gpu else "std::chrono (wall-clock, hook ggml_backend_sched_eval_callback)"
    kernel_label = "kernels OpenCL reels, ex kernel_gemm_noshuffle_q4_0_f32 = GEMM natif Q4_0 sur Adreno" \
        if is_gpu else "ops ggml reelles (ex MUL_MAT, RMS_NORM) — pas de notion de kernel GPU cote CPU"
    L = [f"LAYER {fp['layer']} ({backend})", "─" * 40]
    L.append(f"N OPS           {fp['n_ops']} (mesure)")
    L.append(f"TOTAL MS        {fp['total_ms']:.3f} (mesure reelle, {timer_note})")
    L.append(f"KERNELS         {', '.join(fp['kernels'])} (mesure — {kernel_label})")
    L.append(f"OP LA PLUS LENTE {fp['slowest_op']} via {fp['slowest_kernel']} "
             f"({fp['slowest_ms']:.3f} ms, output={fp['slowest_output_size']})")
    L.append("")
    L.append(f"{backend} FITNESS      mesure directement ci-dessus (comparable a la fiche "
             "produite par le meme pipeline sur un AUTRE backend, MEME modele/prompt, "
             "pour une comparaison directe par couche)")
    return "\n".join(L)


def render_memory_timeline(rows):
    """Repond a la demande explicite 'il manque la ram, debit, et qui
    consomme quoi et quand' — seulement disponible sur une trace CPU
    (cpu_profiling.csv, colonnes rss_kb/t_since_start_ms/tensor_bytes), pas
    sur une trace GPU (cl_profiling.csv n'a pas ces colonnes, dit
    explicitement pourquoi plutot que d'inventer)."""
    with_rss = [r for r in rows if r["rss_kb"] is not None]
    if not with_rss:
        return ("## RAM / debit dans le temps\n\n"
                "Non disponible sur cette trace — colonnes rss_kb/"
                "t_since_start_ms/tensor_bytes absentes (trace GPU "
                "cl_profiling.csv : CL_QUEUE_PROFILING_ENABLE ne donne pas "
                "acces au RSS process, seulement au temps d'execution "
                "kernel). Utiliser cpu_profiling.csv "
                "(GGML_CPU_PROFILE=1) pour ce rapport.")

    rss_start = with_rss[0]["rss_kb"]
    rss_end = with_rss[-1]["rss_kb"]
    rss_peak = max(r["rss_kb"] for r in with_rss)
    peak_row = max(with_rss, key=lambda r: r["rss_kb"])

    # QUI consomme le plus de bande passante (proxy : octets de sortie
    # cumules par op, PAS le nombre d'appels — une op appelee peu souvent
    # mais sur de gros tenseurs peut dominer la bande passante reelle sans
    # dominer le temps CPU).
    bytes_by_op = defaultdict(int)
    for r in rows:
        if r["tensor_bytes"] is not None:
            bytes_by_op[re.sub(r"-\d+$", "", r["op"])] += r["tensor_bytes"]

    L = ["## RAM / débit dans le temps (mesuré, trace CPU)", "",
        f"- RSS process : {rss_start/1024:.1f} Mo au debut -> "
        f"{rss_end/1024:.1f} Mo a la fin (delta {(rss_end-rss_start)/1024:+.1f} Mo "
        f"sur {len(with_rss)} evenements) — croissance attendue = KV cache "
        f"qui grandit a chaque token genere, PAS une fuite si le delta reste "
        f"proportionnel au nombre de tokens.",
        f"- Pic RSS : {rss_peak/1024:.1f} Mo, atteint sur l'op `{peak_row['op']}` "
        f"({peak_row['kernel']}) a t={peak_row['t_since_start_ms']:.1f} ms depuis le debut du run.",
        "",
        "### QUI consomme le plus (octets cumulés par type d'op, proxy bande passante DDR)",
        "",
        "**Limite honnête** : `tensor_bytes` = `ggml_nbytes()` du tenseur DESTINATION "
        "de l'op — pour un `VIEW`/`SET_ROWS` sur le cache KV (`cache_k_lN`/`cache_v_lN`), "
        "cela reflete la taille DECLAREE du tenseur vue (souvent le buffer KV entier "
        "pre-alloue a n_ctx), PAS le nombre d'octets reellement ecrits par CET appel "
        "(un update KV par token n'ecrit qu'une seule ligne). Ne pas lire ces lignes "
        "comme \"chaque token deplace 844 Mo\" — c'est la taille du buffer, pas le trafic "
        "reel de cet appel. Fiable en revanche pour les MUL_MAT/gros tenseurs calcules "
        "en entier a chaque appel.",
        "",
        "| op (sans suffixe couche) | octets cumulés (déclarés) | Mo |", "|---|---:|---:|"]
    for op, tot in sorted(bytes_by_op.items(), key=lambda kv: -kv[1])[:12]:
        L.append(f"| {op} | {tot} | {tot/1e6:.2f} |")
    return "\n".join(L)


def render_summary(rows):
    backend = "CPU" if any(r["rss_kb"] is not None for r in rows) else "GPU"
    layers = sorted(set(r["layer"] for r in rows if r["layer"] is not None))
    off_layer = [r for r in rows if r["layer"] is None]
    total_ms = sum(r["ms"] for r in rows)
    kernel_totals = defaultdict(float)
    kernel_counts = defaultdict(int)
    for r in rows:
        kernel_totals[r["kernel"]] += r["ms"]
        kernel_counts[r["kernel"]] += 1

    title = ("Profil CPU (ggml_backend_sched_eval_callback) — mesure reelle" if backend == "CPU"
             else "Profil GPU (OpenCL/Adreno) — mesure reelle")
    L = [f"# {title}", "",
        f"- {len(rows)} evenements kernel, {total_ms:.2f} ms cumules, "
        f"{len(layers)} couches distinctes identifiees, "
        f"{len(off_layer)} evenements hors-couche (embedding/lm_head/norm final/...)",
        "",
        "## Kernels les plus couteux (cumule sur toute la trace)", "",
        "| kernel | n appels | ms cumules | ms moyen |", "|---|---:|---:|---:|"]
    for k, tot in sorted(kernel_totals.items(), key=lambda kv: -kv[1])[:15]:
        n = kernel_counts[k]
        L.append(f"| {k} | {n} | {tot:.2f} | {tot/n:.4f} |")

    L.append("")
    L.append(render_memory_timeline(rows))
    L.append("")
    L.append("## Fiche par couche (echantillon — 3 premieres + 3 dernieres)")
    sample = layers[:3] + (layers[-3:] if len(layers) > 6 else [])
    for layer in sample:
        L.append("")
        L.append(render_layer_fingerprint_card(build_layer_fingerprint(rows, layer), backend=backend))

    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", help="cl_profiling.csv pulle depuis le device")
    ap.add_argument("--out", help="chemin du rapport .md (sinon stdout)")
    args = ap.parse_args()

    rows = parse_csv(args.csv_path)
    if not rows:
        print(f"[erreur] aucune ligne exploitable dans {args.csv_path}", file=sys.stderr)
        sys.exit(1)
    report = render_summary(rows)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[out] {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
