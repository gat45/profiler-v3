#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""live_monitor.py — monitoring EN TEMPS REEL (pas capture->parse->rapport
apres coup) de ce qui se passe sur le device pendant une inference.

Contrairement a parse_hexagon_profile.py (qui lit un fichier .log complet
DEJA termine), ce script lance le run sur le device via `adb shell`, garde le
pipe stdout OUVERT, et parse chaque ligne "profile-op" AU FUR ET A MESURE
qu'elle est produite par le DSP — donc pendant que le modele tourne encore,
pas apres.

Reutilise le meme parseur (parse_op_line) que parse_hexagon_profile.py — zero
nouveau C++, zero nouvelle instrumentation device : la seule difference est
qu'on ne referme pas le fichier avant de le lire.

Limite honnete : necessite GGML_HEXAGON_PROFILE=1 (pas =3, trop volumineux
pour un affichage temps reel lisible — voir parse_hexagon_profile.py pour les
evenements trace-evt HVX/HMX/DMA plus fins, capturables seulement en mode
post-hoc pour l'instant).

Usage :
  python3 live_monitor.py \\
      --bin /data/local/tmp/rt_clean/bin \\
      --model ../models/Qwen3-0.9B-A0.6B.i1-Q4_0.gguf \\
      --prompt "Explique la thermodynamique." -n 64 \\
      --save-trace device_logs/live_session.jsonl
"""
import argparse
import json
import re
import subprocess
import sys
import time

from parse_hexagon_profile import parse_op_line, LAYER_RE

REFRESH_EVERY_N_OPS = 40


def find_adb():
    import shutil
    for candidate in ("adb", "adb.exe"):
        p = shutil.which(candidate)
        if p:
            return p
    # Fallback connu dans ce projet (natif Windows, contourne WSL) :
    known = r"E:\oneplus\geniex_harness\tools\platform-tools\adb.exe"
    import os
    if os.path.exists(known):
        return known
    raise RuntimeError("adb introuvable — passer --adb <chemin> explicitement")


def build_remote_cmd(bin_dir, model, prompt, n_predict, extra_env):
    env_str = " ".join(f"{k}={v}" for k, v in extra_env.items())
    return (
        f"cd {bin_dir} && LD_LIBRARY_PATH={bin_dir} {env_str} "
        f"./llama-cli --single-turn -m {model} -ngl 99 -lv 5 "
        f"-p '{prompt}' -n {n_predict} 2>&1"
    )


class LiveState:
    def __init__(self):
        self.per_layer = {}  # layer -> {"n_ops":int,"total_us":float,"weight_types":set,"last_kernel":str}
        self.n_ops_total = 0
        self.n_tokens_seen = 0
        self.t_start = time.time()
        self.last_layer_seen = None
        self.all_events = []  # pour --save-trace

    def ingest(self, kind, info):
        self.n_ops_total += 1
        if kind == "batch":
            # Un OPBATCH == un passage complet du graphe == ~1 token en decode.
            self.n_tokens_seen += 1
            return
        layer = info.get("layer")
        self.all_events.append(info)
        if layer is not None:
            self.last_layer_seen = layer
            pl = self.per_layer.setdefault(
                layer, {"n_ops": 0, "total_us": 0.0, "weight_types": set(),
                        "last_kernel": ""})
            pl["n_ops"] += 1
            pl["total_us"] += info.get("usec", 0.0) or 0.0
            if info.get("weight_type"):
                pl["weight_types"].add(info["weight_type"])
            pl["last_kernel"] = info.get("opname", "?")

    def render(self):
        elapsed = time.time() - self.t_start
        tps = self.n_tokens_seen / elapsed if elapsed > 0 else 0.0
        lines = [
            "\033[2J\033[H",  # clear screen + curseur en haut (ANSI, terminal reel requis)
            f"=== LIVE MONITOR — t={elapsed:5.1f}s | {self.n_ops_total} ops vus | "
            f"~{self.n_tokens_seen} tokens | ~{tps:4.1f} t/s (estime, brut) ===",
            f"Derniere couche active : {self.last_layer_seen}",
            "",
            f"{'layer':>6} {'n_ops':>7} {'total_us':>10} {'weight_types':<20} {'dernier kernel'}",
        ]
        for layer in sorted(self.per_layer, key=lambda x: (x is None, x)):
            pl = self.per_layer[layer]
            wt = ",".join(sorted(pl["weight_types"])) or "?"
            lines.append(f"{layer!s:>6} {pl['n_ops']:>7} {pl['total_us']:>10.0f} "
                         f"{wt:<20} {pl['last_kernel']}")
        lines.append("")
        lines.append("(Ctrl+C pour arreter — un rapport final sera affiche)")
        return "\n".join(lines)


def run_live(cmd_argv, save_trace_path=None):
    state = LiveState()
    proc = subprocess.Popen(cmd_argv, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    ops_since_refresh = 0
    try:
        for line in proc.stdout:
            m = re.search(r"profile-op\s+(.*)$", line)
            if not m:
                continue
            info = parse_op_line(m.group(1))
            if info is None:
                continue
            state.ingest(info.get("kind", "op"), info)
            ops_since_refresh += 1
            if ops_since_refresh >= REFRESH_EVERY_N_OPS:
                sys.stdout.write(state.render())
                sys.stdout.flush()
                ops_since_refresh = 0
    except KeyboardInterrupt:
        proc.terminate()
    finally:
        proc.wait()
    print(state.render())
    print("\n=== SESSION TERMINEE ===")
    if save_trace_path:
        # Traduit le format brut de parse_op_line vers le schema attendu par
        # profile_model.py::aggregate_live_trace ({"layer","kernel","latency_us"}),
        # exactement comme le fait parse_hexagon_profile.py::parse_log().
        with open(save_trace_path, "w", encoding="utf-8") as f:
            for e in state.all_events:
                f.write(json.dumps({"layer": e.get("layer"),
                                    "kernel": e.get("opname"),
                                    "latency_us": float(e.get("usec", 0.0))}) + "\n")
        print(f"[trace] {len(state.all_events)} events sauves -> {save_trace_path} "
              f"(exploitable ensuite par profile_model.py --live-trace)")
    return state


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", required=True, help="dossier binaire device, ex /data/local/tmp/rt_clean/bin")
    ap.add_argument("--model", required=True, help="chemin du .gguf sur le device (relatif a --bin ou absolu)")
    ap.add_argument("--prompt", default="Explique brievement ce que tu es.")
    ap.add_argument("-n", "--n-predict", type=int, default=64)
    ap.add_argument("--oppoll", type=int, default=1)
    ap.add_argument("--adb", help="chemin adb.exe (sinon auto-detecte)")
    ap.add_argument("--save-trace", help="chemin JSONL local pour sauver tous les events (optionnel)")
    args = ap.parse_args()

    adb = args.adb or find_adb()
    remote_cmd = build_remote_cmd(
        args.bin, args.model, args.prompt, args.n_predict,
        {"GGML_HEXAGON_PROFILE": "1", "GGML_HEXAGON_OPPOLL": str(args.oppoll)})
    print(f"[live_monitor] adb={adb}")
    print(f"[live_monitor] commande distante : {remote_cmd}")
    run_live([adb, "shell", remote_cmd], save_trace_path=args.save_trace)


if __name__ == "__main__":
    main()
