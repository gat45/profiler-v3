#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""android_telemetry.py — lecture directe Android/kernel, EN COMPLEMENT des
profileurs ggml (HTP/GPU/CPU/ping-pong). Question explicite de l'utilisateur :
"on peut pas passer directement par android ou le kernel ?"

Reponse : OUI, verifie reellement sur ce device (uid=2000 shell, PAS root) :
- /proc/meminfo             -> memoire globale du systeme (sans root)
- /proc/<pid>/status        -> VmRSS/VmHWM de N'IMPORTE QUEL process par PID
                               (contrairement a smaps_rollup, PERMISSION
                               DENIED sur un pid etranger sans root — status
                               est moins restreint, verifie empiriquement)
- /sys/class/kgsl/kgsl-3d0/gpubusy -> charge GPU reelle (2 entiers : busy_ns,
                               total_ns sur la fenetre de mesure du kernel)
- dumpsys gpu               -> memoire GPU PAR PROCESSUS ("Proc <pid> total:
                               <bytes>") — le "qui consomme quoi" cote GPU,
                               sans instrumentation ggml, verifie fonctionnel
- simpleperf existe MAIS les compteurs PMU testes (task-clock) sont refuses
  ("not supported on the device") sans root -> pas de bande passante DDR
  directe accessible par ce chemin sur ce device tel quel.

Usage :
  python3 android_telemetry.py --pid <PID_llama> --interval 0.2 --duration 10 \\
      --out device_logs/android_telemetry.jsonl
  # PID inconnu a l'avance -> lance et resout automatiquement :
  python3 android_telemetry.py --proc-name llama --interval 0.2 --duration 10
"""
import argparse
import json
import re
import subprocess
import sys
import time


def find_adb():
    import shutil
    p = shutil.which("adb") or shutil.which("adb.exe")
    if p:
        return p
    known = r"E:\oneplus\geniex_harness\tools\platform-tools\adb.exe"
    import os
    if os.path.exists(known):
        return known
    raise RuntimeError("adb introuvable")


def adb_shell(adb, cmd):
    return subprocess.run([adb, "shell", cmd], capture_output=True, text=True,
                          timeout=15).stdout


def find_pid_by_name(adb, name):
    out = adb_shell(adb, f"pidof {name}")
    out = out.strip().split()
    return int(out[0]) if out else None


def read_meminfo(adb):
    """Retourne un dict {cle: valeur_kb} depuis /proc/meminfo (global,
    systeme entier, PAS specifique a un process)."""
    out = adb_shell(adb, "cat /proc/meminfo")
    d = {}
    for line in out.splitlines():
        m = re.match(r"(\w+):\s+(\d+)\s*kB", line)
        if m:
            d[m.group(1)] = int(m.group(2))
    return d


def read_proc_status(adb, pid):
    """VmRSS/VmHWM (pic RSS depuis le lancement) pour UN PID donne — marche
    sur un process appartenant a un autre utilisateur (verifie : /proc/1/status
    lisible en shell), contrairement a smaps_rollup qui est refuse."""
    out = adb_shell(adb, f"cat /proc/{pid}/status 2>/dev/null")
    d = {}
    for key in ("VmRSS", "VmHWM", "VmSize", "Threads"):
        m = re.search(rf"{key}:\s+(\d+)", out)
        if m:
            d[key] = int(m.group(1))
    return d if d else None


def read_gpu_busy(adb):
    """(busy_ns, total_ns) sur la fenetre de mesure interne du kernel kgsl —
    busy_ns/total_ns = taux d'utilisation GPU instantane reel."""
    out = adb_shell(adb, "cat /sys/class/kgsl/kgsl-3d0/gpubusy 2>/dev/null").split()
    if len(out) == 2:
        try:
            return int(out[0]), int(out[1])
        except ValueError:
            return None
    return None


def read_gpu_mem_by_proc(adb):
    """dumpsys gpu -> {pid: bytes}. C'est le 'qui consomme quoi' cote GPU,
    SANS avoir besoin d'instrumenter ggml-opencl.cpp — vue systeme complete
    (tous les process, pas seulement le notre)."""
    out = adb_shell(adb, "dumpsys gpu 2>/dev/null")
    d = {}
    for line in out.splitlines():
        m = re.match(r"Proc (\d+) total: (\d+)", line.strip())
        if m:
            d[int(m.group(1))] = int(m.group(2))
    return d


def sample(adb, pid=None):
    row = {"t": time.time(), "meminfo": read_meminfo(adb),
           "gpu_busy": read_gpu_busy(adb), "gpu_mem_by_proc": read_gpu_mem_by_proc(adb)}
    if pid:
        row["proc_status"] = read_proc_status(adb, pid)
        row["proc_gpu_mem_bytes"] = row["gpu_mem_by_proc"].get(pid)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", type=int, help="PID a suivre specifiquement (VmRSS + GPU mem)")
    ap.add_argument("--proc-name", help="resout le PID via 'pidof <nom>' au lieu de --pid")
    ap.add_argument("--interval", type=float, default=0.5, help="secondes entre echantillons")
    ap.add_argument("--duration", type=float, default=10.0, help="duree totale en secondes")
    ap.add_argument("--out", help="fichier JSONL (sinon stdout)")
    args = ap.parse_args()

    adb = find_adb()
    pid = args.pid
    if args.proc_name and not pid:
        pid = find_pid_by_name(adb, args.proc_name)
        if not pid:
            print(f"[avertissement] aucun process nomme '{args.proc_name}' trouve "
                  "au moment du lancement — lance ce script APRES avoir demarre "
                  "le run a profiler, ou repasse --pid explicitement", file=sys.stderr)

    out_f = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    t_end = time.time() + args.duration
    n = 0
    try:
        while time.time() < t_end:
            row = sample(adb, pid)
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            n += 1
            time.sleep(args.interval)
    finally:
        if args.out:
            out_f.close()
    print(f"[android_telemetry] {n} echantillons ecrits", file=sys.stderr)


if __name__ == "__main__":
    main()
