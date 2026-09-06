#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""telemetry_dashboard.py — afficher sur le PC, en direct, la telemetrie
servie par TelemetryServer.kt (app Android hw_monitor) via `adb forward`.

Repond a la demande explicite : "trouve comment je peux afficher ces infos
sur mon pc". Pas besoin de dependance externe (requests) — urllib suffit vu
le volume (un GET toutes les ~0.5-1s).

Prerequis :
  1. App hw_monitor installee et lancee sur le device (le serveur demarre
     automatiquement a l'ouverture de l'app, ou via le bouton).
  2. adb forward tcp:8082 tcp:8082   (fait automatiquement par ce script si
     absent, via adb.exe deja localise ailleurs dans profiler_v3).

Usage :
  python3 telemetry_dashboard.py                    # etat global seul
  python3 telemetry_dashboard.py --name llama        # + process llama trouve automatiquement
  python3 telemetry_dashboard.py --pid 12345          # + process precis
  python3 telemetry_dashboard.py --interval 0.5 --log device_logs/telemetry_session.jsonl
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
import urllib.error

PORT = 8082
BASE_URL = f"http://127.0.0.1:{PORT}"


def find_adb():
    import shutil
    p = shutil.which("adb") or shutil.which("adb.exe")
    if p:
        return p
    import os
    known = r"E:\oneplus\geniex_harness\tools\platform-tools\adb.exe"
    if os.path.exists(known):
        return known
    raise RuntimeError("adb introuvable")


def ensure_forward(adb):
    subprocess.run([adb, "forward", f"tcp:{PORT}", f"tcp:{PORT}"],
                   capture_output=True, text=True)


def fetch(path):
    try:
        with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=3) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, TimeoutError, ConnectionRefusedError) as e:
        return {"_fetch_error": str(e)}


def fmt(v, suffix=""):
    return f"{v}{suffix}" if v is not None else "n/a"


def render(sample):
    lines = ["\033[2J\033[H", "=== TELEMETRIE HW MONITOR (device reel, via adb forward) ===", ""]
    if "_fetch_error" in sample:
        lines.append(f"[ERREUR] {sample['_fetch_error']}")
        lines.append("-> verifier : app hw_monitor lancee ? 'adb forward tcp:8082 tcp:8082' actif ?")
        return "\n".join(lines)

    lines.append(f"ts        : {sample.get('ts_iso')}")
    lines.append(f"CPU %     : {fmt(sample.get('cpuPct'), '%')}   "
                 f"(su/root requis — 'null' = pas encore accorde a l'app dans Magisk)")
    lines.append(f"GPU busy  : {fmt(sample.get('gpuBusyPctInstant'), '%')}   "
                 f"(sans root — direct kgsl/gpubusy)")
    lines.append(f"NPU temp  : {fmt(sample.get('npuTempC'), '°C')}   (su/root requis)")
    lines.append(f"RAM       : {fmt(sample.get('ramPct'), '%')}   (su/root requis)")
    lines.append(f"BATT      : {fmt(sample.get('battPct'), '%')}   (su/root requis)")
    lines.append(f"llamaPids : {sample.get('llamaPids')}")
    if sample.get("gpuBusyError"):
        lines.append(f"[gpuBusyError] {sample['gpuBusyError']}")
    if sample.get("cmdlineScanError"):
        lines.append(f"[cmdlineScanError] {sample['cmdlineScanError']}")

    proc = sample.get("process")
    if proc:
        lines.append("")
        lines.append(f"--- Process PID {proc['pid']} ---")
        lines.append(f"VmRSS      : {fmt(proc.get('vmRssKb'), ' Ko')}")
        lines.append(f"VmHWM (pic): {fmt(proc.get('vmHwmKb'), ' Ko')}")
        lines.append(f"GPU mem    : {fmt(proc.get('gpuMemBytes'), ' octets')}")
        lines.append(f"source     : {proc.get('source')}")
        if proc.get("error"):
            lines.append(f"[error] {proc['error']}")

    lines.append("")
    lines.append("(Ctrl+C pour arreter)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", type=int)
    ap.add_argument("--name")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--log", help="chemin JSONL pour sauver chaque echantillon")
    args = ap.parse_args()

    adb = find_adb()
    ensure_forward(adb)

    if not fetch("/health").get("ok"):
        print("[erreur] serveur injoignable sur 127.0.0.1:8082 — "
              "lance l'app hw_monitor sur le device (adb shell am start "
              "-n com.geniex.hwmonitor/.MainActivity), puis relance ce script.",
              file=sys.stderr)
        sys.exit(1)

    log_f = open(args.log, "w", encoding="utf-8") if args.log else None
    query = f"?pid={args.pid}" if args.pid else (f"?name={args.name}" if args.name else "")
    try:
        while True:
            sample = fetch(f"/telemetry{query}")
            sys.stdout.write(render(sample))
            sys.stdout.flush()
            if log_f:
                log_f.write(json.dumps(sample) + "\n")
                log_f.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if log_f:
            log_f.close()


if __name__ == "__main__":
    main()
