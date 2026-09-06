#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""parse_hexagon_profile.py — PREMIER producteur REEL de trace live par couche
(2026-09-06), fermant le gap documente dans AUDIT_COMPLET_PROFILAGE_PAR_COUCHE_
INEXISTANT_20260906.md.

DECOUVERTE : l'instrumentation par-op existe DEJA dans le runtime deploye
(ggml-hexagon.cpp:ggml_hexagon_dump_op_prof / ggml_hexagon_dump_batch_prof),
avec de VRAIS compteurs cycles/PMU cote DSP (rsp.cycles_start/stop, pd.usecs).
Elle etait juste :
1. Jamais activee (env var GGML_HEXAGON_PROFILE=1, jamais documentee/utilisee
   dans aucun rapport avant aujourd'hui) — confirme empiriquement : 0 occurrence
   de "GGML_HEXAGON_PROFILE" dans tout bench_results/ avant ce script.
2. Filtree par le niveau de log par defaut de llama-cli (GGML_LOG_DEBUG ne
   s'affiche qu'avec -lv 5 / --log-verbosity 5, sinon invisible silencieusement).
3. Jamais parsee vers le format JSONL attendu par
   profile_model.py::aggregate_live_trace() ({"layer":int,"kernel":str,
   "latency_us":float}) — c'est ce que fait ce script.

Usage :
  # 1. capturer un run reel sur device avec l'instrumentation activee :
  adb shell "cd /data/local/tmp/rt_clean/bin && \\
      LD_LIBRARY_PATH=/data/local/tmp/rt_clean/bin GGML_HEXAGON_PROFILE=1 \\
      GGML_HEXAGON_OPPOLL=1 ./llama-cli --single-turn -m ../models/X.gguf \\
      -ngl 99 -lv 5 -p '...' -n 64 > /data/local/tmp/prof.log 2>&1"
  adb pull /data/local/tmp/prof.log .

  # 2. parser vers JSONL exploitable par profile_model_corrige.py --live-trace :
  py parse_hexagon_profile.py prof.log --out prof_live_trace.jsonl

  # 3. profiler avec la VRAIE trace :
  py profile_model_corrige.py model.gguf --live-trace prof_live_trace.jsonl \\
      --emit-tensor-type-file overrides.txt

Format de ligne source (confirme empiriquement le 2026-09-06, Qwen3-0.9B-A0.6B,
7394 lignes 'profile-op' capturees en 8 tokens) :
  "<ts> D ggml-hex: HTP0 profile-op <OPNAME>|<tensor names>|<dims>|<types>|
   <strides>|<path> vtcm <bytes>|usec <N> cycles <N> start <N> mhz <F>"
  ou pour le batch entier :
  "<ts> D ggml-hex: HTP0 profile-op OPBATCH|----|n-ops <N>|----|----|----|
   usec <N> cycles <N> start <N> mhz <F>"
"""
import argparse
import json
import re
from collections import Counter

LINE_RE = re.compile(r"profile-op\s+(.*)$")
USEC_RE = re.compile(r"\busec\s+(\d+)")
CYCLES_RE = re.compile(r"\bcycles\s+(\d+)")
LAYER_RE = re.compile(r"blk\.(\d+)\.")
NOPS_RE = re.compile(r"n-ops\s+(\d+)")

# AJOUT 2026-09-06 : format "trace-evt" confirme IDENTIQUE au parseur officiel
# upstream (llama.cpp/scripts/snapdragon/ggml-hexagon-profile.py, verifie par
# lecture directe du script). Necessite GGML_HEXAGON_PROFILE=3 (pas =1) —
# active un volume de log ~10x plus gros (80 Mo pour 8 tokens sur un petit
# modele) : a n'activer que pour une capture ciblee, pas par defaut. Donne
# acces a deux axes reclames dans la discussion 2026-09-06 et absents de
# GGML_HEXAGON_PROFILE=1 : (1) DMA en tant que VRAI compteur d'evenements,
# distinct du proxy VTCM ; (2) repartition HVX vs HMX ("execution_engine"),
# confirmee empiriquement tres asymetrique sur Qwen3-0.9B (220856 HVX_COMP
# vs 4892 HMX_COMP dans un run de 8 tokens).
TRACE_EVT_RE = re.compile(
    r"trace-evt\s+(?P<event>[A-Z_0-9]+):\s+thread\s+(?P<thread>\d+)\s+"
    r"info\s+(?P<info>\d+)\s+(?P<state>start|stop)\s+(?P<cycles>\d+)")
OP_START_RE = re.compile(r"\bstart\s+(\d+)")


def parse_op_line(rest):
    """rest = tout apres 'profile-op '. Retourne un dict ou None si non-parsable."""
    fields = rest.split("|")
    if len(fields) < 2:
        return None
    opname = fields[0]
    tensors = fields[1] if len(fields) > 1 else ""
    last = fields[-1]
    m_usec = USEC_RE.search(last)
    if not m_usec:
        return None
    usec = int(m_usec.group(1))
    m_cyc = CYCLES_RE.search(last)
    cycles = int(m_cyc.group(1)) if m_cyc else None

    m_start = OP_START_RE.search(last)
    cycles_start = int(m_start.group(1)) if m_start else None

    if opname == "OPBATCH":
        m_n = NOPS_RE.search(rest)
        return {"kind": "batch", "n_ops": int(m_n.group(1)) if m_n else None,
                "usec": usec, "cycles": cycles, "cycles_start": cycles_start}

    # path/vtcm est l'avant-derniere section (ex: "hvx-tiled vtcm 2727936" ou "----")
    path_field = fields[-2] if len(fields) >= 3 else "----"
    path = path_field.split(" vtcm ")[0].strip() if path_field != "----" else None
    m_vtcm = re.search(r"vtcm\s+(\d+)", path_field)
    vtcm_bytes = int(m_vtcm.group(1)) if m_vtcm else None

    layers_found = [int(x) for x in LAYER_RE.findall(tensors)]
    # Une op peut referencer plusieurs tenseurs blk.N. (ex: MUL_MAT_NX fusionne
    # plusieurs projections attn du MEME layer) — verifie qu'ils concordent,
    # sinon prend le premier et le signale (ne devrait pas arriver en pratique).
    layer = layers_found[0] if layers_found else None
    inconsistent = len(set(layers_found)) > 1

    # AJOUT 2026-09-06 : capture du "fingerprint" complet demande (shape reelle,
    # type reel PAR OPERANDE — pas juste le type GGUF annonce du poids — cf.
    # discussion "Q4 poids != Q4 activation"). dims/types sont dans les champs
    # 2 et 3 (index 2/3 de la ligne source), formes "M:N x M:N -> M:N" et
    # "type x type -> type". VERIFIE EMPIRIQUEMENT (2026-09-06, grep exhaustif
    # sur prof_test_qwen09b.log) : sur CE fork/chemin GGML-HTP, l'activation
    # reste f32 (ou f16 pour le KV cache) — jamais observee en "Q8_0_TILED"
    # (0 occurrence). Cette affirmation specifique (parfois relayee) ne
    # correspond donc PAS a ce runtime — peut-etre vraie sur un chemin QNN/
    # QAIRT different, non verifie ici.
    dims = fields[2] if len(fields) > 2 else None
    types_field = fields[3] if len(fields) > 3 else None
    weight_type = None
    act_types = []
    out_type = None
    if types_field and " -> " in types_field:
        ins, out_type = types_field.split(" -> ", 1)
        operand_types = [t.strip() for t in ins.split(" x ")]
        # convention observee : le(s) premier(s) operande(s) qui matchent un
        # type de quant GGML (q4_0/q4_1/q8_0/...) sont les POIDS ; le reste
        # (f32/f16/i32/i64) sont activations/indices.
        quant_types = {"q4_0", "q4_1", "q4_k", "q5_k", "q6_k", "q8_0",
                       "q2_k", "q3_k", "iq2_xxs", "iq3_xxs", "iq1_s", "mxfp4"}
        for t in operand_types:
            if t.lower() in quant_types and weight_type is None:
                weight_type = t
            else:
                act_types.append(t)

    return {"kind": "op", "opname": opname, "layer": layer, "usec": usec,
            "cycles": cycles, "cycles_start": cycles_start,
            "path": path, "vtcm_bytes": vtcm_bytes,
            "dims": dims, "weight_type": weight_type, "act_types": act_types,
            "out_type": out_type, "n_layer_refs": len(set(layers_found)),
            "inconsistent_layer": inconsistent}


class CycleUnwrapper:
    """Reconstruit un compteur de cycles 64-bit a partir de valeurs 32-bit
    tronquees. IDENTIQUE a la classe du parseur officiel upstream
    (llama.cpp/scripts/snapdragon/ggml-hexagon-profile.py) — reprise ici
    2026-09-06 apres avoir identifie que notre premiere tentative de mapping
    trace-evt->batch par bisect naif comparait deux domaines de compteur
    incompatibles (OPBATCH ~6.5e11 vs trace-evt ~1e9). La cle : seeder avec
    initial_val decompose en (bits bas 32 = last_raw, bits hauts = high_part),
    ce qui permet de comparer des valeurs 32-bit ulterieures au meme domaine."""
    def __init__(self, initial_val=None):
        if initial_val is not None:
            self.last_raw = initial_val & 0xFFFFFFFF
            self.high_part = initial_val & 0xFFFFFFFF00000000
        else:
            self.last_raw = None
            self.high_part = 0

    def unwrap(self, raw):
        if self.last_raw is None:
            self.last_raw = raw
            return raw
        diff = raw - self.last_raw
        if diff < -0x80000000:
            self.high_part += 0x100000000
        elif diff > 0x80000000:
            self.high_part -= 0x100000000
        self.last_raw = raw
        return raw + self.high_part


def parse_log(path):
    """Retourne (events_jsonl_ready, batches) :
    - events_jsonl_ready : liste de dict {"layer","kernel","latency_us"} —
      format EXACT attendu par pm.aggregate_live_trace(), pret a serialiser.
    - batches : liste de dict {"n_ops","usec_batch","usec_ops_sum","overhead_us",
      "layer_guess"} — le "cout d'arete/orchestration" mesure directement :
      overhead_us = temps du batch NON explique par la somme des ops qu'il
      contient (dispatch FastRPC, synchronisation, latence de queue).
    """
    events = []
    batches = []
    cur_batch = None
    cur_batch_ops_usec = 0
    cur_batch_layers = []
    n_op_lines = 0
    n_unparsed = 0
    trace_evt_counts = Counter()

    # AJOUT 2026-09-06 : mapping trace-evt -> OP precis (donc par couche),
    # avec le VRAI CycleUnwrapper (identique au parseur officiel upstream) —
    # remplace la tentative naive abandonnee plus tot (bisect direct entre
    # deux domaines de cycles incompatibles, ~6.5e11 vs ~1e9). Un seul device
    # "HTP0" observe a ce jour dans nos captures ; garde une cle par device
    # par coherence avec l'officiel, sans complexifier pour du multi-device
    # jamais rencontre.
    unwrappers = {}          # device -> CycleUnwrapper (reseed a chaque OPBATCH)
    last_batch_start = {}    # device -> cycles_start brut du dernier OPBATCH
    trace_unwrappers = {}    # (device, thread) -> CycleUnwrapper
    ops_for_mapping = []     # {device, start, end, opname, layer}
    device = "HTP0"          # seul device vu dans nos captures a ce jour

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                mt = TRACE_EVT_RE.search(line)
                if mt and mt.group("state") == "start":
                    thread = int(mt.group("thread"))
                    raw_cyc = int(mt.group("cycles"))
                    key = (device, thread)
                    if key not in trace_unwrappers:
                        trace_unwrappers[key] = CycleUnwrapper(last_batch_start.get(device))
                    unwrapped = trace_unwrappers[key].unwrap(raw_cyc)
                    trace_evt_counts[mt.group("event")] += 1
                    ops_for_mapping.append({"kind": "trace", "device": device,
                                            "cyc": unwrapped, "event": mt.group("event")})
                continue
            rec = parse_op_line(m.group(1))
            if rec is None:
                n_unparsed += 1
                continue
            if rec["kind"] == "batch":
                if cur_batch is not None:
                    overhead = cur_batch["usec"] - cur_batch_ops_usec
                    batches.append({
                        "n_ops": cur_batch["n_ops"], "usec_batch": cur_batch["usec"],
                        "usec_ops_sum": cur_batch_ops_usec,
                        "overhead_us": overhead,
                        "layers": sorted(set(cur_batch_layers)),
                        "cycles_start": cur_batch.get("cycles_start"),
                        "cycles": cur_batch.get("cycles"),
                    })
                cur_batch = rec
                cur_batch_ops_usec = 0
                cur_batch_layers = []
                # Reseed l'unwrapper d'ops sur ce nouveau OPBATCH (comme
                # l'officiel) + efface les trace_unwrappers de ce device (un
                # nouveau batch = un nouveau point d'ancrage, l'ancien n'est
                # plus valide pour les traces suivantes).
                if rec.get("cycles_start") is not None:
                    unwrappers[device] = CycleUnwrapper(rec["cycles_start"])
                    last_batch_start[device] = rec["cycles_start"]
                    for k in list(trace_unwrappers.keys()):
                        if k[0] == device:
                            del trace_unwrappers[k]
                continue
            # rec["kind"] == "op"
            n_op_lines += 1
            cur_batch_ops_usec += rec["usec"]
            fingerprint_extra = {"vtcm_bytes": rec.get("vtcm_bytes"),
                                 "dims": rec.get("dims"),
                                 "weight_type": rec.get("weight_type"),
                                 "act_types": rec.get("act_types"),
                                 "out_type": rec.get("out_type")}
            if rec.get("cycles_start") is not None and device in unwrappers:
                start_c = unwrappers[device].unwrap(rec["cycles_start"])
                end_c = start_c + (rec["cycles"] or 0)
                ops_for_mapping.append({"kind": "op", "device": device,
                                        "start": start_c, "end": end_c,
                                        "opname": rec["opname"], "layer": rec["layer"]})
            if rec["layer"] is not None:
                cur_batch_layers.append(rec["layer"])
                events.append({"layer": rec["layer"], "kernel": rec["opname"],
                               "latency_us": float(rec["usec"]),
                               "path": rec["path"], **fingerprint_extra})
            else:
                # norm/embed/lm_head/sans-couche (ex operations sur KV cache
                # standalone) : garde l'evenement sans layer, aggregate_live_trace
                # l'ignore pour per_layer mais le compte dans by_kernel global.
                events.append({"kernel": rec["opname"], "latency_us": float(rec["usec"]),
                               "path": rec["path"], **fingerprint_extra})

    if cur_batch is not None:
        overhead = cur_batch["usec"] - cur_batch_ops_usec
        batches.append({
            "n_ops": cur_batch["n_ops"], "usec_batch": cur_batch["usec"],
            "usec_ops_sum": cur_batch_ops_usec, "overhead_us": overhead,
            "layers": sorted(set(cur_batch_layers)),
            "cycles_start": cur_batch.get("cycles_start"),
            "cycles": cur_batch.get("cycles"),
        })

    # Bisect trace-evt (maintenant dans le MEME domaine de cycles que les ops
    # grace au CycleUnwrapper) dans la fenetre [start,end] de chaque op, puis
    # agrege PAR COUCHE (pas par op individuel — trop de bruit, la couche est
    # l'unite deja calibree ~1 batch/couche aujourd'hui).
    ops_only = [o for o in ops_for_mapping if o["kind"] == "op" and o["layer"] is not None]
    ops_only.sort(key=lambda o: o["start"])
    starts = [o["start"] for o in ops_only]
    per_layer_engine = {}
    import bisect as _bisect
    for item in ops_for_mapping:
        if item["kind"] != "trace":
            continue
        idx = _bisect.bisect_right(starts, item["cyc"]) - 1
        if idx < 0:
            continue
        o = ops_only[idx]
        if o["start"] <= item["cyc"] <= o["end"]:
            per_layer_engine.setdefault(o["layer"], Counter())[item["event"]] += 1

    n_trace_mapped = sum(sum(c.values()) for c in per_layer_engine.values())

    return events, batches, {"n_op_lines": n_op_lines, "n_unparsed": n_unparsed,
                              "n_batches": len(batches),
                              "trace_evt_counts": dict(trace_evt_counts),
                              "per_layer_engine": {k: dict(v) for k, v in
                                                   per_layer_engine.items()},
                              "n_trace_evt_mapped": n_trace_mapped}


def render_per_layer_engine_report(per_layer_engine, n_total_trace_evt):
    """RESOLU 2026-09-06 : mapping trace-evt -> op -> couche, avec le vrai
    CycleUnwrapper (identique au parseur officiel upstream) — remplace la
    tentative naive abandonnee precedemment (bisect direct entre deux domaines
    de cycles incompatibles). Premiere mesure REELLE de la repartition
    HVX/HMX/DMA PAR COUCHE, pas seulement globale."""
    if not per_layer_engine:
        return ("[moteur par couche] aucun mapping — verifier que la capture "
                "utilise GGML_HEXAGON_PROFILE=3 (pas =1) et contient des OPBATCH "
                "avec un champ 'start'.")
    total_mapped = sum(sum(c.values()) for c in per_layer_engine.values())
    L = ["## Moteur d'execution (HVX/HMX/DMA) PAR COUCHE — mapping REEL "
        "(CycleUnwrapper)", ""]
    L.append(f"- {total_mapped}/{n_total_trace_evt} evenements mappes a une "
             f"couche precise ({total_mapped/max(n_total_trace_evt,1)*100:.1f}%)")
    L.append("")
    L.append("| couche | HVX | HMX | DMA | autres | total |")
    L.append("|---:|---:|---:|---:|---:|---:|")
    for layer in sorted(per_layer_engine.keys()):
        c = per_layer_engine[layer]
        hvx = sum(v for k, v in c.items() if k.startswith("HVX"))
        hmx = sum(v for k, v in c.items() if k.startswith("HMX"))
        dma = c.get("DMA", 0)
        total = sum(c.values())
        other = total - hvx - hmx - dma
        L.append(f"| {layer} | {hvx} | {hmx} | {dma} | {other} | {total} |")
    L.append("")
    L.append("Methode : CycleUnwrapper (identique au parseur officiel upstream "
             "llama.cpp/scripts/snapdragon/ggml-hexagon-profile.py) reseede a "
             "chaque OPBATCH, puis bisect de chaque evenement trace-evt dans la "
             "fenetre [start,end] de l'op qui l'englobe. Un evenement non mappe "
             "(cf. pourcentage ci-dessus) tombe hors de toute fenetre d'op connue "
             "— possible si le pas de temps du dernier OPBATCH avant la fin de "
             "capture n'a pas ete ferme par un OPBATCH suivant.")
    return "\n".join(L)


def render_trace_evt_report(counts):
    if not counts:
        return ("[trace-evt] aucun evenement trouve — necessite "
                "GGML_HEXAGON_PROFILE=3 (pas =1) a la capture.")
    total = sum(counts.values())
    hvx = sum(v for k, v in counts.items() if k.startswith("HVX"))
    hmx = sum(v for k, v in counts.items() if k.startswith("HMX"))
    dma = counts.get("DMA", 0)
    L = ["## Repartition HVX/HMX/DMA MESUREE (GGML_HEXAGON_PROFILE=3)", ""]
    L.append(f"- {total} evenements trace-evt captures")
    L.append(f"- **HVX (compute vectoriel) : {hvx} ({hvx/total*100:.1f}%)** — "
             f"{'domine tres largement' if hvx > hmx*10 else 'comparable a HMX'}")
    L.append(f"- **HMX (matrix accelerator) : {hmx} ({hmx/total*100:.1f}%)**")
    L.append(f"- **DMA (compteur reel, distinct du proxy VTCM) : {dma} "
             f"({dma/total*100:.1f}%)**")
    L.append("")
    L.append("| type evenement | count | part |")
    L.append("|---|---:|---:|")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        L.append(f"| {k} | {v} | {v/total*100:.1f}% |")
    L.append("")
    L.append("[LIMITE] Agregation GLOBALE sur toute la capture, pas encore "
             "mappee a un op/couche precis (necessiterait le mapping par "
             "cycle_start/cycle_stop + bisect que fait le parseur officiel "
             "upstream — non implemente ici, scope volontairement limite).")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Fiche par layer (fingerprint) — reponse concrete a la discussion 2026-09-06
# sur le "Layer Cost Vector"/architecture profileur->predicteur. Champs REMPLIS
# uniquement avec des donnees mesurees (op/shape/type/vtcm/usec, deja capturees
# ci-dessus) ; les champs qui necessiteraient une trace GPU/CPU equivalente
# (qui n'existe pas — GGML_HEXAGON_PROFILE ne couvre QUE HTP) sont marques
# explicitement "NON MESURABLE", jamais devines ni extrapoles.
# ---------------------------------------------------------------------------
def build_layer_fingerprint(events, layer):
    """Agrege les evenements REELS d'un layer donne en une fiche. events =
    liste retournee par parse_log() (contient deja dims/weight_type/
    act_types/vtcm_bytes/path/latency_us par op)."""
    ops = [e for e in events if e.get("layer") == layer]
    if not ops:
        return None
    total_us = sum(e["latency_us"] for e in ops)
    total_vtcm = sum(e.get("vtcm_bytes") or 0 for e in ops)
    weight_types = sorted(set(e["weight_type"] for e in ops if e.get("weight_type")))
    act_types = sorted(set(t for e in ops for t in (e.get("act_types") or [])))
    repack_paths = sorted(set(e["path"] for e in ops if e.get("path")))
    slowest = max(ops, key=lambda e: e["latency_us"])
    return {
        "layer": layer, "n_ops": len(ops), "total_us": total_us,
        "total_vtcm_bytes": total_vtcm, "weight_types": weight_types,
        "act_types": act_types, "repack_paths": repack_paths,
        "slowest_op": slowest["kernel"], "slowest_op_us": slowest["latency_us"],
        "slowest_op_dims": slowest.get("dims"),
        "slowest_op_weight": slowest.get("weight_type"),
        "slowest_op_act": slowest.get("act_types"),
    }


def render_layer_fingerprint_card(fp):
    if fp is None:
        return "(aucune donnee pour ce layer dans cette trace)"
    L = [f"LAYER {fp['layer']}", "─" * 40]
    L.append(f"N OPS           {fp['n_ops']} (mesure)")
    L.append(f"TOTAL US        {fp['total_us']:.0f} (mesure, somme reelle)")
    L.append(f"WEIGHT TYPES    {', '.join(fp['weight_types']) or '(aucun poids quantise identifie)'} (mesure)")
    L.append(f"ACT TYPES       {', '.join(fp['act_types']) or '?'} (mesure — PAS suppose depuis le "
             f"nom GGUF du poids, lu directement dans l'appel kernel reel)")
    L.append(f"REPACK/PATH     {', '.join(fp['repack_paths']) or '?'} (mesure — hvx-tiled/row-block/"
             f"hvx = chemins d'execution reels observes, pas un flag theorique)")
    L.append(f"VTCM CUMULE     {fp['total_vtcm_bytes']} octets (mesure — proxy DMA/pression memoire, "
             f"PAS un compteur DMA direct)")
    L.append(f"OP LA PLUS LENTE {fp['slowest_op']} {fp['slowest_op_dims']} "
             f"({fp['slowest_op_us']:.0f}us, poids={fp['slowest_op_weight']}, "
             f"act={fp['slowest_op_act']})")
    L.append("")
    L.append("HTP FITNESS      mesure directement ci-dessus (c'est le SEUL backend trace)")
    L.append("GPU FITNESS      NON MESURABLE — aucune trace OpenCL equivalente a "
             "GGML_HEXAGON_PROFILE n'existe dans ce projet a ce jour")
    L.append("CPU FITNESS      NON MESURABLE — meme limite, aucune instrumentation "
             "per-op CPU comparable captee")
    L.append("PREDICTED (autres backends) : NON PRODUIT — inventer un chiffre sans "
             "trace reelle equivalente serait une fausse precision, exactement le "
             "risque signale par le red-team de la discussion du 2026-09-06")
    return "\n".join(L)


def render_batch_overhead_report(batches):
    """C'est ICI que se trouve le 'cout d'arete' — pas une couche individuelle,
    mais le cout d'orchestration/dispatch PAR BATCH (qui correspond a ~1 layer
    d'apres la calibration existante 29 dispatchs/28 couches), mesure comme
    la difference entre le temps du batch entier et la somme de ses ops."""
    if not batches:
        return "[batch] aucun batch OPBATCH trouve dans ce log."
    overheads = [b["overhead_us"] for b in batches]
    total_batch = sum(b["usec_batch"] for b in batches)
    total_ops = sum(b["usec_ops_sum"] for b in batches)
    total_overhead = sum(overheads)
    L = ["## Cout d'orchestration/dispatch MESURE (par batch, PREMIERE mesure reelle)", ""]
    L.append(f"- {len(batches)} batches, {total_batch} us cumules (batch), "
             f"{total_ops} us cumules (somme des ops)")
    L.append(f"- **overhead total mesure : {total_overhead} us "
             f"({total_overhead/total_batch*100:.1f}% du temps total batch)**")
    L.append(f"- overhead moyen/batch : {total_overhead/len(batches):.1f} us "
             f"(min {min(overheads)}, max {max(overheads)})")
    L.append("")
    L.append("Lecture : ce n'est PAS le cout de calcul (deja compte dans usec_ops_sum) "
             "— c'est le temps de dispatch FastRPC/synchronisation/queue non explique "
             "par le calcul lui-meme. C'est la premiere mesure directe de ce terme, "
             "la ou le modele L3 (profile_model.py) supposait une constante fixe "
             "(DISPATCH_US=215us/layer) jamais verifiee sur ce point precis.")
    L.append("")
    L.append("[AXE CONFIRME 2026-09-06] prefill vs decode : mesure sur Marco-Nano-V2, "
             "meme modele — **overhead 2.3% en prefill (long prompt, batche) vs "
             "11.6% en decode (token par token)**, soit un facteur x5. Le prefill "
             "amortit bien mieux le cout de dispatch par batch (plus d'ops par "
             "dispatch), le decode le paie a chaque token. Un seul overhead moyen "
             "(485us/batch, mesure precedemment) melange les deux phases sans les "
             "distinguer — a garder separe pour toute future calibration.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="log device capture avec GGML_HEXAGON_PROFILE=1 -lv 5")
    ap.add_argument("--out", required=True, help="fichier JSONL de sortie "
                    "(--live-trace de profile_model_corrige.py)")
    ap.add_argument("--fingerprint-layer", type=int, metavar="N",
                    help="affiche la fiche complete (fingerprint) du layer N "
                         "a partir des donnees reellement mesurees")
    args = ap.parse_args()

    events, batches, stats = parse_log(args.log)
    with open(args.out, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    print(f"[parse] {stats['n_op_lines']} lignes op parsees, "
          f"{stats['n_unparsed']} non-parsables, {stats['n_batches']} batches")
    print(f"[out] {len(events)} evenements ecrits dans {args.out}")
    print()
    print(render_batch_overhead_report(batches))
    print()
    print(render_trace_evt_report(stats.get("trace_evt_counts", {})))
    print()
    print(render_per_layer_engine_report(stats.get("per_layer_engine", {}),
                                         sum(stats.get("trace_evt_counts", {}).values())))

    if args.fingerprint_layer is not None:
        print()
        fp = build_layer_fingerprint(events, args.fingerprint_layer)
        print(render_layer_fingerprint_card(fp))


if __name__ == "__main__":
    main()
