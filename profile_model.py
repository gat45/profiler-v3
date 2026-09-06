#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PROFILE_MODEL — profileur universel (GGUF | safetensors) pour SM8850/HTP.

Charge UNIQUEMENT les en-têtes (aucun poids) :
  - .gguf                    -> header GGUF (formes + types réels)
  - dossier safetensors      -> config.json + headers de chaque shard
  - repo HuggingFace (--hf)  -> config.json + index + headers (range request)

Calcule pour TOUT modèle (dense, MoE, hybride, multimodal) :
  - inventaire par famille/couche, matrices quant par tenseur
  - tailles de déploiement (F16..IQ2) vs budget RAM
  - actifs/token, trafic décode, ms/token GGML vs QAIRT, t/s estimés
  - KV cache, lm_head, coût orchestration MoE (ops × 215 µs)
  - plan quant par tenseur sous contrainte RAM (floors par famille)

Calibration device (mesurée 2026-08/09) :
  BW_eff GGML ~30 Go/s (d2_layer_bytes 32.4) · QAIRT 74 Go/s
  Qwen 9B dense : 11.26 t/s @ 5.06 Go/token -> BW_eff 57 Go/s
  Marco MoE     : 42.1  t/s @ 0.34 Go/token -> BW_eff 14.3 Go/s
  fixe par-op HTP ~215 µs (ARGSORT, structurel)

Usage :
  py profile_model.py modele.gguf
  py profile_model.py <dossier_safetensors> [--shards a.bin,b.bin]
  py profile_model.py --hf google/gemma-4-26B-A4B
  [--budget-gb 9.0] [--ctx 4096] [--out DIR] [--json]
"""
import argparse
import json
import os
import re
import struct
import sys
import urllib.request
import urllib.error
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Calibration SM8850 (mesures device)
# ---------------------------------------------------------------------------
GGML_BW = 30.0            # Go/s calibré GGML (borne basse d2_layer_bytes)
GGML_BW_HI = 32.4         # Go/s calibration nominale
QAIRT_BW = 74.0           # Go/s QAIRT estimé
BW_EFF_DENSE = 57.0       # Qwen 9B dense mesuré (11.26 t/s x 5.06 Go)
BW_EFF_MOE = 14.3         # Marco MoE mesuré (42.1 t/s x 0.34 Go)
FIXE_PAR_OP_MS = 0.215    # coût fixe par-op HTP (ms, ARGSORT mesuré)
DEFAULT_RAM_BUDGET_GB = 9.0

# Octets par poids par format GGUF (block_bytes / block_weights)
BPW = {
    "F16": 2.0, "Q8_0": 34/32, "Q6_K": 210/256, "Q5_K": 176/256,
    "Q4_K": 144/256, "Q4_0": 18/32, "Q3_K": 110/256, "Q2_K": 84/256,
    "IQ3_XXS": 120/256, "IQ2_XXS": 66/256, "IQ1_S": 50/256,
}
ORDER = ["F16", "Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q4_0",
         "Q3_K", "Q2_K", "IQ3_XXS", "IQ2_XXS", "IQ1_S"]

# Familles (ordre = priorité de classification)
FAMILY_RULES = [
    ("vision", ("vision", "embed_vision")),
    ("lm_head", ("lm_head",)),
    ("embed", ("token_embd", "embed_tokens", "wte")),
    ("norm", ("norm", "layer_scalar", "rms", "scale", "gamma")),
    ("moe", ("exps", "experts", "router", "gate_up_proj", "per_expert")),
    ("attn", ("attn", "self_attn", "q_proj", "k_proj", "v_proj", "o_proj",
              "wq", "wk", "wv", "wo", "c_attn", "c_proj")),
    ("mlp", ("mlp", "ffn", "gate_proj", "up_proj", "down_proj", "feed_forward")),
    ("ssm", ("ssm", "conv1d", "a_proj", "d_proj", "dt_proj",
             "bc_proj", "bd_proj", "x_proj", "in_proj", "out_proj")),
    ("mtp", ("mtp",)),
]

# Floors de précision par famille (plancher de sécurité qualité)
FLOOR_WBITS = {"norm": 16, "lm_head": 8, "embed": 8, "attn": 4,
               "mlp": 4, "moe": 4, "ssm": 4, "router": 8, "other": 8}
WBITS = {"F16": 16, "Q8_0": 8, "Q6_K": 6, "Q5_K": 5, "Q4_K": 4, "Q4_0": 4,
         "Q3_K": 3, "Q2_K": 2, "IQ3_XXS": 3, "IQ2_XXS": 2, "IQ1_S": 1}

GGUF_MAGIC = 0x46554747
V_TYPES = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",6:"Q5_0",7:"Q5_1",8:"Q8_0",
           9:"Q8_1",10:"Q2_K",11:"Q3_K",12:"Q4_K",13:"Q5_K",14:"Q6_K",
           15:"Q8_K",16:"IQ2_XXS",17:"IQ2_XS",18:"Q3_K_XS",19:"IQ3_XXS",
           20:"IQ1_S",21:"IQ4_NL",22:"IQ3_S",23:"IQ2_S",24:"IQ4_XS",
           25:"I8",26:"I1",27:"I64",28:"I32",30:"MXFP4"}
BLOCK_BYTES = {2:18, 3:20, 6:22, 7:24, 8:34,
               10:84, 11:110, 12:144, 13:176, 14:210, 15:260,
               16:66, 17:84, 18:180, 19:120, 20:50, 21:40, 22:140, 23:70,
               24:88, 30:34}
BLOCK_WEIGHTS = {2:32, 3:32, 6:32, 7:32, 8:32, 30:32}
TYPE_SIZES = {0:4, 1:2, 25:1, 26:1, 27:8, 28:4}


def ggml_type_size(t):
    if t in TYPE_SIZES:
        return TYPE_SIZES[t]
    if t in BLOCK_BYTES:
        return BLOCK_BYTES[t] / BLOCK_WEIGHTS.get(t, 256)
    raise ValueError(f"type ggml inconnu {t}")


# ---------------------------------------------------------------------------
# Chargeurs : GGUF | safetensors local | HF distant (headers only)
# ---------------------------------------------------------------------------
def read_gguf_header(path):
    f = open(path, "rb")
    try:
        return _read_gguf_header_inner(f)
    finally:
        f.close()


def _read_gguf_header_inner(f):
    magic, ver, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
    if magic != GGUF_MAGIC:
        raise ValueError("pas un GGUF")

    def rstr():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")

    def rval(vtype):
        if vtype == 0: return struct.unpack("<B", f.read(1))[0]
        if vtype == 1: return struct.unpack("<b", f.read(1))[0]
        if vtype == 2: return struct.unpack("<H", f.read(2))[0]
        if vtype == 3: return struct.unpack("<h", f.read(2))[0]
        if vtype == 4: return struct.unpack("<I", f.read(4))[0]
        if vtype == 5: return struct.unpack("<i", f.read(4))[0]
        if vtype == 6: return struct.unpack("<f", f.read(4))[0]
        if vtype == 7: return struct.unpack("<B", f.read(1))[0]
        if vtype == 8: return rstr()
        if vtype == 9:
            et, cnt = struct.unpack("<IQ", f.read(12))
            if et == 8:
                return [rstr() for _ in range(cnt)]
            # Tableaux numériques (ex. head_count_kv, sliding_window_pattern,
            # token_type) : on décode les valeurs réelles au lieu du placeholder
            # "<array NxE>" — requis pour les archis per-layer (gemma4).
            sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1}[et]
            fmt = {0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<?"}[et]
            return [struct.unpack(fmt, f.read(sz))[0] for _ in range(cnt)]
        if vtype == 10: return struct.unpack("<Q", f.read(8))[0]
        if vtype == 11: return struct.unpack("<q", f.read(8))[0]
        if vtype == 12: return struct.unpack("<d", f.read(8))[0]
        raise ValueError(f"kv type {vtype}")

    meta = {}
    for _ in range(n_kv):
        k = rstr()
        meta[k] = rval(struct.unpack("<I", f.read(4))[0])
    tensors = []
    for _ in range(n_tensors):
        name = rstr()
        ndims = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(ndims)]
        ttype = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        n = 1
        for d in dims:
            n *= d
        try:
            nbytes = n * ggml_type_size(ttype)
        except ValueError:
            # type ggml inconnu (quant récente non cataloguée) : ne pas
            # planter tout le fichier, estimer avec 4.5 bit/poids moyen
            # et marquer le tenseur comme ESTIMÉ.
            print(f"  [warn] type ggml {ttype} inconnu pour '{name}' — "
                  f"taille ESTIMÉE (ajouter à BLOCK_BYTES/V_TYPES pour "
                  f"une valeur exacte)")
            nbytes = n * 4.5 / 8.0
            ttype = -abs(ttype) - 1000  # marqueur "estimé" côté V_TYPES.get -> "?"
        tensors.append((name, dims, ttype, nbytes))
    return meta, tensors


def parse_safetensors_header(data):
    n = struct.unpack("<Q", data[:8])[0]
    hdr = json.loads(data[8:8 + n])
    out = {}
    for name, info in hdr.items():
        if name == "__metadata__":
            continue
        ne = 1
        for d in info["shape"]:
            ne *= d
        dtype = info.get("dtype", "BF16")
        nb = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8}.get(dtype, 2)
        out[name] = {"dtype": dtype, "shape": info["shape"],
                     "bytes": ne * nb, "elems": ne}
    return out


def load_safetensors_local(directory, shards_override=None):
    cfg = {}
    cfg_path = os.path.join(directory, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
    shards = shards_override or []
    if not shards:
        for pat in ("model-*.safetensors", "*.safetensors", "*_head.bin"):
            pat_re = "^" + re.escape(pat).replace("\\*", ".*") + "$"
            hits = sorted([os.path.join(directory, x) for x in os.listdir(directory)
                           if re.match(pat_re, x)])
            if hits:
                shards = hits
                break
    tensors = {}
    for s in shards:
        with open(s, "rb") as f:
            head = f.read(8)
            ln = struct.unpack("<Q", head)[0]
            f.seek(0)
            data = f.read(8 + min(ln, 200_000_000))
        tensors.update(parse_safetensors_header(data))
    return cfg, tensors, [os.path.basename(s) for s in shards]


def _http_get(url, binary=True):
    req = urllib.request.Request(url, headers={"User-Agent": "profile-model"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def load_hf(repo_id):
    base = f"https://huggingface.co/{repo_id}/resolve/main"
    cfg = json.loads(_http_get(f"https://huggingface.co/{repo_id}/raw/main/config.json"))
    try:
        index = json.loads(_http_get(f"{base}/model.safetensors.index.json"))
        wm = index.get("weight_map", {})
        shard_names = sorted(set(wm.values()))
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        # repo non-shardé : un seul model.safetensors
        print("  [hf] pas d'index (repo non-shardé) -> model.safetensors unique")
        shard_names = ["model.safetensors"]
    tensors = {}
    for sn in shard_names:
        # range request en 2 temps : 8 octets de longueur, puis juste le header
        req = urllib.request.Request(f"{base}/{sn}",
                                     headers={"User-Agent": "profile-model",
                                              "Range": "bytes=0-7"})
        with urllib.request.urlopen(req, timeout=60) as r:
            head = r.read()
        ln = struct.unpack("<Q", head)[0]
        req = urllib.request.Request(f"{base}/{sn}",
                                     headers={"User-Agent": "profile-model",
                                              "Range": f"bytes=0-{7 + ln}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        tensors.update(parse_safetensors_header(data))
        print(f"  [hf] header {sn} : {len([k for k in tensors])} tensors cumulés")
    return cfg, tensors, shard_names


# ---------------------------------------------------------------------------
# Classification générique
# ---------------------------------------------------------------------------
def family_of(name):
    for fam, keys in FAMILY_RULES:
        if any(k in name for k in keys):
            return fam
    return "other"


def classify(name):
    low = name.lower()
    if re.fullmatch(r"output\.weight", name) or low == "lm_head.weight":
        fam = "lm_head"
    else:
        fam = family_of(name)
    is_moe = "exps" in low or "experts" in low or "gate_up_proj" in low
    m = re.search(r"(?:blk|layers|mtp\.layers)\.(\d+)\.", name)
    layer = int(m.group(1)) if m else None
    return {"family": fam, "layer": layer, "moe": is_moe, "name": name}


# ---------------------------------------------------------------------------
# Analyse
# ---------------------------------------------------------------------------
def analyze(cfg, tensors, gguf_meta=None, shard_names=None, is_gguf=False, ctx=4096):
    # ---- Architecture (config ou métadonnées GGUF) ----
    tc = cfg.get("text_config", cfg) if isinstance(cfg, dict) else {}
    vc = cfg.get("vision_config", {}) if isinstance(cfg, dict) else {}
    # clés GGUF préfixées par l'archi (ex qwen3moe.block_count) -> générique
    gm = dict(gguf_meta or {})
    arch = gm.get("general.architecture", "")
    if arch:
        for src in (f"{arch}.", "llama."):
            for dst_key, src_key in (("block_count", "block_count"),
                                     ("embedding_length", "embedding_length"),
                                     ("expert_count", "expert_count"),
                                     ("expert_used_count", "expert_used_count"),
                                     ("vocab_size", "vocab_size"),
                                     ("attn_heads", "attention.head_count"),
                                     ("attn_kv", "attention.head_count_kv"),
                                     ("key_length", "attention.key_length"),
                                     ("value_length", "attention.value_length"),
                                     ("sliding_window", "attention.sliding_window")):
                if gm.get(src + src_key) is not None:
                    gm[dst_key] = gm.get(src + src_key)
            if "block_count" in gm:
                break
    def _scalar(v):
        """Réduit une métadonnée GGUF éventuellement per-layer (liste) à un
        scalaire : gemma4 stocke head_count_kv par layer (ex 2/8)."""
        if isinstance(v, list):
            nums = [x for x in v if isinstance(x, (int, float))]
            return max(nums) if nums else 0
        return v if isinstance(v, (int, float)) else 0

    n_layer = tc.get("num_hidden_layers") or tc.get("n_layer") or gm.get("block_count") or 0
    hidden = tc.get("hidden_size") or tc.get("n_embd") or gm.get("embedding_length") or 0
    n_experts = tc.get("num_experts") or gm.get("expert_count") or 0
    top_k = tc.get("top_k_experts") or gm.get("expert_used_count") or 0
    vocab = tc.get("vocab_size") or gm.get("vocab_size") or 0
    if not vocab and gguf_meta:
        # tokenizer.ggml.tokens (list) ou token_type (array n x 5)
        toks = gguf_meta.get("tokenizer.ggml.tokens")
        if isinstance(toks, list):
            vocab = len(toks)
        tt = gguf_meta.get("tokenizer.ggml.token_type")
        if isinstance(tt, list):
            vocab = len(tt) if not vocab else vocab
        elif isinstance(tt, str) and tt.startswith("<array"):
            m = re.match(r"<array (\d+)x", tt)
            if m and not vocab:
                vocab = int(m.group(1))
    tied = tc.get("tie_word_embeddings", False)
    layer_types = tc.get("layer_types", []) or []
    window = tc.get("sliding_window") or gm.get("sliding_window") or 0
    n_heads = tc.get("num_attention_heads") or gm.get("attn_heads") or 0
    n_kv = _scalar(tc.get("num_key_value_heads") or gm.get("attn_kv") or n_heads)
    if not isinstance(n_kv, (int, float)) or n_kv == 0:
        n_kv = n_heads
    head_dim = tc.get("head_dim") or gm.get("key_length") or 0
    if not head_dim and hidden and n_heads:
        head_dim = hidden // n_heads
    is_moe = n_experts > 0

    full_attn = {i for i, t in enumerate(layer_types) if t == "full_attention"}
    sliding_attn = {i for i, t in enumerate(layer_types) if t == "sliding_attention"}
    # gemma4 GGUF : pas de layer_types, mais sliding_window_pattern per-layer
    # (False = full attention, True = sliding) + head_count_kv per-layer.
    # La clé GGUF est préfixée par l'archi (ex. gemma4.attention.*).
    if not layer_types:
        pat = (gm.get("sliding_window_pattern")
               or gm.get(f"{arch}.attention.sliding_window_pattern")
               or gm.get("llama.attention.sliding_window_pattern"))
        if isinstance(pat, list):
            full_attn = {i for i, v in enumerate(pat) if not v}
            sliding_attn = {i for i, v in enumerate(pat) if bool(v)}

    # ---- Inventaire ----
    rows = []
    for name, info in tensors.items():
        r = classify(name)
        r["elems"] = info.get("elems", 0)
        if is_gguf:
            # GGUF : octets réels du fichier (type ggml) + équivalent BF16
            r["bytes_real"] = info["bytes"]
            r["bytes_bf16"] = r["elems"] * 2.0
            r["gguf_type"] = V_TYPES.get(info.get("ttype", -1), "?")
        else:
            r["bytes_bf16"] = info.get("bytes", r["elems"] * 2.0)
        rows.append(r)

    total_bf16 = sum(r["bytes_bf16"] for r in rows)
    total_real = sum(r.get("bytes_real", r["bytes_bf16"]) for r in rows)
    # BF16 total indépendant : elems × 2 (même pour GGUF quantisé)
    total_bf16 = sum(r["elems"] * 2.0 for r in rows)

    # ---- Par famille ----
    by_fam = defaultdict(lambda: [0, 0.0])
    for r in rows:
        by_fam[r["family"]][0] += 1
        by_fam[r["family"]][1] += r["bytes_bf16"]

    # ---- Par layer ----
    by_layer = defaultdict(lambda: [0, 0.0])
    for r in rows:
        if r["layer"] is not None and r["family"] != "vision":
            by_layer[r["layer"]][0] += 1
            by_layer[r["layer"]][1] += r["bytes_bf16"]

    # ---- Actifs/token (decode) ----
    # attn + mlp dense (toutes les couches) + router + top-k experts + lm_head
    def fam_bytes(fam):
        return by_fam.get(fam, [0, 0.0])[1]

    attn_b = fam_bytes("attn"); mlp_b = fam_bytes("mlp"); moe_b = fam_bytes("moe")
    router_b = sum(r["bytes_bf16"] for r in rows if "router" in r["name"])
    lm_b = fam_bytes("lm_head")
    # tied : pas d'output.weight séparé -> lm_head = embed_tokens
    has_output = any(r["name"] == "output.weight" for r in rows)
    if not lm_b and (tied or not has_output) and fam_bytes("embed"):
        lm_b = fam_bytes("embed")
    norm_b = fam_bytes("norm"); ssm_b = fam_bytes("ssm")

    active_bf16 = attn_b + mlp_b + ssm_b + norm_b + router_b + lm_b
    if is_moe and moe_b and top_k:
        active_bf16 += moe_b * top_k / n_experts

    # ---- Tailles de déploiement par format ----
    sizes = {}
    for fmt in ORDER:
        tot = sum(r["bytes_bf16"] * BPW[fmt] / 2.0 for r in rows)
        sizes[fmt] = tot / 2**30
    # tailles texte seul (sans vision)
    text_rows = [r for r in rows if r["family"] != "vision"]
    sizes_text = {}
    for fmt in ORDER:
        tot = sum(r["bytes_bf16"] * BPW[fmt] / 2.0 for r in text_rows)
        sizes_text[fmt] = tot / 2**30

    # ---- Trafic/token + débit ----
    def traffic_gb(fmt):
        b = active_bf16 * BPW[fmt] / 2.0
        return b / 2**30

    def tps_from(traffic_gib, bw):
        if traffic_gib <= 0 or bw <= 0:
            return 0.0
        return bw / (traffic_gib * 1.0737)

    res = {}
    for fmt in ORDER:
        tg = traffic_gb(fmt)
        res[fmt] = {
            "traffic_gib": tg,
            "t_ggml_ms": tg * 1.0737 / GGML_BW * 1000,
            "t_ggml_hi_ms": tg * 1.0737 / GGML_BW_HI * 1000,
            "t_qairt_ms": tg * 1.0737 / QAIRT_BW * 1000,
            "tps_ggml": tps_from(tg, GGML_BW),
            "tps_qairt": tps_from(tg, QAIRT_BW),
            "tps_dense_like": tps_from(tg, BW_EFF_DENSE),
            "tps_moe_like": tps_from(tg, BW_EFF_MOE),
        }

    # ---- KV cache ----
    kv_per_token = 0.0
    if n_layer and n_kv and head_dim:
        # 2 octets fp16, k+v = 2 tenseurs
        kv_per_token = n_layer * n_kv * head_dim * 2 * 2  # octets/token
        kv_ctx4096 = kv_per_token * 4096 / 2**20
    else:
        kv_ctx4096 = 0.0
        kv_per_token = 0.0

    # ---- Orchestration MoE (ops/token x fixe HTP) ----
    ops_per_token = n_layer * 4                       # attn q/k/v/o
    if is_moe:
        ops_per_token += n_layer * (1 + top_k)        # router + experts
    else:
        ops_per_token += n_layer * 3                  # mlp gate/up/down
    orchestration_ms = ops_per_token * FIXE_PAR_OP_MS

    a = {
        "arch": {
            "n_layer": n_layer, "hidden": hidden, "vocab": vocab,
            "n_experts": n_experts, "top_k": top_k, "is_moe": is_moe,
            "tied": tied, "n_heads": n_heads, "n_kv": n_kv,
            "head_dim": head_dim, "full_attn": sorted(full_attn),
            "sliding_attn": sorted(sliding_attn), "window": window,
            "has_vision": bool(vc) or any(r["family"] == "vision" for r in rows),
        },
        "total_bf16_gib": total_bf16 / 2**30,
        "total_real_gib": total_real / 2**30,
        "by_family": {k: [v[0], v[1] / 2**30] for k, v in by_fam.items()},
        "by_layer": {str(k): v[1] / 2**30 for k, v in by_layer.items()},
        "n_tensors": len(rows),
        "active_bf16_gib": active_bf16 / 2**30,
        "active_bf16_b": active_bf16 / 2 / 1e9,
        "sizes_gib": sizes, "sizes_text_gib": sizes_text,
        "formats": res,
        "kv_per_token_kb": kv_per_token / 1024,
        "kv_ctx4096_mb": kv_ctx4096,
        "ops_per_token": ops_per_token,
        "orchestration_ms": orchestration_ms,
        "lm_head_gib": lm_b / 2**30,
        "shard_names": shard_names or [],
        "is_gguf": is_gguf,
    }
    a = {**a, "l3": simulate_l3(a, ctx)}
    return a


# ---------------------------------------------------------------------------
# NIVEAU 3 — Simulateur Hextimate/VTCM (reverse Qualcomm, datavorous + d2-planner)
# ---------------------------------------------------------------------------
# Constantes hardware par défaut = calibration SM8850 (calib_constants.py du
# workspace + RE datavorous). Surchageables via --device-config <json> pour
# calibrer un autre device Hexagon sans toucher au code.
DEVICE_DEFAULTS = {
    "GGML_BW": 30.0,            # Go/s calibré GGML (borne basse d2_layer_bytes)
    "GGML_BW_HI": 32.4,         # Go/s calibration nominale
    "QAIRT_BW": 74.0,           # Go/s QAIRT estimé
    "BW_EFF_DENSE": 57.0,       # Qwen 9B dense mesuré (11.26 t/s x 5.06 Go)
    "BW_EFF_MOE": 14.3,         # Marco MoE mesuré (42.1 t/s x 0.34 Go)
    "FIXE_PAR_OP_MS": 0.215,    # coût fixe par-op HTP (ms, ARGSORT mesuré)
    "VTCM_CAPACITY_MB": 8.0,    # scratchpad Hexagon (blob QAIRT réel : 8 MiB)
    "VTCM_THRESHOLD": 0.75,     # seuil activation spill/fill observé (~6 Mo)
    "PEAK_COMPUTE_TFLOPS": 61.65,  # HTP v81 INT8 (estimé Gen4x1.37)
    "BW_EFF_GBS": 50.0,         # BW effective decode mesurée terrain
    "DISPATCH_US": 215.0,       # fixe par-op FastRPC mesuré (ARGSORT 215 µs)
    "THERMAL_TPS_FACTOR": 0.65,  # après ~120 s charge continue
    # Coefficient MUL_MAT_ID ∝ hidden (validation 2026-09-03, pr28202 HTP) :
    #   Marco-Nano Q4_0 h1024 : p50 43 µs/op · Gemma-26B-A4B Q2_K h2816 : p50
    #   143 µs/op → p50 ≈ 43 + (hidden-1024) × 0.0558 µs (≈ +2.75× hidden →
    #   ~×3.3 mesuré, le surcroît Q2_K vs Q4_0 est dans MMID_US_AT_HIDDEN).
    "MMID_US_AT_HIDDEN_1024": 43.0,   # p50 MUL_MAT_ID @ hidden 1024 (Marco)
    "MMID_US_PER_HIDDEN": 0.0558,     # pente µs par unité de hidden au-delà de 1024
    "MMID_N_OPS_DECODE": 1,           # ops expert/token par layer en decode
}
DEVICE_NAME = "SM8850 (défaut intégré)"

# Copie mutable utilisée par le reste du module — modifiée par
# load_device_profile(). Garder les mêmes noms que les anciennes globales
# pour ne rien casser côté simulate_l3()/render().
GGML_BW = DEVICE_DEFAULTS["GGML_BW"]
GGML_BW_HI = DEVICE_DEFAULTS["GGML_BW_HI"]
QAIRT_BW = DEVICE_DEFAULTS["QAIRT_BW"]
BW_EFF_DENSE = DEVICE_DEFAULTS["BW_EFF_DENSE"]
BW_EFF_MOE = DEVICE_DEFAULTS["BW_EFF_MOE"]
FIXE_PAR_OP_MS = DEVICE_DEFAULTS["FIXE_PAR_OP_MS"]
VTCM_CAPACITY_MB = DEVICE_DEFAULTS["VTCM_CAPACITY_MB"]
VTCM_THRESHOLD = DEVICE_DEFAULTS["VTCM_THRESHOLD"]
PEAK_COMPUTE_TFLOPS = DEVICE_DEFAULTS["PEAK_COMPUTE_TFLOPS"]
BW_EFF_GBS = DEVICE_DEFAULTS["BW_EFF_GBS"]
DISPATCH_US = DEVICE_DEFAULTS["DISPATCH_US"]
THERMAL_TPS_FACTOR = DEVICE_DEFAULTS["THERMAL_TPS_FACTOR"]
MMID_US_AT_HIDDEN_1024 = DEVICE_DEFAULTS["MMID_US_AT_HIDDEN_1024"]
MMID_US_PER_HIDDEN = DEVICE_DEFAULTS["MMID_US_PER_HIDDEN"]
MMID_N_OPS_DECODE = DEVICE_DEFAULTS["MMID_N_OPS_DECODE"]
DRAM_TRAFFIC_MIN, DRAM_TRAFFIC_MAX = 1.0, 33.0


def load_device_profile(path):
    """Charge un JSON {constante: valeur} et écrase les globales du module.
    Clés acceptées = clés de DEVICE_DEFAULTS. Clés absentes -> valeur SM8850
    conservée. Retourne le dict effectivement appliqué (pour le rapport)."""
    global GGML_BW, GGML_BW_HI, QAIRT_BW, BW_EFF_DENSE, BW_EFF_MOE, \
        FIXE_PAR_OP_MS, VTCM_CAPACITY_MB, VTCM_THRESHOLD, \
        PEAK_COMPUTE_TFLOPS, BW_EFF_GBS, DISPATCH_US, THERMAL_TPS_FACTOR, \
        MMID_US_AT_HIDDEN_1024, MMID_US_PER_HIDDEN, MMID_N_OPS_DECODE, \
        DEVICE_NAME
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    unknown = set(cfg) - set(DEVICE_DEFAULTS) - {"device_name"}
    if unknown:
        print(f"  [device-config] clés ignorées (inconnues) : {sorted(unknown)}")
    applied = dict(DEVICE_DEFAULTS)
    for k in DEVICE_DEFAULTS:
        if k in cfg:
            applied[k] = float(cfg[k])
    DEVICE_NAME = cfg.get("device_name", os.path.basename(path))
    GGML_BW = applied["GGML_BW"]; GGML_BW_HI = applied["GGML_BW_HI"]
    QAIRT_BW = applied["QAIRT_BW"]; BW_EFF_DENSE = applied["BW_EFF_DENSE"]
    BW_EFF_MOE = applied["BW_EFF_MOE"]; FIXE_PAR_OP_MS = applied["FIXE_PAR_OP_MS"]
    VTCM_CAPACITY_MB = applied["VTCM_CAPACITY_MB"]
    VTCM_THRESHOLD = applied["VTCM_THRESHOLD"]
    PEAK_COMPUTE_TFLOPS = applied["PEAK_COMPUTE_TFLOPS"]
    BW_EFF_GBS = applied["BW_EFF_GBS"]; DISPATCH_US = applied["DISPATCH_US"]
    THERMAL_TPS_FACTOR = applied["THERMAL_TPS_FACTOR"]
    MMID_US_AT_HIDDEN_1024 = applied["MMID_US_AT_HIDDEN_1024"]
    MMID_US_PER_HIDDEN = applied["MMID_US_PER_HIDDEN"]
    MMID_N_OPS_DECODE = applied["MMID_N_OPS_DECODE"]
    return applied


def spill_fill_factor(working_set_mb):
    """Équation du RE (datavorous) : 1.0 si ≤ seuil VTCM, sinon 2.0+overflow.
    S'applique aux ACTIVATIONS qui ne tiennent pas dans le scratchpad —
    jamais aux poids (toujours streamés DDR → VTCM par op, chemin normal)."""
    thresh = VTCM_CAPACITY_MB * VTCM_THRESHOLD
    if working_set_mb <= thresh:
        return 1.0, 0.0
    overflow = (working_set_mb - thresh) / working_set_mb
    return min(2.0 + overflow, 3.0), overflow


def simulate_l3(a, ctx):
    """Per-layer : poids Q4 streamés DDR, KV, activations (spill si > VTCM),
    roofline, dispatch calibré Marco (29 ops/token mesurées ≈ 1/layer),
    thermique → latence totale + t/s (borne L3)."""
    arch = a["arch"]
    n = max(arch["n_layer"], 1)
    rows = []
    total_ms = 0.0
    total_flops = 0.0
    total_bytes = 0.0
    for L in range(n):
        # bytes BF16 actifs de la couche (attn + mlp + topk experts)
        # NOTE: by_family stocke des GiB -> *2**30 pour revenir en octets
        fam = a["by_family"]
        attn = fam.get("attn", [0, 0])[1] * 2**30 / n
        mlp = fam.get("mlp", [0, 0])[1] * 2**30 / n
        moe = fam.get("moe", [0, 0])[1] * 2**30 / n
        n_exp = max(arch["n_experts"], 1)
        topk = arch["top_k"] or 1
        moe_act = moe * min(topk / n_exp, 1.0) if arch["is_moe"] else 0.0
        bf16_act = attn + mlp + moe_act
        q4_bytes = bf16_act * BPW["Q4_0"] / 2.0          # octets Q4_0

        # flops decode (GEMV 1 token : 2×N poids)
        flops = bf16_act * 2.0
        total_flops += flops

        # KV de la couche : full = ctx entier, sliding = min(ctx, window)
        kv_dim = arch["n_kv"] * (arch["head_dim"] or 1)
        if L in arch["sliding_attn"] and arch["window"]:
            k = min(ctx, arch["window"])
        else:
            k = ctx
        kv_bytes = 2 * kv_dim * k * 2.0                    # K+V, fp16

        # Activations decode : 1 token, hidden fp16 -> minuscules (≈ KB)
        act_bytes = (arch["hidden"] or 2816) * 2 * 4      # 4 buffers intermédiaires
        ws_act_mb = act_bytes / 1e6
        sf, overflow = spill_fill_factor(ws_act_mb)        # ≈ 1.0 en decode

        # Roofline (équations Hextimate : max(compute, memory) + dispatch)
        compute_ms = flops / (PEAK_COMPUTE_TFLOPS * 1e12) * 1000.0
        # poids streamés 1× + KV lus : PAS de multiplicateur spill sur les poids
        mem_ms = (q4_bytes + kv_bytes) / (BW_EFF_GBS * 1e9) * 1000.0
        mem_ms_spill = mem_ms + act_bytes / (BW_EFF_GBS * 1e9) * 1000.0 * sf
        total_bytes += q4_bytes + kv_bytes

        # dispatch calibré : Marco réel = 29 ops/token / 28 layers ≈ 1 op/layer
        # (batch HTP OPBATCH=512 → coût par-op 215 µs mesuré ARGSORT)
        ops = 1
        dispatch_ms = ops * DISPATCH_US / 1000.0

        # MUL_MAT_ID (experts MoE) : coût expert par architecture (validation
        # 2026-09-03 Marco vs Gemma, p50). Le kernel expert s'ajoute AU-dessus
        # du dispatch générique du layer (les ~730 RPC/token = orchestration
        # de TOUTES les micro-ops, pas seulement l'expert) : on le modélise
        # donc comme terme ADDITIF séparé, jamais en remplacement du dispatch.
        # p50 ≈ MMID_US_AT_HIDDEN_1024 + (hidden−1024)×MMID_US_PER_HIDDEN µs/op
        # (Marco h1024 → 43 µs ; Gemma h2816 → 143 µs, ×~3.3 pour ×2.75 hidden,
        # surcroît Q2_K vs Q4_0 capté dans le terme constant). En dense 0.
        mmid_us_op = MMID_US_AT_HIDDEN_1024
        if arch.get("hidden"):
            mmid_us_op += max(arch["hidden"] - 1024, 0) * MMID_US_PER_HIDDEN
        mmid_ms = 0.0
        if arch.get("is_moe") and arch.get("top_k"):
            mmid_ms = MMID_N_OPS_DECODE * mmid_us_op / 1000.0

        lat = max(compute_ms, mem_ms_spill) + dispatch_ms + mmid_ms
        total_ms += lat
        rows.append({
            "layer": L,
            "type": "full" if L in arch["full_attn"] else
                   ("sliding" if L in arch["sliding_attn"] else "dense"),
            "bf16_act_mb": bf16_act / 2**20,
            "q4_mb": q4_bytes / 2**20,
            "flops_m": flops / 1e6,
            "kv_mb": kv_bytes / 2**20,
            "act_mb": act_bytes / 1e6,
            "spill": sf,
            "overflow": overflow,
            "compute_ms": compute_ms,
            "mem_ms": mem_ms,
            "mem_ms_spill": mem_ms_spill,
            "dispatch_ms": dispatch_ms,
            "mmid_ms": mmid_ms,
            "mmid_us_op": mmid_us_op if arch.get("is_moe") else None,
            "lat_ms": lat,
        })
    # ---- lm_head : plafond indépendant (vocab 262144 tied) ----
    lm_bf16 = a.get("lm_head_gib", 0.0) * 2**30
    if lm_bf16 > 0:
        lm_q4 = lm_bf16 * BPW["Q4_0"] / 2.0
        lm_flops = lm_bf16 * 2.0
        lm_mem_ms = lm_q4 / (BW_EFF_GBS * 1e9) * 1000.0
        lm_dispatch = DISPATCH_US / 1000.0          # 1 op (top-k/vocab)
        lm_lat = max(lm_flops / (PEAK_COMPUTE_TFLOPS * 1e12) * 1000.0,
                     lm_mem_ms) + lm_dispatch
        total_flops += lm_flops
        total_bytes += lm_q4
        total_ms += lm_lat
        rows.append({
            "layer": "lm_head", "type": "tied",
            "bf16_act_mb": lm_bf16 / 2**20, "q4_mb": lm_q4 / 2**20,
            "flops_m": lm_flops / 1e6, "kv_mb": 0.0, "act_mb": 0.0,
            "spill": 1.0, "overflow": 0.0,
            "compute_ms": lm_flops / (PEAK_COMPUTE_TFLOPS * 1e12) * 1000.0,
            "mem_ms": lm_mem_ms, "mem_ms_spill": lm_mem_ms,
            "dispatch_ms": lm_dispatch, "lat_ms": lm_lat,
        })
    cold_tps = 1000.0 / total_ms if total_ms > 0 else 0.0
    return {
        "rows": rows,
        "total_ms": total_ms,
        "total_flops_m": total_flops / 1e6,
        "total_bytes_gb": total_bytes / 1e9,
        "cold_tps": cold_tps,
        "thermal_tps": cold_tps * THERMAL_TPS_FACTOR,
        "bottleneck": "MEMORY" if total_bytes / max(total_flops, 1) > \
            1.0 / (PEAK_COMPUTE_TFLOPS * 1e12 / (BW_EFF_GBS * 1e9)) else "COMPUTE",
        "spill_layers": sum(1 for r in rows if r["spill"] > 1.0),
    }


# ---------------------------------------------------------------------------
# Plan quant sous contrainte RAM (floors par famille)
# ---------------------------------------------------------------------------
def quant_plan(a, budget_gb):
    """Choisit le format le plus petit par famille qui tient dans le budget."""
    plan = {}
    totals = {}
    fams = sorted(a["by_family"].keys())
    # Ordre de dégradation : descendre les familles les plus lourdes d'abord
    weight_order = sorted(fams, key=lambda f: -a["by_family"][f][1])
    for budget_name, budget in (("RAM dispo", budget_gb),
                                ("Q4_0 uniforme", a["sizes_gib"]["Q4_0"])):
        pass
    # allocation gloutonne : part de la meilleure précision, dégrade si dépassement
    alloc = {}
    for fam in fams:
        floor = FLOOR_WBITS.get(fam, 8)
        cands = [f for f in ORDER if WBITS[f] >= floor and f != "F16" or f == "F16"]
        cands = [f for f in ORDER if WBITS[f] >= min(floor, 16)]
        alloc[fam] = "F16" if floor >= 16 else "Q8_0" if floor >= 8 else "Q4_0"
    # itérer les dégradations tant que ça ne tient pas
    degraded = True
    guard = 0
    while degraded and guard < 60:
        degraded = False
        guard += 1
        total = sum(a["by_family"][f][1] * BPW[alloc[f]] / 2.0
                    for f in fams)
        if total <= budget_gb:
            break
        # dégrader la famille la plus lourde encore dégradable
        for fam in weight_order:
            idx = ORDER.index(alloc[fam])
            floor = FLOOR_WBITS.get(fam, 8)
            cands = [f for f in ORDER[idx+1:] if WBITS[f] >= min(floor, 16)]
            if cands:
                alloc[fam] = cands[-1]  # le plus petit candidat
                degraded = True
                break
    total = sum(a["by_family"][f][1] * BPW[alloc[f]] / 2.0
                for f in fams)
    return {"alloc": alloc, "total_gib": total, "fits": total <= budget_gb}


def quant_plan_adaptive(a, budget_gb, live=None, max_spill_freq=0.15):
    """Comme quant_plan(), mais si une trace live montre une fréquence de
    spill au-delà de max_spill_freq, dégrade en plus la famille la plus
    lourde (au-delà de la seule contrainte RAM) pour réduire la pression
    VTCM implicite.

    LIMITE ASSUMÉE [H] : la trace ne mesure la pression VTCM que pour la
    config actuellement exécutée sur device — ce plan ne fait qu'extrapoler
    "moins de poids actifs -> probablement moins de pression" ; ça ne
    modélise pas les vrais leviers de pression VTCM (taille de tuile,
    OPBATCH, ordonnancement), qui sont hors du périmètre header-only de ce
    script. À valider en rejouant une trace sur le plan proposé."""
    plan = quant_plan(a, budget_gb)
    if not live or live.get("spill_freq", 0.0) <= max_spill_freq:
        plan["adaptive_note"] = ("aucune dégradation additionnelle : pas de "
                                  "trace live ou spill_freq sous le seuil")
        return plan
    fams = sorted(a["by_family"].keys(), key=lambda f: -a["by_family"][f][1])
    alloc = dict(plan["alloc"])
    guard = 0
    while live["spill_freq"] > max_spill_freq and guard < 60:
        guard += 1
        moved = False
        for fam in fams:
            idx = ORDER.index(alloc[fam])
            floor = FLOOR_WBITS.get(fam, 8)
            cands = [f for f in ORDER[idx + 1:] if WBITS[f] >= min(floor, 16)]
            if cands:
                alloc[fam] = cands[-1]
                moved = True
                break
        if not moved:
            break
        # heuristique : chaque cran de dégradation supplémentaire réduit
        # la fréquence de spill estimée de ~20% (à recalibrer sur trace réelle)
        live = {**live, "spill_freq": live["spill_freq"] * 0.8}
    total = sum(a["by_family"][f][1] * BPW[alloc[f]] / 2.0 for f in fams)
    return {"alloc": alloc, "total_gib": total, "fits": total <= budget_gb,
            "adaptive_note": f"dégradé sous contrainte spill_freq "
                              f"(seuil {max_spill_freq:.0%}), {guard} cran(s)"}


# ---------------------------------------------------------------------------
# BOUCLE STATIQUE -> DYNAMIQUE : ingestion trace live + recalibration
# ---------------------------------------------------------------------------
# Schéma de trace attendu (JSONL, une ligne par événement kernel) :
#   {"run": 0, "layer": 12, "kernel": "MUL_MAT", "latency_us": 143.2,
#    "vtcm_peak_mb": 5.8, "spill": false, "bytes_ddr": 2621440}
# "run" (optionnel) groupe les événements d'un même forward pass/token pour
# calculer un débit t/s mesuré comparable à cold_tps/thermal_tps du L3.
# Voir INSTRUMENTATION.md pour les points d'accroche côté device
# (ggml-hexagon.cpp / htp-ops.h / vtcm-utils.h ou API de profiling QNN).

def load_live_trace(path):
    events = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [live] ligne {lineno} ignorée (JSON invalide) : {e}")
    if not events:
        raise ValueError(f"trace live vide ou illisible : {path}")
    return events


def _percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def aggregate_live_trace(events):
    by_kernel = defaultdict(list)
    vtcm_peaks = []
    spills = 0
    total_ddr_bytes = 0.0
    total_latency_us = 0.0
    runs = defaultdict(float)
    # Variante MEMPOOL (hook porté 2026-09-03) : kernel "BATCH" = une
    # invocation FastRPC réelle (doorbell du graphe entier) avec latency_us =
    # durée AP-side mesurée + dsp_exec_us + n_ops. En decode, un token ≈ un
    # BATCH : la série BATCH EST le t/s wall — pas besoin de --measured-tps.
    batch_us = []
    batch_dsp_us = []
    # Par layer : sommes de latence par kernel (pour MoE/MUL_MAT_ID first-order)
    per_layer = defaultdict(lambda: {"n": 0, "us": 0.0,
                                     "kernels_us": defaultdict(float),
                                     "kernels_n": defaultdict(int)})
    # Répartition par chemin exécuté (hook enrichi 2026-09-03, verdict REVISE
    # fermé) : pour chaque MUL_MAT/MUL_MAT_ID l'op émet AU POP sa latence réelle
    # + path (hmx-tiled / hvx-tiled / hvx-flat / hvx-ddr) + tailles VTCM réelles
    # (w_mb = poids src0, a_mb = act src1, d_mb = dst). Les anciennes traces
    # (événements MUL_MAT_VTCM_FIT à 0 µs) restent supportées : ignorées des
    # stats de latence mais comptées comme décisions.
    # NOTE : dans la variante mempool les événements par-op MUL_MAT portent le
    # path + tailles mais PAS de latence (indisponible : transport en un seul
    # doorbell) — ils ne sont donc PAS ajoutés à by_kernel ni per_layer pour
    # ne pas polluer les stats temporelles (by_path les compte, sans latence).
    by_path = defaultdict(list)          # path -> [latency_us]
    fit_decisions = 0                    # anciennes traces : événements 0 µs
    spill_fit = 0                        # anciennes traces : repli flat à 0 µs
    for e in events:
        lat = float(e.get("latency_us", 0.0))
        kern = e.get("kernel", "?")
        if kern == "BATCH":
            # événement d'invocation mempool : ne pas le mélanger aux kernels
            # par-op (pas de layer, pas de chemin) — il sert de mesure wall.
            batch_us.append(lat)
            dsp = e.get("dsp_exec_us")
            if dsp:
                batch_dsp_us.append(float(dsp))
            total_latency_us += lat
            continue
        by_kernel[kern].append(lat)
        vtc = e.get("vtcm_mb", e.get("vtcm_peak_mb"))
        if vtc is not None:
            vtcm_peaks.append(float(vtc))
        if e.get("spill"):
            spills += 1
        total_ddr_bytes += float(e.get("bytes_ddr", 0.0))
        total_latency_us += lat
        if "run" in e:
            runs[e["run"]] += lat
        path = e.get("path")
        if path:
            if "latency_us" in e:
                # dspqueue enrichi : path + latence réelle au pop.
                by_path[path].append(lat)
            else:
                # mempool : événement de plan (path + tailles, sans latence par-
                # op possible) — compté comme décision, pas comme mesure.
                fit_decisions += 1
                if e.get("spill"):
                    spill_fit += 1
        if kern.endswith("VTCM_FIT"):
            fit_decisions += 1
            if e.get("spill"):
                spill_fit += 1
        lay = e.get("layer")
        if isinstance(lay, int):
            pl = per_layer[lay]
            pl["n"] += 1
            pl["us"] += lat
            if lat > 0:
                pl["kernels_us"][kern] += lat
                pl["kernels_n"][kern] += 1

    per_kernel = {}
    for k, lats in by_kernel.items():
        per_kernel[k] = {
            "count": len(lats), "mean_us": sum(lats) / len(lats),
            "p50_us": _percentile(lats, 0.50), "p95_us": _percentile(lats, 0.95),
            "max_us": max(lats),
        }
    # Agréger la somme (pas la moyenne) par kernel : le vrai poids dans le
    # temps de la passe = Σ latency_us (le MUL_MAT_ID "~50 %" du Gemma).
    per_kernel_sum_us = {k: sum(lats) for k, lats in by_kernel.items()}
    n = len(events)
    measured_bw_gbs = (total_ddr_bytes / (total_latency_us / 1e6) / 1e9
                        if total_latency_us > 0 else 0.0)
    measured_tps_per_run = ({r: 1e6 / lat_us for r, lat_us in runs.items() if lat_us > 0}
                             if runs else {})

    # ---- MoE / MUL_MAT_ID first-order (verdict governance 2026-09-03) ----
    # MUL_MAT_ID = le kernel expert (routing sparse) : sur Gemma-26B-A4B il
    # pèse ~50 % du temps kernel global et 75-85 % du temps PAR layer MoE.
    # Sur Marco-Nano (qwen3moe) le kernel exécuté est souvent la variante
    # fusionnée MUL_MAT_ID_NX (graphe routé, pr28202) : on agrège les DEUX
    # dans le budget expert, sinon le MoE est sous-compté (11.8 % au lieu de
    # ~33 % mesuré 2026-09-03). ARGSORT = tri de routage (top-k experts) :
    # sur Marco il pèse 34 % du temps kernel (262.9 µs/op HTP, 1876 ops) —
    # compté séparément comme signal de routage MoE.
    mmid_keys = [k for k in per_kernel_sum_us
                 if k in ("MUL_MAT_ID", "MUL_MAT_ID_NX")]
    mmid_us = sum(per_kernel_sum_us.get(k, 0.0) for k in mmid_keys)
    argsort_us = per_kernel_sum_us.get("ARGSORT", 0.0)
    kernel_time_us = sum(v for k, v in per_kernel_sum_us.items()
                         if k != "MUL_MAT_VTCM_FIT")   # événement 0 µs (classification)
    per_layer_out = {}
    for lay, pl in sorted(per_layer.items()):
        ku = dict(pl["kernels_us"])
        kn = dict(pl["kernels_n"])
        # budget expert = MUL_MAT_ID + variante fusionnée NX (Marco/qwen3moe)
        mmid = sum(ku.get(k, 0.0) for k in ("MUL_MAT_ID", "MUL_MAT_ID_NX"))
        mmid_n = sum(kn.get(k, 0) for k in ("MUL_MAT_ID", "MUL_MAT_ID_NX"))
        per_layer_out[str(lay)] = {
            "events": pl["n"], "total_us": round(pl["us"], 1),
            "mmid_us": round(mmid, 1), "mmid_n": mmid_n,
            "mmid_ratio_layer": round(mmid / pl["us"], 4) if pl["us"] > 0 else 0.0,
            "kernels_us": {k: round(v, 1) for k, v in
                            sorted(ku.items(), key=lambda kv: -kv[1])[:6]},
        }
    # Résumé par chemin (hook enrichi) : latence moyenne + part des ops.
    n_path = sum(len(v) for v in by_path.values())
    by_path_out = {p: {"n": len(v), "mean_us": round(sum(v) / len(v), 1),
                       "p50_us": _percentile(v, 0.50), "p95_us": _percentile(v, 0.95)}
                   for p, v in sorted(by_path.items(),
                                      key=lambda kv: -sum(kv[1]))}
    return {
        "n_events": n,
        "per_kernel": per_kernel,
        "per_kernel_sum_us": {k: round(v, 1) for k, v in
                               sorted(per_kernel_sum_us.items(),
                                      key=lambda kv: -kv[1])},
        "per_layer": per_layer_out,
        "mmid_us": round(mmid_us, 1),
        "mmid_share_kernel": round(mmid_us / kernel_time_us, 4)
                              if kernel_time_us > 0 else 0.0,
        "mmid_share_plain": round(per_kernel_sum_us.get("MUL_MAT_ID", 0.0)
                                  / kernel_time_us, 4) if kernel_time_us > 0 else 0.0,
        "mmid_share_nx": round(per_kernel_sum_us.get("MUL_MAT_ID_NX", 0.0)
                               / kernel_time_us, 4) if kernel_time_us > 0 else 0.0,
        "argsort_us": round(argsort_us, 1),
        "argsort_share_kernel": round(argsort_us / kernel_time_us, 4)
                                if kernel_time_us > 0 else 0.0,
        "kernel_time_us": round(kernel_time_us, 1),
        "spill_freq": spills / n if n else 0.0,
        "spill_mm": spill_fit,          # décisions flat à 0 µs (anciennes traces)
        "fit_decisions": fit_decisions,
        "vtcm_peak_mean_mb": sum(vtcm_peaks) / len(vtcm_peaks) if vtcm_peaks else 0.0,
        "vtcm_peak_max_mb": max(vtcm_peaks) if vtcm_peaks else 0.0,
        "vtcm_peak_p95_mb": _percentile(vtcm_peaks, 0.95) if vtcm_peaks else 0.0,
        "by_path": by_path_out,
        "n_path_events": n_path,
        "measured_bw_gbs": measured_bw_gbs,
        "measured_tps_by_run": measured_tps_per_run,
        "mean_latency_us": total_latency_us / n if n else 0.0,
        # Variante mempool : événements BATCH (une invocation FastRPC réelle).
        # En decode 1 token ≈ 1 BATCH → la série BATCH EST le t/s wall :
        # median + mean per-batch µs, dérivables sans --measured-tps.
        "batch_n": len(batch_us),
        "batch_mean_us": (sum(batch_us) / len(batch_us)) if batch_us else 0.0,
        "batch_p50_us": _percentile(batch_us, 0.50) if batch_us else 0.0,
        "batch_tps_p50": (1e6 / _percentile(batch_us, 0.50)) if batch_us else 0.0,
        "batch_dsp_mean_us": (sum(batch_dsp_us) / len(batch_dsp_us))
                              if batch_dsp_us else 0.0,
    }


def classify_bottleneck(live, cmp_=None):
    """Classification modèle → bottleneck (matrice modèle→goulot, verdict
    governance 2026-09-03) :

      MOE_COMPUTE_DOMINATED    MUL_MAT_ID dominant (Gemma-26B-A4B : ~50 % kernel,
                               75-85 % par layer) → optimiser le chemin expert
      MOE_ROUTING_SORT_DOMINATED ARGSORT (tri top-k experts) > matmul expert
                               (Marco-Nano : 34 % kernel, 262.9 µs/op HTP)
      DISPATCH_DOMINATED       wall/L3 < 0.85 → micro-ops/orchestration (Qwen)
      MEMORY_MOVEMENT_DOMINATED CPY/CONCAT/GET_ROWS dominants (transferts)
      DENSE_COMPUTE_DOMINATED  MUL_MAT (dense) dominant, sans MoE
      VTCM_SPILL_DOMINATED     ≥ 50 % des MUL_MAT en repli flat/DDR exécuté
                               (hook enrichi 2026-09-03, ferme le REVISE)
      MIXED                    aucun kernel > seuil, ni signal dispatch
    """
    if not live:
        return "UNKNOWN", {}
    mmid = live.get("mmid_share_kernel", 0.0)
    mmid_us = live.get("mmid_us", 0.0)
    sum_us = live["per_kernel_sum_us"]
    kernel_top = max(sum_us, key=sum_us.get) if sum_us else "?"
    kernel_time = live.get("kernel_time_us", 0.0) or 1.0
    argsort = live.get("argsort_share_kernel", 0.0)
    argsort_us = live.get("argsort_us", 0.0)
    # parts par famille de kernels (sommes réelles, pas moyennes)
    def share(*names):
        return sum(v for k, v in sum_us.items()
                   if any(n in k for n in names)) / kernel_time

    cpy = share("CPY", "CONCAT", "GET_ROWS", "SET_ROWS", "CONT", "REPEAT")
    dense_mm = share("MUL_MAT") - mmid
    wall_ratio = (cmp_ or {}).get("wall_ratio")
    # Chemins exécutés (hook enrichi 2026-09-03) : spill RÉEL = ops parties en
    # hvx-flat/hvx-ddr (le layout tuilé ne tenait pas dans le VTCM).
    bp = live.get("by_path") or {}
    n_path = live.get("n_path_events", 0) or 1
    flat_n = sum(s["n"] for p, s in bp.items() if p in ("hvx-flat", "hvx-ddr"))
    flat_share = flat_n / n_path if n_path else 0.0
    # Anciennes traces : spill_fit = décisions flat à 0 µs du precompute.
    flat_share_legacy = (live.get("spill_mm", 0) /
                         max(live.get("fit_decisions", 1), 1))
    spill_signal = max(flat_share, flat_share_legacy)
    evidence = {"mmid_share_kernel": round(mmid, 4),
                "mmid_us": round(mmid_us, 1),
                "argsort_share_kernel": round(argsort, 4),
                "argsort_us": round(argsort_us, 1),
                "kernel_top": kernel_top,
                "kernel_time_us": round(live.get("kernel_time_us", 0.0), 1),
                "wall_ratio": wall_ratio,
                "cpy_share": round(cpy, 4),
                "dense_mul_mat_share": round(max(dense_mm, 0.0), 4),
                "spill_share_real": round(flat_share, 4),
                "mm_paths": {p: s["n"] for p, s in bp.items()}}
    if spill_signal >= 0.50:
        cls = "VTCM_SPILL_DOMINATED"
        evidence["bottleneck_kernel"] = "MUL_MAT hvx-flat/ddr"
        evidence["note"] = (f"{spill_signal:.0%} des MUL_MAT exécutés en repli "
                             f"flat/DDR (layout tuilé > budget VTCM) → tiling/"
                             f"placement/working-set, pas compute ni BW.")
    elif argsort >= 0.20 and argsort >= mmid * 0.85:
        # Marco-Nano (qwen3moe) : ARGSORT (tri top-k experts) domine OU est
        # co-dominant du budget kernel (tolérance 15 % : sur device réel,
        # ARGSORT 33.7 % vs MUL_MAT_ID 33.9 % bruitent à ±1 pt run-to-run —
        # un tri à ≥20 % du kernel reste massivement actionnable). Levier :
        # tri/politique de routing, pas le kernel matmul ni BW.
        cls = "MOE_ROUTING_SORT_DOMINATED"
        evidence["bottleneck_kernel"] = "ARGSORT"
        evidence["note"] = (f"ARGSORT {argsort:.0%} du temps kernel (routing "
                             f"top-k experts, {argsort_us:.0f} µs cumulés) > "
                             f"matmul expert {mmid:.0%} → optimiser le tri/"
                             f"routing MoE (départage, top-k natif HTP), pas BW.")
    elif mmid >= 0.30:
        cls = "MOE_COMPUTE_DOMINATED"
        evidence["bottleneck_kernel"] = "MUL_MAT_ID"
        evidence["note"] = (f"MUL_MAT_ID {mmid:.0%} du temps kernel "
                             f"→ optimiser le chemin expert (MUL_MAT_ID/"
                             f"réduction experts actifs), pas BW.")
    elif wall_ratio is not None and wall_ratio < 0.85:
        cls = "DISPATCH_DOMINATED"
        evidence["note"] = (f"wall/L3 {wall_ratio:.2f} : orchestration "
                             f"(micro-ops/passe), pas mémoire ni compute.")
    elif cpy >= 0.25:
        cls = "MEMORY_MOVEMENT_DOMINATED"
        evidence["note"] = f"CPY/CONCAT/GET_ROWS {cpy:.0%} → optimiser DMA/layout."
    elif dense_mm >= 0.30 and mmid < 0.30:
        cls = "DENSE_COMPUTE_DOMINATED"
        evidence["note"] = "MUL_MAT dense dominant → optimiser le kernel matmul."
    else:
        cls = "MIXED"
        evidence["note"] = "aucun kernel > 30 % ni signal dispatch — profil mélangé."
    return cls, evidence


def render_moe_layers(live):
    """Tableau par layer : coût MUL_MAT_ID first-order (verdict ACCEPT)."""
    pl = live.get("per_layer", {})
    if not pl:
        return ""
    L = ["", "### 7bis. MoE par layer — MUL_MAT_ID first-order", ""]
    L.append("| layer | evts | total layer µs | MUL_MAT_ID n | MUL_MAT_ID µs | "
             "ratio layer |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for lay, p in pl.items():
        L.append(f"| {int(lay):02d} | {p['events']} | {p['total_us']:.0f} | "
                 f"{p['mmid_n']} | {p['mmid_us']:.0f} | {p['mmid_ratio_layer']:.1%} |")
    L.append("")
    L.append("Lecture : si ratio layer élevé (~80 %), le layer est MoE/expert-"
             "bound : le modèle de coût doit être T_layer = T_MUL_MAT_ID + "
             "T_attn + T_norm + T_routing + T_dispatch, pas FLOPs/HTP_peak.")
    return "\n".join(L)


def compare_live_vs_l3(a, live, measured_tps_wall=None):
    """Écarts mesuré vs modélisé + suggestion de recalibration (ratio simple,
    pas de fit statistique — un seul point de mesure ne justifie pas mieux).

    measured_tps_wall : débit wall réel (llama-bench tgNN), la seule vérité
    terrain exploitable quand la trace n'a pas de BATCH — l'écart L3-vs-wall
    est le signal dominant.

    Variante mempool : la trace porte des événements BATCH (une invocation
    FastRPC réelle, 1 token decode ≈ 1 BATCH) → batch_tps_p50 est le t/s wall
    MESURÉ PAR LA TRACE, sans --measured-tps. Il prend la priorité."""
    l3 = a["l3"]
    predicted_bw = BW_EFF_GBS
    measured_bw = live["measured_bw_gbs"]
    bw_ratio = measured_bw / predicted_bw if predicted_bw > 0 else 0.0
    predicted_spill_freq = l3["spill_layers"] / max(len(l3["rows"]), 1)
    measured_spill_freq = live["spill_freq"]
    predicted_cold_tps = l3["cold_tps"]
    suggestions = {}
    if measured_bw > 0 and abs(bw_ratio - 1.0) > 0.15:
        suggestions["BW_EFF_GBS"] = round(measured_bw, 2)
    if abs(measured_spill_freq - predicted_spill_freq) > 0.10:
        # seuil trop bas -> spill mesuré plus fréquent que prévu (baisser le
        # seuil relatif) ; trop haut -> le modèle est pessimiste (le monter)
        direction = -1 if measured_spill_freq > predicted_spill_freq else 1
        suggestions["VTCM_THRESHOLD"] = round(
            min(max(VTCM_THRESHOLD + direction * 0.05, 0.3), 0.95), 2)
    # ---- Signal wall : écart L3 vs débit réel mesuré ----
    # Le L3 suppose 1 dispatch/layer (~26 ops) alors que la trace réelle montre
    # des centaines d'ops/passe decode (micro-ops ADD/ROPE/SET_ROWS) : pour un
    # petit modèle dense le vrai goulot est l'orchestration, pas la mémoire.
    wall_ratio = None
    mmid_share = live.get("mmid_share_kernel", 0.0)
    # Trace mempool : les événements BATCH donnent le t/s wall MESURÉ PAR LA
    # TRACE (1 token decode ≈ 1 invocation). Prioritaire sur --measured-tps.
    batch_tps = live.get("batch_tps_p50", 0.0)
    wall_src = "llama-bench"
    if batch_tps and batch_tps > 0:
        measured_tps_wall = batch_tps
        wall_src = "trace BATCH (p50)"
    if measured_tps_wall and measured_tps_wall > 0 and predicted_cold_tps > 0:
        wall_ratio = measured_tps_wall / predicted_cold_tps
        if wall_ratio < 0.85:
            if mmid_share >= 0.30:
                # MoE/expert-bound (Gemma) : le gap wall provient du chemin
                # MUL_MAT_ID (coût réel/expert + orchestration expert), pas
                # d'une mauvaise BW_EFF ni d'un "petit modèle".
                suggestions["MUL_MAT_ID_UNDERESTIMATED"] = {
                    "wall_vs_l3": round(wall_ratio, 3),
                    "note": ("L3 {:.1f} t/s mais wall {:.1f} t/s : modèle "
                             "MoE expert-bound (MUL_MAT_ID {:.0%} du temps "
                             "kernel) — affiner le coût MUL_MAT_ID par expert "
                             "actif et le dispatch expert, pas BW_EFF."
                             ).format(predicted_cold_tps, measured_tps_wall,
                                      mmid_share),
                }
            else:
                # sous-perf réelle => recalibrer BW_EFF vers le bas est trompeur
                # (le goulot est le dispatch/orchestration) : on le signale.
                suggestions["DISPATCH_DOMINATED"] = {
                    "wall_vs_l3": round(wall_ratio, 3),
                    "note": ("L3 {:.1f} t/s mais wall {:.1f} t/s : petit modèle "
                             "orchestration-bound (micro-ops/passe > 1/layer "
                             "modélisé). Ajuster le coût dispatch, pas BW_EFF."
                             ).format(predicted_cold_tps, measured_tps_wall),
                }
        elif wall_ratio > 1.18:
            suggestions["BW_EFF_GBS_UNDERESTIMATED"] = {
                "wall_vs_l3": round(wall_ratio, 3),
                "note": "device plus rapide que le L3 (thermique favorable ?)",
            }
    return {
        "predicted_bw_gbs": predicted_bw, "measured_bw_gbs": measured_bw,
        "bw_ratio": bw_ratio,
        "predicted_spill_freq": predicted_spill_freq,
        "measured_spill_freq": measured_spill_freq,
        "predicted_cold_tps": predicted_cold_tps,
        "measured_tps_by_run": live["measured_tps_by_run"],
        "measured_tps_wall": measured_tps_wall,
        "wall_src": wall_src,
        "wall_ratio": wall_ratio,
        "suggested_recalibration": suggestions,
    }


def render_live_section(live, cmp_):
    L = ["", "## 7. Trace live — mesuré vs modèle L3", ""]
    L.append(f"- événements : {live['n_events']} · fréquence spill mesurée : "
             f"{live['spill_freq']:.1%} (modèle L3 : {cmp_['predicted_spill_freq']:.1%})")
    L.append(f"- VTCM pic : moyenne {live['vtcm_peak_mean_mb']:.2f} Mo · "
             f"p95 {live['vtcm_peak_p95_mb']:.2f} Mo · max {live['vtcm_peak_max_mb']:.2f} Mo")
    L.append(f"- BW DDR mesurée : {cmp_['measured_bw_gbs']:.1f} Go/s vs modélisée "
             f"{cmp_['predicted_bw_gbs']:.1f} Go/s (ratio {cmp_['bw_ratio']:.2f})")
    if live.get("mmid_share_kernel", 0.0) > 0:
        L.append(f"- **MUL_MAT_ID (experts MoE, +NX fusionné) : "
                 f"{live['mmid_us']:.0f} µs cumulés = "
                 f"{live['mmid_share_kernel']:.1%} du temps kernel** "
                 f"(plain {live.get('mmid_share_plain', 0):.1%} + "
                 f"NX {live.get('mmid_share_nx', 0):.1%})")
    if live.get("argsort_share_kernel", 0.0) > 0.05:
        L.append(f"- **ARGSORT (routing top-k experts) : "
                 f"{live['argsort_us']:.0f} µs cumulés = "
                 f"{live['argsort_share_kernel']:.1%} du temps kernel** ")
    if cmp_["measured_tps_by_run"]:
        tps_vals = list(cmp_["measured_tps_by_run"].values())
        L.append(f"- t/s mesuré (moy. sur {len(tps_vals)} run(s)) : "
                 f"{sum(tps_vals)/len(tps_vals):.1f} vs t/s L3 froid modélisé : "
                 f"{cmp_['predicted_cold_tps']:.1f}")
    if live.get("batch_n"):
        L.append(f"- **t/s wall (trace BATCH, {live['batch_n']} invocations, "
                 f"p50 {live['batch_mean_us']:.0f} µs) : "
                 f"{cmp_.get('measured_tps_wall', 0):.1f} vs L3 froid "
                 f"{cmp_['predicted_cold_tps']:.1f} "
                 f"(ratio {cmp_['wall_ratio']:.2f} · source "
                 f"{cmp_.get('wall_src', '?')})")
    if (cmp_.get("measured_tps_wall") and cmp_.get("wall_ratio") is not None
            and not live.get("batch_n")):
        L.append(f"- **t/s wall réel ({cmp_.get('wall_src', 'llama-bench')}) : "
                 f"{cmp_['measured_tps_wall']:.1f} "
                 f"vs t/s L3 froid : {cmp_['predicted_cold_tps']:.1f} "
                 f"(ratio {cmp_['wall_ratio']:.2f})")
    L.append("")
    L.append("| kernel | n | moy µs | p50 µs | p95 µs | max µs | Σ µs |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    sums = live["per_kernel_sum_us"]
    fit_n = 0
    for k, s in sorted(live["per_kernel"].items(), key=lambda kv: -kv[1]["mean_us"]):
        if k.endswith("VTCM_FIT"):
            # événement de classification/décision VTCM (latence 0) : on le
            # compte à part — ce n'est pas un kernel réellement exécuté.
            fit_n += s["count"]
            continue
        L.append(f"| {k} | {s['count']} | {s['mean_us']:.1f} | {s['p50_us']:.1f} | "
                 f"{s['p95_us']:.1f} | {s['max_us']:.1f} | {sums.get(k, 0):.0f} |")
    if fit_n:
        L.append("")
        L.append(f"*{fit_n} événements MUL_MAT_VTCM_FIT exclus (0 µs : décision "
                 "fit/flat VTCM des anciennes traces, pas un kernel exécuté).*")
    bp = live.get("by_path")
    if bp:
        L.append("")
        L.append("| chemin exécuté | n ops | moy µs | p50 µs | p95 µs |")
        L.append("|---|---:|---:|---:|---:|")
        for p, s in bp.items():
            L.append(f"| {p} | {s['n']} | {s['mean_us']:.1f} | {s['p50_us']:.1f} | "
                     f"{s['p95_us']:.1f} |")
        L.append("")
        L.append("Chemins (hook enrichi 2026-09-03) : hmx-tiled = tuilé HMX "
                 "(VTCM) · hvx-tiled = tuilé HVX (VTCM) · hvx-flat/hvx-ddr = "
                 "repli DDR/flat (spill réel). Latence par op = chemin exécuté, "
                 "plus de 0 µs de classification.")
    L.append("")
    cls, evidence = classify_bottleneck(live, cmp_)
    L.append(f"**MODEL CLASS : {cls}** — {evidence['note']}")
    L.append(f"MUL_MAT_ID {evidence['mmid_share_kernel']:.1%} · ARGSORT "
             f"{evidence['argsort_share_kernel']:.1%} · kernel top : "
             f"{evidence['kernel_top']} · wall_ratio : "
             f"{evidence['wall_ratio'] if evidence['wall_ratio'] is not None else '—'}")
    L.append("")
    L.append(render_moe_layers(live))
    if cmp_["suggested_recalibration"]:
        L.append(f"**Recalibration suggérée** (écart > seuil) : "
                 f"{cmp_['suggested_recalibration']}")
        L.append("À injecter dans un --device-config pour les prochains runs "
                 "(les clés DISPATCH_DOMINATED/BW_EFF_GBS_UNDERESTIMATED sont "
                 "diagnostiques : à traduire en constante device).")
    else:
        L.append("**Modèle L3 cohérent avec la trace** (écarts sous seuil, "
                 "pas de recalibration suggérée).")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------
def render(a, model_name, budget_gb, ctx, out_path=None):
    L = []
    L.append(f"# PROFILE — {model_name}")
    L.append("")
    L.append(f"- format : {'GGUF (types réels)' if a['is_gguf'] else 'safetensors (BF16)'} · "
             f"{a['n_tensors']} tenseurs · BF16 {a['total_bf16_gib']:.1f} GiB"
             + (f" · fichier {a['total_real_gib']:.1f} GiB" if a['is_gguf'] else ""))
    arch = a["arch"]
    L.append(f"- archi : {arch['n_layer']} couches · hidden {arch['hidden']} · "
             f"vocab {arch['vocab']} · tied {arch['tied']} · vision {arch['has_vision']}")
    if arch["is_moe"]:
        L.append(f"- MoE : {arch['n_experts']} experts top-{arch['top_k']}")
    if arch["full_attn"] or arch["sliding_attn"]:
        L.append(f"- attention : {len(arch['full_attn'])} full "
                 f"{sorted(arch['full_attn'])} · {len(arch['sliding_attn'])} sliding "
                 f"(win {arch['window']})")
    L.append("")
    L.append("## 1. Répartition par famille (BF16)")
    L.append("| famille | tensors | GiB | part % |")
    L.append("|---|---:|---:|---:|")
    for fam, (n, gib) in sorted(a["by_family"].items(), key=lambda kv: -kv[1][1]):
        L.append(f"| {fam} | {n} | {gib:.2f} | {gib/a['total_bf16_gib']*100:.1f} |")
    L.append("")
    L.append("## 2. Tailles de déploiement (GiB)")
    L.append(f"| format | complet | texte seul | vs {budget_gb} Go libres |")
    L.append("|---|---:|---:|---|")
    for fmt in ORDER:
        tot = a["sizes_gib"][fmt]
        txt = a["sizes_text_gib"][fmt]
        ok = "OUI" if txt <= budget_gb else "NON"
        L.append(f"| {fmt} | {tot:.1f} | {txt:.1f} | {ok} |")
    L.append("")
    L.append("## 3. Actifs / trafic décode / débit")
    L.append(f"- actifs/token : {a['active_bf16_b']:.2f}B poids BF16 "
             f"({a['active_bf16_gib']:.2f} GiB)")
    L.append(f"- KV cache/token : {a['kv_per_token_kb']:.0f} Ko "
             f"(@ctx {ctx} : {a['kv_ctx4096_mb']:.0f} Mo)")
    L.append(f"- lm_head : {a['lm_head_gib']:.2f} GiB (Q4_0 "
             f"{a['lm_head_gib']*BPW['Q4_0']/2*1024:.0f} Mo/token)")
    L.append(f"- orchestration : {a['ops_per_token']} ops/token × 215 µs = "
             f"{a['orchestration_ms']:.1f} ms/token (borne fixe HTP)")
    L.append("")
    L.append("| format | trafic Q4_0 GiB/token | GGML 30 Go/s | GGML 32,4 | QAIRT 74 | "
             "régime dense 57 | régime MoE 14,3 |")
    L.append("|---|---:|---:|---:|---:|---:|---:|")
    for fmt in ORDER:
        r = a["formats"][fmt]
        L.append(f"| {fmt} | {r['traffic_gib']:.2f} | {r['tps_ggml']:.1f} | "
                 f"{1000/r['t_ggml_hi_ms']:.1f} | {r['tps_qairt']:.1f} | "
                 f"{r['tps_dense_like']:.1f} | {r['tps_moe_like']:.1f} |")
    L.append("")
    L.append("## 4. Plan quant sous contrainte RAM")
    plan = quant_plan(a, budget_gb)
    L.append(f"| famille | format plan | GiB |")
    L.append("|---|---:|---:|")
    for fam, fmt in sorted(plan["alloc"].items(), key=lambda kv: -a["by_family"][kv[0]][1]):
        gib = a["by_family"][fam][1] * BPW[fmt] / 2.0
        L.append(f"| {fam} | {fmt} | {gib:.2f} |")
    L.append(f"**Total plan : {plan['total_gib']:.1f} GiB "
             f"→ {'RENTRE' if plan['fits'] else 'NE RENTRE PAS'} "
             f"dans {budget_gb} Go libres**")
    L.append("")
    L.append("## 5. Verdict")
    q4 = a["sizes_gib"]["Q4_0"]
    if q4 <= budget_gb:
        L.append(f"- Q4_0 complet {q4:.1f} GiB ≤ {budget_gb} Go → **PROFILABLE HTP** "
                 f"(Q4_0 = seule précision NPU)")
    else:
        L.append(f"- Q4_0 {q4:.1f} GiB > {budget_gb} Go → **OOM HTP**. "
                 f"Seule voie : <4 bit GPU/CPU (IQ2/IQ3 ≈ "
                 f"{a['sizes_gib']['IQ2_XXS']:.1f} GiB) — hors HTP, qualité dégradée.")
    est = a["formats"]["Q4_0"]
    if arch["is_moe"]:
        L.append(f"- débit decode probable Q4_0 : "
                 f"{est['tps_moe_like']:.1f} (MoE) à {est['tps_dense_like']:.1f} t/s "
                 f"(orchestration-bound, ARGSORT/MUL_MAT_ID)")
    else:
        L.append(f"- débit decode probable Q4_0 : {est['tps_dense_like']:.1f}-"
                 f"{est['tps_ggml']:.1f} t/s (trafic-bound GGML)")
    L.append("")
    L.append("## 6. Simulation Hextimate L3 (RE Qualcomm : roofline + VTCM + spill + dispatch)")
    l3 = a["l3"]
    L.append(f"- modèle : T_layer = max(compute, mémoire×spill) + dispatch "
             f"· VTCM {VTCM_CAPACITY_MB:.0f} Mo seuil {VTCM_THRESHOLD:.0%} · "
             f"peak {PEAK_COMPUTE_TFLOPS} TFLOPS · BW_eff {BW_EFF_GBS} Go/s · "
             f"dispatch {DISPATCH_US:.0f} µs/op · thermique x{THERMAL_TPS_FACTOR}")
    if a.get("arch", {}).get("is_moe"):
        mmid_op = (MMID_US_AT_HIDDEN_1024
                   + max(a["arch"].get("hidden", 1024) - 1024, 0)
                   * MMID_US_PER_HIDDEN)
        L.append(f"- **MoE : coût MUL_MAT_ID par layer = dispatch + expert "
                 f"{mmid_op:.0f} µs/op (calibré hidden "
                 f"{a['arch'].get('hidden', '?')} : 43 µs@1024 + 0.0558 µs/"
                 f"unité — Marco vs Gemma 2026-09-03)**")
    L.append(f"- layers simulés : {len(l3['rows'])} · spill actif sur "
             f"{l3['spill_layers']} layers · latence totale {l3['total_ms']:.1f} ms/token")
    L.append(f"- flops/token {l3['total_flops_m']:.0f} MFLOPS · bytes/token "
             f"{l3['total_bytes_gb']*1024:.0f} Mo · intensité arithmétique "
             f"{l3['total_flops_m']*1e6/max(l3['total_bytes_gb']*1e9,1):.2f} FLOPS/octet")
    L.append(f"- **t/s L3 (froid) : {l3['cold_tps']:.1f} · soutenu (thermique) : "
             f"{l3['thermal_tps']:.1f} · bottleneck dominant : {l3['bottleneck']}")
    L.append("")
    is_moe = a.get("arch", {}).get("is_moe")
    L.append("| layer | type | poids Q4 Mo | KV Mo | spill | "
             "compute ms | mémoire ms | dispatch ms | expert ms | latence ms |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in l3["rows"]:
        lay = f"{r['layer']:02d}" if isinstance(r['layer'], int) else str(r['layer'])
        mmid_s = f"{r['mmid_ms']:.3f}" if r.get('mmid_ms') else "—"
        L.append(f"| {lay} | {r['type']:<7} | {r['q4_mb']:.0f} | "
                 f"{r['kv_mb']:.1f} | x{r['spill']:.2f} | "
                 f"{r['compute_ms']*1000:.1f} µs | {r['mem_ms_spill']*1000:.0f} µs | "
                 f"{r['dispatch_ms']:.2f} | {mmid_s} | {r['lat_ms']:.2f} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(prog="profile_model",
                                 description="Profileur universel GGUF/safetensors SM8850")
    ap.add_argument("path", nargs="?", help="fichier .gguf OU dossier safetensors")
    ap.add_argument("--hf", help="repo HuggingFace (headers only, ex google/gemma-4-26B-A4B)")
    ap.add_argument("--shards", help="liste de shards séparés par des virgules (override)")
    ap.add_argument("--budget-gb", type=float, default=DEFAULT_RAM_BUDGET_GB)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--out", help="dossier de sortie (rapport .md + .json)")
    ap.add_argument("--scan", help="dossier à scanner : profile TOUS les .gguf + dossiers safetensors")
    ap.add_argument("--device-config", help="JSON de constantes device (voir "
                     "--write-default-device-config) ; défaut = calibration SM8850")
    ap.add_argument("--write-default-device-config", metavar="PATH",
                     help="écrit les constantes SM8850 par défaut dans PATH et quitte "
                          "(gabarit à copier/adapter pour un autre device)")
    ap.add_argument("--live-trace", help="JSONL d'événements mesurés sur device "
                     "(voir INSTRUMENTATION.md) : compare au modèle L3 et suggère "
                     "une recalibration")
    ap.add_argument("--max-spill-freq", type=float, default=0.15,
                    help="seuil spill_freq pour le plan quant adaptatif")
    ap.add_argument("--measured-tps", type=float, default=None,
                    help="débit wall réel du decode (sortie llama-bench tgNN) : "
                         "comparé au L3 (section 7). La trace n'ayant pas de "
                         "champ run, c'est la seule vérité terrain du débit.")
    args = ap.parse_args()

    if args.write_default_device_config:
        with open(args.write_default_device_config, "w", encoding="utf-8") as f:
            json.dump({"device_name": "SM8850", **DEVICE_DEFAULTS}, f, indent=2)
        print(f"[device-config] gabarit écrit : {args.write_default_device_config}")
        return

    if args.device_config:
        applied = load_device_profile(args.device_config)
        print(f"[device-config] {DEVICE_NAME} chargé : {applied}")

    if args.scan:
        root = args.scan
        found = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ("node_modules", ".git", "__pycache__",
                                                            "_toolchains", ".runtime_backup")]
            for fn in filenames:
                if fn.lower().endswith(".gguf"):
                    found.append(os.path.join(dirpath, fn))
                elif fn == "config.json" and any(x.endswith(".safetensors") or x.endswith("_head.bin")
                                                 for x in filenames):
                    found.append(dirpath)
        seen = set()
        uniq = []
        for f in sorted(found):
            key = os.path.normcase(f)
            if key not in seen:
                seen.add(key)
                uniq.append(f)
        print(f"[scan] {len(uniq)} modèle(s) trouvé(s) dans {root}")
        summary = []
        for m in uniq:
            try:
                print(f"\n===== {m} =====")
                _run_one(m, args)
                summary.append(m)
            except Exception as e:
                print(f"[scan] ÉCHEC {m}: {e}")
        print(f"\n[scan] TERMINÉ : {len(summary)}/{len(uniq)} profilés")
        return

    _run_one(args.path, args)


def _run_one(path, args):
    shards = []
    if args.hf:
        print(f"[load] HF {args.hf} (headers only)...")
        cfg, tensors, shards = load_hf(args.hf)
        model_name = args.hf.split("/")[-1]
        is_gguf = False
    elif path and path.lower().endswith(".gguf"):
        print(f"[load] GGUF {path}...")
        meta, tlist = read_gguf_header(path)
        tensors = {}
        for name, dims, ttype, nb in tlist:
            ne = 1
            for d in dims:
                ne *= d
            tensors[name] = {"dtype": V_TYPES.get(ttype, "?"), "shape": dims,
                             "bytes": nb, "elems": ne, "ttype": ttype}
        cfg = {}
        for k in ("general.name", "general.architecture"):
            print(f"  [gguf] {k} = {meta.get(k)}")
        model_name = meta.get("general.name") or os.path.basename(path)
        is_gguf = True
    elif path and os.path.isdir(path):
        print(f"[load] safetensors {path}...")
        cfg, tensors, shards = load_safetensors_local(
            path, args.shards.split(",") if args.shards else None)
        model_name = cfg.get("_name_or_path", os.path.basename(path.rstrip("/\\")))
        meta = None
        is_gguf = False
    else:
        sys.exit("usage : profile_model.py <modele.gguf|dossier> [--hf REPO] "
                 "[--budget-gb 9.0] [--ctx 4096] [--out DIR] [--scan DIR]")

    a = analyze(cfg, tensors, gguf_meta=meta if is_gguf else None,
                shard_names=shards, is_gguf=is_gguf, ctx=args.ctx)
    report = render(a, model_name, args.budget_gb, args.ctx)

    live = None
    if getattr(args, "live_trace", None):
        events = load_live_trace(args.live_trace)
        live = aggregate_live_trace(events)
        cmp_ = compare_live_vs_l3(a, live,
                                 measured_tps_wall=getattr(args, "measured_tps", None))
        report += "\n" + render_live_section(live, cmp_)
        adapt = quant_plan_adaptive(a, args.budget_gb, live=live,
                                     max_spill_freq=args.max_spill_freq)
        report += (f"\n\n**Plan de quant adaptatif** ({adapt['adaptive_note']}) : "
                    f"{adapt['alloc']} — total {adapt['total_gib']:.1f} GiB\n")
        a["live"] = live
        a["live_vs_l3"] = cmp_
        a["quant_plan_adaptive"] = {"alloc": adapt["alloc"],
                                     "total_gib": adapt["total_gib"],
                                     "note": adapt["adaptive_note"]}

    print(report)
    print()

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        md = os.path.join(args.out, f"PROFILE_{model_name.replace('/', '_')}.md")
        with open(md, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        js = os.path.join(args.out, f"PROFILE_{model_name.replace('/', '_')}.json")
        with open(js, "w", encoding="utf-8") as f:
            json.dump(a, f, indent=2, ensure_ascii=False)
        print(f"[out] {md}\n[out] {js}")


if __name__ == "__main__":
    main()