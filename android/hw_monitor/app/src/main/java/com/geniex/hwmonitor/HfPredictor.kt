package com.geniex.hwmonitor

import java.io.ByteArrayInputStream
import java.io.DataInputStream
import java.net.HttpURLConnection
import java.net.URL
import java.nio.ByteBuffer
import java.nio.ByteOrder

// ===========================================================================
// HfPredictor — port Kotlin de profiler_v3/predict_from_hf.py : predire un
// modele HuggingFace SANS TELECHARGER LES POIDS, directement depuis l'app.
// Demande explicite : "predire un modele HuggingFace SANS le telecharger,
// l'apk doit pouvoir le faire".
//
// HONNETETE : ceci est une version SIMPLIFIEE du modele L3 complet de
// profile_model.py (qui fait ~1500 lignes avec modelisation du spill VTCM,
// du cout de dispatch par-op, du cout MoE detaille, des familles par
// architecture specifique SSM/vision/etc.). Ici : lecture GGUF fidele
// (meme parseur binaire, meme table de types), RAM par famille EXACTE
// (tailles reelles declarees dans le fichier, pas une re-simulation de
// quantization), mais le debit predit utilise une formule roofline
// SIMPLIFIEE (bande passante effective unique / octets-par-token), sans le
// detail spill/dispatch/MoE-cost du simulateur Python. Le facteur de
// correction MoE/dense (ROUTING_CALIBRATION/DENSE_CALIBRATION) est repris
// A L'IDENTIQUE (mêmes points de calibration, mêmes mesures reelles).
// ===========================================================================

data class GgufTensor(val name: String, val dims: List<Long>, val ggmlType: Int, val nbytes: Double)

data class HfPrediction(
    val repoId: String,
    val filename: String,
    val architecture: String,
    val nLayers: Int,
    val hiddenSize: Int,
    val nExperts: Int,
    val topK: Int,
    val totalBytes: Double,
    val bytesByFamily: Map<String, Double>,
    val bytesPerTokenGb: Double,
    val l3ColdTpsRaw: Double,
    val correctionFactor: Double,
    val correctionConfidence: String,
    val correctionNote: String,
    val correctedTps: Double,
    val error: String? = null,
)

object HfPredictor {

    // -- types GGUF (identique a profile_model.py::V_TYPES/BLOCK_BYTES/BLOCK_WEIGHTS) --
    private val GGUF_MAGIC = 0x46554747
    private val blockBytes = mapOf(2 to 18, 3 to 20, 6 to 22, 7 to 24, 8 to 34,
        10 to 84, 11 to 110, 12 to 144, 13 to 176, 14 to 210, 15 to 260,
        16 to 66, 17 to 84, 18 to 180, 19 to 120, 20 to 50, 21 to 40, 22 to 140,
        23 to 70, 24 to 88, 30 to 34)
    private val blockWeights = mapOf(2 to 32, 3 to 32, 6 to 32, 7 to 32, 8 to 32, 30 to 32)
    private val typeSizes = mapOf(0 to 4, 1 to 2, 25 to 1, 26 to 1, 27 to 8, 28 to 4)

    private fun ggmlTypeSize(t: Int): Double {
        typeSizes[t]?.let { return it.toDouble() }
        val bb = blockBytes[t] ?: throw IllegalArgumentException("type ggml $t inconnu")
        val bw = blockWeights[t] ?: 256 // la plupart des quant K-blocks font 256 poids/bloc
        return bb.toDouble() / bw.toDouble()
    }

    // -- Calibration MoE/dense — IDENTIQUE a predict_from_hf.py, meme jour, memes mesures --
    private data class RoutingPoint(val nExperts: Int, val factor: Double, val model: String)
    private val routingCalibration = listOf(
        RoutingPoint(2, 62.6 / 42.5, "Qwen3-0.9B-A0.6B"),
        RoutingPoint(3, 54.1 / 42.5, "Huihui-MoE-1.2B"),
        RoutingPoint(64, 62.2 / 71.3, "EuroMoE-2.6B (n=3)"),
        RoutingPoint(232, 19.9 / 42.5, "Marco-Nano-V2"),
    )
    private data class DensePoint(val bytesPerTokenGb: Double, val factor: Double, val model: String)
    private val denseCalibration = listOf(
        DensePoint(1.472, 38.2 / 28.6, "Qwen3-1.7B-Q4_0"),
        DensePoint(2.935, 20.0 / 15.3, "Qwen3-4B-Q4_0"),
    )
    private const val L3_REFERENCE_TPS = 42.5 // meme reference que ROUTING_CALIBRATION Python

    private fun routingFactor(nExperts: Int): Triple<Double, String, String> {
        if (nExperts <= 1) return Triple(1.0, "n/a", "modele dense, correction routing non applicable")
        val pts = routingCalibration.sortedBy { it.nExperts }
        pts.firstOrNull { it.nExperts == nExperts }?.let {
            return Triple(it.factor, "measured_exact", "correspond exactement a ${it.model}")
        }
        val lo = pts.first(); val hi = pts.last()
        if (nExperts < lo.nExperts) return Triple(lo.factor, "extrapolated_unreliable",
            "< point calibre le plus bas (${lo.nExperts}, ${lo.model}) — peu fiable")
        if (nExperts > hi.nExperts) return Triple(hi.factor, "extrapolated_unreliable",
            "> point calibre le plus haut (${hi.nExperts}, ${hi.model}) — peu fiable")
        val (a, b) = pts.zipWithNext().first { (a, b) -> nExperts in a.nExperts..b.nExperts }
        val t = (Math.log(nExperts.toDouble()) - Math.log(a.nExperts.toDouble())) /
                (Math.log(b.nExperts.toDouble()) - Math.log(a.nExperts.toDouble()))
        val factor = a.factor + t * (b.factor - a.factor)
        return Triple(factor, "interpolated", "interpole entre ${a.model} et ${b.model} — 4 points seulement")
    }

    private fun denseFactor(bytesPerTokenGb: Double): Triple<Double, String, String> {
        val pts = denseCalibration.sortedBy { it.bytesPerTokenGb }
        pts.firstOrNull { Math.abs(it.bytesPerTokenGb - bytesPerTokenGb) < 1e-6 }?.let {
            return Triple(it.factor, "measured_exact", "correspond exactement a ${it.model}")
        }
        val lo = pts.first(); val hi = pts.last()
        if (bytesPerTokenGb < lo.bytesPerTokenGb) return Triple(lo.factor, "extrapolated_unreliable",
            "< ${lo.bytesPerTokenGb} Go/token (${lo.model}) — 2 points seulement, peu fiable hors plage")
        if (bytesPerTokenGb > hi.bytesPerTokenGb) return Triple(hi.factor, "extrapolated_unreliable",
            "> ${hi.bytesPerTokenGb} Go/token (${hi.model}) — 2 points seulement, peu fiable hors plage")
        val t = (Math.log(bytesPerTokenGb) - Math.log(lo.bytesPerTokenGb)) /
                (Math.log(hi.bytesPerTokenGb) - Math.log(lo.bytesPerTokenGb))
        val factor = lo.factor + t * (hi.factor - lo.factor)
        return Triple(factor, "interpolated", "interpole entre ${lo.model} et ${hi.model} — 2 points seulement")
    }

    // -- Recherche HuggingFace (nouveau, demande explicite "il doit avoir une
    // fonction de recherche sur huggingface") — utilise l'API publique HF,
    // aucune auth necessaire pour une recherche/listing en lecture seule. --

    data class HfSearchHit(val id: String, val downloads: Long, val likes: Long)

    private fun httpGetString(url: String): String {
        val conn = URL(url).openConnection() as HttpURLConnection
        conn.setRequestProperty("User-Agent", "hw_monitor-android")
        conn.connectTimeout = 10000
        conn.readTimeout = 20000
        conn.connect()
        val code = conn.responseCode
        val stream = if (code in 200..299) conn.inputStream else conn.errorStream
        val text = stream.bufferedReader().use { it.readText() }
        if (code !in 200..299) throw RuntimeException("HTTP $code pour $url : $text")
        return text
    }

    /** Recherche de modeles HF par mot-cle, filtre GGUF. Equivalent leger de
     *  taper la recherche sur huggingface.co/models, directement depuis
     *  l'app — aucun telechargement de poids, juste des metadonnees. */
    fun searchModels(query: String, limit: Int = 15): List<HfSearchHit> {
        val q = java.net.URLEncoder.encode(query, "UTF-8")
        val url = "https://huggingface.co/api/models?search=$q&filter=gguf&limit=$limit&sort=downloads&direction=-1"
        val json = org.json.JSONArray(httpGetString(url))
        val out = mutableListOf<HfSearchHit>()
        for (i in 0 until json.length()) {
            val o = json.getJSONObject(i)
            out.add(HfSearchHit(
                id = o.optString("id", "?"),
                downloads = o.optLong("downloads", 0),
                likes = o.optLong("likes", 0),
            ))
        }
        return out
    }

    /** Liste les fichiers .gguf d'un repo precis (pour choisir --file avant
     *  d'appeler predict()) — equivalent de list_hf_gguf_files() Python. */
    fun listGgufFiles(repoId: String): List<String> {
        val url = "https://huggingface.co/api/models/$repoId"
        val json = org.json.JSONObject(httpGetString(url))
        val siblings = json.optJSONArray("siblings") ?: return emptyList()
        val out = mutableListOf<String>()
        for (i in 0 until siblings.length()) {
            val name = siblings.getJSONObject(i).optString("rfilename", "")
            if (name.lowercase().endsWith(".gguf")) out.add(name)
        }
        return out
    }

    // -- Lecture GGUF distante : UNE requete HTTP Range, AUCUN poids telecharge --
    fun fetchGgufHeaderRemote(repoId: String, filename: String, maxHeaderBytes: Int = 8 * 1024 * 1024): ByteArray {
        val url = URL("https://huggingface.co/$repoId/resolve/main/$filename")
        val conn = url.openConnection() as HttpURLConnection
        conn.setRequestProperty("User-Agent", "hw_monitor-android")
        conn.setRequestProperty("Range", "bytes=0-${maxHeaderBytes - 1}")
        conn.connectTimeout = 15000
        conn.readTimeout = 30000
        conn.connect()
        val code = conn.responseCode
        if (code != 200 && code != 206) throw RuntimeException("HTTP $code pour $url")
        return conn.inputStream.readBytes()
    }

    private class LEReader(data: ByteArray) {
        private val buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN)
        fun u8(): Int = buf.get().toInt() and 0xFF
        fun i8(): Int = buf.get().toInt()
        fun u16(): Int = buf.short.toInt() and 0xFFFF
        fun i16(): Int = buf.short.toInt()
        fun u32(): Long = buf.int.toLong() and 0xFFFFFFFFL
        fun i32(): Int = buf.int
        fun f32(): Float = buf.float
        fun u64(): Long = buf.long
        fun i64(): Long = buf.long
        fun f64(): Double = buf.double
        fun bytes(n: Int): ByteArray { val b = ByteArray(n); buf.get(b); return b }
        fun str(): String { val n = u64().toInt(); return String(bytes(n), Charsets.UTF_8) }
    }

    /** Parseur GGUF fidele a profile_model.py::_read_gguf_header_inner —
     *  memes types, meme ordre de lecture. Retourne (meta, tensors). */
    fun parseGgufHeader(data: ByteArray): Pair<Map<String, Any?>, List<GgufTensor>> {
        val r = LEReader(data)
        val magic = r.u32()
        r.u32() // version, non utilisee
        val nTensors = r.u64()
        val nKv = r.u64()
        if (magic != GGUF_MAGIC.toLong()) throw IllegalArgumentException("pas un GGUF (magic=$magic)")

        fun readValue(vtype: Int): Any? = when (vtype) {
            0 -> r.u8(); 1 -> r.i8(); 2 -> r.u16(); 3 -> r.i16()
            4 -> r.u32(); 5 -> r.i32(); 6 -> r.f32(); 7 -> r.u8()
            8 -> r.str()
            9 -> {
                val et = r.u32().toInt(); val cnt = r.u64()
                if (et == 8) List(cnt.toInt()) { r.str() }
                else List(cnt.toInt()) { readValue(et) }
            }
            10 -> r.u64(); 11 -> r.i64(); 12 -> r.f64()
            else -> throw IllegalArgumentException("kv type $vtype")
        }

        val meta = mutableMapOf<String, Any?>()
        repeat(nKv.toInt()) {
            val k = r.str()
            val vtype = r.u32().toInt()
            meta[k] = readValue(vtype)
        }
        val tensors = mutableListOf<GgufTensor>()
        repeat(nTensors.toInt()) {
            val name = r.str()
            val ndims = r.u32().toInt()
            val dims = List(ndims) { r.u64() }
            val ttype = r.u32().toInt()
            r.u64() // offset, non utilise ici
            var n = 1.0
            for (d in dims) n *= d.toDouble()
            val nbytes = try { n * ggmlTypeSize(ttype) } catch (_: Exception) { n * 4.5 / 8.0 }
            tensors.add(GgufTensor(name, dims, ttype, nbytes))
        }
        return meta to tensors
    }

    private fun familyOf(name: String): String {
        val low = name.lowercase()
        return when {
            "vision" in low -> "vision"
            "lm_head" in low || name == "output.weight" -> "lm_head"
            "token_embd" in low || "embed_tokens" in low || "wte" in low -> "embed"
            "norm" in low || "rms" in low || "scale" in low || "gamma" in low -> "norm"
            "exps" in low || "experts" in low || "router" in low || "gate_up_proj" in low -> "moe"
            "attn" in low || "self_attn" in low || "q_proj" in low || "k_proj" in low ||
                "v_proj" in low || "o_proj" in low -> "attn"
            "mlp" in low || "ffn" in low || "gate_proj" in low || "up_proj" in low ||
                "down_proj" in low -> "mlp"
            "ssm" in low || "conv1d" in low -> "ssm"
            else -> "other"
        }
    }

    /** Meme pipeline que predict() mais sur un GGUF DEJA PRESENT localement
     *  sur le device (ex /data/local/tmp/...) — lit juste les premiers
     *  octets du fichier, jamais le poids complet.
     *
     *  BUG REEL trouve et corrige (2026-09-06, "j'ai des erreur permission
     *  denied quand je selectionne un model") : cette fonction lisait le
     *  fichier en Java pur (FileInputStream), sans le fallback su deja
     *  utilise partout ailleurs (FileBrowser.kt, ProcessMemoryMonitor.kt) —
     *  pour un GGUF dans un dossier root-only (ex /data/local/tmp lui-meme,
     *  cf. le meme probleme de permission deja rencontre sur le listing),
     *  la lecture directe echoue. Fix : meme pattern try-direct-puis-su,
     *  via `dd` pour extraire juste les premiers octets sans tout lire. */
    fun predictLocalFile(path: String, maxHeaderBytes: Int = 32 * 1024 * 1024): HfPrediction {
        val f = java.io.File(path)
        val direct = try {
            if (f.exists() && f.canRead()) {
                val toRead = minOf(f.length(), maxHeaderBytes.toLong()).toInt()
                java.io.FileInputStream(f).use { it.readBytes(toRead) }
            } else null
        } catch (_: Exception) { null }

        val header = if (direct != null && direct.isNotEmpty()) direct
            else readHeaderViaSu(path, maxHeaderBytes)
                ?: throw java.io.IOException("lecture impossible (direct ET su ont echoue) pour $path — " +
                        "root accorde a l'app dans Magisk ? fichier existe ?")
        return try {
            predictFromHeader(path, f.name, header)
        } catch (e: java.nio.BufferUnderflowException) {
            // En-tete GGUF plus gros que maxHeaderBytes (modele avec BEAUCOUP
            // de tenseurs/metadonnees, ex un gros MoE type Gemma-26B-A4B) —
            // erreur claire au lieu de laisser fuiter l'exception brute.
            throw java.io.IOException("en-tete GGUF incomplet dans les " +
                    "${maxHeaderBytes / 1024 / 1024} Mo lus (modele avec beaucoup de " +
                    "tenseurs/metadonnees) — augmenter maxHeaderBytes")
        }
    }

    /** Extrait les premiers octets d'un fichier via `su -c dd` — contourne
     *  les permissions POSIX qui bloquent une lecture Java directe (meme
     *  cause que le bug de listing de FileBrowser.kt : /data/local/tmp et
     *  dossiers similaires sont root-only pour une app tierce). Lit les
     *  octets bruts du stdout du process (PAS via un Reader texte, qui
     *  corromprait les octets binaires du GGUF). */
    private fun readHeaderViaSu(path: String, maxHeaderBytes: Int): ByteArray? {
        return try {
            val mb = (maxHeaderBytes / (1024 * 1024)).coerceAtLeast(1)
            val escaped = path.replace("'", "'\\''")
            val p = ProcessBuilder("su", "-c", "dd if='$escaped' bs=1M count=$mb 2>/dev/null")
                .start()
            val bytes = p.inputStream.readBytes(maxHeaderBytes)
            p.waitFor()
            if (bytes.isEmpty()) null else bytes
        } catch (_: Exception) {
            null
        }
    }

    private fun java.io.InputStream.readBytes(n: Int): ByteArray {
        val buf = ByteArray(n)
        var off = 0
        while (off < n) {
            val r = read(buf, off, n - off)
            if (r < 0) break
            off += r
        }
        return if (off == n) buf else buf.copyOf(off)
    }

    /** Pipeline complet : en-tete distant -> architecture -> RAM par famille
     *  -> debit L3 simplifie -> correction MoE/dense calibree. */
    fun predict(repoId: String, filename: String): HfPrediction {
        val header = fetchGgufHeaderRemote(repoId, filename)
        return predictFromHeader(repoId, filename, header)
    }

    private fun predictFromHeader(repoId: String, filename: String, header: ByteArray): HfPrediction {
        val (meta, tensors) = parseGgufHeader(header)

        val arch = meta["general.architecture"] as? String ?: ""
        fun metaInt(vararg keys: String): Int? {
            for (k in keys) (meta[k] as? Number)?.let { return it.toInt() }
            return null
        }
        val nLayers = metaInt("$arch.block_count", "block_count") ?: 0
        val hiddenSize = metaInt("$arch.embedding_length", "embedding_length") ?: 0
        val nExperts = metaInt("$arch.expert_count", "expert_count") ?: 0
        val topK = metaInt("$arch.expert_used_count", "expert_used_count") ?: 0

        val byFamily = mutableMapOf<String, Double>()
        var totalBytes = 0.0
        for (t in tensors) {
            val fam = familyOf(t.name)
            byFamily[fam] = (byFamily[fam] ?: 0.0) + t.nbytes
            totalBytes += t.nbytes
        }

        // Roofline SIMPLIFIE (voir avertissement en tete de fichier) :
        // bytes/token approxime par (poids attn+mlp+moe)/n_layers, KV cache
        // et cout de dispatch NON modelises ici (contrairement a simulate_l3
        // complet cote Python) — uniquement pour situer un ordre de grandeur
        // AVANT d'aller profiler pour de vrai avec profiler.py.
        val streamableBytes = (byFamily["attn"] ?: 0.0) + (byFamily["mlp"] ?: 0.0) + (byFamily["moe"] ?: 0.0)
        val bytesPerTokenGb = if (nLayers > 0) (streamableBytes / nLayers) / 1e9 * nLayers / nLayers
            else streamableBytes / 1e9
        // note : bytes/token pour un modele dense ~ streamableBytes total / 1e9
        // (tout le poids est lu une fois par token en decode) — pas divise par
        // n_layers (chaque couche EST lue une fois par token, la somme totale
        // EST le trafic par token).
        val bytesPerTokenGbFinal = streamableBytes / 1e9
        val l3ColdTpsRaw = if (bytesPerTokenGbFinal > 0) L3_REFERENCE_TPS * (0.34 / bytesPerTokenGbFinal) else 0.0
        // 0.34 Go/token = reference Marco-Nano (cf. calibration Python
        // BW_EFF_MOE/simulate_l3) donnant 42.5 t/s dans le modele L3 original —
        // meme point d'ancrage que ROUTING_CALIBRATION.

        val (factor, confidence, note) = if (nExperts > 1) routingFactor(nExperts)
            else denseFactor(bytesPerTokenGbFinal)
        val correctedTps = l3ColdTpsRaw * factor

        return HfPrediction(repoId, filename, arch, nLayers, hiddenSize, nExperts, topK,
            totalBytes, byFamily, bytesPerTokenGbFinal, l3ColdTpsRaw, factor, confidence, note, correctedTps)
    }
}
