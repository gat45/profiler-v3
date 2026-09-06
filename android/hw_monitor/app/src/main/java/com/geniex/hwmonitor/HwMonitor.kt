package com.geniex.hwmonitor

import java.io.BufferedReader
import java.io.InputStreamReader

// ===========================================================================
// HwMonitor — collecte temps réel CPU/GPU/NPU/RAM/batt/thermique via root.
// Le script hwmon.sh est poussé dans /data/local/tmp (une passe ~0.6s).
// Sortie : lignes "clé=valeur" parsées en Map. Best-effort : jamais d'exception.
//
// REVISE (2026-09-06) suite a "on ne voit pas tous les coeurs, les % et
// temperature de tous" : hwmon.sh boucle maintenant sur TOUS les coeurs
// (cpuN_freq/cpuN_pct pour chaque N present) et TOUTES les zones thermiques
// (temp_<nom_zone>), plus generique qu'avant (3 coeurs + 4 zones fixes).
// `raw` expose la map complete pour un affichage dynamique cote UI, les
// champs nommes existants restent pour compatibilite ascendante.
// ===========================================================================

data class HwSample(
    val cpuPct: Int? = null,
    val cpu0Freq: Long? = null,
    val cpu4Freq: Long? = null,
    val cpu8Freq: Long? = null,
    val gpuFreq: Long? = null,
    val gpuPct: Int? = null,
    val npuTemp: Int? = null,   // °C zone nsphvx (proxy activité NPU)
    val npuTemp2: Int? = null,  // °C zone qmx
    val gpuTemp: Int? = null,
    val cpuTemp: Int? = null,
    val ramPct: Int? = null,
    val ramAvailMb: Long? = null,
    val battPct: Int? = null,
    val battW: Int? = null,
    val fastrpcRate: Int? = null,   // signaux fastrpc/s (activité NPU directe)
    val raw: Map<String, String> = emptyMap(), // TOUTES les cles brutes (cpuN_*, temp_*)
    val ts: Long = System.currentTimeMillis(),
) {
    // Proxy d'activité NPU : priorité au comptage fastrpc réel si >0,
    // sinon proxy thermique (45°C≈idle, 90°C≈charge pleine).
    val npuPct: Int?
        get() = when {
            (fastrpcRate ?: 0) > 0 -> ((fastrpcRate ?: 0) * 100 / 200).coerceIn(0, 100)
            npuTemp != null -> ((npuTemp!! - 45).coerceIn(0, 45) * 100 / 45)
            else -> null
        }

    /** Liste (coeur, freqMHz, pct) pour TOUS les coeurs presents dans `raw`,
     *  tries par numero de coeur. */
    fun allCores(): List<Triple<Int, Long?, Int?>> {
        val re = Regex("""^cpu(\d+)_freq$""")
        return raw.keys.mapNotNull { k -> re.find(k)?.groupValues?.get(1)?.toIntOrNull() }
            .sorted()
            .map { n -> Triple(n, raw["cpu${n}_freq"]?.toLongOrNull(), raw["cpu${n}_pct"]?.toIntOrNull()) }
    }

    /** Liste (nom_zone, temperature_C) pour TOUTES les zones thermiques. */
    fun allTemps(): List<Pair<String, Int>> {
        return raw.entries.filter { it.key.startsWith("temp_") }
            .mapNotNull { (k, v) -> v.toIntOrNull()?.let { k.removePrefix("temp_") to it } }
            .sortedBy { it.first }
    }
}

object HwMonitor {
    const val SCRIPT = "/data/local/tmp/hwmon.sh"
    const val SHELL = "su"

    fun collect(): HwSample {
        val out = runSu("sh $SCRIPT")
        // comptage fastrpc sur la même fenêtre (best-effort, tracerfs root)
        val fastrpc = countFastrpc()
        return parse(out, fastrpc)
    }

    /** Compte les signaux fastrpc sur ~0.5s via tracepoint (proxy NPU direct).
     *  Retourne un taux /s, ou null si tracerfs indisponible. */
    fun countFastrpc(): Int? {
        val tracer = "/sys/kernel/tracing"
        if (runSu("test -w $tracer/tracing_on && echo ok")?.contains("ok") != true) return null
        val script = "echo 0 > $tracer/tracing_on;" +
            "echo > $tracer/trace;" +
            "echo 'fastrpc_dspsignal' > $tracer/set_event;" +
            "echo 1 > $tracer/tracing_on;" +
            "sleep 1;" +
            "echo 0 > $tracer/tracing_on;" +
            "grep -c fastrpc_dspsignal $tracer/trace;" +
            "echo '0' > $tracer/set_event"
        val out = runSu(script) ?: return null
        val n = out.trim().toIntOrNull() ?: return null
        return n
    }

    fun runSu(cmd: String): String? {
        return try {
            val p = ProcessBuilder(SHELL, "-c", cmd).redirectErrorStream(true).start()
            val r = BufferedReader(InputStreamReader(p.inputStream))
            val sb = StringBuilder()
            var line: String?
            while (r.readLine().also { line = it } != null) sb.append(line).append('\n')
            p.waitFor()
            sb.toString()
        } catch (e: Exception) {
            null
        }
    }

    fun parse(raw: String?, fastrpcRate: Int? = null): HwSample {
        if (raw == null) return HwSample(fastrpcRate = fastrpcRate)
        val m = mutableMapOf<String, String>()
        for (ln in raw.lines()) {
            val i = ln.indexOf('=')
            if (i > 0) m[ln.substring(0, i).trim()] = ln.substring(i + 1).trim()
        }
        fun int(k: String): Int? = m[k]?.toIntOrNull()
        fun long(k: String): Long? = m[k]?.toLongOrNull()
        // Alias retro-compatibles : hwmon.sh emet maintenant temp_<zone>
        // generique (pas npu_temp/gpu_temp/cpu_temp fixes) — on retrouve la
        // bonne zone par sous-chaine du nom, meme mapping qu'avant.
        fun tempBySubstring(vararg needles: String): Int? {
            for ((k, v) in m) {
                if (!k.startsWith("temp_")) continue
                val zone = k.removePrefix("temp_")
                if (needles.any { zone.contains(it, ignoreCase = true) }) return v.toIntOrNull()
            }
            return null
        }
        return HwSample(
            cpuPct = int("cpu_pct"),
            cpu0Freq = long("cpu0_freq"),
            cpu4Freq = long("cpu4_freq"),
            cpu8Freq = long("cpu8_freq"),
            gpuFreq = long("gpu_freq"),
            gpuPct = int("gpu_pct"),
            npuTemp = tempBySubstring("nsphvx"),
            npuTemp2 = tempBySubstring("qmx"),
            gpuTemp = tempBySubstring("gpuss"),
            cpuTemp = tempBySubstring("cpullc"),
            ramPct = int("ram_pct"),
            ramAvailMb = long("ram_avail_mb"),
            battPct = int("batt_pct"),
            battW = int("batt_w"),
            fastrpcRate = fastrpcRate,
            raw = m,
        )
    }

    // Signale si le script est présent sur le device (sinon le pousser)
    fun scriptAvailable(): Boolean {
        val out = runSu("ls $SCRIPT 2>/dev/null") ?: return false
        return out.contains("hwmon.sh")
    }
}
