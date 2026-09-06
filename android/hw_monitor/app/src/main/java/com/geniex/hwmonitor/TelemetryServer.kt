package com.geniex.hwmonitor

import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStream
import java.net.ServerSocket
import java.net.Socket
import java.net.SocketTimeoutException
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicLong

// ===========================================================================
// TelemetryServer — expose HwMonitor (global, root) + ProcessMemoryMonitor
// (par-pid, sans root) en JSON sur localhost, pour reponse a la demande
// explicite : "je veut pouvoir lancer le truc via une commande a distance ou
// du tel" (MCP/PC + declenchement depuis le device lui-meme).
//
// Zero dependance externe (pas de NanoHTTPD/Ktor) : java.net.ServerSocket
// brut, pour eviter tout risque de resolution Gradle sur cette machine.
// Ecoute sur 127.0.0.1 UNIQUEMENT (jamais expose au reseau) — acces externe
// via `adb forward tcp:8082 tcp:8082` puis http://127.0.0.1:8082/... cote
// PC (c'est ce que le wrapper MCP appelle).
//
// Endpoints :
//   GET  /telemetry             -> HwMonitor.collect() (etat global, si su
//                                   dispo) + liste de pid connus (llama*)
//   GET  /telemetry?pid=NNN     -> + ProcessMemoryMonitor.sample(NNN)
//   GET  /telemetry?name=llama  -> resout le pid via un scan de /proc,
//                                   puis meme chose que ?pid=
//   POST /run  {cmd:"..."}      -> lance une commande shell EN ARRIERE-PLAN
//                                   (ex demarrer un run llama-cli deja
//                                   present sur le device) et retourne son
//                                   pid immediatement (async, pas d'attente)
//   GET  /run/status?pid=NNN    -> le process tourne-t-il encore (via
//                                   /proc/<pid> exists)
//   GET  /health                -> {"ok":true} — sonde simple
//
// Enrichissement (suite a la demande "ameliore enrichie precise") :
//   - gpuBusyPct calcule (delta de 2 lectures kgsl/gpubusy espacees de
//     ~150ms), pas juste le compteur brut cumulatif.
//   - horodatage ISO-8601 en plus du epoch ms.
//   - liste de TOUS les pid dont le nom de cmdline contient "llama", pas
//     un seul pid suppose.
// ===========================================================================

object TelemetryServer {
    private const val PORT = 8082
    private var serverSocket: ServerSocket? = null
    private val running = AtomicBoolean(false)
    private val pool = Executors.newCachedThreadPool()
    private val runningJobs = mutableMapOf<Int, Process>()

    fun start() {
        if (running.getAndSet(true)) return
        pool.execute {
            try {
                val ss = ServerSocket(PORT, 16, java.net.InetAddress.getByName("127.0.0.1"))
                serverSocket = ss
                while (running.get()) {
                    val client = try { ss.accept() } catch (e: Exception) { break }
                    pool.execute { handle(client) }
                }
            } catch (e: Exception) {
                running.set(false)
            }
        }
    }

    fun stop() {
        running.set(false)
        try { serverSocket?.close() } catch (_: Exception) {}
    }

    fun isRunning(): Boolean = running.get()

    private fun handle(client: Socket) {
        try {
            client.soTimeout = 5000
            val reader = BufferedReader(InputStreamReader(client.getInputStream()))
            val requestLine = reader.readLine() ?: return
            val parts = requestLine.split(" ")
            if (parts.size < 2) return
            val method = parts[0]
            val target = parts[1]

            var contentLength = 0
            var line: String?
            while (reader.readLine().also { line = it } != null && line!!.isNotEmpty()) {
                if (line!!.startsWith("Content-Length:", ignoreCase = true)) {
                    contentLength = line!!.substringAfter(":").trim().toIntOrNull() ?: 0
                }
            }
            val body = if (contentLength > 0) {
                val buf = CharArray(contentLength)
                reader.read(buf, 0, contentLength)
                String(buf)
            } else ""

            val (path, query) = splitTarget(target)
            val params = parseQuery(query)

            val (status, json) = route(method, path, params, body)
            writeResponse(client.getOutputStream(), status, json)
        } catch (_: SocketTimeoutException) {
            // client lent/mort — ignore
        } catch (_: Exception) {
            // ne jamais faire planter le serveur pour une requete malformee
        } finally {
            try { client.close() } catch (_: Exception) {}
        }
    }

    private fun route(method: String, path: String, params: Map<String, String>, body: String): Pair<Int, String> {
        return when {
            path == "/health" -> 200 to """{"ok":true,"port":$PORT}"""
            path == "/telemetry" && method == "GET" -> 200 to telemetryJson(params)
            path == "/run" && method == "POST" -> runCommand(body)
            path == "/run/status" && method == "GET" -> runStatus(params)
            else -> 404 to """{"error":"not found","path":"$path"}"""
        }
    }

    // -- /telemetry ----------------------------------------------------

    private fun telemetryJson(params: Map<String, String>): String {
        val hw = HwMonitor.collect()
        val gpuBusyPct = measureGpuBusyPct()
        val targetPid = params["pid"]?.toIntOrNull()
            ?: params["name"]?.let { findPidByCmdlineSubstring(it) }
        val procPart = targetPid?.let { pid ->
            val s = ProcessMemoryMonitor.sample(pid)
            ""","process":{"pid":${s.pid},"vmRssKb":${s.vmRssKb ?: "null"},"vmHwmKb":${s.vmHwmKb ?: "null"},"gpuMemBytes":${s.gpuMemBytes ?: "null"},"source":"${s.source}","error":${s.error?.let { "\"${it.replace("\"", "'")}\"" } ?: "null"}}"""
        } ?: ""
        val allLlamaPids = findAllPidsByCmdlineSubstring("llama")
        val isoNow = java.time.Instant.now().toString()
        return """{
            |"ts_ms":${System.currentTimeMillis()},
            |"ts_iso":"$isoNow",
            |"cpuPct":${hw.cpuPct ?: "null"},
            |"cpu4FreqHz":${hw.cpu4Freq ?: "null"},
            |"cpu8FreqHz":${hw.cpu8Freq ?: "null"},
            |"gpuFreqHz":${hw.gpuFreq ?: "null"},
            |"gpuBusyPctInstant":${gpuBusyPct ?: "null"},
            |"npuTempC":${hw.npuTemp ?: "null"},
            |"gpuTempC":${hw.gpuTemp ?: "null"},
            |"cpuTempC":${hw.cpuTemp ?: "null"},
            |"ramPct":${hw.ramPct ?: "null"},
            |"ramAvailMb":${hw.ramAvailMb ?: "null"},
            |"battPct":${hw.battPct ?: "null"},
            |"battW":${hw.battW ?: "null"},
            |"fastrpcRatePerSec":${hw.fastrpcRate ?: "null"},
            |"llamaPids":[${allLlamaPids.joinToString(",")}],
            |"gpuBusyError":${lastGpuBusyError?.let { "\"${it.replace("\"", "'")}\"" } ?: "null"},
            |"cmdlineScanError":${lastCmdlineScanError?.let { "\"${it.replace("\"", "'")}\"" } ?: "null"}
            |$procPart
            |}""".trimMargin().replace("\n", "")
    }

    /** Deux lectures espacees de kgsl/gpubusy -> % instantane, plus precis
     *  qu'un compteur cumulatif brut depuis le boot. */
    @Volatile private var lastGpuBusyError: String? = null
    @Volatile private var lastCmdlineScanError: String? = null

    private fun measureGpuBusyPct(): Int? {
        return try {
            val f = java.io.File("/sys/class/kgsl/kgsl-3d0/gpubusy")
            fun read(): Pair<Long, Long>? {
                val parts = f.readText().trim().split(Regex("\\s+"))
                if (parts.size != 2) return null
                return (parts[0].toLongOrNull() ?: return null) to (parts[1].toLongOrNull() ?: return null)
            }
            val a = read() ?: run { lastGpuBusyError = "parse_fail_a"; return null }
            Thread.sleep(150)
            val b = read() ?: run { lastGpuBusyError = "parse_fail_b"; return null }
            val dBusy = b.first - a.first
            val dTotal = b.second - a.second
            lastGpuBusyError = null
            if (dTotal > 0) ((dBusy * 100) / dTotal).toInt().coerceIn(0, 100) else null
        } catch (e: Exception) {
            lastGpuBusyError = "${e.javaClass.simpleName}: ${e.message}"
            null
        }
    }

    private fun findPidByCmdlineSubstring(needle: String): Int? =
        findAllPidsByCmdlineSubstring(needle).firstOrNull()

    private fun findAllPidsByCmdlineSubstring(needle: String): List<Int> {
        val result = mutableListOf<Int>()
        val procDir = java.io.File("/proc")
        val pids = procDir.listFiles { f -> f.name.toIntOrNull() != null }
            ?: run { lastCmdlineScanError = "listFiles(/proc) == null"; return result }
        var denied = 0
        for (pf in pids) {
            val pid = pf.name.toIntOrNull() ?: continue
            val cmdlineFile = java.io.File(pf, "cmdline")
            try {
                val cmdline = cmdlineFile.readText().replace(' ', ' ')
                if (cmdline.contains(needle)) result.add(pid)
            } catch (_: Exception) {
                denied++ // process disparu OU lecture refusee (SELinux/hidepid)
            }
        }
        lastCmdlineScanError = if (result.isEmpty() && denied > 0)
            "0 match sur ${pids.size} pid listes, $denied lectures refusees " +
            "(SELinux/hidepid probable sur /proc/<pid>/cmdline d'un autre uid)" else null
        return result
    }

    // -- /run (declenchement a distance) --------------------------------

    /** body attendu : {"cmd":"cd /data/local/tmp/... && ./llama cli ..."}
     *  Lance via `sh -c` en arriere-plan, retourne le pid immediatement.
     *  ATTENTION : execute n'importe quelle commande shell — le serveur
     *  n'ecoute QUE sur 127.0.0.1, jamais expose reseau, mais reste un
     *  vecteur d'execution de commande a garder conscient (usage local/dev
     *  uniquement, jamais a exposer au-dela d'adb forward). */
    private fun runCommand(body: String): Pair<Int, String> {
        val cmd = Regex(""""cmd"\s*:\s*"((?:[^"\\]|\\.)*)"""").find(body)
            ?.groupValues?.get(1)?.replace("\\\"", "\"")
            ?: return 400 to """{"error":"body must be {\"cmd\":\"...\"}"}"""
        return try {
            // Process.pid() (java.lang) indisponible dans ce SDK Android tel
            // que configure ici (unresolved a la compilation) -> on recupere
            // le pid en le faisant s'auto-imprimer via $$ (pid du shell),
            // puis `exec` remplace le shell par la commande reelle SANS
            // changer de pid (exec ne fork pas) -> le pid imprime EST celui
            // du process final, pas juste celui du sh intermediaire.
            val p = ProcessBuilder("sh", "-c", "echo \$\$; exec $cmd")
                .redirectErrorStream(true).start()
            val firstLine = BufferedReader(InputStreamReader(p.inputStream)).readLine()
            val pid = firstLine?.trim()?.toIntOrNull()
                ?: return 500 to """{"error":"could not read pid from launched process"}"""
            runningJobs[pid] = p
            200 to """{"started":true,"pid":$pid}"""
        } catch (e: Exception) {
            500 to """{"error":"${e.message}"}"""
        }
    }

    private fun runStatus(params: Map<String, String>): Pair<Int, String> {
        val pid = params["pid"]?.toIntOrNull() ?: return 400 to """{"error":"missing pid"}"""
        val alive = java.io.File("/proc/$pid").exists()
        return 200 to """{"pid":$pid,"alive":$alive}"""
    }

    // -- utilitaires HTTP minimalistes ----------------------------------

    private fun splitTarget(target: String): Pair<String, String> {
        val i = target.indexOf('?')
        return if (i < 0) target to "" else target.substring(0, i) to target.substring(i + 1)
    }

    private fun parseQuery(query: String): Map<String, String> {
        if (query.isEmpty()) return emptyMap()
        return query.split("&").mapNotNull {
            val kv = it.split("=", limit = 2)
            if (kv.size == 2) java.net.URLDecoder.decode(kv[0], "UTF-8") to java.net.URLDecoder.decode(kv[1], "UTF-8") else null
        }.toMap()
    }

    private fun writeResponse(out: OutputStream, status: Int, jsonBody: String) {
        val statusText = if (status == 200) "OK" else if (status == 404) "Not Found" else if (status == 400) "Bad Request" else "Error"
        val bytes = jsonBody.toByteArray(Charsets.UTF_8)
        val header = "HTTP/1.1 $status $statusText\r\n" +
                "Content-Type: application/json\r\n" +
                "Content-Length: ${bytes.size}\r\n" +
                "Connection: close\r\n\r\n"
        out.write(header.toByteArray(Charsets.UTF_8))
        out.write(bytes)
        out.flush()
    }
}
