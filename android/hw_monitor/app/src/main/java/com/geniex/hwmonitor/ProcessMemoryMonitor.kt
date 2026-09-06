package com.geniex.hwmonitor

import java.io.BufferedReader
import java.io.File
import java.io.InputStreamReader

// ===========================================================================
// ProcessMemoryMonitor — complement de HwMonitor (etat GLOBAL, root via su)
// : celui-ci lit l'attribution PAR PROCESSUS (RAM host + memoire GPU).
//
// CORRECTION 2026-09-06 (teste reellement sur device, PAS juste suppose) :
// l'hypothese initiale "marche sans root, verifie en adb shell" etait
// INCOMPLETE — verifie en adb shell (uid=2000, DOMAINE "shell") ne prouve PAS
// que ca marche depuis un process d'app installee (domaine SELinux
// "untrusted_app", bien plus restreint). Teste directement (run-as ET
// process reel de l'app) :
//   - /proc/<pid>/status  d'un pid ETRANGER -> "Permission denied" meme sous
//     run-as -> BLOQUE pour untrusted_app, PAS un bug de code.
//   - dumpsys gpu         -> "Error ... FAILED_TRANSACTION" sous run-as pour
//     le service 'gpu' -> BLOQUE de la meme facon.
//   - /proc/self/status (SOI-MEME) et les fichiers globaux (/proc/meminfo,
//     /sys/class/kgsl/kgsl-3d0/gpubusy) restent lisibles sans su : verifie
//     et confirme fonctionnel EN VRAI depuis le process de l'app (pas juste
//     run-as) — voir TelemetryServer.measureGpuBusyPct(), qui a fonctionne.
//
// CONSEQUENCE : la lecture cross-process (RSS/memoire GPU d'un AUTRE
// process, ex llama-cli lance separement) necessite soit root (via su, ce
// device a Magisk pour le shell adb mais PAS encore accorde a cette app —
// action manuelle unique requise : ouvrir Magisk, accorder root a
// com.geniex.hwmonitor), soit rester sur le chemin PC+adb
// (profiler_v3/android_telemetry.py, qui tourne dans le domaine "shell" et
// n'a PAS cette restriction, verifie fonctionnel).
//
// Ce fichier essaie D'ABORD la lecture directe (rapide, pas de fork), et
// BASCULE sur `su` si elle echoue ET que su est disponible — fonctionne des
// que l'utilisateur accorde root a l'app dans Magisk, sans autre changement.
// ===========================================================================

data class ProcessMemorySample(
    val pid: Int,
    val vmRssKb: Long? = null,
    val vmHwmKb: Long? = null,   // pic RSS depuis le lancement du process
    val gpuMemBytes: Long? = null,
    val source: String = "none", // "direct" | "su" | "none" — QUELLE methode a fourni la valeur
    val error: String? = null,   // dernier echec rencontre, jamais un null muet
    val ts: Long = System.currentTimeMillis(),
)

object ProcessMemoryMonitor {

    private fun runSu(cmd: String): String? {
        return try {
            val p = ProcessBuilder("su", "-c", cmd).redirectErrorStream(true).start()
            val out = BufferedReader(InputStreamReader(p.inputStream)).readText()
            p.waitFor()
            out
        } catch (_: Exception) {
            null
        }
    }

    /** Lecture directe /proc/<pid>/status. Marche pour son PROPRE pid sans
     *  root ; refusee (Permission denied) pour un pid etranger sous
     *  untrusted_app (verifie empiriquement) — retourne null dans ce cas,
     *  PAS une exception avalee silencieusement en amont. */
    private fun readStatusDirect(pid: Int): Pair<Long?, Long?>? {
        val f = File("/proc/$pid/status")
        if (!f.exists()) return null
        return try {
            var rss: Long? = null
            var hwm: Long? = null
            f.bufferedReader().use { r ->
                r.forEachLine { line ->
                    when {
                        line.startsWith("VmRSS:") -> rss = line.filter { it.isDigit() }.toLongOrNull()
                        line.startsWith("VmHWM:") -> hwm = line.filter { it.isDigit() }.toLongOrNull()
                    }
                }
            }
            if (rss == null && hwm == null) null else rss to hwm
        } catch (_: Exception) {
            null // Permission denied la plupart du temps sur un pid etranger
        }
    }

    private fun readStatusSu(pid: Int): Pair<Long?, Long?>? {
        val out = runSu("cat /proc/$pid/status") ?: return null
        if (out.contains("No such file") || out.contains("Permission denied")) return null
        val rss = Regex("""VmRSS:\s+(\d+)""").find(out)?.groupValues?.get(1)?.toLongOrNull()
        val hwm = Regex("""VmHWM:\s+(\d+)""").find(out)?.groupValues?.get(1)?.toLongOrNull()
        return if (rss == null && hwm == null) null else rss to hwm
    }

    private fun readGpuMemDirect(pid: Int): Long? {
        return try {
            val p = ProcessBuilder("dumpsys", "gpu").redirectErrorStream(true).start()
            val out = BufferedReader(InputStreamReader(p.inputStream)).readText()
            p.waitFor()
            if (out.contains("FAILED_TRANSACTION") || out.contains("PERMISSION_DENIED")) return null
            Regex("""Proc $pid total: (\d+)""").find(out)?.groupValues?.get(1)?.toLongOrNull()
        } catch (_: Exception) {
            null
        }
    }

    private fun readGpuMemSu(pid: Int): Long? {
        val out = runSu("dumpsys gpu") ?: return null
        return Regex("""Proc $pid total: (\d+)""").find(out)?.groupValues?.get(1)?.toLongOrNull()
    }

    /** Echantillon complet pour un pid — essaie direct D'ABORD (rapide,
     *  fonctionne pour self ou si le device n'a pas de restriction stricte),
     *  bascule sur su SI necessaire et disponible. `source`/`error` disent
     *  explicitement ce qui s'est passe plutot qu'un null ambigu. */
    fun sample(pid: Int): ProcessMemorySample {
        var source = "none"
        var error: String? = null

        var rssHwm = readStatusDirect(pid)
        if (rssHwm != null) {
            source = "direct"
        } else {
            rssHwm = readStatusSu(pid)
            if (rssHwm != null) {
                source = "su"
            } else {
                error = "status inaccessible en direct (SELinux untrusted_app probable " +
                        "sur un pid etranger) ET via su (root non accorde a cette app dans " +
                        "Magisk, ou su indisponible)"
            }
        }

        var gpuMem = readGpuMemDirect(pid)
        if (gpuMem == null) {
            gpuMem = readGpuMemSu(pid)
        }

        return ProcessMemorySample(pid = pid, vmRssKb = rssHwm?.first, vmHwmKb = rssHwm?.second,
            gpuMemBytes = gpuMem, source = source, error = error)
    }

    /** Raccourci pour le process courant (l'app elle-meme) — toujours
     *  "direct", jamais besoin de su pour lire ses propres fichiers proc. */
    fun sampleSelf(): ProcessMemorySample = sample(android.os.Process.myPid())
}
