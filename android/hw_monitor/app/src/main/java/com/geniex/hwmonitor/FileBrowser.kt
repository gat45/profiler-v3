package com.geniex.hwmonitor

import java.io.BufferedReader
import java.io.InputStreamReader

// ===========================================================================
// FileBrowser — navigation reelle du filesystem EN ROOT, demande explicite
// ("je veut naviguer dans mes dossier et en root"). Une app normale
// (untrusted_app) ne peut lister QUE son propre bac a sable + les zones
// publiques (sdcard) — pour voir /data/local/tmp (ou vivent les GGUF de
// test) ou n'importe quel autre dossier du systeme, il faut passer par su.
//
// Essaie D'ABORD un listing direct (rapide, marche pour /sdcard et les
// dossiers de l'app), bascule sur `su -c ls` si refuse ET que root est
// accorde a l'app dans Magisk — meme pattern de fallback que
// ProcessMemoryMonitor.kt.
// ===========================================================================

data class FileEntry(val name: String, val isDir: Boolean, val sizeBytes: Long)

object FileBrowser {

    @Volatile var lastSuRawOutput: String? = null
        private set

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

    /** Liste un dossier. Retourne (entries, source) ou source = "direct"/"su"/"none".
     *
     *  BUG REEL trouve et corrige (2026-09-06) : `File.listFiles()` sur
     *  Android retourne un tableau VIDE (pas null) pour un dossier dont la
     *  permission POSIX interdit la lecture (ex /data/local/tmp lui-meme,
     *  mode drwxrwx--x, "other" n'a que le bit x pas r) — teste
     *  empiriquement via `su -c ls` qui, lui, listait bien du contenu. Le
     *  code d'origine prenait ce tableau vide comme un succes "direct" et
     *  ne tentait JAMAIS le fallback su. Fix : verifier explicitement
     *  f.canRead() (reflete le bit de permission POSIX reel) avant de faire
     *  confiance a un resultat vide. */
    fun list(path: String): Pair<List<FileEntry>, String> {
        val f = java.io.File(path)
        val direct = try { f.listFiles() } catch (_: Exception) { null }
        // NOTE : canRead() s'est revele PEU FIABLE ici (retourne parfois true
        // pour un dossier que listFiles() renvoie quand meme vide par
        // permission) — au lieu de lui faire confiance, on tente TOUJOURS su
        // en complement si le resultat direct est vide, et on garde su s'il
        // rapporte plus d'entrees. Seul un resultat direct NON VIDE est
        // retourne immediatement (evite un aller-retour su inutile).
        if (direct != null && direct.isNotEmpty()) {
            val entries = direct.map { FileEntry(it.name, it.isDirectory, if (it.isFile) it.length() else 0L) }
                .sortedWith(compareByDescending<FileEntry> { it.isDir }.thenBy { it.name.lowercase() })
            return entries to "direct"
        }

        // Fallback su : `ls -la` parse. BUG REEL trouve et corrige : le `ls`
        // toybox de ce device Android imprime la date en UN SEUL format ISO
        // "YYYY-MM-DD HH:MM" (2 tokens), pas le classique BSD "Mon DD HH:MM"
        // (3 tokens) suppose initialement -> 8 colonnes reelles, pas 9.
        // L'ancien code (limit=9, name=parts[8]) rejetait TOUTES les lignes
        // silencieusement (parts.size toujours 8 < 9), d'ou un dossier qui
        // semblait vide alors que `su -c ls` donnait bien du contenu (verifie
        // en comparant le raw output capture via lastSuRawOutput).
        // Colonnes reelles : perms links owner group size date time name
        val out = runSu("ls -la '${path.replace("'", "'\\''")}'") ?: return emptyList<FileEntry>() to "none"
        val entries = mutableListOf<FileEntry>()
        for (line in out.lines()) {
            val trimmed = line.trim()
            if (trimmed.isEmpty() || trimmed.startsWith("total ")) continue
            val parts = trimmed.split(Regex("\\s+"), limit = 8)
            if (parts.size < 8) continue
            val perms = parts[0]
            val name = parts[7]
            if (name == "." || name == "..") continue
            val isDir = perms.startsWith("d")
            val size = parts[4].toLongOrNull() ?: 0L
            entries.add(FileEntry(name, isDir, size))
        }
        val sorted = entries.sortedWith(compareByDescending<FileEntry> { it.isDir }.thenBy { it.name.lowercase() })
        lastSuRawOutput = out
        return sorted to (if (sorted.isNotEmpty() || out.isBlank()) "su" else "none")
    }

    fun parentOf(path: String): String {
        if (path == "/" || path.isEmpty()) return "/"
        val trimmed = path.trimEnd('/')
        val idx = trimmed.lastIndexOf('/')
        return if (idx <= 0) "/" else trimmed.substring(0, idx)
    }
}
