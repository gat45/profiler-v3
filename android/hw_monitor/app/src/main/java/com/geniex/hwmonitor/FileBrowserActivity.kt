package com.geniex.hwmonitor

import android.app.AlertDialog
import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.ListView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

// ===========================================================================
// FileBrowserActivity — "je veut naviguer dans mes dossier et en root, je
// veut choisir mes model" : navigation reelle du filesystem (via
// FileBrowser.kt, root si accorde a l'app), tap sur un .gguf -> lance
// HfPredictor.predictLocalFile() DIRECTEMENT dans l'app (pas besoin de
// taper un chemin a la main sur PC/curl).
// ===========================================================================

class FileBrowserActivity : AppCompatActivity() {

    private lateinit var tvPath: TextView
    private lateinit var tvSource: TextView
    private lateinit var listView: ListView
    // Point de depart : /sdcard, TOUJOURS listable sans root. /data/local/tmp
    // (racine) refuse le listing direct meme sans root sur certains devices
    // (permissions restrictives sur ce dossier precis) meme si SES SOUS-
    // DOSSIERS (ex /data/local/tmp/sweep) restent listables normalement —
    // verifie empiriquement, pas un bug de ce code. Naviguer manuellement
    // vers un sous-dossier connu fonctionne ; lister /data/local/tmp lui-
    // meme necessite le root (accorder l'app dans Magisk).
    private var currentPath = "/sdcard"
    private var currentEntries: List<FileEntry> = emptyList()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_file_browser)
        tvPath = findViewById(R.id.tvPath)
        tvSource = findViewById(R.id.tvSource)
        listView = findViewById(R.id.listView)

        listView.setOnItemClickListener { _, _, position, _ ->
            if (position == 0 && currentPath != "/") {
                navigateTo(FileBrowser.parentOf(currentPath))
                return@setOnItemClickListener
            }
            val idx = if (currentPath != "/") position - 1 else position
            val entry = currentEntries.getOrNull(idx) ?: return@setOnItemClickListener
            val fullPath = if (currentPath.endsWith("/")) currentPath + entry.name else "$currentPath/${entry.name}"
            if (entry.isDir) {
                navigateTo(fullPath)
            } else if (entry.name.lowercase().endsWith(".gguf")) {
                predictAndShow(fullPath)
            } else {
                Toast.makeText(this, "${entry.name} : pas un .gguf", Toast.LENGTH_SHORT).show()
            }
        }

        navigateTo(currentPath)
    }

    private fun navigateTo(path: String) {
        currentPath = path
        tvPath.text = path
        lifecycleScope.launch {
            val (entries, source) = withContext(Dispatchers.IO) { FileBrowser.list(path) }
            currentEntries = entries
            tvSource.text = "source : $source" +
                if (source == "none") "  (rien lu — root non accorde a l'app et lecture directe refusee)" else ""
            val labels = mutableListOf<String>()
            if (path != "/") labels.add(".. (dossier parent)")
            for (e in entries) {
                val suffix = if (e.isDir) "/" else "  (${e.sizeBytes / 1024 / 1024} Mo)"
                val marker = if (!e.isDir && e.name.lowercase().endsWith(".gguf")) "★ " else "  "
                labels.add("$marker${e.name}$suffix")
            }
            listView.adapter = ArrayAdapter(this@FileBrowserActivity,
                android.R.layout.simple_list_item_1, labels)
        }
    }

    private fun predictAndShow(path: String) {
        Toast.makeText(this, "Analyse de $path...", Toast.LENGTH_SHORT).show()
        lifecycleScope.launch {
            var errorMsg: String? = null
            val result: HfPrediction? = withContext(Dispatchers.IO) {
                try {
                    HfPredictor.predictLocalFile(path)
                } catch (e: Exception) {
                    errorMsg = e.message ?: e.javaClass.simpleName
                    null
                }
            }
            val text = if (result != null) {
                """Modele : ${result.filename}
                   |Architecture : ${result.architecture}
                   |Couches : ${result.nLayers}  Hidden : ${result.hiddenSize}
                   |Experts : ${result.nExperts} (top-${result.topK})
                   |Taille totale : ${"%.2f".format(result.totalBytes / 1e9)} Go
                   |Debit L3 brut : ${"%.1f".format(result.l3ColdTpsRaw)} t/s
                   |Facteur correction : x${"%.2f".format(result.correctionFactor)} (${result.correctionConfidence})
                   |Debit corrige : ${"%.1f".format(result.correctedTps)} t/s
                   |
                   |ATTENTION : modele roofline SIMPLIFIE, pas le simulateur
                   |L3 complet — utiliser profiler_v3/predict_from_hf.py sur
                   |PC pour une decision fine.""".trimMargin()
            } else {
                "Erreur : $errorMsg"
            }
            AlertDialog.Builder(this@FileBrowserActivity)
                .setTitle("Prediction")
                .setMessage(text)
                .setPositiveButton("OK", null)
                .show()
        }
    }
}
