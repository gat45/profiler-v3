package com.geniex.hwmonitor

import android.app.AlertDialog
import android.os.Bundle
import android.view.KeyEvent
import android.view.inputmethod.EditorInfo
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.EditText
import android.widget.ListView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.OnBackPressedCallback
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

// ===========================================================================
// FileBrowserActivity — "je veut naviguer dans mes dossier et en root, je
// veut choisir mes model" : navigation reelle du filesystem (via
// FileBrowser.kt, root si accorde a l'app), tap sur un .gguf -> lance
// HfPredictor.predictLocalFile() DIRECTEMENT dans l'app.
//
// REVISE (2026-09-06) suite a "ton navigateur a pas d'options pour voir
// les dossiers ni de retour" : ajout d'une barre d'adresse editable (taper
// un chemin directement), d'un bouton "Retour" explicite EN PLUS de
// l'entree ".." dans la liste, du bouton systeme Retour (gere pour
// remonter d'un niveau plutot que fermer l'activity), et de raccourcis
// vers les emplacements utiles (/sdcard, /data/local/tmp, /).
// ===========================================================================

class FileBrowserActivity : AppCompatActivity() {

    private lateinit var tvSource: TextView
    private lateinit var etPath: EditText
    private lateinit var listView: ListView
    private var currentPath = "/sdcard"
    private var currentEntries: List<FileEntry> = emptyList()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_file_browser)
        tvSource = findViewById(R.id.tvSource)
        etPath = findViewById(R.id.etPath)
        listView = findViewById(R.id.listView)

        findViewById<Button>(R.id.btnUp).setOnClickListener { goUp() }
        findViewById<Button>(R.id.btnGo).setOnClickListener { navigateTo(etPath.text.toString().trim().ifEmpty { "/" }) }
        findViewById<Button>(R.id.btnSdcard).setOnClickListener { navigateTo("/sdcard") }
        findViewById<Button>(R.id.btnDataLocalTmp).setOnClickListener { navigateTo("/data/local/tmp") }
        findViewById<Button>(R.id.btnRoot).setOnClickListener { navigateTo("/") }

        etPath.setOnEditorActionListener { _, actionId, event ->
            if (actionId == EditorInfo.IME_ACTION_GO ||
                (event?.keyCode == KeyEvent.KEYCODE_ENTER && event.action == KeyEvent.ACTION_DOWN)) {
                navigateTo(etPath.text.toString().trim().ifEmpty { "/" })
                true
            } else false
        }

        listView.setOnItemClickListener { _, _, position, _ ->
            if (position == 0 && currentPath != "/") {
                goUp()
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

        // Bouton systeme "Retour" -> remonte d'un niveau (pas de fermeture
        // brutale de l'activity tant qu'on n'est pas a la racine "/").
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (currentPath != "/") goUp() else {
                    isEnabled = false
                    onBackPressedDispatcher.onBackPressed()
                }
            }
        })

        navigateTo(currentPath)
    }

    private fun goUp() = navigateTo(FileBrowser.parentOf(currentPath))

    private fun navigateTo(path: String) {
        currentPath = path
        etPath.setText(path)
        lifecycleScope.launch {
            val (entries, source) = withContext(Dispatchers.IO) { FileBrowser.list(path) }
            currentEntries = entries
            tvSource.text = "source : $source  •  ${entries.size} elements" +
                if (source == "none") "  (rien lu — root non accorde a l'app et lecture directe refusee)" else ""
            val labels = mutableListOf<String>()
            if (path != "/") labels.add("⬅ .. (dossier parent)")
            for (e in entries) {
                val suffix = if (e.isDir) "/" else "  (${e.sizeBytes / 1024 / 1024} Mo)"
                val marker = if (!e.isDir && e.name.lowercase().endsWith(".gguf")) "★ " else if (e.isDir) "📁 " else "  "
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
