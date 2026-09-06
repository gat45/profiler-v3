package com.geniex.hwmonitor

import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : AppCompatActivity() {

    private lateinit var tvCores: TextView
    private lateinit var tvGpu: TextView
    private lateinit var tvNpu: TextView
    private lateinit var tvTemp: TextView
    private lateinit var tvRam: TextView
    private lateinit var tvBatt: TextView
    private lateinit var tvStatus: TextView
    private lateinit var tvServer: TextView
    private lateinit var btnToggleServer: Button
    private var job: Job? = null
    private var running = false

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        tvCores = findViewById(R.id.tvCores)
        tvGpu = findViewById(R.id.tvGpu)
        tvNpu = findViewById(R.id.tvNpu)
        tvTemp = findViewById(R.id.tvTemp)
        tvRam = findViewById(R.id.tvRam)
        tvBatt = findViewById(R.id.tvBatt)
        tvStatus = findViewById(R.id.tvStatus)
        tvServer = findViewById(R.id.tvServer)
        btnToggleServer = findViewById(R.id.btnToggleServer)

        // Demarre automatiquement au lancement de l'app — repond a la
        // demande "je veut pouvoir lancer le truc via une commande a
        // distance ou du tel" : au lancement de l'app (depuis le tel OU
        // via `adb shell am start`), le serveur est deja pret pour
        // `adb forward tcp:8082 tcp:8082` cote PC.
        TelemetryServer.start()
        btnToggleServer.setOnClickListener {
            if (TelemetryServer.isRunning()) TelemetryServer.stop() else TelemetryServer.start()
            renderServerStatus()
        }
        findViewById<Button>(R.id.btnBrowseFiles).setOnClickListener {
            startActivity(android.content.Intent(this, FileBrowserActivity::class.java))
        }
        renderServerStatus()
    }

    private fun renderServerStatus() {
        tvServer.text = if (TelemetryServer.isRunning())
            "SERVEUR : ● actif sur 127.0.0.1:8082 (adb forward tcp:8082 tcp:8082)"
        else
            "SERVEUR : ○ arrete"
    }

    override fun onStart() {
        super.onStart()
        startLoop()
    }

    override fun onStop() {
        super.onStop()
        job?.cancel()
        job = null
        running = false
    }

    private fun startLoop() {
        if (running) return
        running = true
        job = lifecycleScope.launch {
            while (isActive) {
                // scriptAvailable() fait un appel su bloquant — BUG REEL
                // trouve et corrige (demande "ca reffeche pas") : l'ancienne
                // version l'appelait depuis render() APRES le retour du
                // withContext(IO), donc SUR LE THREAD PRINCIPAL, a chaque
                // tick (1 Hz) — un appel su bloquant sur le thread UI cause
                // des a-coups/gel visibles. Deplace ici, dans le meme bloc IO.
                val (s, scriptOk) = withContext(Dispatchers.IO) {
                    HwMonitor.collect() to HwMonitor.scriptAvailable()
                }
                render(s, scriptOk)
                delay(1000)
            }
        }
    }

    private fun render(s: HwSample, scriptOk: Boolean) {
        val gpu = s.gpuPct ?: -1
        val npu = s.npuPct ?: -1

        // TOUS les coeurs (demande explicite "on ne voit pas tous les coeurs,
        // les % et temperature de tous") — une ligne par coeur present,
        // plus l'agregat global en tete.
        val cores = s.allCores()
        val coreLines = StringBuilder("CPU global : ${s.cpuPct?.let { "$it%" } ?: "n/a"}\n")
        for ((n, freq, pct) in cores) {
            coreLines.append("  core$n : ${pct?.let { "$it%" } ?: "n/a"}  ${fmtMHz(freq)}\n")
        }
        if (cores.isEmpty()) coreLines.append("  (aucun coeur lu — script hwmon.sh a jour ? root accorde ?)")
        tvCores.text = coreLines.toString().trimEnd()

        tvGpu.text = "GPU  : ${if (gpu >= 0) "$gpu%" else "n/a"}  " + fmtMHz(s.gpuFreq)
        tvNpu.text = "NPU  : ${if (npu >= 0) "$npu%" else "n/a"}  " +
                "${s.npuTemp?.let { "${it}°C" } ?: ""}"

        // TOUTES les zones thermiques (pas juste 3 fixes)
        val temps = s.allTemps()
        tvTemp.text = if (temps.isEmpty()) "TEMP : n/a"
            else "TEMP (${temps.size} zones) :\n" + temps.joinToString("\n") { (name, t) -> "  $name : ${t}°C" }

        tvRam.text = "RAM  : ${s.ramPct ?: 0}%  (${s.ramAvailMb ?: 0} MB libres)"
        tvBatt.text = "BATT : ${s.battPct ?: 0}%  ${s.battW ?: 0} W"
        tvStatus.text = if (scriptOk) "● temps réel (1 Hz)" else "⚠ script absent: ${HwMonitor.SCRIPT}"
    }

    private fun fmtMHz(hz: Long?): String =
        if (hz != null) "${hz / 1000}MHz" else "-"
}