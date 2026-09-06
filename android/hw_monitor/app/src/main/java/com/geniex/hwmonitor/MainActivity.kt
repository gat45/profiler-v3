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

    private lateinit var tvCpu: TextView
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
        tvCpu = findViewById(R.id.tvCpu)
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
                val s = withContext(Dispatchers.IO) { HwMonitor.collect() }
                render(s)
                delay(1000)
            }
        }
    }

    private fun render(s: HwSample) {
        val cpu = s.cpuPct ?: -1
        val gpu = s.gpuPct ?: -1
        val npu = s.npuPct ?: -1
        tvCpu.text = "CPU  : ${if (cpu >= 0) "$cpu%" else "n/a"}  " +
                "${fmtMHz(s.cpu4Freq)} / ${fmtMHz(s.cpu8Freq)}"
        tvGpu.text = "GPU  : ${if (gpu >= 0) "$gpu%" else "n/a"}  " + fmtMHz(s.gpuFreq)
        tvNpu.text = "NPU  : ${if (npu >= 0) "$npu%" else "n/a"}  " +
                "${s.npuTemp?.let { "${it}°C" } ?: ""}"
        val temps = listOfNotNull(
            s.cpuTemp?.let { "CPU ${it}°" },
            s.gpuTemp?.let { "GPU ${it}°" },
            s.npuTemp?.let { "NPU ${it}°" },
        )
        tvTemp.text = "TEMP : " + (temps.joinToString("  ") ?: "n/a")
        tvRam.text = "RAM  : ${s.ramPct ?: 0}%  (${s.ramAvailMb ?: 0} MB libres)"
        tvBatt.text = "BATT : ${s.battPct ?: 0}%  ${s.battW ?: 0} W"
        val scriptOk = HwMonitor.scriptAvailable()
        tvStatus.text = if (scriptOk) "● temps réel (1 Hz)" else "⚠ script absent: ${HwMonitor.SCRIPT}"
    }

    private fun fmtMHz(hz: Long?): String =
        if (hz != null) "${hz / 1000}MHz" else "-"
}