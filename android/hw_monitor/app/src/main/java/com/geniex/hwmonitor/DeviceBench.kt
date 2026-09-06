package com.geniex.hwmonitor

// ===========================================================================
// DeviceBench — benchmark du telephone lui-meme, pour produire un fichier
// --device-config compatible avec profiler_v3/profile_model.py::DEVICE_DEFAULTS.
//
// Demande explicite : "le projet doit bench le tel (pour d'autres snapdragon
// on a deja les scripts normalement)" — objectif : que CETTE app, installee
// sur N'IMPORTE QUEL device Snapdragon (pas seulement le SM8850 de
// developpement), puisse produire un point de depart de calibration SANS
// avoir besoin de tout le stack llama.cpp/ggml-hexagon deja compile pour ce
// device precis.
//
// HONNETETE explicite (comme partout ailleurs dans ce projet) : seule une
// PARTIE des constantes de DEVICE_DEFAULTS est mesurable depuis du code
// Kotlin pur (bande passante memoire CPU, debit compute flottant CPU) — ce
// sont des PROXIES cross-device raisonnables, PAS les memes grandeurs
// physiques que BW_EFF_GBS/PEAK_COMPUTE_TFLOPS reelles du HTP (qui
// necessitent un vrai run ggml-hexagon instrumente pour etre mesurees,
// cf. parse_hexagon_profile.py). Chaque valeur produite est marquee
// "measured" ou "not_measured_by_app" (garde la valeur par defaut Python),
// jamais fabriquee.
// ===========================================================================

data class BenchResult(
    val memBandwidthGBps: Double,
    val cpuComputeGflops: Double,
    val durationMs: Long,
)

object DeviceBench {

    /** Bande passante memoire CPU (lecture sequentielle d'un gros buffer,
     *  plusieurs passes pour depasser le cache L2/L3 et approcher la DDR
     *  reelle). Proxy raisonnable pour GGML_BW/BW_EFF_GBS SUR UN NOUVEAU
     *  device sans mesure LLM reelle encore disponible — PAS la bande
     *  passante DMA HTP reelle (chemin materiel different). */
    fun measureMemoryBandwidthGBps(bufferSizeMb: Int = 64, passes: Int = 20): Double {
        val size = bufferSizeMb * 1024 * 1024
        val src = ByteArray(size)
        val dst = ByteArray(size)
        java.util.Random(42).nextBytes(src)

        // Chauffe (JIT/ART, pages) — pas chronometree, evite de mesurer la
        // compilation JIT elle-meme plutot que la bande passante reelle.
        repeat(3) { System.arraycopy(src, 0, dst, 0, size) }

        // System.arraycopy() sur des tableaux primitifs est implemente en
        // memcpy natif (intrinsic JIT/ART) — c'est CA qui mesure vraiment la
        // bande passante memoire, contrairement a une boucle element-par-
        // element dominee par l'overhead d'appel de methode (bug reel trouve
        // et corrige : premiere version donnait 0.28 Go/s, absurdement bas
        // pour un flagship — signe evident d'un bench qui mesure autre chose
        // que la memoire).
        val t0 = System.nanoTime()
        repeat(passes) { System.arraycopy(src, 0, dst, 0, size) }
        val elapsedNs = System.nanoTime() - t0

        // read+write du meme volume par passe -> 2x le volume "utile" transite
        val totalBytes = size.toLong() * passes * 2
        val seconds = elapsedNs / 1e9
        return (totalBytes / 1e9) / seconds
    }

    /** Debit de calcul flottant CPU (multiplication-addition en boucle
     *  serree, un seul thread) — proxy grossier de puissance de calcul CPU,
     *  PAS le debit INT8 du HTP (unite materielle totalement differente,
     *  seule une vraie inference NPU peut mesurer PEAK_COMPUTE_TFLOPS). */
    fun measureCpuComputeGflops(iterations: Long = 200_000_000L): Double {
        var a = 1.0000001
        var b = 0.9999999
        val t0 = System.nanoTime()
        var i = 0L
        while (i < iterations) {
            a = a * b + 1e-9
            b = b * a + 1e-9
            i++
        }
        val elapsedNs = System.nanoTime() - t0
        if (a == Double.NaN.toDouble()) println("impossible: $a $b")
        // 2 FLOP (mul+add) x 2 (deux variables mises a jour) par iteration
        val flops = iterations * 4.0
        return (flops / (elapsedNs / 1e9)) / 1e9
    }

    fun run(): BenchResult {
        val t0 = System.currentTimeMillis()
        val bw = measureMemoryBandwidthGBps()
        val gflops = measureCpuComputeGflops()
        return BenchResult(bw, gflops, System.currentTimeMillis() - t0)
    }

    /** Produit un JSON compatible avec profile_model.py::load_device_profile
     *  (memes cles que DEVICE_DEFAULTS). Les cles non mesurables depuis
     *  l'app sont explicitement absentes (load_device_profile garde alors
     *  la valeur par defaut SM8850 codee en dur cote Python — comportement
     *  documente, pas un oubli). */
    fun toDeviceConfigJson(r: BenchResult, deviceModel: String): String {
        return """{
            |"_source": "DeviceBench.kt (hw_monitor app) — proxies CPU, PAS mesure HTP reelle",
            |"_device_model": "$deviceModel",
            |"_bench_duration_ms": ${r.durationMs},
            |"_note": "GGML_BW/BW_EFF_GBS ci-dessous sont des PROXIES bande passante memoire CPU, a considerer comme point de depart SEULEMENT sur un device sans mesure LLM reelle. Remplacer par une vraie calibration (profile_model.py --live-trace + --measured-tps) des que possible.",
            |"GGML_BW": ${r.memBandwidthGBps},
            |"BW_EFF_GBS": ${r.memBandwidthGBps * 0.5},
            |"_cpu_compute_gflops_measured": ${r.cpuComputeGflops},
            |"_not_measured_by_app": ["QAIRT_BW", "BW_EFF_DENSE", "BW_EFF_MOE", "FIXE_PAR_OP_MS", "VTCM_CAPACITY_MB", "VTCM_THRESHOLD", "PEAK_COMPUTE_TFLOPS", "DISPATCH_US", "THERMAL_TPS_FACTOR"]
            |}""".trimMargin().replace("\n", "")
    }
}
