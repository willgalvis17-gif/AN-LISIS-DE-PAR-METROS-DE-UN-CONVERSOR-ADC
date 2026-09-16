# ============================================================
# ADC_testing_v5.py
# Raspberry Pi Pico 2 W (RP2350) - MicroPython
# Dynamic ADC characterization using FFT (Memory-Optimized)
#
# Universidad Militar Nueva Granada
# Programa de Ingeniería en Telecomunicaciones
# Comunicaciones Digitales
#
# Main objectives:
#   - Acquire ADC samples with a software sampling marker on GP15.
#   - Obtain the time-domain waveform and frequency spectrum.
#   - Measure fundamental frequency and dynamic ADC metrics.
#   - Compare Hann and rectangular windows.
#   - Study Nyquist and aliasing by changing the analog input frequency.
#   - Distinguish acquired record length N_s from FFT length N_FFT (M).
# ============================================================

from machine import ADC, Pin
import utime
import array
import math
import cmath
import gc

# ------------------------------------------------------------
# Hardware and acquisition configuration
# ------------------------------------------------------------

ADC_PIN = 27
SAMPLE_MARKER_PIN = 15

ADC_BITS = 12        # N = 12 bits de resolución del ADC
VREF = 3.3

N_SAMPLES = 512      # N_s = Número de muestras capturadas (modificable por el estudiante)
N_FFT = 1024         # M = Puntos de la FFT (N_FFT >= N_s)
FS_TARGET = 2000
DT_US = int(round(1_000_000 / FS_TARGET))

DEFAULT_INPUT_FREQ_HZ = 250.0
DEFAULT_WINDOW = "hann"

NUM_HARMONICS = 5

MAIN_LOBE_HALF_WIDTH = 4
GUARD_HALF_WIDTH = 8
HARMONIC_SEARCH_HALF_WIDTH = 2

adc = ADC(Pin(ADC_PIN))
sample_marker = Pin(SAMPLE_MARKER_PIN, Pin.OUT)
sample_marker.value(0)


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def mean(values):
    return sum(values) / len(values)


def median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    m = n // 2
    if n % 2:
        return ordered[m]
    return 0.5 * (ordered[m - 1] + ordered[m])


def std_population(values, avg=None):
    if not values:
        return 0.0
    if avg is None:
        avg = mean(values)
    return math.sqrt(sum((x - avg) ** 2 for x in values) / len(values))


def db10_ratio(num, den):
    if num <= 0 or den <= 0:
        return -200.0
    return 10.0 * math.log10(num / den)


def db20_ratio(num, den):
    if num <= 0 or den <= 0:
        return -200.0
    return 20.0 * math.log10(num / den)


def alias_to_first_nyquist_zone(freq, fs):
    if fs <= 0:
        return 0.0
    f = abs(freq) % fs
    if f > fs / 2.0:
        f = fs - f
    return f


def bin_region(center, half_width, max_bin):
    start = max(1, center - half_width)
    stop = min(max_bin, center + half_width)
    return set(range(start, stop + 1))


def window_enbw_bins(window):
    sw = sum(window)
    sw2 = sum(w * w for w in window)
    if sw <= 0:
        return 1.0
    return len(window) * sw2 / (sw * sw)


def safe_label_frequency(freq):
    text = "{:.3f}".format(freq)
    while text.endswith("0"):
        text = text[:-1]
    if text.endswith("."):
        text += "0"
    return text.replace(".", "p") + "Hz"


def get_run_configuration():
    print()
    print("--- Test configuration ---")

    txt = input(
        "Input frequency from generator [Hz] "
        "(Enter = {:.0f}): ".format(DEFAULT_INPUT_FREQ_HZ)
    ).strip()

    if txt:
        try:
            input_freq_hz = float(txt)
        except ValueError:
            print("Invalid value. Using default frequency.")
            input_freq_hz = DEFAULT_INPUT_FREQ_HZ
    else:
        input_freq_hz = DEFAULT_INPUT_FREQ_HZ

    txt = input(
        "Window hann/rectangular "
        "(Enter = {}): ".format(DEFAULT_WINDOW)
    ).strip().lower()

    if not txt:
        window_type = DEFAULT_WINDOW
    elif txt in ("hann", "rectangular"):
        window_type = txt
    else:
        print("Invalid window. Using default window.")
        window_type = DEFAULT_WINDOW

    return input_freq_hz, window_type


# ------------------------------------------------------------
# ADC acquisition
# ------------------------------------------------------------

def acquire_data():
    samples = array.array("H", [0] * N_SAMPLES)
    times_us = array.array("I", [0] * N_SAMPLES)

    gc.collect()
    gc.disable()

    start = utime.ticks_us()

    for i in range(N_SAMPLES):
        target = utime.ticks_add(start, i * DT_US)
        while utime.ticks_diff(target, utime.ticks_us()) > 0:
            pass

        sample_marker.value(1)
        t_sample = utime.ticks_us()
        times_us[i] = utime.ticks_diff(t_sample, start)
        samples[i] = adc.read_u16()
        sample_marker.value(0)

    sample_marker.value(0)
    gc.enable()

    elapsed_us = times_us[-1] - times_us[0]
    fs_real = (N_SAMPLES - 1) * 1_000_000.0 / elapsed_us if elapsed_us > 0 else 0.0

    intervals_us = [times_us[i] - times_us[i - 1] for i in range(1, N_SAMPLES)]
    mean_dt_us = mean(intervals_us)
    jitter_us = std_population(intervals_us, mean_dt_us)

    fs_error_pct = (
        100.0 * (fs_real - FS_TARGET) / FS_TARGET if FS_TARGET > 0 else 0.0
    )

    print()
    print("--- Sampling ---")
    print("Target Fs              : {:.3f} Hz".format(FS_TARGET))
    print("Measured Fs            : {:.3f} Hz".format(fs_real))
    print("Fs error               : {:+.4f} %".format(fs_error_pct))
    print("Mean Ts                : {:.3f} us".format(mean_dt_us))
    print("Timing jitter (std)    : {:.3f} us".format(jitter_us))

    return samples, times_us, fs_real, jitter_us


# ------------------------------------------------------------
# Manual radix-2 FFT
# ------------------------------------------------------------

def fft_manual(x, n_fft):
    if n_fft < len(x):
        raise ValueError("N_FFT must be >= number of acquired samples")
    if n_fft & (n_fft - 1):
        raise ValueError("N_FFT must be a power of two")

    def bit_reversal(n, log_n):
        rev = 0
        for i in range(log_n):
            if (n >> i) & 1:
                rev |= 1 << (log_n - 1 - i)
        return rev

    X = [complex(v, 0.0) for v in x]
    X.extend([0j] * (n_fft - len(x)))

    log_n = int(math.log2(n_fft))

    for i in range(n_fft):
        j = bit_reversal(i, log_n)
        if j > i:
            X[i], X[j] = X[j], X[i]

    for stage in range(log_n):
        m = 2 ** (stage + 1)
        half_m = m // 2
        w_m = cmath.exp(-2j * math.pi / m)

        for k in range(0, n_fft, m):
            w = 1.0 + 0j
            for j in range(half_m):
                t = w * X[k + j + half_m]
                u = X[k + j]
                X[k + j] = u + t
                X[k + j + half_m] = u - t
                w *= w_m

    return X


# ------------------------------------------------------------
# Spectral analysis (Preallocated to prevent MemoryError)
# ------------------------------------------------------------

def analyze_fft(
    fft_result,
    fs_real,
    window,
    input_freq_hz,
    window_type,
    jitter_us,
    n_fft=N_FFT
):
    nyquist_bin = n_fft // 2
    max_bin = nyquist_bin
    num_bins = max_bin + 1

    # Preasignación de tamaño fijo para evitar fragmentación
    raw_power = [0.0] * num_bins
    for k in range(num_bins):
        raw_power[k] = abs(fft_result[k]) ** 2

    # Búsqueda de fundamental (excluyendo DC)
    fundamental_bin = max(range(1, num_bins), key=lambda k: raw_power[k])

    delta = 0.0
    if 1 <= fundamental_bin < max_bin:
        a = raw_power[fundamental_bin - 1]
        b = raw_power[fundamental_bin]
        c = raw_power[fundamental_bin + 1]
        denom = a - 2.0 * b + c
        if abs(denom) > 1e-30:
            delta = 0.5 * (a - c) / denom
            if delta > 0.5:
                delta = 0.5
            elif delta < -0.5:
                delta = -0.5

    fundamental_freq = (fundamental_bin + delta) * fs_real / n_fft

    window_sum = sum(window)
    full_scale_peak = VREF / 2.0
    full_scale_rms = full_scale_peak / math.sqrt(2.0)

    # Preasignación directa: evita redimensionamientos dinámicos en el heap
    frequencies = [0.0] * num_bins
    magnitudes_vpeak = [0.0] * num_bins
    magnitudes_dbfs = [0.0] * num_bins

    for k in range(num_bins):
        frequencies[k] = k * fs_real / n_fft
        mag = (
            abs(fft_result[k]) / window_sum
            if (k == 0 or k == nyquist_bin)
            else 2.0 * abs(fft_result[k]) / window_sum
        )
        magnitudes_vpeak[k] = mag
        magnitudes_dbfs[k] = db20_ratio(mag, full_scale_peak)

    window_power_sum = sum(w * w for w in window)
    spectral_power = [0.0] * num_bins
    for k in range(num_bins):
        factor = 1.0 if (k == 0 or k == nyquist_bin) else 2.0
        spectral_power[k] = factor * raw_power[k] / (n_fft * window_power_sum)

    signal_bins = bin_region(fundamental_bin, MAIN_LOBE_HALF_WIDTH, max_bin)
    signal_guard = bin_region(fundamental_bin, GUARD_HALF_WIDTH, max_bin)

    harmonic_bins = set()
    harmonic_guard = set()
    harmonic_info = []

    for h in range(2, NUM_HARMONICS + 1):
        expected_freq = alias_to_first_nyquist_zone(h * fundamental_freq, fs_real)
        if expected_freq <= 0.0:
            continue

        expected_bin = int(round(expected_freq * n_fft / fs_real))
        expected_bin = max(1, min(max_bin, expected_bin))

        search_start = max(1, expected_bin - HARMONIC_SEARCH_HALF_WIDTH)
        search_stop = min(max_bin, expected_bin + HARMONIC_SEARCH_HALF_WIDTH)

        harmonic_bin = max(range(search_start, search_stop + 1), key=lambda k: raw_power[k])

        this_power_region = bin_region(harmonic_bin, MAIN_LOBE_HALF_WIDTH, max_bin)
        this_guard_region = bin_region(harmonic_bin, GUARD_HALF_WIDTH, max_bin)

        this_power_region -= signal_bins
        this_guard_region -= signal_guard
        this_power_region -= harmonic_bins
        this_guard_region -= harmonic_guard

        if this_power_region:
            harmonic_bins |= this_power_region
            harmonic_guard |= this_guard_region
            harmonic_info.append((h, harmonic_bin, harmonic_bin * fs_real / n_fft))

    all_positive_bins = set(range(1, num_bins))
    noise_bins = all_positive_bins - signal_guard - harmonic_guard

    signal_power = sum(spectral_power[k] for k in signal_bins)
    harmonic_power = sum(spectral_power[k] for k in harmonic_bins)
    noise_power = sum(spectral_power[k] for k in noise_bins)

    signal_rms = math.sqrt(signal_power) if signal_power > 0 else 0.0
    signal_peak = math.sqrt(2.0) * signal_rms
    signal_vpp = 2.0 * signal_peak
    noise_rms = math.sqrt(noise_power) if noise_power > 0 else 0.0

    # Dynamic metrics
    snr_db = db10_ratio(signal_power, noise_power)
    thd_db = db10_ratio(harmonic_power, signal_power)
    thd_percent = (
        100.0 * math.sqrt(harmonic_power / signal_power)
        if signal_power > 0 and harmonic_power > 0
        else 0.0
    )

    thdn_power = harmonic_power + noise_power
    thdn_db = db10_ratio(thdn_power, signal_power)
    thdn_percent = (
        100.0 * math.sqrt(thdn_power / signal_power)
        if signal_power > 0 and thdn_power > 0
        else 0.0
    )

    sinad_db = db10_ratio(signal_power, thdn_power)
    enob = (sinad_db - 1.76) / 6.02

    signal_dbfs = db20_ratio(signal_rms, full_scale_rms)
    integrated_noise_dbfs = db20_ratio(noise_rms, full_scale_rms)

    noise_bin_dbfs = [magnitudes_dbfs[k] for k in noise_bins]
    measured_noise_floor_dbfs = median(noise_bin_dbfs) if noise_bin_dbfs else -200.0

    ideal_lsb = VREF / (2 ** ADC_BITS)
    ideal_quant_noise_rms = ideal_lsb / math.sqrt(12.0)
    ideal_snr_db = 6.02 * ADC_BITS + 1.76

    independent_bins = N_SAMPLES / 2.0
    ideal_fft_noise_floor_dbfs = (
        -ideal_snr_db - 10.0 * math.log10(independent_bins)
        if independent_bins > 0
        else -200.0
    )

    enbw_bins = window_enbw_bins(window)
    window_adjusted_noise_floor_dbfs = (
        ideal_fft_noise_floor_dbfs + 10.0 * math.log10(enbw_bins)
    )

    # SFDR
    spur_candidates = all_positive_bins - signal_guard
    if spur_candidates:
        largest_spur_bin = max(spur_candidates, key=lambda k: magnitudes_vpeak[k])
        largest_spur_freq = frequencies[largest_spur_bin]
        largest_spur_dbfs = magnitudes_dbfs[largest_spur_bin]
        fundamental_peak_dbfs = magnitudes_dbfs[fundamental_bin]
        sfdr_db = fundamental_peak_dbfs - largest_spur_dbfs
    else:
        largest_spur_bin = 0
        largest_spur_freq = 0.0
        largest_spur_dbfs = -200.0
        sfdr_db = 200.0

    expected_alias_hz = alias_to_first_nyquist_zone(input_freq_hz, fs_real)
    alias_error_hz = fundamental_freq - expected_alias_hz
    samples_per_period = fs_real / input_freq_hz if input_freq_hz > 0 else 0.0
    nyquist_hz = fs_real / 2.0

    jitter_s = jitter_us * 1e-6
    if input_freq_hz > 0 and jitter_s > 0:
        snr_jitter_db = -20.0 * math.log10(2.0 * math.pi * input_freq_hz * jitter_s)
    else:
        snr_jitter_db = 200.0

    # Clasificación de bins preasignada
    bin_types = ["guard"] * num_bins
    bin_types[0] = "dc"
    for k in signal_bins:
        bin_types[k] = "signal"
    for k in harmonic_bins:
        bin_types[k] = "harmonic"
    for k in noise_bins:
        bin_types[k] = "noise"

    # Reporte por consola
    print()
    print("--- Frequency interpretation ---")
    print("Declared input frequency : {:.3f} Hz".format(input_freq_hz))
    print("Measured Nyquist freq.   : {:.3f} Hz".format(nyquist_hz))
    print("Expected alias frequency : {:.3f} Hz".format(expected_alias_hz))
    print("FFT fundamental          : {:.3f} Hz".format(fundamental_freq))
    print("FFT - expected alias     : {:+.3f} Hz".format(alias_error_hz))
    print("Samples per input period : {:.3f}".format(samples_per_period))

    if input_freq_hz > nyquist_hz:
        print("WARNING: input is above Nyquist. The FFT displays its alias.")

    print()
    print("--- Spectral analysis ---")
    print("Window                  : {}".format(window_type))
    print("Fundamental frequency   : {:.3f} Hz".format(fundamental_freq))
    print("Signal amplitude        : {:.6f} Vrms".format(signal_rms))
    print("Signal amplitude        : {:.6f} Vpeak".format(signal_peak))
    print("Signal amplitude        : {:.6f} Vpp".format(signal_vpp))
    print("Signal level            : {:.2f} dBFS".format(signal_dbfs))
    print("Integrated noise        : {:.6f} Vrms".format(noise_rms))
    print("Integrated noise        : {:.2f} dBFS".format(integrated_noise_dbfs))
    print("Measured FFT noise floor: {:.2f} dBFS/bin".format(measured_noise_floor_dbfs))
    print("Ideal quantization noise: {:.6f} Vrms".format(ideal_quant_noise_rms))
    print("Ideal SNR ({} bit)      : {:.2f} dB".format(ADC_BITS, ideal_snr_db))
    print("Ideal FFT noise floor   : {:.2f} dBFS/bin".format(ideal_fft_noise_floor_dbfs))
    print("Window ENBW             : {:.3f} bins".format(enbw_bins))
    print("Window-adjusted floor   : {:.2f} dBFS/bin".format(window_adjusted_noise_floor_dbfs))
    print("SNR                     : {:.2f} dB".format(snr_db))
    print("THD                     : {:.2f} dB ({:.3f} %)".format(thd_db, thd_percent))
    print("THD+N                   : {:.2f} dB ({:.3f} %)".format(thdn_db, thdn_percent))
    print("SINAD                   : {:.2f} dB".format(sinad_db))
    print("ENOB                    : {:.2f} bits".format(enob))
    print("SFDR                    : {:.2f} dBc".format(sfdr_db))
    print("Largest spur            : {:.3f} Hz, {:.2f} dBFS".format(largest_spur_freq, largest_spur_dbfs))
    print("Jitter-limited SNR est. : {:.2f} dB".format(snr_jitter_db))

    return {
        "frequencies": frequencies,
        "magnitudes_vpeak": magnitudes_vpeak,
        "magnitudes_dbfs": magnitudes_dbfs,
        "bin_types": bin_types,
        "input_freq_hz": input_freq_hz,
        "expected_alias_hz": expected_alias_hz,
        "fundamental_freq": fundamental_freq,
        "alias_error_hz": alias_error_hz,
        "samples_per_period": samples_per_period,
        "nyquist_hz": nyquist_hz,
        "signal_rms": signal_rms,
        "signal_peak": signal_peak,
        "signal_vpp": signal_vpp,
        "signal_dbfs": signal_dbfs,
        "noise_rms": noise_rms,
        "integrated_noise_dbfs": integrated_noise_dbfs,
        "measured_noise_floor_dbfs": measured_noise_floor_dbfs,
        "ideal_quant_noise_rms": ideal_quant_noise_rms,
        "ideal_snr_db": ideal_snr_db,
        "ideal_fft_noise_floor_dbfs": ideal_fft_noise_floor_dbfs,
        "window_enbw_bins": enbw_bins,
        "window_adjusted_noise_floor_dbfs": window_adjusted_noise_floor_dbfs,
        "snr_db": snr_db,
        "thd_db": thd_db,
        "thd_percent": thd_percent,
        "thdn_db": thdn_db,
        "thdn_percent": thdn_percent,
        "sinad_db": sinad_db,
        "enob": enob,
        "sfdr_db": sfdr_db,
        "largest_spur_freq": largest_spur_freq,
        "largest_spur_dbfs": largest_spur_dbfs,
        "snr_jitter_db": snr_jitter_db,
    }


# ------------------------------------------------------------
# File output
# ------------------------------------------------------------

def make_filenames(input_freq_hz, window_type):
    label = safe_label_frequency(input_freq_hz) + "_" + window_type
    return (
        "adc_samples_" + label + ".csv",
        "adc_fft_" + label + ".csv",
        "adc_summary_" + label + ".csv",
    )


def save_samples_csv(filename, samples, times_us, vref=VREF):
    with open(filename, "w") as f:
        f.write("sample,time_s,raw_u16,code12,voltage_V\n")
        for i in range(len(samples)):
            code12 = samples[i] >> 4
            v = (samples[i] / 65535.0) * vref
            f.write("{},{:.9f},{},{},{:.6f}\n".format(
                i, times_us[i] / 1_000_000.0, samples[i], code12, v
            ))


def save_fft_csv(filename, results):
    with open(filename, "w") as f:
        f.write("frequency_Hz,magnitude_Vpeak,magnitude_dBFS,bin_type\n")
        for freq, mag, mag_dbfs, b_type in zip(
            results["frequencies"],
            results["magnitudes_vpeak"],
            results["magnitudes_dbfs"],
            results["bin_types"],
        ):
            f.write("{:.6f},{:.8f},{:.4f},{}\n".format(freq, mag, mag_dbfs, b_type))


def save_summary_csv(filename, results, fs_real, jitter_us, window_type):
    with open(filename, "w") as f:
        f.write(
            "input_frequency_Hz,measured_Fs_Hz,nyquist_Hz,window,N_SAMPLES,N_FFT,"
            "physical_resolution_Hz,FFT_spacing_Hz,expected_alias_Hz,"
            "FFT_fundamental_Hz,frequency_error_Hz,samples_per_period,"
            "timing_jitter_us,signal_dBFS,integrated_noise_Vrms,"
            "integrated_noise_dBFS,measured_noise_floor_dBFS_per_bin,"
            "ideal_SNR_dB,ideal_noise_floor_dBFS_per_bin,SNR_dB,THD_dB,"
            "THD_percent,THDN_dB,THDN_percent,SINAD_dB,ENOB_bits,SFDR_dBc,"
            "largest_spur_Hz,largest_spur_dBFS,jitter_limited_SNR_dB\n"
        )
        f.write(
            "{:.6f},{:.6f},{:.6f},{},{},{},{:.6f},{:.6f},{:.6f},{:.6f},"
            "{:.6f},{:.6f},{:.6f},{:.6f},{:.9f},{:.6f},{:.6f},{:.6f},"
            "{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},"
            "{:.6f},{:.6f},{:.6f},{:.6f}\n".format(
                results["input_freq_hz"],
                fs_real,
                results["nyquist_hz"],
                window_type,
                N_SAMPLES,
                N_FFT,
                fs_real / N_SAMPLES,
                fs_real / N_FFT,
                results["expected_alias_hz"],
                results["fundamental_freq"],
                results["alias_error_hz"],
                results["samples_per_period"],
                jitter_us,
                results["signal_dbfs"],
                results["noise_rms"],
                results["integrated_noise_dbfs"],
                results["measured_noise_floor_dbfs"],
                results["ideal_snr_db"],
                results["window_adjusted_noise_floor_dbfs"],
                results["snr_db"],
                results["thd_db"],
                results["thd_percent"],
                results["thdn_db"],
                results["thdn_percent"],
                results["sinad_db"],
                results["enob"],
                results["sfdr_db"],
                results["largest_spur_freq"],
                results["largest_spur_dbfs"],
                results["snr_jitter_db"],
            )
        )


# ------------------------------------------------------------
# Main program (Optimized Memory Lifecycle)
# ------------------------------------------------------------

def main():
    print("====================================================")
    print("ADC dynamic characterization - FFT")
    print("Raspberry Pi Pico 2 W - RP2350")
    print("====================================================")
    print("ADC input              : GP{}".format(ADC_PIN))
    print("Sampling marker        : GP{}".format(SAMPLE_MARKER_PIN))
    print("N_SAMPLES (Ns)         : {}".format(N_SAMPLES))
    print("N_FFT (M)              : {}".format(N_FFT))
    print("Target Fs              : {} Hz".format(FS_TARGET))
    print("Approx. physical df    : {:.4f} Hz".format(FS_TARGET / N_SAMPLES))
    print("Approx. FFT spacing    : {:.4f} Hz".format(FS_TARGET / N_FFT))

    input_freq_hz, window_type = get_run_configuration()
    samples_file, fft_file, summary_file = make_filenames(input_freq_hz, window_type)

    samples, times_us, fs_real, jitter_us = acquire_data()

    # Conversion in-situ a voltajes para ahorrar memoria
    n_s = len(samples)
    voltages = [(samples[i] / 65535.0) * VREF for i in range(n_s)]
    avg_dc = mean(voltages)
    print("DC offset              : {:.6f} V".format(avg_dc))

    # Restar continua in-situ
    for i in range(n_s):
        voltages[i] -= avg_dc

    # Generacion de ventana y ventaneo in-situ
    if window_type == "hann":
        window = [0.5 * (1.0 - math.cos(2.0 * math.pi * i / n_s)) for i in range(n_s)]
    else:
        window = [1.0] * n_s

    for i in range(n_s):
        voltages[i] *= window[i]

    gc.collect()

    fft_result = fft_manual(voltages, N_FFT)
    del voltages
    gc.collect()

    results = analyze_fft(
        fft_result,
        fs_real,
        window,
        input_freq_hz,
        window_type,
        jitter_us,
        N_FFT,
    )
    del fft_result
    gc.collect()

    save_samples_csv(samples_file, samples, times_us, VREF)
    del samples, times_us
    gc.collect()

    save_fft_csv(fft_file, results)
    save_summary_csv(summary_file, results, fs_real, jitter_us, window_type)

    print()
    print("--- Output files ---")
    print("Time-domain data       :", samples_file)
    print("Frequency-domain data  :", fft_file)
    print("Summary                :", summary_file)
    print("Analysis completed successfully.")


if __name__ == "__main__":
    main()