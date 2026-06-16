import numpy as np
from nmrdfrommd.fourier import fourier_transform

def test_fourier_transform_frequency_axis():
    """Frequency axis matches FFT definition for uniform sampling."""
    t_ps = np.linspace(0, 10, 100)
    signal = np.zeros_like(t_ps)

    data = np.column_stack((t_ps, signal))

    out = fourier_transform(data)

    dt_ps = t_ps[1] - t_ps[0]
    dt = dt_ps * 1e-12

    expected = np.fft.rfftfreq(len(t_ps), dt) / 1e6  # MHz

    np.testing.assert_allclose(out[:, 0], expected)

def test_fourier_transform_sine_peak():
    """A sine wave produces a dominant peak at the correct frequency."""
    t_ps = np.linspace(0, 1e6, 2048)

    t = t_ps * 1e-12  # ps → s

    f_mhz = 5.0

    # cleaner signal: integer-ish number of oscillations reduces leakage
    signal = np.sin(2 * np.pi * f_mhz * 1e6 * t)

    data = np.column_stack((t_ps, signal))

    out = fourier_transform(data)

    freqs = out[:, 0]
    spectrum = np.abs(out[:, 1])

    peak_freq = freqs[np.argmax(spectrum)]

    # allow small numerical + leakage tolerance
    np.testing.assert_allclose(peak_freq, f_mhz, rtol=5e-2)
