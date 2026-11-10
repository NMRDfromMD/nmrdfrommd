import numpy as np
from nmrdfrommd.fourier import compute_spectral_density, fourier_transform


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

def test_compute_spectral_density_matches_fourier_transform():
    """For dim=1, compute_spectral_density reduces to a single fourier_transform call."""
    t_ps = np.linspace(0, 10, 100)
    signal = np.exp(-t_ps / 2.0)
    gij = signal[np.newaxis, :]  # shape (1, n_frames)

    f, J = compute_spectral_density(t_ps, gij, dim=1)

    expected = fourier_transform(np.column_stack((t_ps, signal)))
    np.testing.assert_allclose(f, expected[:, 0])
    np.testing.assert_allclose(J[0], np.real(expected[:, 1]))

def test_compute_spectral_density_multiple_components():
    """Each component m is transformed independently; shapes and per-row results match."""
    t_ps = np.linspace(0, 10, 64)
    gij = np.stack([
        np.exp(-t_ps / 2.0),
        np.exp(-t_ps / 4.0),
        np.zeros_like(t_ps),
    ])

    f, J = compute_spectral_density(t_ps, gij, dim=3)

    assert J.shape == (3, len(f))
    for m in range(3):
        expected = fourier_transform(np.column_stack((t_ps, gij[m])))
        np.testing.assert_allclose(J[m], np.real(expected[:, 1]))

def test_compute_spectral_density_zero_signal_gives_zero_spectrum():
    """A flat zero correlation function has zero spectral density everywhere."""
    t_ps = np.linspace(0, 10, 50)
    gij = np.zeros((1, len(t_ps)))

    _, J = compute_spectral_density(t_ps, gij, dim=1)

    np.testing.assert_allclose(J, 0.0)
