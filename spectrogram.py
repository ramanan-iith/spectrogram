import wave
import numpy as np
import matplotlib.pyplot as plt

def read_wave_file(filepath):
    """Read a WAV file"""
    with wave.open(filepath, 'rb') as audio_file:
        sample_rate = audio_file.getframerate()
        num_frames = audio_file.getnframes()
        audio_data = audio_file.readframes(num_frames)
        audio_data = np.array([int.from_bytes(audio_data[i:i+2], byteorder='little', signed=True) for i in range(0, len(audio_data), 2)])
    return sample_rate, num_frames, audio_data

def manual_dft(signal, N):
    # Calculate the step size for 50% overlap
    step_size = N // 2

    # Determine the number of windows considering the overlap
    num_windows = (len(signal) - N) // step_size + 1

    # Precompute the Hanning window
    hanning_window = np.hanning(N)

    # Precompute the exponential factors for the DFT
    exp_factors = np.array([np.exp(-2j * np.pi * k * n / N) for k in range(N) for n in range(N)]).reshape(N, N)

    dft_result = np.zeros((num_windows, N), dtype=complex)

    # Loop over each window with 50% overlap
    for i in range(num_windows):
        window_start = i * step_size
        window_end = window_start + N
        if window_end > len(signal):
            break  # Avoid going out of bounds at the end of the signal

        window = signal[window_start:window_end]
        windowed_signal = window * hanning_window  # Apply a Hanning window to reduce spectral leakage

        # Compute the DFT for the current window
        for k in range(N):
            # Vectorized computation of the DFT using precomputed exponential factors
            dft_result[i, k] = np.sum(windowed_signal * exp_factors[k])

    return dft_result



def plot_audio_and_spectrogram(audio_data, dft_magnitude, sample_rate, N, num_frames):
    """Plot the audio waveform and its spectrogram."""
    time_audio = np.linspace(0, num_frames / sample_rate, num=len(audio_data))
    time_vector = np.linspace(0, num_frames / sample_rate, num=dft_magnitude.shape[0])
    freq_vector = np.fft.fftfreq(N, d=1 / sample_rate)[:N // 2]

    fig, ax = plt.subplots(2, 1, figsize=(12, 8))

    # Plot audio waveform
    ax[0].plot(time_audio, audio_data)
    ax[0].set_title('Audio Waveform')
    ax[0].set_xlabel('Time (seconds)')
    ax[0].set_ylabel('Amplitude')

    # Plot spectrogram
    spectrogram_image = ax[1].imshow(dft_magnitude[:, :N//2].T, aspect='auto', origin='lower', cmap='gray_r', extent=[time_vector[0], time_vector[-1], freq_vector.min(), freq_vector.max()])
    ax[1].set_title('Spectrogram')
    ax[1].set_xlabel('Time (seconds)')
    ax[1].set_ylabel('Frequency (Hz)')

    # Add colorbar to the figure
    fig.colorbar(spectrogram_image, ax=ax[1], label='Magnitude')

    plt.tight_layout()
    plt.ylim(0, 7000)
    plt.show()

def main():
    filepath = 'audios/KAN_M_WIKI_00014.wav' # Place your audio file here
    N = 1024  # Window size for DFT. This can be increased / decreased based on the desired frequency resolution
    sample_rate, num_frames, audio_data = read_wave_file(filepath)
    duration_seconds = num_frames / sample_rate
    dft_result = manual_dft(audio_data, N)
    dft_magnitude = np.abs(np.log(dft_result+0.00001))
    plot_audio_and_spectrogram(audio_data, dft_magnitude, sample_rate, N, num_frames)

if __name__ == "__main__":
    main()
