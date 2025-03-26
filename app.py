import streamlit as st
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import tempfile

# --- UI Setup ---
st.set_page_config(page_title="SoundLab Pro Ultimate", layout="wide")
st.title("SoundLab Pro – Ultimate Session Generator")

# --- Frequency Sliders ---
st.subheader("Frequencies (Hz)")
freq1 = st.slider("Frequency 1", 20, 20000, 528)
freq2 = st.slider("Frequency 2", 20, 20000, 963)
freq3 = st.slider("Frequency 3", 20, 20000, 40)

# --- Duration ---
duration_min = st.slider("Duration (minutes)", 1, 60, 5)
duration = duration_min * 60
sample_rate = 44100
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

# --- Base Signal ---
tone1 = np.sin(2 * np.pi * freq1 * t)
tone2 = np.sin(2 * np.pi * freq2 * t)
tone3 = np.sin(2 * np.pi * freq3 * t)
signal = (tone1 + tone2 + tone3) / 3
left = right = signal

# --- FX Controls ---
st.subheader("Audio FX")
fx = {
    "1. Amplitude Modulation (Hz)": None,
    "2. Reverb": None,
    "3. Echo": None,
    "4. Stereo Panning": None,
    "5. Isochronic Pulses": None,
    "6. Binaural Beats": None,
    "7. Chorus": None,
    "8. Flanger": None,
    "9. Tremolo": None,
    "10. Lowpass Filter": None,
    "11. Highpass Filter": None,
    "12. Distortion": None,
    "13. Noise Layer": None
}

cols = st.columns(4)
fx_enabled = {}
sliders = {}

for i, key in enumerate(fx.keys()):
    with cols[i % 4]:
        fx_enabled[key] = st.checkbox(key)
        if key == "1. Amplitude Modulation (Hz)":
            sliders["am_rate"] = st.slider("AM Rate", 0.0, 30.0, 0.0)
        elif key == "5. Isochronic Pulses":
            sliders["pulse_rate"] = st.slider("Pulse Rate", 1, 20, 10)
        elif key == "6. Binaural Beats":
            sliders["bb_diff"] = st.slider("BB Frequency Diff", 0, 50, 10)

# --- FX Processing ---
if fx_enabled["1. Amplitude Modulation (Hz)"] and sliders["am_rate"] > 0:
    am = 0.5 * (1 + np.sin(2 * np.pi * sliders["am_rate"] * t))
    signal *= am

if fx_enabled["2. Reverb"]:
    decay = np.exp(-0.0005 * np.arange(len(signal)))
    signal += np.convolve(signal, decay, mode='full')[:len(signal)] * 0.5

if fx_enabled["3. Echo"]:
    delay = int(0.3 * sample_rate)
    echo_sig = np.zeros_like(signal)
    echo_sig[delay:] = signal[:-delay]
    signal += 0.5 * echo_sig

if fx_enabled["4. Stereo Panning"]:
    pan = np.sin(2 * np.pi * 0.25 * t)
    left = (1 - pan) * signal
    right = (1 + pan) * signal

if fx_enabled["5. Isochronic Pulses"]:
    pulse = 0.5 * (1 + np.sign(np.sin(2 * np.pi * sliders["pulse_rate"] * t)))
    left *= pulse
    right *= pulse

if fx_enabled["6. Binaural Beats"]:
    left = np.sin(2 * np.pi * freq1 * t)
    right = np.sin(2 * np.pi * (freq1 + sliders["bb_diff"]) * t)

if fx_enabled["7. Chorus"]:
    left += 0.02 * np.roll(left, 200)
    right += 0.02 * np.roll(right, 200)

if fx_enabled["8. Flanger"]:
    flanged = np.roll(signal, 400) * 0.5
    left += flanged
    right += flanged

if fx_enabled["9. Tremolo"]:
    trem = 0.5 * (1 + np.sin(2 * np.pi * 5 * t))
    left *= trem
    right *= trem

if fx_enabled["10. Lowpass Filter"]:
    kernel = np.ones(100) / 100
    left = np.convolve(left, kernel, mode="same")
    right = np.convolve(right, kernel, mode="same")

if fx_enabled["11. Highpass Filter"]:
    b, a = butter(1, 0.01, btype='high')
    left = filtfilt(b, a, left)
    right = filtfilt(b, a, right)

if fx_enabled["12. Distortion"]:
    left = np.tanh(left * 5)
    right = np.tanh(right * 5)

if fx_enabled["13. Noise Layer"]:
    noise = np.random.normal(0, 0.03, size=t.shape)
    left += noise
    right += noise

# --- Normalize ---
max_val = max(np.max(np.abs(left)), np.max(np.abs(right)))
if max_val > 0:
    left /= max_val
    right /= max_val

# --- Waveform Visualization ---
st.subheader("Waveform Visualizer")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
ax1.plot(t[:1000], left[:1000], color='blue')
ax1.set_title("Left Channel")
ax2.plot(t[:1000], right[:1000], color='red')
ax2.set_title("Right Channel")
fig.tight_layout()
st.pyplot(fig)

# --- Export & Playback ---
stereo = np.stack((left, right), axis=-1)
output = np.int16(stereo * 32767)

if st.button("Generate Preview"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        write(f.name, sample_rate, output)
        st.audio(f.name)
        st.download_button("Download .WAV", data=open(f.name, "rb").read(), file_name="soundlab_output.wav")
