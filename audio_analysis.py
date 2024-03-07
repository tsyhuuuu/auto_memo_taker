import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
# from scipy.fftpack.realtransforms import dct

import warnings
warnings.filterwarnings('ignore')

file_name = './mp3_source/sample3.wav'


wav, sr = librosa.load(file_name)
plt.figure(figsize=(10, 5))
# librosa.display.waveshow(wav, sr=sr)
# plt.show()

data, fs = sf.read(file_name)

t = np.arange(0, len(data)/fs, 1/fs)

# Cut out a part of the speech waveform
center = 25000
cuttime = 0.04  # second
x = data[int(center - cuttime/2*fs) : int(center + cuttime/2*fs)]
time = t[int(center - cuttime/2*fs) : int(center + cuttime/2*fs)]

# plt.plot(time*1000, x)
# plt.xlabel("time [ms]")
# plt.ylabel("amplitude")
# plt.show()


hamming = np.hamming(len(x))
x = x*hamming

# Fourier amplitude spectrum
N = 2048
spec = np.abs(np.fft.fft(x, N))[:N//2]
fscale = np.fft.fftfreq(N, d = 1.0/fs)[:N//2]

# plt.plot(fscale, spec)
# plt.xlabel("frequency [Hz]")
# plt.ylabel("amplitude spectrum")
# plt.show()


def hz2mel(f):
    return 2595 * np.log(f/700 + 1)

def mel2hz(m):
    return 700 * (np.exp(m/2595) - 1)

def melFilterBank(fs, N, numChannels):
    fmax = fs / 2
    melmax = hz2mel(fmax)
    nmax = N // 2
    df = fs / N
    dmel = melmax / (numChannels + 1)
    melcenters = np.arange(1, numChannels + 1) * dmel
    fcenters = mel2hz(melcenters)
    indexcenter = np.round(fcenters / df)
    indexstart = np.hstack(([0], indexcenter[0:numChannels - 1]))
    indexstop = np.hstack((indexcenter[1:numChannels], [nmax]))
    filterbank = np.zeros((numChannels, nmax))
    # print(indexstop)
    for c in range(0, numChannels):
        increment= 1.0 / (indexcenter[c] - indexstart[c])
        for i in range(int(indexstart[c]), int(indexcenter[c])):
            filterbank[c, i] = (i - indexstart[c]) * increment
        decrement = 1.0 / (indexstop[c] - indexcenter[c])
        for i in range(int(indexcenter[c]), int(indexstop[c])):
            filterbank[c, i] = 1.0 - ((i - indexcenter[c]) * decrement)

    return filterbank, fcenters

numChannels = 20
df = fs / N
filterbank, fcenters = melFilterBank(fs, N, numChannels)

# for c in np.arange(0, numChannels):
#     plt.plot(np.arange(0, N / 2) * df, filterbank[c])

# plt.title('Mel filter bank')
# plt.xlabel('Frequency[Hz]')
# plt.show()

mspec = np.dot(spec, filterbank.T)
plt.figure(figsize=(13, 5))

plt.plot(fscale, 10*np.log10(spec), label='Original Spectrum')
plt.plot(fcenters, 10*np.log10(mspec), "o-", label='Mel Spectrum')
plt.xlabel("frequency[Hz]")
plt.ylabel('Amplitude[dB]')
plt.legend()
plt.show()