import pandas as pd
import os
import time
import numpy as np
import pandas as pd
import speechpy
from scipy.io import wavfile
from scipy.signal import get_window
from scipy.fftpack import fft
import io
import warnings


import matplotlib.pyplot as plt

class AudioFeatureExtractor:

        
    def __init__(self, sample_rate=44100, hop_size=12, FFT_size=2048):
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.FFT_size = FFT_size
        self.threshold = 0.1
        self.filterbank = 10
        self.dct_filter_num = 40
        self.bark_pointz = self.calculate_bark_points()
        
    def calculate_bark_points(self):
        low_freq = 0
        high_freq = self.sample_rate / 2
        fmin_bark = self.freq_to_bark(low_freq)
        fmax_bark = self.freq_to_bark(high_freq)
        bark_pointz = np.linspace(fmin_bark, fmax_bark, self.filterbank + 4)
        return bark_pointz

    def pre_emphasis(self, audio_signal, alpha=0.97):
        emphasized_signal = np.copy(audio_signal)
        for i in range(1, len(audio_signal)):
            emphasized_signal[i] = audio_signal[i] - alpha * audio_signal[i - 1]
        return emphasized_signal

    def normalize_audio(self, audio):
        audio = audio / np.max(np.abs(audio))
        return audio

    def frame_audio(self, audio, FFT_size, hop_size):
        audio = np.pad(audio, int(FFT_size / 2), mode='reflect')
        frame_len = np.round(self.sample_rate * hop_size / 1000).astype(int)
        frame_num = int((len(audio) - FFT_size) / frame_len) + 1
        frames = np.zeros((frame_num, FFT_size))
        for n in range(frame_num):
            frames[n] = audio[n * frame_len: n * frame_len + FFT_size]
        return frames

    def apply_threshold(self, audio_signal):
        awal = 0
        audiohasil = audio_signal
        for x in range(len(audio_signal)):
            if np.abs(audio_signal[x]) >= self.threshold:
                awal = x
                break
        audiohasil = audio_signal[awal: len(audio_signal)]

        for x in range(len(audiohasil)):
            if np.abs(audiohasil[x]) >= self.threshold:
                akhir = x
        audiohasil2 = audiohasil[0: akhir]
        return audiohasil2

    def freq_to_bark(self, freq):
        return 6.0 * np.arcsinh(freq / 600.0)

    def bark_to_freq(self, bark):
        return 600.0 * np.sinh(bark / 6.0)

    def bark_points(self, low_freq, high_freq):
        fmin_bark = self.freq_to_bark(low_freq)
        fmax_bark = self.freq_to_bark(high_freq)
        bark_pointz = np.linspace(fmin_bark, fmax_bark, self.filterbank + 4)
        freqs = self.bark_to_freq(bark_pointz)
        return np.floor((self.FFT_size + 1) * (freqs / self.sample_rate)).astype(int), freqs

    def Fm(self, fb, fc):
        if fc - 2.5 <= fb <= fc - 0.5:
            return 10 ** (2.5 * (fb - fc + 0.5))
        elif fc - 0.5 < fb < fc + 0.5:
            return 1
        elif fc + 0.5 <= fb <= fc + 1.3:
            return 10 ** (-2.5 * (fb - fc - 0.5))
        else:
            return 0

    def bark_filterbank(self, filter_points):
        filters = np.zeros([self.filterbank, int(self.FFT_size // 2 + 1)])
        for i in range(2, self.filterbank + 2):
            for j in range(int(filter_points[i - 2]), int(filter_points[i + 2])):
                fc = self.bark_pointz[i]  # Use self to access the attribute
                fb = self.freq_to_bark((j * self.sample_rate) / (self.FFT_size + 1))
                filters[i - 2, j] = self.Fm(fb, fc)
        return np.abs(filters)


    def replaceZeroes(self, data):
        min_nonzero = np.min(data[np.nonzero(data)])
        data[data == 0] = min_nonzero
        return data

    def extract_features(self, audio_path):
        sample_rate, audio = wavfile.read(audio_path)
        if len(audio.shape) > 1:
            audio = self.normalize_audio(audio[:, 0])
        else:
            audio = self.normalize_audio(audio)

        audio = self.apply_threshold(audio)
        audio_framed = self.frame_audio(audio, self.FFT_size, self.hop_size)
        window = get_window("hamming", self.FFT_size, fftbins=True)
        audio_win = audio_framed * window
        audio_winT = np.transpose(audio_win)
        audio_fft = np.empty((int(1 + self.FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')
        for n in range(audio_fft.shape[1]):
            audio_fft[:, n] = fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]
        audio_fft = np.transpose(audio_fft)

        audio_power = np.square(np.abs(audio_fft))
        low_freq = 0
        high_freq = self.sample_rate / 2
        filter_points, bark_freqs = self.bark_points(low_freq, high_freq)
        filters = self.bark_filterbank(filter_points)
        enorm = 2.0 / (bark_freqs[2:self.filterbank + 2] - bark_freqs[:self.filterbank])
        filters *= enorm[:, np.newaxis]
        audio_filtered = np.dot(filters, np.transpose(audio_power))

        prob = self.replaceZeroes(audio_filtered)
        audio_log = 10.0 * np.log10(audio_filtered)

        def dct(dct_filter_num, filter_len):
            basis = np.empty((dct_filter_num, filter_len))
            basis[0, :] = 1.0 / np.sqrt(filter_len)
            samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)
            for i in range(1, dct_filter_num):
                basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
            return basis

        dct_filters = dct(self.dct_filter_num, self.filterbank)
        cepstral_coefficents = np.dot(dct_filters, audio_log)
        cepstral_coefficents = speechpy.processing.cmvn(cepstral_coefficents, True)

        fiturmean = np.empty((self.dct_filter_num, 1))
        for xpos in range(len(cepstral_coefficents)):
            sigmax = np.sum(cepstral_coefficents[xpos, :])
            fiturmean[xpos, 0] = sigmax / len(np.transpose(cepstral_coefficents))

        return fiturmean
    
if __name__ == "__main__":
    extractor = AudioFeatureExtractor()
    test_folder = "D:\\Python Semester 5\\pemsu\\Test"  # Specify the root folder containing class subfolders
    output_csv = "Testing_Data.csv"  # Specify the output CSV file

    features_list = []  # To store all features
    classes = []  # To store class labels

    for class_folder in os.listdir(test_folder):
        class_path = os.path.join(test_folder, class_folder)

        if os.path.isdir(class_path):
            class_label = class_folder  # Use the class folder name as the label

            for file_name in os.listdir(class_path):
                if file_name.endswith(".wav"):
                    audio_path = os.path.join(class_path, file_name)

                    print("Processing:", audio_path)
                    features = extractor.extract_features(audio_path)

                    # Reshape the features to 1D
                    features = features.flatten()
                    features_list.append(features)
                    classes.append(class_label)

                    print("--Done--")

    # Create a Pandas DataFrame for the features
    feature_df = pd.DataFrame(features_list, columns=[f"feature_{i+1}" for i in range(len(features_list[0]))])
    
    # Add a "class" column for the class labels
    feature_df["class"] = classes

    # Save the DataFrame to a CSV file
    feature_df.to_csv(output_csv, index=False)

    print("All data extraction completed. CSV file saved.")

