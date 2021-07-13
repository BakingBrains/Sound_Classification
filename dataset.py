from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import torch
import os


class UrbanSoundDataset(Dataset):

    def __init__(self, annotation_files, audio_dir, transformation,
                 target_sample_rate, num_samples, device):
        self.annotations = pd.read_csv(annotation_files)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        audio_sample_path = self._get_audio_sample_path(item)
        label = self._get_audio_sample_label(item)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

    def _cut_if_necessary(self, signal):
        # signal -> (1, num_sample)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        len_signal = signal.shape[1]
        if len_signal < self.num_samples: # apply right pad
            num_missing_samples = self.num_samples - len_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        # signal = (channels, num_samples) -> (2, 16000) -> (1, 16000)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, item):
        fold = f"fold{self.annotations.iloc[item, 5]}"
        path = os.path.join(self.audio_dir, fold, self.annotations.iloc[item, 0])
        return path

    def _get_audio_sample_label(self, item):
        return self.annotations.iloc[item, 6]


if __name__ == "__main__":
    ANNOTATIONS_FILE = 'UrbanSound8K/metadata/UrbanSound8K.csv'
    AUDIO_DIR = 'UrbanSound8K/audio'
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = UrbanSoundDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram,
                            SAMPLE_RATE, NUM_SAMPLES, device)

    print(f"There are {len(usd)} samples in dataset")
    signal, label = usd[1]
    # a = 1 # dummy