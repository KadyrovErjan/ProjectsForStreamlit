# from fastapi import FastAPI, HTTPException, UploadFile, File
# import torch
# import torch.nn as nn
# from torchaudio import transforms
# import torch.nn.functional as F
# import io
# import soundfile as sf
# import streamlit as st
#
#
# class EnvironmentAudio(nn.Module):
#     def __init__(self, num_classes=50):
#         super(EnvironmentAudio, self).__init__()
#         self.first = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#
#             nn.AdaptiveAvgPool2d((8, 8))
#         )
#
#         self.flatten = nn.Flatten()
#
#         self.second = nn.Sequential(
#             nn.Linear(64 * 8 * 8, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_classes)
#         )
#
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.first(x)
#         x = self.flatten(x)
#         x = self.second(x)
#         return x
#
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# sr = 22050
# transform = transforms.MelSpectrogram(
#     sample_rate=sr,
#     n_mels=64
# )
#
#
# labels = torch.load('environment_labels.pth')
#
# model = EnvironmentAudio()
# model.load_state_dict((torch.load('environment_model.pth', map_location=device)))
# model.to(device)
# model.eval()
#
# max_len = 500
#
# def change_audio(waveform, sample_rate):
#     # waveform: torch.Tensor [samples]
#     if sample_rate != sr:
#         resample = transforms.Resample(orig_freq=sample_rate, new_freq=sr)
#         # waveform должен иметь форму [1, N] для resample
#         if waveform.ndim == 1:
#             waveform = waveform.unsqueeze(0)
#         waveform = resample(waveform)
#         waveform = waveform.squeeze(0)
#
#     # строим мел-спектрограмму
#     spec = transform(waveform).squeeze(0)
#
#     # выравниваем длину
#     if spec.shape[1] > max_len:
#         spec = spec[:, :max_len]
#     elif spec.shape[1] < max_len:
#         spec = F.pad(spec, (0, max_len - spec.shape[1]))
#
#     return spec
#
#
#
# torch_audio = FastAPI(title='Environment_audio sounds')
#
#
#
# def environment_audio():
#     st.title('Environment Urban')
#     st.text('Загрузите аудио файл')
#
#     audio_file = st.file_uploader('Выбериту файл', type='wav')
#
#     if not audio_file:
#         st.warning('Загрузите .wav файл')
#     else:
#         st.audio(audio_file)
#     if st.button('Распознать'):
#         try:
#             data = audio_file.read()
#             if not data:
#                 raise HTTPException(status_code=400, detail='Empty file')
#
#             waveform, sample_rate = sf.read(io.BytesIO(data), dtype='float32')
#             waveform = torch.tensor(waveform, dtype=torch.float32)
#             if waveform.ndim > 1:
#                 waveform = waveform.mean(dim=1)  # делаем моно
#
#             spec = change_audio(waveform, sample_rate).unsqueeze(0).to(device)
#
#             with torch.no_grad():
#                 y_pred = model(spec)
#                 pred_idx = torch.argmax(y_pred, dim=1).item()
#                 predicted_class = labels[pred_idx]
#
#             st.success({"Index": pred_idx, "Sound": predicted_class})
#
#         except Exception as e:
#             st.exception(e)

import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import transforms
import soundfile as sf
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# ------------------ Модель ------------------
class EnvironmentAudio(nn.Module):
    def __init__(self, num_classes=50):
        super(EnvironmentAudio, self).__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((8, 8))
        )

        self.flatten = nn.Flatten()

        self.second = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.first(x)
        x = self.flatten(x)
        x = self.second(x)
        return x


# ------------------ Настройки ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sr = 22050
max_len = 500

transform = transforms.MelSpectrogram(sample_rate=sr, n_mels=64)

labels = torch.load("environment_labels.pth")

model = EnvironmentAudio()
model.load_state_dict(torch.load("environment_model.pth", map_location=device))
model.to(device)
model.eval()


# ------------------ Обработка аудио ------------------
def change_audio(waveform, sample_rate):
    if sample_rate != sr:
        resample = transforms.Resample(orig_freq=sample_rate, new_freq=sr)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        waveform = resample(waveform)
        waveform = waveform.squeeze(0)

    spec = transform(waveform).squeeze(0)

    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]
    elif spec.shape[1] < max_len:
        spec = F.pad(spec, (0, max_len - spec.shape[1]))

    return spec


# ------------------ Интерфейс Streamlit ------------------
def environment_audio():
    st.set_page_config(page_title="Environment Sound Classifier", page_icon="🎧", layout="centered")

    st.markdown(
        """
        <h1 style='text-align:center; color:#1E88E5;'>🎶 Environment Sound Classifier</h1>
        <p style='text-align:center; color:gray;'>Upload a .wav file and let the neural network guess what sound it is!</p>
        """,
        unsafe_allow_html=True,
    )

    # Добавим небольшой стиль
    st.markdown(
        """
        <style>
        .stButton>button {
            background-color: #1E88E5;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 100%;
            font-size: 18px;
        }
        .stButton>button:hover {
            background-color: #1565C0;
        }
        .result-card {
            background-color: #E3F2FD;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.write("")

    audio_file = st.file_uploader("🎵 Upload your .wav audio file", type=["wav"])

    if not audio_file:
        st.info("⬆️ Please upload a `.wav` audio file to get started.")
        return

    st.audio(audio_file, format="audio/wav")

    if st.button("🔍 Recognize Sound"):
        try:
            data = audio_file.read()
            if not data:
                raise HTTPException(status_code=400, detail="Empty file")

            waveform, sample_rate = sf.read(io.BytesIO(data), dtype="float32")
            waveform = torch.tensor(waveform, dtype=torch.float32)

            if waveform.ndim > 1:
                waveform = waveform.mean(dim=1)

            spec = change_audio(waveform, sample_rate).unsqueeze(0).to(device)

            with torch.no_grad():
                y_pred = model(spec)
                pred_idx = torch.argmax(y_pred, dim=1).item()
                predicted_class = labels[pred_idx]

            # Визуализация спектрограммы
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.imshow(spec.cpu().squeeze().numpy(), origin="lower", aspect="auto")
            ax.set_title("Mel-Spectrogram")
            ax.set_xlabel("Time")
            ax.set_ylabel("Mel Frequency")
            st.pyplot(fig)

            st.markdown(
                f"""
                <div class='result-card'>
                    <h3>✅ Prediction Result</h3>
                    <h2 style='color:#1E88E5;'>{predicted_class}</h2>
                    <p style='color:gray;'>Class Index: <b>{pred_idx}</b></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.exception(e)


# ------------------ Запуск ------------------
if __name__ == "__main__":
    environment_audio()

