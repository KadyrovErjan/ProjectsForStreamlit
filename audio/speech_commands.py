# from fastapi import FastAPI
# import torch
# import torch.nn as nn
# from torchaudio import transforms
# import torch.nn.functional as F
# import io
# import soundfile as sf
# import streamlit as st
#
# class CheckAudio(nn.Module):
#     def __init__(self, num_classes=35):
#         super().__init__()
#         self.first = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.AdaptiveAvgPool2d((8, 8))
#         )
#
#         self.second = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(32 * 8 * 8, 128),
#             nn.ReLU(),
#             nn.Linear(128, 35),
#         )
#
#     def forward(self, x):
#         x = x.unsqueeze(1)
#         x = self.first(x)
#         x = self.second(x)
#         return x
#
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# labels = torch.load('speech_label.pth')
# model = CheckAudio()
# model.load_state_dict(torch.load('speech_model.pth', map_location=device))
# model.to(device)
# model.eval()
#
# transform = transforms.MelSpectrogram(
#     sample_rate=16000,
#     n_mels=64,
# )
#
# max_len = 100
#
#
# def change_audio_format(waveform, sample_rate):
#     if sample_rate != 16000:
#         new_sr = transforms.Resample(orig_freq=sample_rate, new_freq=16000)
#         waveform = new_sr(torch.tensor(waveform))
#
#     spec = transform(waveform).squeeze(0)
#
#     if spec.shape[1] > max_len:
#         spec = spec[:, :max_len]
#
#     elif spec.shape[1] < max_len:
#         count_diff = max_len - spec.shape[1]
#         spec = F.pad(spec, (0, count_diff))
#
#     return spec
#
#
# check_audio = FastAPI(title='Audio')
#
# def speech_audio():
#     name = st.radio("Choose input method:", ["Upload", "Record"], horizontal=True)
#     if name == 'Upload':
#         st.title("🎧 Speech Commands")
#         st.text('Загрузите аудио файл')
#
#         audio_file = st.file_uploader('Выбериту файл', type='wav')
#
#         if not audio_file:
#             st.warning('Загрузите .wav файл')
#         else:
#             st.audio(audio_file)
#         if st.button('Распознать'):
#                 try:
#                     data =  audio_file.read()
#
#                     wf, sr = sf.read(io.BytesIO(data), dtype='float32')
#                     wf = torch.tensor(wf).T
#
#                     spec = change_audio_format(wf, sr).unsqueeze(0).to(device)
#
#                     with torch.no_grad():
#                         y_pred = model(spec)
#                         pred_idx = torch.argmax(y_pred, dim=1).item()
#                         pred_class = labels[pred_idx]
#                         st.success({'Модель думает, что это команда': pred_class})
#
#                 except Exception as e:
#                     st.exception(f'{e}')
#
#
#     if name == 'Record':
#         st.title("🎧 Speech Commands")
#         st.info(f'Скажи слово из этого списка: {labels}')
#
#         audio_record = st.audio_input('Скажите слово')
#
#         st.audio(audio_record)
#         if st.button('Распознать'):
#             try:
#                 data = audio_record.read()
#
#                 wf, sr = sf.read(io.BytesIO(data), dtype='float32')
#                 wf = torch.tensor(wf).T
#
#                 spec = change_audio_format(wf, sr).unsqueeze(0).to(device)
#
#                 with torch.no_grad():
#                     y_pred = model(spec)
#                     pred_idx = torch.argmax(y_pred, dim=1).item()
#                     pred_class = labels[pred_idx]
#                     st.success({'Модель думает, что это команда': pred_class})
#
#             except Exception as e:
#                 st.exception(f'{e}')

import streamlit as st
import torch
import torch.nn.functional as F
import io
import soundfile as sf
from torchaudio import transforms
from fastapi import FastAPI
import torch.nn as nn


# ------------------- Модель -------------------
class CheckAudio(nn.Module):
    def __init__(self, num_classes=35):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.first(x)
        x = self.second(x)
        return x


# ------------------- Настройки -------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

labels = torch.load('speech_label.pth')
model = CheckAudio()
model.load_state_dict(torch.load('speech_model.pth', map_location=device))
model.to(device)
model.eval()

transform = transforms.MelSpectrogram(sample_rate=16000, n_mels=64)
max_len = 100


def change_audio_format(waveform, sample_rate):
    if sample_rate != 16000:
        waveform = transforms.Resample(orig_freq=sample_rate, new_freq=16000)(torch.tensor(waveform))
    spec = transform(waveform).squeeze(0)
    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]
    elif spec.shape[1] < max_len:
        spec = F.pad(spec, (0, max_len - spec.shape[1]))
    return spec


check_audio = FastAPI(title='Speech Commands Recognition')


# ------------------- Интерфейс -------------------
def speech_audio():
    st.title("🗣️ Speech Commands Recognition")
    st.markdown("""
    Эта модель определяет, **какое слово вы произнесли** из набора коротких команд (например, "yes", "no", "stop", "go").

    🎧 Модель обучена на датасете **Google Speech Commands**.  
    Используется **Mel-спектрограмма + сверточная нейросеть (CNN)**.
    """)

    st.divider()
    method = st.radio("🎙️ Выберите способ ввода:", ["📤 Загрузить аудио", "🎤 Записать голос"], horizontal=True)
    st.divider()

    if method == "📤 Загрузить аудио":
        st.subheader("📥 Загрузите WAV файл")
        audio_file = st.file_uploader("Выберите .wav файл", type="wav")

        if audio_file:
            st.audio(audio_file, format="audio/wav")
            if st.button("🚀 Распознать", use_container_width=True, type="primary"):
                try:
                    data = audio_file.read()
                    wf, sr = sf.read(io.BytesIO(data), dtype="float32")
                    wf = torch.tensor(wf).T

                    spec = change_audio_format(wf, sr).unsqueeze(0).to(device)
                    with torch.no_grad():
                        y_pred = model(spec)
                        pred_idx = torch.argmax(y_pred, dim=1).item()
                        pred_class = labels[pred_idx]

                    st.success(f"✅ Модель думает, что это команда: **{pred_class.upper()}**")
                except Exception as e:
                    st.error(f"⚠️ Ошибка: {e}")

        else:
            st.info("👆 Загрузите .wav файл, чтобы продолжить.")

    elif method == "🎤 Записать голос":
        st.subheader("🎙️ Запишите короткое слово")
        st.info(f"Попробуйте сказать одно из слов: {', '.join(labels[:10])} ...")

        audio_record = st.audio_input("Нажмите для записи")

        if audio_record:
            st.audio(audio_record)
            if st.button("🚀 Распознать", use_container_width=True, type="primary"):
                try:
                    data = audio_record.read()
                    wf, sr = sf.read(io.BytesIO(data), dtype="float32")
                    wf = torch.tensor(wf).T

                    spec = change_audio_format(wf, sr).unsqueeze(0).to(device)
                    with torch.no_grad():
                        y_pred = model(spec)
                        pred_idx = torch.argmax(y_pred, dim=1).item()
                        pred_class = labels[pred_idx]

                    st.success(f"🧠 Модель думает, что это команда: **{pred_class.upper()}**")
                except Exception as e:
                    st.error(f"⚠️ Ошибка: {e}")
        else:
            st.info("🎤 Нажмите кнопку, чтобы записать свой голос.")
