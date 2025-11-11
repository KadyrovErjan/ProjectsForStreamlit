import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import soundfile as sf
from torchaudio import transforms
from fastapi import FastAPI, HTTPException

# ------------------- Модель -------------------
class PlaceAudio(nn.Module):
    def __init__(self, num_classes=10):
        super(PlaceAudio, self).__init__()
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
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.first(x)
        x = self.flatten(x)
        x = self.second(x)
        return x


# ------------------- Настройки -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    win_length=1024,
    hop_length=256,
    n_mels=64,
    f_min=0,
    f_max=8000,
    power=2.0
)

labels = torch.load("region_labels.pth")
num_classes = len(labels)

model = PlaceAudio(num_classes=num_classes)
checkpoint = torch.load("region_model.pth", map_location=device)

missing, unexpected = model.load_state_dict(checkpoint, strict=False)
print("⚠️ Пропущенные ключи:", missing)
print("⚠️ Неожиданные ключи:", unexpected)

model.to(device)
model.eval()

max_len = 500


# ------------------- Обработка аудио -------------------
def change_audio_format(waveform, sample_rate):
    waveform = torch.tensor(waveform, dtype=torch.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=1)

    if sample_rate != 16000:
        resample = transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resample(waveform)

    spec = transform(waveform).squeeze(0)

    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]
    elif spec.shape[1] < max_len:
        spec = F.pad(spec, (0, max_len - spec.shape[1]))

    return spec


check_audio = FastAPI(title="Region Classifier")


# ------------------- Streamlit UI -------------------
def region_audio():
    st.title("🌍 Region Audio Classification")
    st.markdown("""
    Эта модель определяет, **из какого региона или страны** может быть голос или речь на аудиозаписи.

    🎧 Основана на **сверточной нейросети (CNN)**, которая анализирует **Mel-спектрограмму звука**  
    и находит характерные акустические особенности, присущие разным регионам.
    """)

    st.divider()
    method = st.radio("🎙️ Выберите способ ввода:", ["📤 Загрузить аудио", "🎤 Записать голос"], horizontal=True)
    st.divider()

    if method == "📤 Загрузить аудио":
        st.subheader("📥 Загрузите WAV файл")
        audio_file = st.file_uploader("Выберите .wav файл", type="wav")

        if audio_file:
            st.audio(audio_file, format="audio/wav")
            if st.button("🚀 Определить регион", use_container_width=True, type="primary"):
                try:
                    data = audio_file.read()
                    wf, sr = sf.read(io.BytesIO(data), dtype="float32")
                    spec = change_audio_format(wf, sr).unsqueeze(0).to(device)

                    with torch.no_grad():
                        y_pred = model(spec)
                        pred_idx = torch.argmax(y_pred, dim=1).item()
                        pred_class = labels[pred_idx]

                    st.success(f"🌏 Предположительно, это **{pred_class}** регион")
                except Exception as e:
                    st.error(f"⚠️ Ошибка: {e}")
        else:
            st.info("👆 Загрузите .wav файл, чтобы продолжить.")

    elif method == "🎤 Записать голос":
        st.subheader("🎙️ Запишите короткий образец речи")
        st.info(f"Модель может различать регионы: {', '.join(labels[:10])} ...")

        audio_record = st.audio_input("Нажмите для записи")

        if audio_record:
            st.audio(audio_record)
            if st.button("🚀 Определить регион", use_container_width=True, type="primary"):
                try:
                    data = audio_record.read()
                    wf, sr = sf.read(io.BytesIO(data), dtype="float32")
                    spec = change_audio_format(wf, sr).unsqueeze(0).to(device)

                    with torch.no_grad():
                        y_pred = model(spec)
                        pred_idx = torch.argmax(y_pred, dim=1).item()
                        pred_class = labels[pred_idx]

                    st.success(f"🧠 Модель думает, что это **{pred_class}** регион")
                except Exception as e:
                    st.error(f"⚠️ Ошибка: {e}")
        else:
            st.info("🎤 Нажмите кнопку, чтобы записать свой голос.")
