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
# class UrbanAudio(nn.Module):
#     def __init__(self, num_classes=10):
#         super(UrbanAudio, self).__init__()
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
# labels = torch.load('urban_labels.pth')
#
# model = UrbanAudio()
# model.load_state_dict((torch.load('urban_model.pth', map_location=device)))
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
# torch_audio = FastAPI(title='Urban sounds')
#
#
#
# def urban_audio():
#     st.title('Model Urban')
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


from fastapi import FastAPI, HTTPException
import torch
import torch.nn as nn
from torchaudio import transforms
import torch.nn.functional as F
import io
import soundfile as sf
import streamlit as st


# ==================== 🎵 Модель ====================
class UrbanAudio(nn.Module):
    def __init__(self, num_classes=10):
        super(UrbanAudio, self).__init__()
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


# ==================== ⚙️ Настройки ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_RATE = 22050
MAX_LEN = 500

transform = transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_mels=64
)

labels = torch.load('urban_labels.pth')

model = UrbanAudio(num_classes=len(labels))
model.load_state_dict(torch.load('urban_model.pth', map_location=device))
model.to(device)
model.eval()

torch_audio = FastAPI(title="Urban Sounds Classifier")


# ==================== 🎧 Обработка аудио ====================
def change_audio(waveform, sample_rate):
    """Приводим аудио к нужному формату и строим мел-спектрограмму"""
    waveform = torch.tensor(waveform, dtype=torch.float32)

    # В моно
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=1)

    # Ресемплинг
    if sample_rate != SAMPLE_RATE:
        resample = transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = resample(waveform.unsqueeze(0)).squeeze(0)

    # Спектрограмма
    spec = transform(waveform).squeeze(0)

    # Подгоняем длину
    if spec.shape[1] > MAX_LEN:
        spec = spec[:, :MAX_LEN]
    elif spec.shape[1] < MAX_LEN:
        spec = F.pad(spec, (0, MAX_LEN - spec.shape[1]))

    return spec


# ==================== 🚀 Интерфейс Streamlit ====================
def urban_audio():
    st.set_page_config(page_title="🏙️ Urban Sound Classifier", layout="centered")

    st.title("🏙️ UrbanSound8K Classifier")
    st.markdown(
        """
        Определяет **тип звука из города** по короткому аудио 🎧  
        Примеры: 🚗 Машина, 🐕 Лай собаки, 🚨 Сирена, 🔨 Строительство и т.д.
        """
    )

    st.divider()
    st.subheader("📂 Загрузите WAV-файл")

    audio_file = st.file_uploader("Выберите аудиофайл", type=["wav"])

    if not audio_file:
        st.info("Пожалуйста, выберите .wav файл для анализа.")
        return

    st.audio(audio_file, format="audio/wav")

    if st.button("🔍 Распознать звук", use_container_width=True):
        try:
            data = audio_file.read()
            if not data:
                raise HTTPException(status_code=400, detail="Пустой файл")

            waveform, sample_rate = sf.read(io.BytesIO(data), dtype='float32')

            spec = change_audio(waveform, sample_rate).unsqueeze(0).to(device)

            with torch.no_grad():
                y_pred = model(spec)
                pred_idx = torch.argmax(y_pred, dim=1).item()
                predicted_class = labels[pred_idx]

            st.success(f"✅ Определённый звук: **{predicted_class}**")
            st.caption(f"🎚️ Индекс класса: {pred_idx}")

        except Exception as e:
            st.error("❌ Ошибка при обработке файла:")
            st.exception(e)


# ==================== 🧩 Запуск ====================
if __name__ == "__main__":
    urban_audio()
