# from fastapi import FastAPI, HTTPException, UploadFile, File
# import torch
# import torch.nn as nn
# from torchaudio import transforms
# import torch.nn.functional as F
# import io
# import soundfile as sf
# import streamlit as st
#
# class CheckAudio(nn.Module):
#   def __init__(self):
#     super().__init__()
#     self.first = nn.Sequential(
#         nn.Conv2d(1, 16, kernel_size=3, padding=1),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         nn.Conv2d(16, 32, kernel_size=3, padding=1),
#         nn.ReLU(),
#         nn.MaxPool2d(2),
#         nn.AdaptiveAvgPool2d((8, 8)),
#     )
#
#     self.second = nn.Sequential(
#         nn.Flatten(),
#         nn.Linear(32 * 8 * 8, 128),
#         nn.ReLU(),
#         nn.Linear(128, 10)
#     )
#
#
#   def forward(self, x):
#     x = x.unsqueeze(1)
#     x = self.first(x)
#     x = self.second(x)
#     return x
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# transform = transforms.MelSpectrogram(
#     sample_rate=22050,
#     n_mels=64
# )
# max_len = 500
#
# genres = torch.load('gtzan_labels.pth')
# index_to_label = {ind: lab for ind, lab in enumerate(genres)}
#
# model = CheckAudio()
# model.load_state_dict(torch.load('gtzan_model.pth', map_location=device))
# model.to(device)
# model.eval()
#
# def change_audio(waveform, sr):
#     if sr != 22050:
#         resample = transforms.Resample(orig_freq=sr, new_freq=22050)
#         waveform = resample(torch.tensor(waveform).unsqueeze(0))
#
#     spec = transform(waveform).squeeze(0)
#
#     if spec.shape[1] > max_len:
#         spec = spec[:, :max_len]
#
#     if spec.shape[1] < max_len:
#         count_len = max_len - spec.shape[1]
#         spec = F.pad(spec, (0, count_len))
#
#     return spec
# audio_app = FastAPI()
#
#
# def gtzan_audio():
#     st.title('Model GTZAN')
#     st.text('Загрузите аудио файл')
#
#     audio_file = st.file_uploader('Выбериту файл', type='wav')
#
#     if not audio_file:
#         st.warning('Загрузите .wav файл')
#     else:
#         st.audio(audio_file)
#     if st.button('Распознать'):
#             try:
#                 data = audio_file.read()
#                 if not data:
#                     raise HTTPException(status_code=404, detail='Пустой файл')
#                 wf, sr = sf.read(io.BytesIO(data), dtype='float32')
#                 wf = torch.tensor(wf).T
#
#                 spec = change_audio(wf, sr).unsqueeze(0).to(device)
#
#                 with torch.no_grad():
#                     y_pred = model(spec)
#                     pred_ind = torch.argmax(y_pred, dim=1).item()
#                     pred_class = index_to_label[pred_ind]
#
#                 st.success({f'Индекс: {pred_ind}  Жанр: {pred_class}'})
#             except Exception as e:
#                 st.exception(f'{e}')

from fastapi import FastAPI, HTTPException
import torch
import torch.nn as nn
from torchaudio import transforms
import torch.nn.functional as F
import io
import soundfile as sf
import streamlit as st

# ==================== 🎶 Модель ====================
class CheckAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((8, 8)),
        )

        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.first(x)
        x = self.second(x)
        return x


# ==================== ⚙️ Настройки ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.MelSpectrogram(
    sample_rate=22050,
    n_mels=64
)
max_len = 500

genres = torch.load('gtzan_labels.pth')
index_to_label = {ind: lab for ind, lab in enumerate(genres)}

model = CheckAudio()
model.load_state_dict(torch.load('gtzan_model.pth', map_location=device))
model.to(device)
model.eval()


# ==================== 🎧 Обработка аудио ====================
def change_audio(waveform, sr):
    if sr != 22050:
        resample = transforms.Resample(orig_freq=sr, new_freq=22050)
        waveform = resample(torch.tensor(waveform).unsqueeze(0))

    spec = transform(waveform).squeeze(0)

    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]
    elif spec.shape[1] < max_len:
        spec = F.pad(spec, (0, max_len - spec.shape[1]))

    return spec


# ==================== 🚀 FastAPI + Streamlit ====================
audio_app = FastAPI()


def gtzan_audio():
    st.set_page_config(page_title="🎵 GTZAN Genre Classifier", layout="centered")

    st.title("🎧 GTZAN Music Genre Classifier")
    st.markdown(
        """
        Определяет **жанр музыки** по вашему аудиофайлу.  
        Поддерживаемые жанры:  
        🎸 Rock • 🎹 Classical • 🎤 Pop • 🥁 HipHop • 🎺 Jazz • и другие
        """
    )

    with st.container():
        st.subheader("📂 Загрузите WAV-файл")
        audio_file = st.file_uploader("Выберите файл (.wav)", type=["wav"])

        if not audio_file:
            st.info("Пожалуйста, загрузите аудио для анализа.")
        else:
            st.audio(audio_file, format='audio/wav')

        if st.button("🔍 Распознать жанр", use_container_width=True):
            try:
                data = audio_file.read()
                if not data:
                    raise HTTPException(status_code=400, detail="Пустой файл")

                wf, sr = sf.read(io.BytesIO(data), dtype='float32')
                wf = torch.tensor(wf).T

                spec = change_audio(wf, sr).unsqueeze(0).to(device)

                with torch.no_grad():
                    y_pred = model(spec)
                    pred_ind = torch.argmax(y_pred, dim=1).item()
                    pred_class = index_to_label[pred_ind]

                st.success(f"🎶 **Предсказанный жанр:** `{pred_class}`")
                st.caption(f"📊 Индекс класса: {pred_ind}")

            except Exception as e:
                st.error("❌ Ошибка при обработке файла")
                st.exception(e)


# ==================== 🧩 Запуск ====================
if __name__ == "__main__":
    gtzan_audio()
