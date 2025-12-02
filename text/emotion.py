# from fastapi import FastAPI
# from pydantic import BaseModel
# import torch
# import torch.nn as nn
# from googletrans import Translator
# from torchtext.data import get_tokenizer
# import streamlit as st
# import asyncio
#
#
#
# news_app = FastAPI()
#
# classes = {
#     0: 'admiration',
#     1: 'amusement',
#     2: 'anger',
#     3: 'annoyance',
#     4: 'approval',
#     5: 'caring',
#     6: 'confusion',
#     7: 'curiosity',
#     8: 'desire',
#     9: 'disappointment',
#     10: 'disapproval',
#     11: 'disgust',
#     12: 'embarrassment',
#     13: 'excitement',
#     14: 'fear',
#     15: 'gratitude',
#     16: 'grief',
#     17: 'joy',
#     18: 'love',
#     19: 'nervousness',
#     20: 'optimism',
#     21: 'pride',
#     22: 'realization',
#     23: 'relief',
#     24: 'remorse',
#     25: 'sadness',
#     26: 'surprise',
#     27: 'neutral'
# }
#
#
# class CheckEmotion(nn.Module):
#   def __init__(self, vocab_size):
#     super().__init__()
#     self.emb = nn.Embedding(vocab_size, 64)
#     self.lstm = nn.LSTM(64, 128, batch_first=True)
#     self.lin = nn.Linear(128, 28)
#
#   def forward(self, x):
#     x = self.emb(x)
#     _, (h, c) = self.lstm(x)
#     h = h[-1]
#     x = self.lin(h)
#     return x
#
# vocab = torch.load('emotion_vocab.pth',  weights_only=False)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CheckEmotion(len(vocab))
# model.state_dict(torch.load('emotion_model.pth', map_location=device))
# model.to(device)
# model.eval()
#
# tokenizer = get_tokenizer('basic_english')
#
# def change_audio(text):
#     return [vocab[i] for i in tokenizer(text)]
#
# class TextSchema(BaseModel):
#     word: str
#
#
# translator = Translator()

# def emotion_text():
#     st.title('Emotion AI Model')
#     text = st.text_area("Input some text here", )
#     if st.button('Answer'):
#         async def translate_async(text):
#             return await translator.translate(text, dest='en')
#
#         if text:
#             translated = asyncio.run(translate_async(text))
#             translated_text = translated.text
#
#             num_text = torch.tensor(change_audio(translated_text), dtype=torch.int64).unsqueeze(0).to(device)
#             with torch.no_grad():
#                 pred = model(num_text)
#                 result = torch.argmax(pred, dim=1).item()
#                 st.success({'class': classes[result]})


from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
from googletrans import Translator
from torchtext.data import get_tokenizer
import streamlit as st
import asyncio

# -------------------------------
# FastAPI instance
# -------------------------------
news_app = FastAPI()

# -------------------------------
# Class labels
# -------------------------------
classes = {
    0: 'admiration 😍',
    1: 'amusement 😄',
    2: 'anger 😡',
    3: 'annoyance 😒',
    4: 'approval 👍',
    5: 'caring 🤗',
    6: 'confusion 😕',
    7: 'curiosity 🤔',
    8: 'desire 😏',
    9: 'disappointment 😞',
    10: 'disapproval 👎',
    11: 'disgust 🤮',
    12: 'embarrassment 😳',
    13: 'excitement 🤩',
    14: 'fear 😨',
    15: 'gratitude 🙏',
    16: 'grief 😢',
    17: 'joy 😁',
    18: 'love ❤️',
    19: 'nervousness 😬',
    20: 'optimism 😊',
    21: 'pride 😌',
    22: 'realization 💡',
    23: 'relief 😅',
    24: 'remorse 😔',
    25: 'sadness 😢',
    26: 'surprise 😲',
    27: 'neutral 😐'
}

# -------------------------------
# Model definition
# -------------------------------
class CheckEmotion(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.lin = nn.Linear(128, 28)

    def forward(self, x):
        x = self.emb(x)
        _, (h, _) = self.lstm(x)
        h = h[-1]
        return self.lin(h)

# -------------------------------
# Load vocab + model
# -------------------------------
vocab = torch.load("emotion_vocab.pth", weights_only=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CheckEmotion(len(vocab))

model.load_state_dict(torch.load("emotion_model.pth", map_location=device))
model.to(device)
model.eval()

tokenizer = get_tokenizer("basic_english")

# -------------------------------
# Text to tensor
# -------------------------------
def change_audio(text: str):
    return [vocab[token] for token in tokenizer(text)]

# -------------------------------
# Pydantic schema
# -------------------------------
class TextSchema(BaseModel):
    word: str

translator = Translator()

# -------------------------------
# Streamlit UI
# -------------------------------
def emotion_text():
    st.title("Emotion AI Model")
    text = st.text_area("Input some text here:")

    if st.button("Answer"):
        async def translate_async(t):
            return await translator.translate(t, dest="en")

        if text:
            translated = asyncio.run(translate_async(text))
            translated_text = translated.text

            # Convert to tensor
            tokens = change_audio(translated_text)
            num_text = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0).to(device)

            # Predict
            with torch.no_grad():
                pred = model(num_text)
                result = torch.argmax(pred, dim=1).item()

            st.success({"class": classes[result]})


def emotion_text():
    st.title("😊 Emotion Classification")

    st.markdown("""
    Эта модель определяет **эмоцию текста**, используя нейронную сеть  
    **Embedding → LSTM → Linear**.

    Модель возвращает **28 эмоций**, включая:

    - ❤️ love  
    - 😊 joy  
    - 😡 anger  
    - 😢 sadness  
    - 😱 surprise  
    - 😐 neutral  
    - 😕 confusion  
    - 🤩 excitement  
    - 🙏 gratitude  
    - и другие...

    Текст можно вводить **на любом языке** — перед анализом он автоматически переводится.
    """)

    st.divider()

    st.subheader("📝 Введите текст")
    text = st.text_area(
        "Введите фразу или небольшой абзац, описывающий эмоцию.",
        height=180
    )

    if st.button("🚀 Определить эмоцию", use_container_width=True, type="primary"):

        if not text.strip():
            st.warning("⚠️ Поле ввода пустое. Пожалуйста, введите текст.")
            return

        try:
            # ---- Асинхронный перевод ----
            async def translate_async(tx):
                return await translator.translate(tx, dest="en")

            translated = asyncio.run(translate_async(text))
            translated_text = translated.text

            st.info(f"🌐 Перевод на английский: **{translated_text}**")

            # ---- Токенизация ----
            tokens = change_audio(translated_text)
            if not tokens:
                st.error("⚠️ Текст не распознан. Попробуйте переписать предложение.")
                return

            tensor = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0).to(device)

            # ---- Предсказание ----
            with torch.no_grad():
                pred = model(tensor)
                cls = torch.argmax(pred, dim=1).item()

            st.success(f"🎭 Определённая эмоция: **{classes[cls]}**")

        except Exception as e:
            st.error(f"❌ Ошибка: {e}")
