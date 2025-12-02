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
#     0: 'World',
#     1: 'Sport',
#     2: 'Business',
#     3: 'Sci/Tech'
# }
#
# class ChecNews(nn.Module):
#   def __init__(self, vocab_size):
#     super().__init__()
#     self.emb = nn.Embedding(vocab_size, 64)
#     self.lstm = nn.LSTM(64, 128, batch_first=True)
#     self.lin = nn.Linear(128, 4)
#
#   def forward(self, x):
#     x = self.emb(x)
#     _, (h, c) = self.lstm(x)
#     h = h[-1]
#     x = self.lin(h)
#     return x
#
# vocab = torch.load('news_vocab.pth',  weights_only=False)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ChecNews(len(vocab))
# model.state_dict(torch.load('news_model.pth', map_location=device))
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
#
# def news_text():
#     st.title('News AI Model')
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
#
import streamlit as st
import torch
import torch.nn as nn
import asyncio
from googletrans import Translator
from torchtext.data import get_tokenizer


# ------------------- Модель -------------------
class ChecNews(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.lin = nn.Linear(128, 4)

    def forward(self, x):
        x = self.emb(x)
        _, (h, _) = self.lstm(x)
        h = h[-1]
        x = self.lin(h)
        return x


# ------------------- Настройки -------------------
classes = {
    0: 'World 🌍',
    1: 'Sport ⚽',
    2: 'Business 💼',
    3: 'Sci/Tech 🔬'
}

vocab = torch.load('news_vocab.pth', weights_only=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ChecNews(len(vocab))
model.load_state_dict(torch.load('news_model.pth', map_location=device))
model.to(device)
model.eval()

tokenizer = get_tokenizer('basic_english')
translator = Translator()


def encode_text(text):
    return [vocab[i] for i in tokenizer(text) if i in vocab]


# ------------------- Интерфейс -------------------
def news_text():
    st.title("📰 News Classification AI")

    st.markdown("""
    Эта модель определяет, к какой категории относится **новостной текст**.  
    Используется архитектура **Embedding → LSTM → Linear**, обученная на датасете AG-News.

    Классы:
    - 🌍 *World*
    - ⚽ *Sport*
    - 💼 *Business*
    - 🔬 *Sci/Tech*
    """)

    st.divider()

    st.subheader("📝 Введите текст новости")
    text = st.text_area(
        "Подходит текст на любом языке — система автоматически переведёт.",
        height=200
    )

    if st.button("🚀 Классифицировать новость", use_container_width=True, type="primary"):

        if not text.strip():
            st.warning("⚠️ Пожалуйста, введите текст.")
            return

        try:
            # ---- Асинхронный перевод ----
            async def translate_async(tx):
                return await translator.translate(tx, dest='en')

            translated = asyncio.run(translate_async(text))
            translated_text = translated.text

            st.info(f"🌐 Перевод на английский: **{translated_text}**")

            # ---- Токенизация ----
            encoded = encode_text(translated_text)
            if not encoded:
                st.error("⚠️ После токенизации текст пуст. Вероятно, слова отсутствуют в словаре.")
                return

            tensor = torch.tensor(encoded, dtype=torch.int64).unsqueeze(0).to(device)

            # ---- Предсказание ----
            with torch.no_grad():
                pred = model(tensor)
                cls = torch.argmax(pred, dim=1).item()

            st.success(f"🧠 Класс новости: **{classes[cls]}**")

        except Exception as e:
            st.error(f"❌ Ошибка: {e}")
