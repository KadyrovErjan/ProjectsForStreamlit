# from fastapi import FastAPI
# import torch
# import torch.nn as nn
# from googletrans import Translator
# from torchtext.data import get_tokenizer
# import streamlit as st
# import asyncio
# news_app = FastAPI()
#
# classes = {
#     0: 'Negative',
#     1: 'Positive',
# }
#
# class CheckIMDB(nn.Module):
#   def __init__(self, vocab_size):
#     super().__init__()
#     self.emb = nn.Embedding(vocab_size, 64)
#     self.lstm = nn.LSTM(64, 128, batch_first=True)
#     self.lin = nn.Linear(128, 2)
#
#   def forward(self, x):
#     x = self.emb(x)
#     _, (h, c) = self.lstm(x)
#     h = h[-1]
#     x = self.lin(h)
#     return x
#
# vocab = torch.load('imdb_vocab.pth',  weights_only=False)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CheckIMDB(len(vocab))
# model.state_dict(torch.load('imdb_model.pth', map_location=device))
# model.to(device)
# model.eval()
#
# tokenizer = get_tokenizer('basic_english')
#
# def change_audio(text):
#     return [vocab[i] for i in tokenizer(text)]
#
# # class TextSchema(BaseModel):
# #     word: str
#
#
# translator = Translator()
#
# # @news_app.post('/predict/')
# # async def check_text(text: TextSchema):
# def imdb_text():
#     st.title('IMDB AI Model')
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
#
#
from fastapi import FastAPI
import torch
import torch.nn as nn
from googletrans import Translator
from torchtext.data import get_tokenizer
import streamlit as st
import asyncio

news_app = FastAPI()

# ------------------- Классы -------------------
classes = {
    0: 'Negative 💔',
    1: 'Positive ❤️',
}


# ------------------- Модель -------------------
class CheckIMDB(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.lin = nn.Linear(128, 2)

    def forward(self, x):
        x = self.emb(x)
        _, (h, _) = self.lstm(x)
        h = h[-1]
        x = self.lin(h)
        return x


# ------------------- Настройки -------------------
vocab = torch.load("imdb_vocab.pth", weights_only=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CheckIMDB(len(vocab))
model.load_state_dict(torch.load("imdb_model.pth", map_location=device))
model.to(device)
model.eval()

tokenizer = get_tokenizer("basic_english")
translator = Translator()


def encode_text(text):
    return [vocab[i] for i in tokenizer(text) if i in vocab]


# ------------------- Интерфейс -------------------
def imdb_text():
    st.title("🎬 IMDB Sentiment Analysis")

    st.markdown("""
    Эта модель определяет **тональность текста отзыва о фильме**.  
    Основана на нейронной сети **Embedding → LSTM → Linear**.

    Модель возвращает:
    - ❤️ **Positive**
    - 💔 **Negative**
    """)

    st.divider()

    st.subheader("📝 Введите текст отзыва")
    text = st.text_area(
        "Можно вводить текст на любом языке — модель автоматически переведёт.",
        height=200
    )

    if st.button("🚀 Анализировать отзыв", use_container_width=True, type="primary"):

        if not text.strip():
            st.warning("⚠️ Пожалуйста, введите текст.")
            return

        try:
            # ---- Асинхронный перевод ----
            async def translate_async(tx):
                return await translator.translate(tx, dest="en")

            translated = asyncio.run(translate_async(text))
            translated_text = translated.text

            st.info(f"🌐 Перевод на английский: **{translated_text}**")

            # ---- Кодирование ----
            encoded = encode_text(translated_text)
            if not encoded:
                st.error("⚠️ Токенизация дала пустой результат. Слова не найдены в словаре.")
                return

            tensor = torch.tensor(encoded, dtype=torch.int64).unsqueeze(0).to(device)

            # ---- Предсказание ----
            with torch.no_grad():
                pred = model(tensor)
                cls = torch.argmax(pred, dim=1).item()

            st.success(f"🧠 Результат анализа: **{classes[cls]}**")

        except Exception as e:
            st.error(f"❌ Ошибка: {e}")
