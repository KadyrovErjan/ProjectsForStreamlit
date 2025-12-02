# import sys
# import os
#
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import streamlit as st
#
# # image
# from image.mnist import mnist_image
# from image.intel_image import intel_image
# from image.buildings import buildings_image
# from image.fashion import fashion_image
# from image.cifar100 import cifar100_image
# from image.cifar10 import cifar10_image
#
# # audio
# from audio.region import region_audio
# from audio.gtzan import gtzan_audio
# from audio.speech_commands import speech_audio
# from audio.urban import urban_audio
#
# st.title('AI MODELS')
# with st.sidebar:
#     st.header('AI Models')
#     name = st.radio('Choose', ['MNIST', 'Fashion', 'CIFAR-100',
#                             'Urban', 'GTZAN', 'Speech Commands',
#                             'Intel Image', 'Buildings', 'CIFAR-10', 'Region'])
#
# # images
# if name == 'MNIST':
#     mnist_image()
#
# elif name == 'Buildings':
#     buildings_image()
#
# elif name == 'Fashion':
#     fashion_image()
#
# elif name == 'CIFAR-100':
#     cifar100_image()
#
# elif name == 'CIFAR-10':
#     cifar10_image()
#
# elif name == 'Intel Image':
#     intel_image()
#
# # audio
# elif name == 'Urban':
#     urban_audio()
#
# elif name == 'GTZAN':
#     gtzan_audio()
#
# elif name == 'Speech Commands':
#     speech_audio()
#
# elif name == 'Region':
#     region_audio()
#

import sys
import os
import streamlit as st
from PIL import Image

# Добавляем корень проекта в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------- Импорты моделей ----------
# Image models
from image.mnist import mnist_image
from image.intel_image import intel_image
from image.buildings import buildings_image
from image.fashion import fashion_image
from image.cifar100 import cifar100_image
from image.cifar10 import cifar10_image
from image.command import command_image

# Audio models
from audio.region import region_audio
from audio.gtzan import gtzan_audio
from audio.speech_commands import speech_audio
from audio.urban import urban_audio
from audio.environment import environment_audio

# Text models
from text.imdb import imdb_text
from text.news import news_text
from text.emotion import emotion_text


# ---------- Основной интерфейс ----------
st.set_page_config(
    page_title="AI Models Showcase",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI MODELS DEMO")
st.markdown("""
Добро пожаловать в **AI Models Showcase** — интерактивное демо, где вы можете попробовать разные модели машинного обучения для:
- 🖼️ **распознавания изображений**
- 🔊 **анализа аудио**
""")

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4712/4712107.png", width=100)
st.sidebar.header("AI Models Menu")

# ---------- Меню выбора ----------
name = st.sidebar.radio("Выберите модель:", [
    "MNIST (Digits)",
    "Fashion MNIST",
    "CIFAR-10",
    "CIFAR-100",
    "Intel Image",
    "Buildings",
    "UrbanSound8K",
    "GTZAN (Music Genre)",
    "Speech Commands",
    "Region Classification",
    "Environmental",
    "IMDB",
    "News",
    "Emotion",
    "Command",
])

# ---------- Описание моделей ----------
descriptions = {
    "MNIST (Digits)": "🧮 Классическая модель для распознавания **рукописных цифр (0–9)**. Обучена на чёрно-белых изображениях 28×28 пикселей.",
    "Fashion MNIST": "👕 Распознаёт **одежду и аксессуары** (футболки, обувь, сумки и др.). Альтернатива MNIST.",
    "CIFAR-10": "🦋 Распознаёт **10 категорий объектов** — самолёты, машины, кошки, собаки и др. Цветные изображения 32×32.",
    "CIFAR-100": "🌍 Расширенная версия CIFAR-10 с **100 категориями объектов** (животные, растения, техника и т.п.).",
    "Intel Image": "🏙️ Классифицирует **фотографии местности** (города, леса, горы, пляжи и т.д.) с помощью CNN.",
    "Buildings": "🏢 Распознаёт тип **здания** по изображению — жилое, коммерческое и т.п.",
    "UrbanSound8K": "🎧 Распознаёт **городские звуки** — сирены, лай собак, звуки улицы и т.п.",
    "GTZAN (Music Genre)": "🎵 Классифицирует **жанр музыки** (рок, джаз, классика и др.) по аудиофрагменту.",
    "Speech Commands": "🗣️ Определяет **короткие голосовые команды** вроде “yes”, “no”, “stop”, “go”.",
    "Region Classification": "🌏 Определяет **регион или страну** по особенностям речи (аудио).",
    "Environmental": "🌳 Распознаёт <b>звуки окружающей среды</b> — дождь, ветер, птиц, шаги, транспорт и другие шумы природы и города.",
    "IMDB": "🎬 Определяет **тональность текста** (позитивный/негативный) из отзывов о фильмах IMDB.",
    "Emotion": "😊 Модель анализа эмоций, определяет **23 эмоциональных состояния** по тексту.",
    "News": "📰 Классифицирует новости по **4 категориям**: World, Sport, Business, Sci/Tech." ,
    "Command": "🧾 Универсальная модель, распознающая **26 различных категорий изображений** — машины, еда, дорожные знаки, предметы и элементы окружающей среды.",
}

st.markdown(f"""
### 🧠 Вы выбрали: **{name}**
{descriptions[name]}
""")

st.divider()

# ---------- Запуск выбранной модели ----------
if name == "MNIST (Digits)":
    mnist_image()

elif name == "Fashion MNIST":
    fashion_image()

elif name == "CIFAR-10":
    cifar10_image()

elif name == "CIFAR-100":
    cifar100_image()

elif name == "Intel Image":
    intel_image()

elif name == "Buildings":
    buildings_image()

elif name == "UrbanSound8K":
    urban_audio()

elif name == "GTZAN (Music Genre)":
    gtzan_audio()

elif name == "Speech Commands":
    speech_audio()

elif name == "Region Classification":
    region_audio()

elif name == "Environmental":
    environment_audio()

elif name == "IMDB":
    imdb_text()

elif name == "News":
    news_text()

elif name == "Emotion":
    emotion_text()

elif name == "Command":
    command_image()
