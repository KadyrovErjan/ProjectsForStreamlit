# from fastapi import FastAPI, HTTPException
# import io
# import torch
# from torchvision import transforms
# import torch.nn as nn
# from PIL import Image
# import streamlit as st
#
# transform_gray = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.Grayscale(),
#     transforms.ToTensor(),
# ])
#
# transform_rgb = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
# ])
#
#
# class CheckImageVGGGrey(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.first = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, padding=1),
#             nn.BatchNorm2d(16),
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
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#         )
#         self.second = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128 * 8 * 8, 256),
#             nn.ReLU(),
#             nn.Linear(256, 6)
#         )
#
#     def forward(self, x):
#         x = self.first(x)
#         x = self.second(x)
#         return x
#
#
# class CheckImageVGGRGB(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.first = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#         )
#         self.second = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(512 * 8 * 8, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 6)
#         )
#
#     def forward(self, x):
#         x = self.first(x)
#         x = self.second(x)
#         return x
#
# check_image_app = FastAPI()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_gray= CheckImageVGGGrey()
# model_rgb = CheckImageVGGRGB()
# state_dict_gray = torch.load('model_gray_intel.pth', map_location=device)
# model_gray.load_state_dict(state_dict_gray, strict=False)
# model_gray.to(device)
# model_gray.eval()
# state_dict_rgb = torch.load('model_rgb_intel.pth', map_location=device)
# model_rgb.load_state_dict(state_dict_rgb, strict=False)
# model_rgb.to(device)
# model_rgb.eval()
#
# def intel_image():
#     name = st.radio("Choose input method:", ["GREY", "RGB"], horizontal=True)
#
#
#     class_names=['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
#     if name == 'GREY':
#         st.title('Intel Image AI Classifier')
#         st.text('Upload image with a number, and model will recognize it')
#
#         file = st.file_uploader('Choose of drop an image', type=['svg', 'png', 'jpg', 'jpeg'])
#
#         if not file:
#             st.warning('No file is uploaded')
#         else:
#             st.image(file, caption='Uploaded image')
#             if st.button('Recognize the image'):
#                 try:
#                     image_data = file.read()
#                     if not image_data:
#                         raise HTTPException(status_code=400, detail='No image is given')
#                     img = Image.open(io.BytesIO(image_data))
#                     img_tensor = transform_gray(img).unsqueeze(0).to(device)
#
#                     with torch.no_grad():
#                         y_pred = model_gray(img_tensor)
#                         pred = y_pred.argmax(dim=1).item()
#
#                     st.success({'Prediction': class_names[pred]})
#
#                 except Exception as e:
#                     raise HTTPException(status_code=500, detail=str(e))
#
#     if name == 'RGB':
#         st.title('Intel Image AI Classifier')
#         st.text('Upload image with a number, and model will recognize it')
#
#         file = st.file_uploader('Choose of drop an image', type=['svg', 'png', 'jpg', 'jpeg'])
#
#         if not file:
#             st.warning('No file is uploaded')
#         else:
#             st.image(file, caption='Uploaded image')
#             if st.button('Recognize the image'):
#                 try:
#                     image_data = file.read()
#                     if not image_data:
#                         raise HTTPException(status_code=400, detail='No image is given')
#                     img = Image.open(io.BytesIO(image_data))
#                     img_tensor = transform_rgb(img).unsqueeze(0).to(device)
#
#
#                     with torch.no_grad():
#                         y_pred = model_rgb(img_tensor)
#                         pred = y_pred.argmax(dim=1).item()
#
#                     st.success({'Prediction': class_names[pred]})
#
#                 except Exception as e:
#                     raise HTTPException(status_code=500, detail=str(e))
#
#
#
import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import streamlit as st
from fastapi import HTTPException

# ===========================
# 🎨 Настройки страницы Streamlit
# ===========================
st.set_page_config(
    page_title="Intel Image Классификатор",
    page_icon="🧠",
    layout="centered",
)

# ===========================
# ⚙️ Классы моделей
# ===========================
class CheckImageVGGGrey(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x


class CheckImageVGGRGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, 6)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x


# ===========================
# ⚙️ Преобразования изображений
# ===========================
transform_gray = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

transform_rgb = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# ===========================
# ⚙️ Загрузка моделей
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_gray = CheckImageVGGGrey().to(device)
model_rgb = CheckImageVGGRGB().to(device)

try:
    model_gray.load_state_dict(torch.load("model_gray_intel.pth", map_location=device), strict=False)
    model_rgb.load_state_dict(torch.load("model_rgb_intel.pth", map_location=device), strict=False)
except:
    st.warning("⚠️ Файлы весов модели не найдены. Проверьте, что .pth файлы находятся в той же директории.")

model_gray.eval()
model_rgb.eval()

# ===========================
# 🧠 Основная функция приложения
# ===========================
def intel_image():
    # Заголовок
    st.markdown("<h1 style='text-align:center; color:#00ADB5;'>🏞️ Intel Image Классификатор</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:gray;'>AI-модель распознаёт тип ландшафта или городской среды.</p>", unsafe_allow_html=True)
    st.divider()

    # О модели
    with st.expander("ℹ️ О модели", expanded=False):
        st.write("""
        **Intel Image Классификатор** распознаёт 6 типов сцен:
        - 🏢 Здания  
        - 🌲 Лес  
        - 🧊 Ледник  
        - ⛰️ Горы  
        - 🌊 Море  
        - 🛣️ Улица  

        Можно загрузить изображение в формате **GREY** или **RGB**,  
        и модель предскажет наиболее вероятный класс.
        """)

    # Выбор типа изображения
    st.markdown("### 🎨 Выберите тип изображения")
    name = st.radio("Выберите тип модели:", ["GREY", "RGB"], horizontal=True)
    class_names = ['Здания', 'Лес', 'Ледник', 'Горы', 'Море', 'Улица']

    # Загрузка изображения
    st.markdown("### 📤 Загрузите изображение")
    file = st.file_uploader("Выберите или перетащите изображение", type=['png', 'jpg', 'jpeg'])

    if not file:
        st.info("👆 Пожалуйста, загрузите изображение для начала работы.")
        st.stop()

    # Предпросмотр изображения
    st.image(file, caption="Загруженное изображение", use_column_width=True)

    # Кнопка классификации
    if st.button("🔍 Классифицировать"):
        try:
            image_data = file.read()
            img = Image.open(io.BytesIO(image_data))

            if name == "GREY":
                img_tensor = transform_gray(img).unsqueeze(0).to(device)
                model = model_gray
            else:
                img_tensor = transform_rgb(img).unsqueeze(0).to(device)
                model = model_rgb

            with torch.no_grad():
                y_pred = model(img_tensor)
                pred = y_pred.argmax(dim=1).item()
                prediction = class_names[pred]

            # Отображение результата
            st.success(f"✅ **Распознавание:** {prediction}")
            st.progress((pred + 1) / len(class_names))

            # Эмодзи для наглядности
            emoji_map = {
                "Здания": "🏢",
                "Лес": "🌲",
                "Ледник": "🧊",
                "Горы": "⛰️",
                "Море": "🌊",
                "Улица": "🛣️",
            }
            st.markdown(f"<h3 style='text-align:center;'>{emoji_map[prediction]} {prediction}</h3>", unsafe_allow_html=True)

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


# ===========================
# 🚀 Запуск приложения
# ===========================
if __name__ == "__main__":
    intel_image()
