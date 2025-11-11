# from fastapi import FastAPI, UploadFile, File, HTTPException
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
# class CheckImageAlexGrey(nn.Module):
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
#             nn.Linear(256, 2)
#         )
#
#     def forward(self, x):
#         x = self.first(x)
#         x = self.second(x)
#         return x
#
# class CheckImageAlexRGB(nn.Module):
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
#             nn.Linear(1024, 2)
#         )
#
#     def forward(self, x):
#         x = self.first(x)
#         x = self.second(x)
#         return x
#
#
# check_image_app = FastAPI()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model_gray= CheckImageAlexGrey()
# model_rgb = CheckImageAlexRGB()
# model_gray.load_state_dict(torch.load('model_gray_buildings.pth', map_location=device))
# model_gray.to(device)
# model_gray.eval()
# model_rgb.load_state_dict(torch.load('model_rgb_buildings.pth', map_location=device))
# model_rgb.to(device)
# model_rgb.eval()
#
#
# class_names = [
#      'apartment building',
#      'house',
# ]
# def buildings_image():
#     name = st.radio("Choose input method:", ["Grey", "RGB"], horizontal=True)
#     if name == 'Grey':
#         st.title('Apartment and House Classifier')
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
#         st.title('Apartment and House Classifier')
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

from fastapi import FastAPI, HTTPException
import io
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import streamlit as st


# ==================== 🧠 Модели ====================
class CheckImageAlexGrey(nn.Module):
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
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x


class CheckImageAlexRGB(nn.Module):
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
            nn.Linear(1024, 2)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x


# ==================== ⚙️ Настройки ====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_gray = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

transform_rgb = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Загрузка моделей
model_gray = CheckImageAlexGrey()
model_rgb = CheckImageAlexRGB()
model_gray.load_state_dict(torch.load("model_gray_buildings.pth", map_location=device))
model_rgb.load_state_dict(torch.load("model_rgb_buildings.pth", map_location=device))
model_gray.to(device).eval()
model_rgb.to(device).eval()

class_names = ["🏢 Apartment building", "🏠 House"]

check_image_app = FastAPI(title="Building Classifier")


# ==================== 🏙️ Интерфейс Streamlit ====================
def buildings_image():
    st.set_page_config(page_title="🏠 Building Classifier", layout="centered")

    st.title("🏙️ Apartment vs House Classifier")
    st.markdown(
        """
        Это модель, которая определяет — на фотографии **дом или многоквартирное здание**. 🧱  
        Использует **сверточные нейросети (CNN)**, обученные на разных типах изображений.
        """
    )

    st.divider()
    method = st.radio("📸 Выберите тип изображения:", ["Grayscale (черно-белое)", "RGB (цветное)"], horizontal=True)

    st.write("")

    uploaded_file = st.file_uploader("📂 Загрузите изображение", type=["jpg", "jpeg", "png"])

    if not uploaded_file:
        st.info("Пожалуйста, выберите изображение в формате JPG или PNG.")
        return

    st.image(uploaded_file, caption="🖼️ Загруженное изображение", use_container_width=True)

    if st.button("🔍 Распознать", use_container_width=True):
        try:
            image_bytes = uploaded_file.read()
            if not image_bytes:
                raise HTTPException(status_code=400, detail="Файл пустой")

            img = Image.open(io.BytesIO(image_bytes))

            if method.startswith("Grayscale"):
                img_tensor = transform_gray(img).unsqueeze(0).to(device)
                model = model_gray
            else:
                img_tensor = transform_rgb(img).unsqueeze(0).to(device)
                model = model_rgb

            with torch.no_grad():
                y_pred = model(img_tensor)
                pred_idx = y_pred.argmax(dim=1).item()
                result = class_names[pred_idx]

            st.success(f"✅ Определено: **{result}**")
            st.caption(f"📊 Индекс класса: {pred_idx}")

        except Exception as e:
            st.error("❌ Ошибка при обработке изображения:")
            st.exception(e)
