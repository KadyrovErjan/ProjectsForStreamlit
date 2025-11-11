# from fastapi import FastAPI, UploadFile, File, HTTPException
# import io
# import torch
# from torchvision import transforms
# import torch.nn as nn
# from PIL import Image
# import streamlit as st
#
#
# class FashionCNN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv_block = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Dropout(0.25),
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Dropout(0.25),
#         )
#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * 7 * 7, 128),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(128, 10)
#         )
#
#     def forward(self, x):
#         x = self.conv_block(x)
#         x = self.fc(x)
#         return x
#
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((28, 28)),
#     transforms.ToTensor(),
# ])
#
#
# check_image_app = FastAPI()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = FashionCNN()
# model.load_state_dict(torch.load('fashion_model.pth', map_location=device))
# model.to(device)
# model.eval()
#
#
# class_names = [
#     "T-shirt/top",
#     "Trouser",
#     "Pullover",
#     "Dress",
#     "Coat",
#     "Sandal",
#     "Shirt",
#     "Sneaker",
#     "Bag",
#     "Ankle boot"
# ]
# def fashion_image():
#     st.title('Fashion AI model')
#     st.text('Upload image with a number, and model will recognize it')
#
#     file = st.file_uploader('Choose of drop an image', type=['svg', 'png', 'jpg', 'jpeg'])
#
#     if not file:
#         st.warning('No file is uploaded')
#     else:
#         st.image(file, caption='Uploaded image')
#         if st.button('Recognize the image'):
#             try:
#                 image_data = file.read()
#                 if not image_data:
#                     raise HTTPException(status_code=400, detail='No image is given')
#                 img = Image.open(io.BytesIO(image_data))
#                 img_tensor = transform(img).unsqueeze(0).to(device)
#
#                 with torch.no_grad():
#                     y_pred = model(img_tensor)
#                     pred = y_pred.argmax(dim=1).item()
#
#                 st.success({"Answer": class_names[pred]})
#
#             except Exception as e:
#                 raise HTTPException(status_code=500, detail=str(e))
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
# 🎨 Настройки страницы
# ===========================
st.set_page_config(
    page_title="Fashion AI Классификатор",
    page_icon="👗",
    layout="centered"
)

# ===========================
# ⚙️ Преобразования изображений
# ===========================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# ===========================
# ⚙️ Модель
# ===========================
class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc(x)
        return x

# ===========================
# ⚙️ Загрузка модели
# ===========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FashionCNN().to(device)
model.load_state_dict(torch.load('fashion_model.pth', map_location=device))
model.eval()

# ===========================
# ⚙️ Классы Fashion MNIST
# ===========================
class_names = [
    "Футболка/топ",
    "Брюки",
    "Свитер",
    "Платье",
    "Пальто",
    "Сандалии",
    "Рубашка",
    "Кроссовки",
    "Сумка",
    "Ботильоны"
]

# ===========================
# 🧠 Основная функция Streamlit
# ===========================
def fashion_image():
    st.markdown("<h1 style='text-align:center; color:#FF6F61;'>👗 Fashion AI Классификатор</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:gray;'>Загрузите изображение одежды, и модель определит его класс.</p>", unsafe_allow_html=True)
    st.divider()

    file = st.file_uploader('Выберите изображение или перетащите его сюда', type=['png', 'jpg', 'jpeg'])
    if not file:
        st.info("👆 Загрузите изображение для начала работы.")
        st.stop()

    st.image(file, caption="Загруженное изображение", use_column_width=True)

    if st.button("🔍 Классифицировать"):
        try:
            image_data = file.read()
            img = Image.open(io.BytesIO(image_data))
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                y_pred = model(img_tensor)
                pred = y_pred.argmax(dim=1).item()
                prediction = class_names[pred]

            st.success(f"✅ **Распознавание:** {prediction}")

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# ===========================
# 🚀 Запуск
# ===========================
if __name__ == "__main__":
    fashion_image()

