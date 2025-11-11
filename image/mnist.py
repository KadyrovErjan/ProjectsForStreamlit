# from fastapi import UploadFile, File, HTTPException, APIRouter, FastAPI
# import io
# import torch
# from torchvision import transforms
# import torch.nn as nn
# from PIL import Image
# import streamlit as st
#
# mnist_app = FastAPI()
#
#
# class CheckImage(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(16 * 14 * 14, 64),
#             nn.ReLU(),
#             nn.Linear(64, 10),
#         )
#
#     def forward(self, x):
#         x = self.conv(x)
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
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = CheckImage()
# model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
# model.to(device)
# model.eval()
# def mnist_image():
#     st.title('MNIST Classifier')
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
#                 st.success(f'Prediction: {pred}')
#
#             except Exception as e:
#                 st.exception(f'Error: {e}')
import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from fastapi import HTTPException

# ===========================
# 🎨 Настройка страницы
# ===========================
st.set_page_config(
    page_title="MNIST AI Классификатор",
    page_icon="✏️",
    layout="centered"
)

# ===========================
# ⚙️ Модель
# ===========================
class CheckImage(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# ===========================
# ⚙️ Трансформации
# ===========================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# ===========================
# ⚙️ Загрузка модели
# ===========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CheckImage()
model.load_state_dict(torch.load('mnist_model.pth', map_location=device))
model.to(device)
model.eval()

# ===========================
# 🖌 Основная функция
# ===========================
def mnist_image():
    st.markdown("<h1 style='text-align:center; color:#FF6F61;'>✏️ MNIST AI Классификатор</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:gray;'>Нарисуйте цифру или загрузите изображение, и модель определит её.</p>", unsafe_allow_html=True)
    st.divider()

    method = st.radio("Выберите способ ввода:", ["🎨 Рисовать цифру", "📤 Загрузить изображение"], horizontal=True)

    if method == "🎨 Рисовать цифру":
        canvas_result = st_canvas(
            fill_color="#000000",       # фон
            stroke_width=15,            # толщина кисти
            stroke_color="#FFFFFF",     # цвет цифры
            background_color="#000000", # черный фон
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas",
        )

        if st.button("🔍 Распознать рисунок"):
            if canvas_result.image_data is not None:
                img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
                img = img.resize((28, 28))
                img_tensor = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    y_pred = model(img_tensor)
                    pred = y_pred.argmax(dim=1).item()

                st.success(f'✅ Распознанная цифра: {pred}')

    elif method == "📤 Загрузить изображение":
        file = st.file_uploader('Выберите изображение', type=['png', 'jpg', 'jpeg'])
        if not file:
            st.info("👆 Загрузите изображение для распознавания")
            st.stop()

        st.image(file, caption='Загруженное изображение', use_column_width=True)

        if st.button("🔍 Распознать изображение"):
            try:
                image_data = file.read()
                img = Image.open(io.BytesIO(image_data)).convert("L")
                img_tensor = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    y_pred = model(img_tensor)
                    pred = y_pred.argmax(dim=1).item()

                st.success(f'✅ Распознанная цифра: {pred}')

            except Exception as e:
                st.exception(f"❌ Ошибка: {e}")

# ===========================
# 🚀 Запуск
# ===========================
if __name__ == "__main__":
    mnist_image()
