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
#             nn.Linear(256, 26)
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
#             nn.Linear(1024, 26)
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
# state_dict_gray = torch.load('model_gray_command.pth', map_location=device)
# model_gray.load_state_dict(state_dict_gray, strict=False)
# model_gray.to(device)
# model_gray.eval()
# state_dict_rgb = torch.load('model_rgb_command.pth', map_location=device)
# model_rgb.load_state_dict(state_dict_rgb, strict=False)
# model_rgb.to(device)
# model_rgb.eval()
#
# class_names = ['Audi',
#  'Bmw',
#  'Burger',
#  'Mercedes',
#  'Nissan',
#  'No entry',
#  'No parking',
#  'Pasta',
#  'Pedestrian',
#  'Pizza',
#  'Salad',
#  'Shawarma',
#  'Speed limit',
#  'Stop',
#  'Toyota',
#  'book',
#  'buildings',
#  'forest',
#  'glacier',
#  'glasses',
#  'laptop',
#  'mountain',
#  'mug',
#  'phone',
#  'sea',
#  'street']
# def command_image():
#     name = st.radio("Choose input method:", ["GREY", "RGB"], horizontal=True)
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

from fastapi import HTTPException
import io
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import streamlit as st


# ===========================
# 🎨 Оформление страницы
# ===========================
st.set_page_config(
    page_title="Intel Image AI Classifier",
    page_icon="🧠",
    layout="wide"
)

# ===========================
# 🎨 Кастомный UI
# ===========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2ff, #ffffff);
    font-family: "Inter", sans-serif;
}
.title {
    text-align:center;
    font-size:46px;
    font-weight:800;
    color:#333;
}
.subtitle {
    text-align:center;
    font-size:19px;
    color:#666;
}
.upload-box {
    padding:25px;
    border-radius:18px;
    background:white;
    border:1px solid #dce3ff;
    box-shadow:0 5px 18px rgba(0,0,0,0.06);
}
.pred-box {
    padding:20px;
    border-radius:15px;
    background:#e9f9ee;
    border:1px solid #8bd49b;
    font-size:26px;
    font-weight:700;
    text-align:center;
    color:#166534;
}
.stButton>button {
    border-radius:12px;
    padding:12px 32px;
    font-size:18px;
    font-weight:600;
    background:#4b70f5;
    color:white;
    border:none;
}
.stButton>button:hover {
    background:#3555d4;
}
</style>
""", unsafe_allow_html=True)


# ===========================
# ⚙️ Transforms
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
# ⚙️ Models
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
            nn.Linear(256, 26)
        )

    def forward(self, x):
        return self.second(self.first(x))


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
            nn.Linear(1024, 26)
        )

    def forward(self, x):
        return self.second(self.first(x))


# ===========================
# ⚙️ Load models
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_gray = CheckImageVGGGrey()
model_rgb = CheckImageVGGRGB()

model_gray.load_state_dict(torch.load("model_gray_command.pth", map_location=device))
model_rgb.load_state_dict(torch.load("model_rgb_command.pth", map_location=device))

model_gray.to(device).eval()
model_rgb.to(device).eval()


# ===========================
# 📌 Class labels
# ===========================
class_names = [
    'Audi', 'Bmw', 'Burger', 'Mercedes', 'Nissan', 'No entry', 'No parking', 'Pasta', 'Pedestrian',
    'Pizza', 'Salad', 'Shawarma', 'Speed limit', 'Stop', 'Toyota', 'book', 'buildings', 'forest',
    'glacier', 'glasses', 'laptop', 'mountain', 'mug', 'phone', 'sea', 'street'
]


# ===========================
# 🚀 UI function
# ===========================
def command_image():
    st.markdown("<h1 class='title'>🧠 Command Image AI Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Загрузите изображение — модель определит, что на нём 🤖</p>",
                unsafe_allow_html=True)
    st.divider()

    mode = st.radio("Тип модели:", ["GREY", "RGB"], horizontal=True)

    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    file = st.file_uploader("Загрузите изображение", type=["png", "jpg", "jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)

    if not file:
        st.info("👆 Добавьте изображение, чтобы начать")
        return

    st.image(file, caption="Загруженное изображение", use_column_width=True)

    if st.button("🔍 Определить"):
        try:
            img = Image.open(io.BytesIO(file.read()))

            if mode == "GREY":
                tensor = transform_gray(img).unsqueeze(0).to(device)
                model = model_gray
            else:
                tensor = transform_rgb(img).unsqueeze(0).to(device)
                model = model_rgb

            with torch.no_grad():
                y_pred = model(tensor)
                pred = y_pred.argmax(dim=1).item()

            st.markdown(
                f"<div class='pred-box'>✅ Результат: {class_names[pred]}</div>",
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Ошибка: {e}")


# ===========================
# ▶️ Run
# ===========================
if __name__ == "__main__":
    command_image()
