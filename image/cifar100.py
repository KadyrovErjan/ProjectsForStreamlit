# from fastapi import FastAPI, UploadFile, File, HTTPException
# import io
# import torch
# from torchvision import transforms
# import torch.nn as nn
# from PIL import Image
# import streamlit as st
#
#
#
# transform = transforms.Compose([
#     transforms.Resize((32, 32)),  # добавь это
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
#
# class CheckImageVGG16(nn.Module):
#   def __init__(self):
#     super().__init__()
#     self.first = nn.Sequential(
#         nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), #32
#         nn.Conv2d(16, 32, kernel_size=3, padding=1),nn.ReLU(),
#         nn.MaxPool2d(2),
#
#         nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
#         nn.Conv2d(64, 128, kernel_size=3, padding=1),nn.ReLU(), #16
#         nn.MaxPool2d(2),
#
#         nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
#         nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
#         nn.MaxPool2d(2),
#
#     )
#     self.second = nn.Sequential(
#         nn.Flatten(),
#         nn.Linear(512*4*4, 1024),
#         nn.ReLU(),
#         nn.Dropout(0.5),
#         nn.Linear(1024, 100)
#     )
#
#   def forward(self, x):
#     x = self.first(x)
#     x = self.second(x)
#     return x
#
# check_image_app = FastAPI()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = CheckImageVGG16()
# state = torch.load('model_cifar100.pth', map_location=device)
# model.load_state_dict(state)
# model = model.to(device)
# model.eval()
#
# class_name = [
#     'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
#     'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
#     'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
#     'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
#     'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
#     'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
#     'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy',
#     'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal',
#     'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
#     'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
#     'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
# ]
#
#
# def cifar100_image():
#     st.title('CIFAR100 AI model')
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
#                 st.success({"Answer": class_name[pred]})
#
#             except Exception as e:
#                 raise HTTPException(status_code=500, detail=str(e))
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
    page_title="CIFAR100 AI Классификатор",
    page_icon="🖼️",
    layout="centered"
)

# ===========================
# ⚙️ Преобразования изображений
# ===========================
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ===========================
# ⚙️ Класс модели
# ===========================
class CheckImageVGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 100)
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x

# ===========================
# ⚙️ Загрузка модели
# ===========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CheckImageVGG16().to(device)
state = torch.load('model_cifar100.pth', map_location=device)
model.load_state_dict(state)
model.eval()

# ===========================
# ⚙️ Классы CIFAR100
# ===========================
class_name = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
    'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
    'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
    'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
    'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy',
    'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal',
    'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
    'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]

# ===========================
# 🧠 Основная функция Streamlit
# ===========================
def cifar100_image():
    st.markdown("<h1 style='text-align:center; color:#00ADB5;'>🖼️ CIFAR100 AI Классификатор</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:gray;'>Загрузите изображение, и модель предскажет его класс.</p>", unsafe_allow_html=True)
    st.divider()

    file = st.file_uploader('Выберите или перетащите изображение', type=['png', 'jpg', 'jpeg'])
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
                prediction = class_name[pred]

            st.success(f"✅ **Распознавание:** {prediction}")

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# ===========================
# 🚀 Запуск
# ===========================
if __name__ == "__main__":
    cifar100_image()
