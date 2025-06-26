from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import zipfile
import tempfile
import torch
from torchvision import models, transforms
from typing import List

# Load environment variables
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ResNet and labels
resnet = models.resnet50(pretrained=True)
resnet.eval()
LABELS_PATH = "imagenet_classes.txt"
if not os.path.exists(LABELS_PATH):
    import urllib.request
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        LABELS_PATH
    )
with open(LABELS_PATH) as f:
    labels = [line.strip() for line in f.readlines()]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def analyze_image(image_data):
    try:
        image = Image.open(BytesIO(image_data)).convert('RGB')
    except Exception as e:
        print(f"Warning: Could not open image: {e}")
        return None
    input_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        output = resnet(input_tensor)
    _, predicted = output.max(1)
    idx = predicted.item()
    if idx < len(labels):
        return labels[idx]
    else:
        return labels[-1]  # 'other' or last label

def add_label_to_image(image_data, label):
    image = Image.open(BytesIO(image_data))
    image_with_label = image.copy()
    draw = ImageDraw.Draw(image_with_label)
    try:
        font = ImageFont.truetype("Arial", 36)
    except IOError:
        font = ImageFont.load_default()
    text_width = draw.textlength(label, font=font)
    text_height = 36
    background = Image.new('RGBA', (int(text_width + 20), int(text_height + 20)), (0, 0, 0, 128))
    image_with_label.paste(background, (10, image.height - text_height - 30), background)
    draw.text((20, image.height - text_height - 20), label, fill="white", font=font)
    img_byte_arr = BytesIO()
    image_with_label.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

@app.post("/label-folder")
async def label_folder(files: List[UploadFile] = File(...)):
    labeled_images = []
    for file in files:
        filename = file.filename
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            print(f"Skipping non-image file: {filename}")
            continue
        image_data = await file.read()
        label = analyze_image(image_data)
        if label is None:
            print(f"Skipping file (not a valid image): {filename}")
            continue
        print(f"Label for {filename}: {label}")
        labeled_image_data = add_label_to_image(image_data, label)
        base64_image = base64.b64encode(labeled_image_data).decode('utf-8')
        labeled_images.append({
            "filename": filename,
            "image": base64_image,
            "label": label
        })
    if not labeled_images:
        return {"error": "No valid images uploaded. Please upload .jpg, .jpeg, or .png files."}
    return {"images": labeled_images}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Backend is running"}

@app.get("/")
async def read_root():
    return {"message": "Property Image Labeling API"} 