from fastapi import FastAPI, UploadFile, File, HTTPException
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
from typing import List, Dict, Any
import cv2
import numpy as np
from ultralytics import YOLO
import json

# Load environment variables
load_dotenv()

app = FastAPI(title="Property Labeling Bot", description="ResNet + YOLOv8 Architecture")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
print("Loading ResNet model...")
resnet = models.resnet50(pretrained=True)
resnet.eval()

print("Loading YOLOv8 model...")
yolo_model = YOLO('yolov8n.pt')  # Using nano model for faster inference

# Load ImageNet labels
LABELS_PATH = "imagenet_classes.txt"
if not os.path.exists(LABELS_PATH):
    import urllib.request
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
        LABELS_PATH
    )

with open(LABELS_PATH) as f:
    imagenet_labels = [line.strip() for line in f.readlines()]

# Room classification labels (mapped from ImageNet classes)
ROOM_LABELS = {
    'kitchen': ['kitchen', 'stove', 'refrigerator', 'microwave', 'sink', 'dishwasher'],
    'bathroom': ['bathroom', 'toilet', 'bathtub', 'shower', 'sink'],
    'bedroom': ['bed', 'bedroom', 'mattress', 'pillow'],
    'living_room': ['sofa', 'couch', 'television', 'tv', 'coffee_table'],
    'garage': ['garage', 'car', 'vehicle', 'tool'],
    'dining_room': ['dining', 'table', 'chair', 'plate'],
    'office': ['desk', 'computer', 'laptop', 'office'],
    'laundry': ['washer', 'dryer', 'laundry'],
    'basement': ['basement', 'storage'],
    'attic': ['attic', 'storage']
}

# Preprocessing for ResNet
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def classify_room(image_data: bytes) -> Dict[str, Any]:
    """Classify the room type using ResNet"""
    try:
        image = Image.open(BytesIO(image_data)).convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            output = resnet(input_tensor)
        
        # Get top 5 predictions
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        
        predictions = []
        for i in range(top5_prob.size(0)):
            label = imagenet_labels[top5_catid[i]]
            confidence = top5_prob[i].item()
            predictions.append({
                'label': label,
                'confidence': round(confidence * 100, 2)
            })
        
        # Determine room type based on predictions
        room_type = 'unknown'
        best_confidence = 0
        
        for pred in predictions:
            pred_label = pred['label'].lower()
            for room, keywords in ROOM_LABELS.items():
                if any(keyword in pred_label for keyword in keywords):
                    if pred['confidence'] > best_confidence:
                        room_type = room.replace('_', ' ').title()
                        best_confidence = pred['confidence']
        
        return {
            'room_type': room_type,
            'confidence': best_confidence,
            'predictions': predictions
        }
        
    except Exception as e:
        print(f"Error in room classification: {e}")
        return {
            'room_type': 'unknown',
            'confidence': 0,
            'predictions': []
        }

def detect_objects(image_data: bytes) -> Dict[str, Any]:
    """Detect objects in the image using YOLOv8"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run YOLOv8 inference
        results = yolo_model(image)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get confidence and class
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = yolo_model.names[class_id]
                    
                    detections.append({
                        'class': class_name,
                        'confidence': round(confidence * 100, 2),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
        
        return {
            'objects': detections,
            'count': len(detections)
        }
        
    except Exception as e:
        print(f"Error in object detection: {e}")
        return {
            'objects': [],
            'count': 0
        }

def add_room_label_to_image(image_data: bytes, room_info: Dict[str, Any]) -> bytes:
    """Add room classification label to image"""
    try:
        image = Image.open(BytesIO(image_data)).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("Arial", 24)
        except:
            font = ImageFont.load_default()
        
        # Create label text
        label_text = f"{room_info['room_type']} ({room_info['confidence']:.1f}%)"
        
        # Get text size
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Create background rectangle
        background = Image.new('RGBA', (text_width + 20, text_height + 10), (0, 0, 0, 180))
        image.paste(background, (10, 10), background)
        
        # Draw text
        draw.text((20, 15), label_text, fill="white", font=font)
        
        # Convert back to bytes
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG', quality=95)
        return img_byte_arr.getvalue()
        
    except Exception as e:
        print(f"Error adding room label: {e}")
        return image_data

def add_object_detections_to_image(image_data: bytes, detections: Dict[str, Any]) -> bytes:
    """Add object detection bounding boxes to image"""
    try:
        # Convert PIL image to OpenCV format
        pil_image = Image.open(BytesIO(image_data))
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Draw bounding boxes
        for obj in detections['objects']:
            x1, y1, x2, y2 = obj['bbox']
            label = f"{obj['class']} ({obj['confidence']:.1f}%)"
            
            # Draw rectangle
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(cv_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(cv_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Convert back to PIL and then to bytes
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        img_byte_arr = BytesIO()
        pil_image.save(img_byte_arr, format='JPEG', quality=95)
        return img_byte_arr.getvalue()
        
    except Exception as e:
        print(f"Error adding object detections: {e}")
        return image_data

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    """Upload and process a single image with ResNet room classification"""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    
    try:
        image_data = await file.read()
        
        # Step 1: Room classification with ResNet
        room_info = classify_room(image_data)
        
        # Step 2: Add room label to image
        labeled_image = add_room_label_to_image(image_data, room_info)
        
        # Step 3: Convert to base64
        base64_image = base64.b64encode(labeled_image).decode('utf-8')
        
        return {
            "filename": file.filename,
            "image": base64_image,
            "room_classification": room_info,
            "message": "Image processed with ResNet room classification"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/detect-objects/")
async def detect_objects_endpoint(file: UploadFile = File(...)):
    """Detect objects in an image using YOLOv8"""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    
    try:
        image_data = await file.read()
        
        # Step 1: Object detection with YOLOv8
        detections = detect_objects(image_data)
        
        # Step 2: Add bounding boxes to image
        annotated_image = add_object_detections_to_image(image_data, detections)
        
        # Step 3: Convert to base64
        base64_image = base64.b64encode(annotated_image).decode('utf-8')
        
        return {
            "filename": file.filename,
            "image": base64_image,
            "object_detections": detections,
            "message": "Image processed with YOLOv8 object detection"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/process-complete/")
async def process_complete(file: UploadFile = File(...)):
    """Complete processing: ResNet room classification + YOLOv8 object detection"""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    
    try:
        image_data = await file.read()
        
        # Step 1: Room classification with ResNet
        room_info = classify_room(image_data)
        
        # Step 2: Object detection with YOLOv8
        detections = detect_objects(image_data)
        
        # Step 3: Add room label to image
        image_with_room = add_room_label_to_image(image_data, room_info)
        
        # Step 4: Add object detections to image
        final_image = add_object_detections_to_image(image_with_room, detections)
        
        # Step 5: Convert to base64
        base64_image = base64.b64encode(final_image).decode('utf-8')
        
        return {
            "filename": file.filename,
            "image": base64_image,
            "room_classification": room_info,
            "object_detections": detections,
            "message": "Complete processing: ResNet + YOLOv8"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/upload-zip/")
async def upload_zip(file: UploadFile = File(...)):
    """Upload and process a ZIP file containing multiple images"""
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only ZIP files are allowed")
    
    try:
        zip_data = await file.read()
        labeled_images = []
        
        with zipfile.ZipFile(BytesIO(zip_data), 'r') as zip_file:
            for filename in zip_file.namelist():
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    with zip_file.open(filename) as image_file:
                        image_data = image_file.read()
                        
                        # Process with ResNet room classification
                        room_info = classify_room(image_data)
                        labeled_image = add_room_label_to_image(image_data, room_info)
                        
                        base64_image = base64.b64encode(labeled_image).decode('utf-8')
                        labeled_images.append({
                            "filename": filename,
                            "image": base64_image,
                            "room_classification": room_info
                        })
        
        if not labeled_images:
            raise HTTPException(status_code=400, detail="No valid images found in ZIP file")
        
        return {
            "images": labeled_images,
            "count": len(labeled_images),
            "message": f"Processed {len(labeled_images)} images with ResNet room classification"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing ZIP file: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "message": "Property Labeling Bot Backend is running",
        "models": {
            "resnet": "loaded",
            "yolov8": "loaded"
        }
    }

@app.get("/")
async def read_root():
    return {
        "message": "Property Image Labeling API",
        "architecture": "ResNet (Room Classification) + YOLOv8 (Object Detection)",
        "endpoints": {
            "/upload/": "Upload single image for room classification",
            "/detect-objects/": "Upload single image for object detection",
            "/process-complete/": "Complete processing (ResNet + YOLOv8)",
            "/upload-zip/": "Upload ZIP file for batch processing"
        }
    } 