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

app = FastAPI(title="Property Labeling Bot", description="Enhanced ResNet + YOLOv8 Architecture")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
print("Loading ResNet50 model for room classification...")
resnet = models.resnet50(pretrained=True)
resnet.eval()

print("Loading YOLOv8x model for object detection...")
# Using YOLOv8x for better accuracy (larger model)
yolo_model = YOLO('yolov8x.pt')  # Using extra large model for better detection

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

# Enhanced room classification labels (mapped from ImageNet classes)
ROOM_LABELS = {
    'kitchen': ['kitchen', 'stove', 'refrigerator', 'microwave', 'sink', 'dishwasher', 'oven', 'cooktop', 'range', 'fridge'],
    'bathroom': ['bathroom', 'toilet', 'bathtub', 'shower', 'sink', 'bidet', 'vanity', 'mirror'],
    'bedroom': ['bed', 'bedroom', 'mattress', 'pillow', 'nightstand', 'dresser', 'wardrobe'],
    'living_room': ['sofa', 'couch', 'television', 'tv', 'coffee_table', 'armchair', 'fireplace', 'bookshelf'],
    'garage': ['garage', 'car', 'vehicle', 'tool', 'workbench', 'shelf', 'storage'],
    'dining_room': ['dining', 'table', 'chair', 'plate', 'chandelier', 'buffet', 'sideboard'],
    'office': ['desk', 'computer', 'laptop', 'office', 'chair', 'monitor', 'keyboard'],
    'laundry': ['washer', 'dryer', 'laundry', 'washing_machine'],
    'basement': ['basement', 'storage', 'furnace', 'water_heater'],
    'attic': ['attic', 'storage', 'insulation'],
    'entryway': ['entryway', 'foyer', 'hallway', 'door'],
    'pantry': ['pantry', 'storage', 'shelf', 'cabinet'],
    'mudroom': ['mudroom', 'entryway', 'bench', 'hook'],
    'home_office': ['desk', 'computer', 'office', 'workspace'],
    'game_room': ['game', 'pool_table', 'arcade', 'entertainment'],
    'home_gym': ['gym', 'exercise', 'treadmill', 'weights'],
    'wine_cellar': ['wine', 'cellar', 'storage', 'rack'],
    'media_room': ['media', 'theater', 'projector', 'screen'],
    'sunroom': ['sunroom', 'patio', 'glass', 'outdoor'],
    'utility_room': ['utility', 'furnace', 'water_heater', 'electrical']
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
    """Classify the room type using ResNet50"""
    try:
        image = Image.open(BytesIO(image_data)).convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            output = resnet(input_tensor)
        
        # Get top 10 predictions for better room classification
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top10_prob, top10_catid = torch.topk(probabilities, 10)
        
        predictions = []
        for i in range(top10_prob.size(0)):
            label = imagenet_labels[top10_catid[i]]
            confidence = top10_prob[i].item()
            predictions.append({
                'label': label,
                'confidence': round(confidence * 100, 2)
            })
        
        # Enhanced room type determination
        room_type = 'unknown'
        best_confidence = 0
        room_keywords = {}
        
        # Count keyword matches for each room type
        for pred in predictions:
            pred_label = pred['label'].lower()
            for room, keywords in ROOM_LABELS.items():
                for keyword in keywords:
                    if keyword in pred_label:
                        if room not in room_keywords:
                            room_keywords[room] = []
                        room_keywords[room].append({
                            'keyword': keyword,
                            'confidence': pred['confidence']
                        })
        
        # Determine best room type based on keyword matches and confidence
        for room, matches in room_keywords.items():
            if matches:
                avg_confidence = sum(match['confidence'] for match in matches) / len(matches)
                if avg_confidence > best_confidence:
                    room_type = room.replace('_', ' ').title()
                    best_confidence = avg_confidence
        
        return {
            'room_type': room_type,
            'confidence': best_confidence,
            'predictions': predictions[:5],  # Return top 5 predictions
            'keyword_matches': room_keywords
        }
        
    except Exception as e:
        print(f"Error in room classification: {e}")
        return {
            'room_type': 'unknown',
            'confidence': 0,
            'predictions': [],
            'keyword_matches': {}
        }

def detect_objects(image_data: bytes) -> Dict[str, Any]:
    """Detect objects in the image using YOLOv8x"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run YOLOv8x inference with higher confidence threshold
        results = yolo_model(image, conf=0.25, iou=0.45)  # Lower confidence threshold for more detections
        
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
        
        # Sort detections by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
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
        
        # Color palette for different object classes
        colors = [
            (0, 255, 0),   # Green
            (255, 0, 0),   # Blue
            (0, 0, 255),   # Red
            (255, 255, 0), # Cyan
            (255, 0, 255), # Magenta
            (0, 255, 255), # Yellow
            (128, 0, 128), # Purple
            (255, 165, 0), # Orange
        ]
        
        # Draw bounding boxes
        for i, obj in enumerate(detections['objects']):
            x1, y1, x2, y2 = obj['bbox']
            label = f"{obj['class']} ({obj['confidence']:.1f}%)"
            color = colors[i % len(colors)]
            
            # Draw rectangle
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(cv_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            
            # Draw label text
            cv2.putText(cv_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
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
    """Upload and process a single image with ResNet50 room classification"""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    
    try:
        image_data = await file.read()
        
        # Step 1: Room classification with ResNet50
        room_info = classify_room(image_data)
        
        # Step 2: Add room label to image
        labeled_image = add_room_label_to_image(image_data, room_info)
        
        # Step 3: Convert to base64
        base64_image = base64.b64encode(labeled_image).decode('utf-8')
        
        return {
            "filename": file.filename,
            "image": base64_image,
            "room_classification": room_info,
            "message": "Image processed with ResNet50 room classification"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/detect-objects/")
async def detect_objects_endpoint(file: UploadFile = File(...)):
    """Detect objects in an image using YOLOv8x"""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    
    try:
        image_data = await file.read()
        
        # Step 1: Object detection with YOLOv8x
        detections = detect_objects(image_data)
        
        # Step 2: Add bounding boxes to image
        annotated_image = add_object_detections_to_image(image_data, detections)
        
        # Step 3: Convert to base64
        base64_image = base64.b64encode(annotated_image).decode('utf-8')
        
        return {
            "filename": file.filename,
            "image": base64_image,
            "object_detections": detections,
            "message": "Image processed with YOLOv8x object detection"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/process-complete/")
async def process_complete(file: UploadFile = File(...)):
    """Complete processing: ResNet50 room classification + YOLOv8x object detection"""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Only image files are allowed")
    
    try:
        image_data = await file.read()
        
        # Step 1: Room classification with ResNet50
        room_info = classify_room(image_data)
        
        # Step 2: Object detection with YOLOv8x
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
            "message": "Complete processing: ResNet50 + YOLOv8x"
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
                        
                        # Process with ResNet50 room classification
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
            "message": f"Processed {len(labeled_images)} images with ResNet50 room classification"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing ZIP file: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "message": "Property Labeling Bot Backend is running",
        "models": {
            "resnet50": "loaded",
            "yolov8x": "loaded"
        },
        "architecture": "Enhanced ResNet50 + YOLOv8x"
    }

@app.get("/")
async def read_root():
    return {
        "message": "Property Image Labeling API",
        "architecture": "Enhanced ResNet50 (Room Classification) + YOLOv8x (Object Detection)",
        "models": {
            "resnet50": "Pretrained ResNet50 for room classification",
            "yolov8x": "Pretrained YOLOv8x for object detection"
        },
        "endpoints": {
            "/upload/": "Upload single image for room classification",
            "/detect-objects/": "Upload single image for object detection",
            "/process-complete/": "Complete processing (ResNet50 + YOLOv8x)",
            "/upload-zip/": "Upload ZIP file for batch processing"
        }
    } 