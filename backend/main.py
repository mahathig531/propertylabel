from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import zipfile
import tempfile
import torch
import torch.nn as nn
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

# Trained Room Classifier Model
class TrainedRoomClassifier(nn.Module):
    def __init__(self, num_classes, model_path, class_mapping_path):
        super(TrainedRoomClassifier, self).__init__()
        # Load pretrained ResNet50
        self.resnet = models.resnet50(weights=None)  # We'll load our trained weights
        
        # Replace the final layer to match our training
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        
        # Load class mapping
        with open(class_mapping_path, 'r') as f:
            self.class_to_idx = json.load(f)
        
        # Create reverse mapping
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Set to evaluation mode
        self.eval()
    
    def forward(self, x):
        return self.resnet(x)
    
    def predict(self, image_path, confidence_threshold=0.5):
        """Predict room type for a given image"""
        # Load and preprocess image
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.forward(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            confidence = confidence.item()
            predicted_class = self.idx_to_class[predicted_idx.item()]
            
            # Get top 3 predictions
            top3_prob, top3_idx = torch.topk(probabilities, 3)
            top3_predictions = []
            for i in range(3):
                top3_predictions.append({
                    'class': self.idx_to_class[top3_idx[0][i].item()],
                    'confidence': top3_prob[0][i].item()
                })
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'top3_predictions': top3_predictions,
                'is_home': predicted_class in [
                    'bedroom', 'bathroom', 'kitchen', 'livingroom', 'dining_room',
                    'children_room', 'pantry', 'closet', 'garage', 'nursery', 'gameroom'
                ]
            }

# Load trained room classification model
def load_trained_room_classifier():
    """Load the fine-tuned ResNet model for room classification"""
    try:
        model_path = "best_room_classifier.pth"
        class_mapping_path = "class_mapping.json"
        
        if not os.path.exists(model_path) or not os.path.exists(class_mapping_path):
            print("Warning: Trained model not found, using default ResNet")
            return None
            
        # Load class mapping
        with open(class_mapping_path, 'r') as f:
            class_to_idx = json.load(f)
        
        num_classes = len(class_to_idx)
        model = TrainedRoomClassifier(num_classes, model_path, class_mapping_path)
        model.eval()
        print(f"Loaded trained room classifier with {num_classes} classes")
        return model
    except Exception as e:
        print(f"Error loading trained model: {e}")
        return None

# Initialize trained model
trained_room_classifier = load_trained_room_classifier()

def classify_room_trained(image_data: bytes):
    """Classify room using the trained model"""
    if trained_room_classifier is None:
        return classify_room_resnet(image_data)  # Fallback to default
    
    try:
        # Save image data to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(image_data)
            tmp_path = tmp_file.name
        
        result = trained_room_classifier.predict(tmp_path)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return {
            'room_type': result['predicted_class'],
            'confidence': result['confidence'] * 100,  # Convert to percentage
            'is_home': result['is_home'],
            'top_predictions': result['top3_predictions']
        }
    except Exception as e:
        print(f"Error in trained room classification: {e}")
        return classify_room_resnet(image_data)  # Fallback to default

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

def classify_room_resnet(image_data: bytes) -> Dict[str, Any]:
    """Classify the room type using default ResNet50 (fallback)"""
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

def add_object_detections_to_image(image_data: bytes, detections: Dict[str, Any], highlight_classes: list = None) -> bytes:
    """Add object detection bounding boxes to image"""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Draw bounding boxes
        for obj in detections['objects']:
            class_name = obj['class']
            confidence = obj['confidence']
            bbox = obj['bbox']
            
            # Check if this class should be highlighted
            should_highlight = highlight_classes is None or class_name in highlight_classes
            
            # Set color and thickness based on highlighting
            if should_highlight:
                color = (0, 255, 0)  # Green for highlighted
                thickness = 3
            else:
                color = (128, 128, 128)  # Gray for non-highlighted
                thickness = 1
            
            # Draw bounding box
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
            
            # Add label
            label = f"{class_name} {confidence:.1f}%"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(image, (bbox[0], bbox[1] - label_size[1] - 10), 
                         (bbox[0] + label_size[0], bbox[1]), color, -1)
            
            # Draw label text
            cv2.putText(image, label, (bbox[0], bbox[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Convert back to bytes
        _, buffer = cv2.imencode('.jpg', image)
        return buffer.tobytes()
        
    except Exception as e:
        print(f"Error adding object detections: {e}")
        return image_data

@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    """Upload and classify a single image"""
    try:
        # Read image data
        image_data = await file.read()
        
        # Classify room using trained model
        room_info = classify_room_trained(image_data)
        
        # Add room label to image
        labeled_image = add_room_label_to_image(image_data, room_info)
        
        # Convert to base64
        image_base64 = base64.b64encode(labeled_image).decode('utf-8')
        
        return {
            "success": True,
            "room_info": room_info,
            "image": image_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-objects/")
async def detect_objects_endpoint(file: UploadFile = File(...)):
    """Detect objects in an image"""
    try:
        # Read image data
        image_data = await file.read()
        
        # Detect objects
        detections = detect_objects(image_data)
        
        # Add bounding boxes to image
        labeled_image = add_object_detections_to_image(image_data, detections)
        
        # Convert to base64
        image_base64 = base64.b64encode(labeled_image).decode('utf-8')
        
        return {
            "success": True,
            "detections": detections,
            "image": image_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-complete/")
async def process_complete(file: UploadFile = File(...)):
    """Complete processing: room classification + object detection"""
    try:
        # Read image data
        image_data = await file.read()
        
        # Classify room
        room_info = classify_room_trained(image_data)
        
        # Detect objects
        detections = detect_objects(image_data)
        
        # Add room label first
        image_with_room = add_room_label_to_image(image_data, room_info)
        
        # Then add object detections
        final_image = add_object_detections_to_image(image_with_room, detections)
        
        # Convert to base64
        image_base64 = base64.b64encode(final_image).decode('utf-8')
        
        return {
            "success": True,
            "room_info": room_info,
            "detections": detections,
            "image": image_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-zip/")
async def upload_zip(file: UploadFile = File(...)):
    """Upload a ZIP file containing multiple images"""
    try:
        # Read ZIP file
        zip_data = await file.read()
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save ZIP file
            zip_path = os.path.join(temp_dir, "upload.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_data)
            
            # Extract ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Process all images
            results = []
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            
            for root, dirs, files in os.walk(temp_dir):
                for file_name in files:
                    if any(file_name.lower().endswith(ext) for ext in image_extensions):
                        file_path = os.path.join(root, file_name)
                        
                        # Read image
                        with open(file_path, 'rb') as f:
                            image_data = f.read()
                        
                        # Process image
                        room_info = classify_room_trained(image_data)
                        detections = detect_objects(image_data)
                        
                        # Add labels
                        image_with_room = add_room_label_to_image(image_data, room_info)
                        final_image = add_object_detections_to_image(image_with_room, detections)
                        
                        results.append({
                            "filename": file_name,
                            "room_info": room_info,
                            "detections": detections,
                            "image": base64.b64encode(final_image).decode('utf-8')
                        })
            
            return {
                "success": True,
                "results": results,
                "count": len(results)
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "trained_model_loaded": trained_room_classifier is not None,
        "yolo_model_loaded": yolo_model is not None
    }

@app.get("/")
async def read_root():
    """Root endpoint"""
    return {
        "message": "Property Labeling Bot API",
        "version": "2.0",
        "features": [
            "Trained ResNet50 for room classification (36 categories)",
            "YOLOv8x for object detection",
            "Support for single images and ZIP files",
            "Interactive object highlighting"
        ]
    }

@app.post("/highlight-object/")
async def highlight_object(file: UploadFile = File(...), class_name: str = Body(...)):
    """Highlight specific objects in an image"""
    try:
        # Read image data
        image_data = await file.read()
        
        # Detect objects
        detections = detect_objects(image_data)
        
        # Highlight specific class
        labeled_image = add_object_detections_to_image(image_data, detections, [class_name])
        
        # Convert to base64
        image_base64 = base64.b64encode(labeled_image).decode('utf-8')
        
        return {
            "success": True,
            "detections": detections,
            "highlighted_class": class_name,
            "image": image_base64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 