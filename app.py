import os
import re
import io
import json
import base64
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import models
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__, template_folder='templates')

# =====================================================================
# SYSTEM STATIC DIR CONFIGURATIONS
# =====================================================================
DEVICE = torch.device("cpu")
CLASSIFICATION_WEIGHTS = "cirrhosis_efficientnet_b2.pth"
DETECTION_WEIGHTS = "/media/minahil/Home/University/6th_semester/CV/annotator/week3_outputs/content/runs/detect/train/weights/best.pt"
SEGMENTATION_WEIGHTS = "/home/minahil/Downloads/week4_outputs/content/segmentation_runs/train/weights/best.pth"

# Base Dataset Directories
IMAGE_DIR = os.path.join(os.path.dirname(__file__), 'mri_slices')
DB_FILE = 'annotations_db.json'
EXPORT_DIR = os.path.join(os.path.dirname(__file__), 'saved_annotations')

CLASSES_CLASSIFICATION = ['healthy', 'mild', 'moderate', 'severe']

# Dynamic Configurable Label Map
CLASS_MAPPING = {
    0: {"name": "Healthy", "id": 0},
    1: {"name": "Mild_Cirrhosis", "id": 1},
    2: {"name": "Moderate_Cirrhosis", "id": 2},
    3: {"name": "Severe_Cirrhosis", "id": 3}
}

# Model Lazy Cache Variables
classifier_model = None
segmentation_model = None
detector_model = None

# =====================================================================
# LOCAL UTILITIES & PERSISTENCE MAPPING
# =====================================================================
def load_db():
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_db(data):
    with open(DB_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def extract_label_from_filename(filename):
    match = re.search(r'CLASS_(\d+)', filename, re.IGNORECASE)
    if match:
        class_num = int(match.group(1))
        if class_num in CLASS_MAPPING:
            return CLASS_MAPPING[class_num]
    return {"name": "Healthy", "id": 0}

def serve_numpy_image(img_bgr):
    _, buffer = cv2.imencode('.png', img_bgr)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"

# =====================================================================
# CORE ARCHITECTURE INITS
# =====================================================================
def load_classification_model():
    global classifier_model
    if classifier_model is not None: return classifier_model
    model = models.efficientnet_b2()
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, len(CLASSES_CLASSIFICATION))
    )
    if os.path.exists(CLASSIFICATION_WEIGHTS):
        model.load_state_dict(torch.load(CLASSIFICATION_WEIGHTS, map_location=DEVICE))
    model.eval()
    classifier_model = model
    return classifier_model

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            x = self.ups[idx+1](torch.cat((skip_connection, x), dim=1))
        return self.final_conv(x)

def load_segmentation_model():
    global segmentation_model
    if segmentation_model is not None: return segmentation_model
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    if os.path.exists(SEGMENTATION_WEIGHTS):
        model.load_state_dict(torch.load(SEGMENTATION_WEIGHTS, map_location=DEVICE, weights_only=False))
    model.eval()
    segmentation_model = model
    return segmentation_model

def load_detector_model():
    global detector_model
    if detector_model is not None: return detector_model
    if os.path.exists(DETECTION_WEIGHTS):
        detector_model = YOLO(DETECTION_WEIGHTS)
    return detector_model

classification_transforms = transforms.Compose([
    transforms.Resize((290, 290)),
    transforms.CenterCrop((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

segmentation_transforms = transforms.Compose([
    transforms.Resize((256, 256), Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =====================================================================
# FLASK WEB ENDPOINTS
# =====================================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mri_slices/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

@app.route('/init-data')
def init_data():
    db = load_db()
    images = []
    if os.path.exists(IMAGE_DIR):
        for f in os.listdir(IMAGE_DIR):
            if 'mask' in f.lower(): continue
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(f)
    return jsonify({"images": sorted(images), "db": db})

@app.route('/process-image/<path:filename>')
def process_image(filename):
    db = load_db()
    label_info = extract_label_from_filename(filename)
    is_saved = filename in db
    
    bboxes = []
    labels = []

    if is_saved:
        for item in db[filename]:
            bboxes.append(item['bbox'])
            labels.append(item['label_id'])
    else:
        mask_filename = f"mask_{filename}"
        mask_path = os.path.join(IMAGE_DIR, mask_filename)

        if os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None and np.any(mask_img > 0):
                _, thresh = cv2.threshold(mask_img, 50, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    all_pts = np.vstack(contours)
                    x, y, w, h = cv2.boundingRect(all_pts)
                    bboxes.append([int(x), int(y), int(x + w), int(y + h)])
                    labels.append(label_info["id"])
        
        if not bboxes:
            img_path = os.path.join(IMAGE_DIR, filename)
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    h, w, _ = img.shape
                    bboxes.append([int(w * 0.35), int(h * 0.35), int(w * 0.65), int(h * 0.65)])
                    labels.append(label_info["id"])

    thermal_src = f"/generate-thermal/{filename}"
    return jsonify({
        "thermal_src": thermal_src,
        "suggested_label": label_info["name"],
        "suggested_id": label_info["id"],
        "is_saved": is_saved,
        "bboxes": bboxes,
        "labels": labels
    })

@app.route('/generate-thermal/<path:filename>')
def generate_thermal(filename):
    img_path = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(img_path): return "Image not found", 404
    src_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if src_img is None: return "Invalid image", 400

    thermal_img = cv2.applyColorMap(src_img, cv2.COLORMAP_JET)
    _, img_encoded = cv2.imencode('.png', thermal_img)
    return img_encoded.tobytes(), 200, {'Content-Type': 'image/png'}

@app.route('/save', methods=['POST'])
def save_annotation():
    data = request.json
    filename = data.get("image_name")
    frontend_results = data.get("res", [])
    custom_labels = data.get("custom_labels", {})
    
    if not filename: return jsonify({"status": "error", "message": "Missing image name"}), 400

    # Sync backend mapping with any custom target label variants registered during runtime
    for key, val in custom_labels.items():
        CLASS_MAPPING[int(key)] = {"name": val, "id": int(key)}

    db = load_db()
    entry_list = [{"bbox": entry["bbox"], "label_id": entry["label_id"]} for entry in frontend_results]
        
    db[filename] = entry_list
    save_db(db)
    
    name_part, ext = os.path.splitext(filename)
    individual_json_path = os.path.join(EXPORT_DIR, f"{name_part}.json")
    with open(individual_json_path, 'w') as f:
        json.dump({"image_file": filename, "annotations": entry_list, "custom_label_map": custom_labels}, f, indent=4)

    src_img_path = os.path.join(IMAGE_DIR, filename)
    if os.path.exists(src_img_path):
        annotated_img = cv2.imread(src_img_path)
        if annotated_img is not None:
            for entry in entry_list:
                x1, y1, x2, y2 = entry["bbox"]
                # Use a pseudo-random hash generator for custom labels if index is above 3
                l_id = entry["label_id"]
                if l_id > 3:
                    np.random.seed(l_id)
                    color = tuple(int(c) for c in np.random.randint(50, 255, size=3))
                else:
                    bgr_colors = {0: (238, 211, 34), 1: (21, 204, 250), 2: (68, 68, 239), 3: (0, 0, 255)}
                    color = bgr_colors.get(l_id, (255, 255, 255))

                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
            cv2.imwrite(os.path.join(EXPORT_DIR, f"{name_part}_annotated{ext}"), annotated_img)

    return jsonify({"status": "success"})

# =====================================================================
# LIVE IN-MEMORY BASE64 DECODING INFERENCE PIPELINE
# =====================================================================
@app.route('/evaluate_picked_file/<task_type>', methods=['POST'])
def evaluate_picked_file(task_type):
    data = request.json
    filename = data.get("filename", "").strip()
    file_data_b64 = data.get("file_data", "").strip()

    if not file_data_b64:
        return jsonify({"status": "error", "message": "No file image data payload transferred."})

    try:
        # Decode base64 stream to raw byte chunks completely in-memory
        image_bytes = base64.b64decode(file_data_b64)
        raw_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Convert to numpy arrays for cv2 processing tasks
        open_cv_image = cv2.cvtColor(np.array(raw_image), cv2.COLOR_RGB2BGR)

        # --- TASK PIPELINE 1: CLASSIFICATION ---
        if task_type == 'classification':
            active_classifier = load_classification_model()
            input_tensor = classification_transforms(raw_image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = active_classifier(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            
            confidence, class_idx = torch.max(probabilities, dim=0)
            predicted_class = CLASSES_CLASSIFICATION[class_idx.item()]
            
            vis_tensor = input_tensor.squeeze(0).cpu().permute(1, 2, 0)
            vis_tensor = vis_tensor * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            vis_tensor = torch.clamp(vis_tensor, 0, 1).numpy() * 255.0
            vis_bgr = cv2.cvtColor(vis_tensor.astype(np.uint8), cv2.COLOR_RGB2BGR)
            prob_map = {cls: f"{probabilities[idx].item() * 100:.2f}%" for idx, cls in enumerate(CLASSES_CLASSIFICATION)}
            
            return jsonify({
                'status': 'success',
                'predicted_class': predicted_class,
                'confidence': f"{confidence.item() * 100:.2f}%",
                'probability_map': prob_map,
                'output_image': serve_numpy_image(vis_bgr)
            })

        # --- TASK PIPELINE 2: DETECTOR ---
        elif task_type == 'detection':
            active_detector = load_detector_model()
            # Predict using numpy image array directly in memory context
            results = active_detector.predict(source=open_cv_image, conf=0.25, imgsz=256)
            annotated_image_bgr = results[0].plot()
            return jsonify({'status': 'success', 'output_image': serve_numpy_image(annotated_image_bgr)})

        # --- TASK PIPELINE 3: SEGMENTATION ---
        elif task_type == 'segmentation':
            active_segmenter = load_segmentation_model()
            resized_img = raw_image.resize((256, 256), Image.BILINEAR)
            input_tensor = segmentation_transforms(resized_img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                output = active_segmenter(input_tensor)
                pred_mask = (torch.sigmoid(output) > 0.5).float().squeeze().cpu().numpy()
                
            mask_uint8 = (pred_mask * 255).astype(np.uint8)
            mask_bgr = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
            return jsonify({'status': 'success', 'output_image': serve_numpy_image(mask_bgr)})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    app.run(debug=True, port=5000)