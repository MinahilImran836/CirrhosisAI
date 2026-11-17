# Liver Cirrhosis Detection & Annotation Workspace

A Flask web app for looking at liver MRI slices, annotating them, and running three deep learning models on them: **classification**, **detection** and **segmentation**. Each model grades or localises liver cirrhosis.

## What it does

### 1. Annotator
- Shows every MRI slice in `mri_slices/` (files whose names contain `mask` are skipped).
- Shows each slice next to a **Jet thermal colour map** of it, generated with OpenCV.
- Suggests a label from the filename (`CLASS_1_...` → Mild, `CLASS_2_...` → Moderate, `CLASS_3_...` → Severe).
- Proposes a bounding box automatically:
  - from the matching `mask_<filename>.png`, if there is one
  - otherwise from a default box in the centre of the image
- Lets you draw or edit boxes, pick a label, or **create your own custom labels**.
- **Commit Annotations to Disk** saves:
  - `annotations_db.json`, a combined database of all annotations
  - `saved_annotations/<name>.json`, the annotation for that one image
  - `saved_annotations/<name>_annotated.png`, the image with its boxes drawn on

### 2. Model inference
Upload any image and choose a task:

| Task | Model | Output |
|---|---|---|
| Classification | EfficientNet-B2 (custom head) | Healthy / Mild / Moderate / Severe, with confidence and per-class probabilities |
| Detection | YOLO (Ultralytics) | Image with the detected regions boxed |
| Segmentation | U-Net (built in PyTorch) | Binary mask of the region |

The models load the first time they are used and then stay in memory. Inference runs on the CPU.

## Tech stack
- **Backend:** Python, Flask
- **Deep learning:** PyTorch, Torchvision (EfficientNet-B2, U-Net), Ultralytics YOLO
- **Image processing:** OpenCV, Pillow, NumPy
- **Frontend:** HTML, CSS and plain JavaScript (`templates/index.html`)
- **Storage:** JSON files

## Project structure
```
.
├── app.py                          # Flask server + model code + API routes
├── templates/index.html            # Web UI (annotator + inference)
├── mri_slices/                     # MRI slice images (+ optional mask_*.png)
├── saved_annotations/              # Exported per-image JSON + annotated PNGs
├── cirrhosis_efficientnet_b2.pth   # Classification weights
├── best.pt                         # YOLO detection weights
└── best.pth                        # U-Net segmentation weights
```

## Setup & run
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install flask torch torchvision ultralytics opencv-python pillow numpy
python app.py
```
Then open http://127.0.0.1:5000

> **Note:** In `app.py`, `DETECTION_WEIGHTS` and `SEGMENTATION_WEIGHTS` point to absolute paths on the original machine. To use the weights in this folder, change them to `"best.pt"` and `"best.pth"`.

## API routes
| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Web UI |
| `/init-data` | GET | List the images and the saved annotations |
| `/process-image/<file>` | GET | Suggested label and bounding boxes for an image |
| `/generate-thermal/<file>` | GET | Jet colour-map version of an image |
| `/save` | POST | Save the annotations for an image |
| `/evaluate_picked_file/<task>` | POST | Run `classification`, `detection` or `segmentation` |
