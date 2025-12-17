# AI Board Scanner API

> A FastAPI-based service for converting diagram images into structured JSON representations with spatial mapping, shape detection, OCR, and relationship extraction.

[![GitHub](https://img.shields.io/badge/GitHub-muhammad--asif78%2Fnbdev__testing__simon2-blue)](https://github.com/muhammad-asif78/nbdev_testing_simon2)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Running the API](#running-the-api)
- [API Endpoints](#api-endpoints)
- [Development with nbdev](#development-with-nbdev)
- [Model Weights & Checkpoints](#model-weights--checkpoints)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

The **AI Board Scanner API** is an intelligent diagram processing pipeline that:

1. **Detects shapes** in diagrams using RF-DETR (object detection)
2. **Extracts text** using PaddleOCR
3. **Predicts angles** for rotated shapes using ResNet-based models
4. **Identifies arrows** and their connections between shapes
5. **Extracts colors** from detected shapes
6. **Generates structured JSON** with nodes, edges, and spatial relationships

This project is built using **nbdev**, allowing development in Jupyter notebooks with automatic Python module generation.

---

## ✨ Features

### Core Capabilities

- **🔍 Shape Detection**: Detects 10+ shape types (rectangles, circles, diamonds, arrows, triangles, pentagons, etc.) using RF-DETR
- **📝 OCR Processing**: Extracts text from diagrams with confidence scoring using PaddleOCR
- **📐 Angle Prediction**: Determines rotation angles for shapes using ResNet-18 models and SAM2 segmentation
- **🎨 Color Extraction**: Extracts dominant colors from detected shapes
- **🔗 Relationship Mapping**: Identifies arrow connections between shapes to build edge relationships
- **📊 JSON Output**: Generates structured JSON with canvas dimensions, nodes, edges, and text labels

### API Features

- **Health Check**: `/api/health` - Simple service health status
- **Readiness Check**: `/api/ready` - Validates all models, weights, and dependencies are loaded
- **Diagram Processing**: `/api/diagram-to-json` - Converts uploaded diagrams to JSON

### Supported Input Formats

- Images: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
- Documents: `.pdf` (first page converted to image)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
│                    (pipeline/main.py)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Spatial Pipeline Orchestrator               │
│                (pipeline/spatial_pipeline.py)                │
└─────────────────────────────────────────────────────────────┘
           │                  │                  │
           ▼                  ▼                  ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐
    │ RF-DETR  │      │ PaddleOCR│      │  Angle   │
    │ Detector │      │   OCR    │      │Prediction│
    └──────────┘      └──────────┘      └──────────┘
           │                  │                  │
           └──────────────────┴──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ JSON Builders    │
                    │ (Nodes & Edges)  │
                    └──────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- **Python 3.9+** (tested on 3.9, 3.10, 3.11, 3.12)
- **Git** (for cloning the repository)
- **Virtual environment** (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/muhammad-asif78/nbdev_testing_simon2.git
cd nbdev_testing_simon2
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
# On Linux/Mac:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Install the package in development mode
pip install -e .

# Or install from requirements (if available)
pip install -r requirements.txt
```

### Step 4: Download Model Weights

**⚠️ IMPORTANT**: Model weights and SAM2 checkpoints are **NOT** tracked in Git (they're in `.gitignore`). You need to download them separately.

#### Required Weights:

1. **RF-DETR Weights** (~400MB)
   - Path: `pipeline/weights/pre-trained-model/checkpoint_best_regular.pth`
   - Download from your model source or training artifacts

2. **Angle Detection Models** (~45MB each)
   - Path: `pipeline/weights/angle-models/`
   - Files: `Triangle.pth`, `Triangle_Angle_90_270.pth`, `best_resnet18_arrow.pth`, etc.

3. **SAM2 Checkpoint** (~323MB)
   - Path: `sam2_checkpoints/sam2_hiera_base_plus.pt`
   - Download from [SAM2 official release](https://github.com/facebookresearch/segment-anything-2)

4. **PaddleOCR Models** (auto-downloaded on first run)
   - Path: `pipeline/weights/paddleocr/`

#### Directory Structure for Weights:

```
pipeline/
├── weights/
│   ├── pre-trained-model/
│   │   └── checkpoint_best_regular.pth  # ~400MB
│   ├── angle-models/
│   │   ├── Triangle.pth
│   │   ├── Triangle_Angle_90_270.pth
│   │   ├── best_resnet18_arrow.pth
│   │   ├── best_resnet18_pentagon.pth
│   │   ├── final_racetrack.pth
│   │   └── final_rectangle.pth
│   └── paddleocr/  # Auto-downloaded
sam2_checkpoints/
└── sam2_hiera_base_plus.pt  # ~323MB
```

**Note**: These directories are in `.gitignore` to avoid pushing large binary files to GitHub.

---

## 🏃 Running the API

### Start the Server

From the repository root:

```bash
# Navigate to pipeline directory
cd pipeline

# Start Uvicorn server
uvicorn main:api --host 0.0.0.0 --port 8000 --reload
```

Or from the repository root:

```bash
# Using the virtual environment's uvicorn
.venv/bin/uvicorn pipeline.main:api --host 0.0.0.0 --port 8000 --reload
```

### Access the API

- **API Base URL**: `http://localhost:8000`
- **Interactive Docs (Swagger)**: `http://localhost:8000/docs`
- **ReDoc Documentation**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

---

## 📡 API Endpoints

### 1. Health Check

**GET** `/api/health`

Simple health check to verify the service is running.

**Response:**
```json
{
  "success": true,
  "status": "ok",
  "code": 200,
  "message": "Service is healthy"
}
```

### 2. Readiness Check

**GET** `/api/ready`

Validates that all model weights, dependencies, and packages are properly loaded.

**Response (Ready):**
```json
{
  "success": true,
  "status": "ready",
  "code": 200,
  "message": "All checks passed - weights, packages, and modules are ready"
}
```

**Response (Not Ready - 503):**
```json
{
  "success": false,
  "status": "not_ready",
  "code": 503,
  "message": "Missing required files: RF-DETR weights (/path/to/checkpoint_best_regular.pth)"
}
```

### 3. Diagram to JSON

**POST** `/api/diagram-to-json`

Converts an uploaded diagram image to structured JSON with spatial mapping.

**Request:**
- **Content-Type**: `multipart/form-data`
- **Body**: `file` (image or PDF file)

**Example using cURL:**
```bash
curl -X POST "http://localhost:8000/api/diagram-to-json" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@diagram.png"
```

**Example using Python:**
```python
import requests

url = "http://localhost:8000/api/diagram-to-json"
files = {"file": open("diagram.png", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Response:**
```json
{
  "success": true,
  "status": "ok",
  "code": 200,
  "data": {
    "canvas": {
      "width": 1920,
      "height": 1080
    },
    "nodes": [
      {
        "id": "node_0",
        "type": "rectangle",
        "x": 100,
        "y": 150,
        "width": 200,
        "height": 100,
        "label": "Start Process",
        "color": "#4A90E2",
        "angle": 0,
        "confidence": 0.95
      }
    ],
    "edges": [
      {
        "id": "edge_0",
        "source": "node_0",
        "target": "node_1",
        "type": "arrow",
        "color": "#000000"
      }
    ],
    "text_labels": []
  },
  "metadata": {
    "timestamp": "2025-12-17T18:30:00.000Z",
    "processing_time_ms": 1250,
    "api_version": "1.0.0",
    "warnings": []
  }
}
```

---

## 🔧 Development with nbdev

This project uses **nbdev** for literate programming - all code is developed in Jupyter notebooks and automatically exported to Python modules.

### nbdev Workflow

#### 1. Install nbdev

```bash
pip install nbdev
```

#### 2. Make Changes in Notebooks

All source code lives in `nbs/` directory:

```
nbs/
├── 00_core.ipynb              # Core utilities
├── 12_rf_detr.ipynb           # RF-DETR shape detection
├── 13_routers_spatial.ipynb   # FastAPI routers
├── 14_spatial_pipeline.ipynb  # Main pipeline orchestrator
├── main.ipynb                 # FastAPI application entry point
└── ...
```

#### 3. Export Notebooks to Python Modules

After editing notebooks, export them to Python files:

```bash
nbdev_export
```

This generates/updates files in `pipeline/`:
- `nbs/12_rf_detr.ipynb` → `pipeline/rf_detr/detector.py`
- `nbs/13_routers_spatial.ipynb` → `pipeline/routers/spatial.py`
- `nbs/14_spatial_pipeline.ipynb` → `pipeline/spatial_pipeline.py`
- `nbs/main.ipynb` → `pipeline/main.py`

#### 4. Run Tests

```bash
nbdev_test
```

#### 5. Build Documentation

```bash
nbdev_docs
```

#### 6. Prepare for Release

```bash
nbdev_prepare
```

This runs: `nbdev_export`, `nbdev_test`, and `nbdev_clean`.

### Important nbdev Commands

| Command | Description |
|---------|-------------|
| `nbdev_export` | Export notebooks to Python modules |
| `nbdev_test` | Run tests from notebooks |
| `nbdev_clean` | Clean notebooks (remove outputs) |
| `nbdev_docs` | Build documentation |
| `nbdev_prepare` | Export + test + clean (pre-commit) |
| `nbdev_preview` | Preview documentation locally |

### Editing Code

**⚠️ DO NOT EDIT** files in `pipeline/` directly! They are auto-generated.

**✅ ALWAYS EDIT** the corresponding `.ipynb` files in `nbs/` and run `nbdev_export`.

---

## 📦 Model Weights & Checkpoints

### Why Weights Are Not in Git

Model weights and checkpoints are **large binary files** (400MB - 323MB each) that:
- Would bloat the Git repository
- Are slow to clone/push
- Can be downloaded separately from model sources

### .gitignore Configuration

The following paths are excluded from Git tracking:

```gitignore
# Model weights and checkpoints (large files)
pipeline/weights/
sam2_checkpoints/
*.pth
*.pt

# Virtual environments
.venv/
venv/
env/

# Python cache
__pycache__/
*.pyc
*.pyo
```

### Downloading Weights

1. **RF-DETR Weights**: Download from your training artifacts or model hub
2. **SAM2 Checkpoint**: Download from [SAM2 GitHub](https://github.com/facebookresearch/segment-anything-2/releases)
3. **Angle Models**: Download from your training artifacts

Place them in the correct directories as shown in [Installation](#step-4-download-model-weights).

---

## 📁 Project Structure

```
aiboardscanner-Endpoints/
├── nbs/                          # Jupyter notebooks (source code)
│   ├── 00_core.ipynb
│   ├── 12_rf_detr.ipynb
│   ├── 13_routers_spatial.ipynb
│   ├── 14_spatial_pipeline.ipynb
│   ├── main.ipynb
│   └── ...
├── pipeline/                     # Auto-generated Python modules
│   ├── main.py                   # FastAPI application entry point
│   ├── spatial_pipeline.py       # Main pipeline orchestrator
│   ├── rf_detr/
│   │   └── detector.py           # RF-DETR shape detection
│   ├── routers/
│   │   └── spatial.py            # API endpoints
│   ├── paddle_ocr/
│   │   └── ocr.py                # PaddleOCR text extraction
│   ├── angle_detection/
│   │   ├── angle_predictor.py
│   │   └── sam2_processor.py
│   ├── arrow_processing/
│   ├── shape_processing/
│   ├── json_builders/
│   └── weights/                  # Model weights (gitignored)
│       ├── pre-trained-model/
│       ├── angle-models/
│       └── paddleocr/
├── sam2_checkpoints/             # SAM2 weights (gitignored)
│   └── sam2_hiera_base_plus.pt
├── .gitignore                    # Git ignore rules
├── .gitattributes                # Git LFS configuration
├── settings.ini                  # nbdev configuration
├── setup.py                      # Package setup
├── README.md                     # This file
└── requirements.txt              # Python dependencies
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Edit notebooks** in `nbs/` (not Python files directly!)
4. **Export changes**: `nbdev_export`
5. **Run tests**: `nbdev_test`
6. **Commit changes**: `git commit -m 'Add amazing feature'`
7. **Push to branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### Development Guidelines

- All code changes must be made in `.ipynb` notebooks
- Run `nbdev_prepare` before committing
- Add tests in notebook cells
- Update documentation in notebook markdown cells
- Do not commit model weights or checkpoints

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **RF-DETR**: Object detection model for diagram shapes
- **PaddleOCR**: OCR engine for text extraction
- **SAM2**: Segment Anything Model 2 for segmentation
- **FastAPI**: Modern web framework for building APIs
- **nbdev**: Literate programming system for Python

---

## 📞 Support

For issues, questions, or contributions:

- **GitHub Issues**: [Create an issue](https://github.com/muhammad-asif78/nbdev_testing_simon2/issues)
- **GitHub Repository**: [nbdev_testing_simon2](https://github.com/muhammad-asif78/nbdev_testing_simon2)

---

**Made with ❤️ using nbdev and FastAPI**
