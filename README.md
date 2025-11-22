<<<<<<< HEAD
# 📌 Project Title  
YOLO-based Object Detection using Streamlit (Local Deployment with Conda + VS Code)

---

## 📘 1. Abstract / Introduction  
This project implements a real-time object detection system using the YOLO model trained on the COCO dataset.  
The application is developed in **Python**, executed in **VS Code**, and deployed locally using **Streamlit**.  
The project demonstrates how to use YOLO for image/video inference and how to integrate it with a simple UI.

---

## 📂 2. Dataset & YOLO Model Details  
### **YOLO Model Details**  
- Model used: YOLOv8 / YOLOv11 (choose yours)  
- Pre-trained on: **MS COCO dataset**  
- Number of classes: **80**  
- Common COCO classes: person, car, dog, bicycle, airplane, truck, etc.  

### **Why COCO Dataset?**  
- Standard benchmark dataset  
- Large-scale (200k+ images)  
- Best for training and evaluating detection models  
- Widely used for research + production  

---

## ⚙️ 3. Environment Setup

### **Create Conda Environment**
```bash
conda create -n yoloproject python=3.10 -y
conda activate yoloproject

## 4. CPU Installation Steps

```bash
pip install ultralytics
pip install streamlit
pip install opencv-python
pip install pillow
pip install numpy

```

## 🚀 5. GPU Installation Steps (NVIDIA GPU)
1. Install CUDA Toolkit

2. Install cuDNN

3. Install PyTorch with GPU support:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install streamlit opencv-python pillow numpy

```

## 🖥️ 6. How to Run in VS Code using Conda
Step 1: Open VS Code
Step 2: Open your project folder
Step 3: Select Conda Interpreter

Press Ctrl + Shift + P

Search: Python: Select Interpreter

Choose: yoloproject

Step 4: Run the Streamlit App

```
streamlit run app.py

```

## 🖼️ 8. Output Screenshots (Required)

<img width="1919" height="794" alt="Screenshot 2025-11-18 092457" src="https://github.com/user-attachments/assets/43c35c76-d9a8-4ed1-b9bf-6c311b7922cb" />
<img width="1919" height="929" alt="image" src="https://github.com/user-attachments/assets/475e8abc-eac1-4c3a-8c60-64850e665448" />

---

## ⚡ 9. Enhancements / Innovations Added

Examples you may include:

- Image upload + Video upload support
- Webcam live detection
- Confidence threshold slider
- Bounding box color customization
- FPS counter for performance monitoring
- Added dark/light UI mode
- Support for GPU inference
- Improved UI design with custom CSS

---


## 📁 Required GitHub Repository Structure

📦 YOLO-Object-Detection-Project
├── app.py
├── requirements.txt
├── Screenshots/
├── README.md
├── models/ (if any)
├── utils/ (if any)
└── sample_images/ (optional)

## 🧾 10. Results & Conclusion

- YOLO-based object detection works efficiently on images and videos.  
- Streamlit provides a simple and user-friendly web UI.  
- Real-time detection is smooth on both CPU and GPU systems.  
- The project successfully demonstrates machine learning deployment using local Streamlit.
=======
# YOLOv11 Search App

*Project:* Computer Vision Powered Search Application

---

## AIM

Build an easy-to-use Streamlit web application that runs object detection over a directory of images (using YOLOv11 model weights), saves detection metadata per image, and provides a visual, filterable *search engine* to find images by detected classes and counts.

---

## APPARATUS REQUIRED (Dependencies & Tools)

* Python 3.8+
* Packages (recommended to install into a virtualenv or conda env):

  * streamlit
  * torch (or appropriate backend for YOLOv11; check your model's framework)
  * Pillow (PIL)
  * numpy
  * any additional libs used by src.inference (for example ultralytics/yolov5/yolov11 wrappers)
* Model weights file (example default name used in app): yolo11m.pt
* A folder of images to process (jpg/png).
* (Optional) arial.ttf or system fonts — fallback to default is implemented.

Install example:

bash
python -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install streamlit torch pillow numpy
# install any other project-specific requirements (see requirements.txt if present)


---

## QUICK START (run the app)

1. Put your model weights (e.g. yolo11m.pt) somewhere accessible.
2. Ensure src/ contains the project modules (inference.py, utils.py) referenced by the main app.
3. Run Streamlit:

bash
streamlit run app.py


(You can specify port: streamlit run app.py --server.port 8080.)

The app's entry point is app.py. 

---

## STEP-BY-STEP APPROACH (How the app works)

1. *Initialization*

   * app.py sets up a Streamlit UI and initializes session state keys (metadata, unique classes, search params, display options). 

2. *Process new images (Inference)*

   * User selects "Process new images", provides:

     * Image directory path (path to folder containing images).
     * Model weights path (default yolo11m.pt).
   * When *Start Inference* is pressed:

     * YOLOv11Inference(model_path) (from src.inference) is instantiated.
     * inferencer.process_directory(image_dir) runs detection over every image and builds metadata.
     * Metadata is saved via save_metadata(metadata, image_dir).
   * The app collects unique classes and per-class count options for the search UI. 

3. *Load existing metadata*

   * User can instead load an existing metadata JSON file (e.g. metadata.json) — the app will load it, rebuild class lists, and use it for searching. 

4. *Search UI*

   * Choose search mode:

     * Any of selected classes (OR)
     * All selected classes (AND)
   * Select classes to search for (multi-select).
   * Optionally set per-class count thresholds (e.g. find images with at most N instances of a class).
   * Click *Search Images*. The app filters metadata and returns matching image records. 

5. *Results Display*

   * Results are shown in a responsive grid (user can set number of columns).
   * Options: show bounding boxes, highlight matching classes, download results as JSON.
   * For each image the app draws bounding boxes and labels (confidence), and shows a metadata overlay (class counts). 

---

## FILE DIRECTORIES (Suggested / existing)

Below is a recommended structure based on app.py imports and standard layout:


project-root/
├─ app.py                         # Streamlit app (main). :contentReference[oaicite:7]{index=7}
├─ requirements.txt               # (optional) python deps
├─ yolo11m.pt                     # model weights (user provided)
├─ metadata/                      # (optional) saved metadata JSONs
│  └─ metadata_images_dir.json
├─ images/                        # example input images
│  ├─ img1.jpg
│  └─ ...
└─ src/
   ├─ inference.py                # YOLOv11Inference class — runs detection
   ├─ utils.py                    # save_metadata, load_metadata, helper functions
   └─ (other helper modules)


> Note: app.py expects src.inference.YOLOv11Inference and functions in src.utils to exist and behave as used in the app. 

---

## TYPICAL METADATA JSON FORMAT (example)

When inference completes, the app saves per-image metadata. Example single-entry (pretty-printed JSON):

json
{
  "image_path": "/path/to/images/img_001.jpg",
  "image_width": 1280,
  "image_height": 720,
  "detections": [
    {
      "class": "person",
      "confidence": 0.94,
      "bbox": [100, 50, 220, 400]   // [x1, y1, x2, y2]
    },
    {
      "class": "dog",
      "confidence": 0.87,
      "bbox": [300, 200, 480, 520]
    }
  ],
  "class_counts": {
    "person": 1,
    "dog": 1
  }
}


The app expects metadata as a list of such objects (one per image). When search results are produced, it shows the list and allows downloading a search_results.json.

---

## OUTPUT (What you will see / get)

* *Streamlit UI*

  * Home page with two main modes:

    * Process new images — run detection and save metadata.
    * Load existing metadata — load metadata and search.
  * Search panel with class selection, thresholds, AND/OR mode.
  * Responsive image grid with bounding boxes + metadata overlay.
  * Export option: *Download Results (JSON)* (search result set). 

* *Saved metadata file*

  * JSON file placed into metadata/ (or same images folder — determined by save_metadata implementation).

* **search_results.json**

  * Downloadable file containing the filtered matches (same structure as saved metadata entries).

---
## OUTPUT
<img width="1919" height="794" alt="Screenshot 2025-11-18 092457" src="https://github.com/user-attachments/assets/43c35c76-d9a8-4ed1-b9bf-6c311b7922cb" />
<img width="1919" height="929" alt="image" src="https://github.com/user-attachments/assets/475e8abc-eac1-4c3a-8c60-64850e665448" />


---

## USAGE EXAMPLES

Run inference on images folder:

bash
streamlit run app.py
# in the UI:
# - Choose "Process new images"
# - Image directory path: /home/user/images
# - Model weights path: /home/user/weights/yolo11m.pt
# - Click "Start Inference"


Load existing metadata and search:

bash
streamlit run app.py
# in the UI:
# - Choose "Load existing metadata"
# - Metadata file path: /home/user/metadata/images_metadata.json
# - Click "Load Metadata"
# - Use Search UI to filter and view results


---

## TROUBLESHOOTING / NOTES

* If the app errors when loading fonts, it falls back to PIL default font — bounding box labels will still render. 
* Ensure src.inference and src.utils are on PYTHONPATH (the app adds the project root to sys.path, so placing src/ inside project root should be sufficient). 
* If detections are unexpectedly empty:

  * Confirm the correct model weights and compatible detection code are used.
  * Inspect the outputs of YOLOv11Inference.process_directory directly (add prints or test in REPL).
* For large image sets, inference time will depend on GPU/CPU availability and model size.

---

## INFERENCE
The flask app to detect objects using yolo v11 model was executed successfully.
* Produce a metadata example JSON file ready to drop into the app.
* Create a simplified src/utils.py stub (save/load metadata, get unique classes) or a minimal src/inference.py mock for local testing.
>>>>>>> aecb32736938331880b582bc0d90877d0fe7c572
