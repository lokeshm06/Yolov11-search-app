# 📌 Project Title  
YOLO-based Object Detection using Streamlit (Local Deployment with Conda + VS Code)

---

## 📘 1. Abstract
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
- Step 1: Open VS Code
- Step 2: Open your project folder
- Step 3: Select Conda Interpreter

- Press Ctrl + Shift + P

- Search: Python: Select Interpreter

- Choose: yoloproject

- Step 4: Run the Streamlit App

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

```
📦 YOLO-Object-Detection-Project
├── app.py
├── requirements.txt
├── Screenshots/
├── README.md
├── models/ (if any)
├── utils/ (if any)
└── sample_images/ (optional)
```


## 🧾 10. Results & Conclusion

- YOLO-based object detection works efficiently on images and videos.  
- Streamlit provides a simple and user-friendly web UI.  
- Real-time detection is smooth on both CPU and GPU systems.  
- The project successfully demonstrates machine learning deployment using local Streamlit.
=======


