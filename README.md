# Tree Disease Detection using YOLOv8

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ismat-Samadov/crop_desease_detection/blob/main/crop_desease_detection.ipynb)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-yellow)](https://huggingface.co/IsmatS/crop_desease_detection)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/IsmatS/tree-disease-detector-demo)

A deep learning project for detecting unhealthy/diseased trees in aerial UAV imagery using YOLOv8s architecture. This model achieves 93.3% mAP50 on the PDT (Pests and Diseases Tree) dataset.

![Training Results](static/training_results.png)

## 🎓 Project Overview

This university project demonstrates a complete machine learning pipeline for computer vision-based tree disease detection. The system uses state-of-the-art YOLO architecture to identify unhealthy trees in aerial imagery, which has practical applications in forest management and precision agriculture.

## 🏗️ System Architecture

```mermaid
graph TD
    A[PDT Dataset<br/>HuggingFace Hub] --> B[Data Download<br/>& Extraction]
    B --> C[Dataset Processing<br/>YOLO Format]
    C --> D[Model Training<br/>YOLOv8s]
    D --> E[Model Evaluation<br/>mAP, Precision, Recall]
    E --> F[Model Export<br/>best.pt]
    F --> G[Deployment<br/>HuggingFace Hub]
    G --> H[Web Interface<br/>Gradio Demo]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style H fill:#bfb,stroke:#333,stroke-width:2px
```

## 📊 Training Pipeline

```mermaid
graph LR
    A[Raw Dataset] --> B[Data Preprocessing]
    B --> C[Train/Val/Test Split]
    C --> D[Data Augmentation]
    D --> E[Model Training]
    E --> F[Validation]
    F --> G{Performance<br/>Acceptable?}
    G -->|No| H[Hyperparameter<br/>Tuning]
    H --> E
    G -->|Yes| I[Final Model]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#bbf,stroke:#333,stroke-width:2px
    style I fill:#bfb,stroke:#333,stroke-width:2px
```

## 🔄 Data Processing Workflow

```mermaid
flowchart TD
    A[HuggingFace Dataset] --> B[Download ZIP]
    B --> C[Extract Dataset]
    C --> D{Find YOLO_txt<br/>Directory}
    D --> E[Train Split<br/>4,536 images]
    D --> F[Validation Split<br/>567 images]
    D --> G[Test Split<br/>567 images]
    E --> H[Copy Images & Labels]
    F --> H
    G --> H
    H --> I[Create data.yaml]
    I --> J[YOLO-Ready Dataset]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style J fill:#bfb,stroke:#333,stroke-width:2px
```

## 🔍 YOLOv8 Default Preprocessing Pipeline

```mermaid
flowchart TD
    A[Input Images] --> B[Format Filtering<br/>JPG, JPEG, PNG]
    B --> C[Resizing<br/>640×640 pixels]
    C --> D[Normalization<br/>0-1 Range, RGB Conversion]
    
    D --> E{YOLOv8 Default<br/>Augmentation Pipeline}
    
    E --> F[Image Transforms]
    F --> F1[Horizontal Flip<br/>p=0.5]
    F --> F2[Random Scaling<br/>±50%]
    F --> F3[Random Translation<br/>±10%]
    F --> F4[Random Erasing<br/>p=0.4]
    
    E --> G[Color Adjustments]
    G --> G1[HSV Hue<br/>±1.5%]
    G --> G2[HSV Saturation<br/>±70%]
    G --> G3[HSV Value<br/>±40%]
    
    E --> H[Advanced Augmentation]
    H --> H1[Mosaic<br/>4-image composite]
    H --> H2[Copy-Paste<br/>with flip mode]
    
    E --> I[Albumentations<br/>p=0.01 each]
    I --> I1[Blur<br/>kernel 3-7]
    I --> I2[MedianBlur<br/>kernel 3-7] 
    I --> I3[ToGray<br/>weighted RGB] 
    I --> I4[CLAHE<br/>clip limit 1.0-4.0]
    
    H1 --> J[Mosaic Deactivation<br/>at epoch 41]
    
    F1 --> K[Cache Generation]
    F2 --> K
    F3 --> K
    F4 --> K
    G1 --> K
    G2 --> K
    G3 --> K
    H2 --> K
    J --> K
    I1 --> K
    I2 --> K
    I3 --> K
    I4 --> K
    
    K --> L[Training Ready<br/>Augmented Images]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#bbf,stroke:#333,stroke-width:2px
    style L fill:#bfb,stroke:#333,stroke-width:2px
```

The model automatically applies YOLOv8's sophisticated preprocessing pipeline without explicit coding. These preprocessing stages significantly contribute to the model's robustness and accuracy:

### Basic Processing
- **Format Filtering**: Only JPG, JPEG, and PNG images are processed
- **Resizing**: All images standardized to 640×640 pixels
- **Normalization**: Pixel values normalized to [0-1] range
- **Color Space**: BGR to RGB conversion

### YOLOv8 Default Augmentations
- **Geometric Transforms**:
  - Horizontal flipping (50% probability)
  - Random scaling (±50% variation)
  - Random translation (±10% of image size)
  - Random erasing (40% probability)

- **Color Adjustments**:
  - Hue variation: ±1.5%
  - Saturation variation: ±70%
  - Brightness variation: ±40%

- **Advanced Augmentation**:
  - Mosaic: Combines 4 training images (until epoch 40)
  - Mosaic deactivation: Turned off for final 10 epochs

- **Albumentations Library** (1% probability each):
  - Blur: Random Gaussian blur (kernel 3-7)
  - MedianBlur: Salt and pepper noise reduction (kernel 3-7)
  - ToGray: Grayscale conversion with weighted average
  - CLAHE: Contrast Limited Adaptive Histogram Equalization (8×8 tile grid)

These preprocessing techniques work together to create a robust training dataset that helps the model generalize well to different lighting conditions, perspectives, and image qualities encountered in real-world UAV imagery.

## 🤖 Model Architecture

```mermaid
graph TD
    A[Input Image<br/>640x640] --> B[Backbone<br/>CSPDarknet]
    B --> C[Neck<br/>PANet]
    C --> D[Detection Head]
    D --> E[Bounding Boxes]
    D --> F[Class Scores]
    D --> G[Confidence Scores]
    E --> H[NMS<br/>Post-processing]
    F --> H
    G --> H
    H --> I[Final Detections<br/>Unhealthy Trees]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:2px
    style I fill:#bfb,stroke:#333,stroke-width:2px
```

## 📈 Training Process

```mermaid
graph TD
    A[Initialize YOLOv8s] --> B[Load Dataset]
    B --> C[Configure Training<br/>50 epochs, batch=16]
    C --> D[Train Model<br/>SGD Optimizer]
    D --> E[Monitor Losses<br/>Box, Class, DFL]
    E --> F[Validation<br/>Every Epoch]
    F --> G[Save Best Model<br/>best.pt]
    G --> H[Final Evaluation<br/>Test Set]
    
    subgraph Training Loop
        D --> E
        E --> F
        F --> D
    end
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#bbf,stroke:#333,stroke-width:2px
    style H fill:#bfb,stroke:#333,stroke-width:2px
```

## 🚀 Quick Links

- 🤗 **[Interactive Demo on Hugging Face Spaces](https://huggingface.co/spaces/IsmatS/tree-disease-detector-demo)**
- 🤗 **[Model on Hugging Face Hub](https://huggingface.co/IsmatS/crop_desease_detection)**
- 📓 **[Google Colab Notebook](https://colab.research.google.com/github/Ismat-Samadov/crop_desease_detection/blob/main/crop_desease_detection.ipynb)**

## 🎯 Try It Now!

Experience the model in action with our interactive demo:

<div align="center">
  <a href="https://huggingface.co/spaces/IsmatS/tree-disease-detector-demo">
    <img src="https://img.shields.io/badge/Try%20Demo-Hugging%20Face%20Spaces-blue?style=for-the-badge&logo=huggingface" alt="Demo">
  </a>
</div>

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| mAP50 | 0.933 |
| mAP50-95 | 0.659 |
| Precision | 0.878 |
| Recall | 0.863 |
| Training Time | 24.5 minutes |
| Device | NVIDIA A100-SXM4-40GB |

## 🔍 Detection Examples

Here are some example detections from the model showing unhealthy tree identification:

<div align="center">
<img src="static/pred_2.png" width="45%" alt="Detection Example 2">
<img src="static/pred_3.png" width="45%" alt="Detection Example 3">
</div>

<div align="center">
<img src="static/pred_4.png" width="45%" alt="Detection Example 4">
<img src="static/pred_5.png" width="45%" alt="Detection Example 5">
</div>

<div align="center">
<img src="static/pred_6.png" width="45%" alt="Detection Example 6">
</div>

The model successfully identifies unhealthy trees in various aerial imagery conditions, with confidence scores ranging from 0.32 to 0.86. These examples demonstrate the model's ability to detect multiple diseased trees in a single image with accurate bounding boxes.

## 📋 Test Predictions Preview

The following examples demonstrate the model's real-world performance on diverse test images, showcasing its ability to distinguish between healthy and diseased trees across various conditions:

### Disease Detection Results

<div align="center">
<img src="static/preview/disease prediction.png" width="70%" alt="Disease Prediction Test Results">
</div>

**Disease Detection Analysis:**
- The model successfully identifies multiple diseased trees in aerial imagery
- Bounding boxes accurately localize unhealthy vegetation with high confidence scores
- Detection works effectively across different lighting conditions and tree densities
- Multiple diseased trees can be detected simultaneously in a single image
- Confidence scores typically range from 0.3 to 0.9, indicating reliable detection thresholds

### Health Classification Results

<div align="center">
<img src="static/preview/health prediction.png" width="70%" alt="Health Prediction Test Results">
</div>

**Health Classification Analysis:**
- The model correctly identifies healthy forest areas without false positive detections
- Clean background classification demonstrates the model's ability to distinguish healthy from diseased vegetation
- No erroneous bounding boxes on healthy trees, showing good precision
- Robust performance in dense forest environments with varying canopy coverage
- Effective handling of different aerial perspectives and image resolutions

### Key Performance Insights

These test predictions validate several critical aspects of the model:

1. **Accuracy**: High precision in distinguishing diseased from healthy trees
2. **Robustness**: Consistent performance across different environmental conditions
3. **Scalability**: Effective detection in both sparse and dense forest areas
4. **Reliability**: Stable confidence scoring for practical deployment
5. **Versatility**: Adaptable to various UAV imaging scenarios and altitudes

The test results confirm the model's readiness for real-world applications in precision agriculture, forest management, and environmental monitoring.

## 🌟 Features

- High-accuracy detection of unhealthy trees in aerial imagery
- Optimized for UAV/drone captured images at 640x640 resolution
- Fast inference (~7ms per image on GPU)
- Pre-trained model available on [Hugging Face](https://huggingface.co/IsmatS/crop_desease_detection)
- Interactive web demo on [Hugging Face Spaces](https://huggingface.co/spaces/IsmatS/tree-disease-detector-demo)

## 📁 Project Structure

```
crop_desease_detection/
├── crop_desease_detection.ipynb  # Main training notebook
├── crop_desease_detection.py     # Python implementation
├── LICENSE                       # MIT License
├── README.md                     # This file
└── static/                       # Static assets
    ├── training_results.png      # Model performance visualization
    ├── pred_2.png               # Example detection 2
    ├── pred_3.png               # Example detection 3
    ├── pred_4.png               # Example detection 4
    ├── pred_5.png               # Example detection 5
    └── pred_6.png               # Example detection 6
```

## 🚀 Quick Start

### Installation

```bash
pip install ultralytics torch torchvision opencv-python matplotlib
```

### Using the Pre-trained Model

You can load the model directly from Hugging Face:

```python
from ultralytics import YOLO

# Load model from Hugging Face
model = YOLO('https://huggingface.co/IsmatS/crop_desease_detection/resolve/main/best.pt')

# Or use the model ID
model = YOLO('IsmatS/crop_desease_detection')

# Run inference
results = model('path/to/your/image.jpg')

# Process results
for result in results:
    boxes = result.boxes
    if boxes is not None:
        for box in boxes:
            confidence = box.conf[0]
            bbox = box.xyxy[0]
            print(f"Unhealthy tree detected with {confidence:.2f} confidence")

# Save annotated image
results[0].save('result.jpg')
```

### Web Interface

For a user-friendly interface, visit our [Hugging Face Space](https://huggingface.co/spaces/IsmatS/tree-disease-detector-demo) where you can:
- Upload images directly
- Adjust detection thresholds
- Visualize results instantly
- Download annotated images

### 📋 Step-by-Step Training Process

Based on our training notebook, here's the complete pipeline:

```mermaid
sequenceDiagram
    participant User
    participant Colab
    participant HuggingFace
    participant Model
    
    User->>Colab: Start Training Notebook
    Colab->>HuggingFace: Download PDT Dataset
    HuggingFace-->>Colab: 5.6GB Dataset (ZIP)
    Colab->>Colab: Extract & Process Dataset
    Colab->>Colab: Setup YOLO Format
    Colab->>Model: Initialize YOLOv8s
    Colab->>Model: Train for 50 Epochs
    Model-->>Colab: Training Metrics
    Colab->>Colab: Evaluate Performance
    Colab->>HuggingFace: Upload Trained Model
    User->>HuggingFace: Access Model & Demo
```

## 📊 Dataset

This model was trained on the [PDT (Pests and Diseases Tree) dataset](https://huggingface.co/datasets/qwer0213/PDT_dataset):

- **Training Images**: 4,536
- **Validation Images**: 567
- **Test Images**: 567
- **Resolution**: 640x640 pixels
- **Classes**: 1 (unhealthy trees)

### Dataset Statistics

| Split | Images | Labels | Backgrounds |
|-------|--------|--------|-------------|
| Train | 4,536  | 3,206  | 1,330      |
| Val   | 567    | 399    | 168        |
| Test  | 567    | 390    | 177        |

## 🏗️ Model Architecture

- **Base Model**: YOLOv8s
- **Input Size**: 640x640 pixels
- **Parameters**: 11.1M
- **GFLOPs**: 28.6
- **Layers**: 129

The trained model is available on [Hugging Face Model Hub](https://huggingface.co/IsmatS/crop_desease_detection).

### Training Configuration

```yaml
epochs: 50
batch_size: 16
optimizer: SGD
learning_rate: 0.01
momentum: 0.9
weight_decay: 0.001
device: CUDA (NVIDIA A100-40GB)
```

## 📈 Results

The model achieved excellent performance on the validation set:

- Fast convergence: reached 0.878 precision by epoch 13
- Stable training: consistent improvement without overfitting
- High accuracy: 93.3% mAP50 on validation data

View training results and performance metrics on our [Hugging Face Model Card](https://huggingface.co/IsmatS/crop_desease_detection).

## 📊 Understanding Training Metrics

### Overall Performance

Your model achieved excellent results:
- **93.3% mAP50** - Primary accuracy metric for object detection
- **65.9% mAP50-95** - Stricter accuracy measure using multiple IoU thresholds
- **87.8% Precision** - When detecting a diseased tree, the model is correct 87.8% of the time
- **86.3% Recall** - The model finds 86.3% of all diseased trees in images
- **Training Time**: 24.5 minutes on NVIDIA A100-40GB GPU

### Loss Function Evolution

```mermaid
graph LR
    A[Epoch 1<br/>Box Loss: 3.371<br/>Cls Loss: 2.346<br/>DFL Loss: 2.348] --> B[Epoch 10<br/>Box Loss: 1.8<br/>Cls Loss: 1.2<br/>DFL Loss: 1.5]
    B --> C[Epoch 25<br/>Box Loss: 1.3<br/>Cls Loss: 0.8<br/>DFL Loss: 1.2]
    C --> D[Epoch 50<br/>Box Loss: 1.117<br/>Cls Loss: 0.645<br/>DFL Loss: 1.072]
    
    style A fill:#f99,stroke:#333,stroke-width:2px
    style B fill:#ff9,stroke:#333,stroke-width:2px
    style C fill:#9f9,stroke:#333,stroke-width:2px
    style D fill:#9ff,stroke:#333,stroke-width:2px
```

### Training Progress Analysis

#### Loss Metrics Breakdown

The training process tracked three types of losses that decreased over 50 epochs:

1. **Box Loss (box_loss)**: 3.371 → 1.117
   - Measures bounding box coordinate prediction accuracy
   - Lower values indicate better localization of diseased trees

2. **Classification Loss (cls_loss)**: 2.346 → 0.6453
   - Measures object classification accuracy
   - Significant reduction shows improved disease identification

3. **DFL Loss (Distribution Focal Loss)**: 2.348 → 1.072
   - Helps with precise bounding box regression
   - Steady decrease indicates better boundary detection

#### Evaluation Metrics Evolution

- **mAP50**: Improved from 28.8% (epoch 1) to 93.3% (final)
  - Mean Average Precision at 50% IoU threshold
  - Primary accuracy metric for object detection

- **mAP50-95**: Rose from 12% to 65.9%
  - Average mAP for IoU thresholds from 50% to 95%
  - More stringent metric; 65.9% is excellent

- **Precision**: Reached 87.8%
  - True positives / (True positives + False positives)
  - Low false positive rate

- **Recall**: Achieved 86.3%
  - True positives / (True positives + False negatives)
  - Finds most diseased trees in images

### Training Characteristics

1. **Fast Initial Learning**: Major improvements in first 10 epochs
2. **Stable Plateau**: Performance stabilized around epochs 20-30
3. **Fine-tuning Phase**: Gradual improvements in final epochs
4. **No Overfitting**: Validation metrics continued improving throughout

### Model Efficiency

- **Inference Speed**: ~7ms per image on GPU
- **Model Size**: 11.1M parameters (lightweight)
- **Batch Processing**: 16 images per batch at 640x640 resolution

### Dataset Insights

The model was trained on:
- **Training**: 4,536 images (3,206 with diseased trees, 1,330 healthy backgrounds)
- **Validation**: 567 images (399 with diseased trees, 168 backgrounds)
- **Test**: 567 images (390 with diseased trees, 177 backgrounds)

Background images help the model learn to distinguish healthy from diseased trees, reducing false positives.

## 🔧 Advanced Usage

### Custom Inference Settings

```python
# Adjust detection parameters
results = model.predict(
    source='path/to/image.jpg',
    conf=0.25,  # Confidence threshold
    iou=0.45,   # IoU threshold for NMS
    imgsz=640,  # Inference size
    save=True   # Save results
)
```

### Batch Processing

```python
import glob

# Process multiple images
image_paths = glob.glob('path/to/images/*.jpg')
results = model(image_paths, batch=8)

# Process results
for i, result in enumerate(results):
    print(f"Image {i}: Detected {len(result.boxes)} unhealthy trees")
    result.save(f'result_{i}.jpg')
```

### API Usage

You can also use the model through the Hugging Face Inference API:

```python
import requests

API_URL = "https://api-inference.huggingface.co/models/IsmatS/crop_desease_detection"
headers = {"Authorization": "Bearer YOUR_HF_TOKEN"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

output = query("your_image.jpg")
```

## 🌐 Applications

- **Precision Agriculture**: Early detection of diseased trees in orchards
- **Forest Management**: Large-scale monitoring of forest health
- **Environmental Monitoring**: Tracking disease spread patterns
- **Research**: Studying tree disease progression

## 🔬 Technical Implementation Details

### Data Pipeline Implementation

```mermaid
graph TD
    A[snapshot_download] --> B[ZIP Extraction]
    B --> C[Directory Structure<br/>Exploration]
    C --> D[YOLO Format<br/>Conversion]
    D --> E[data.yaml Creation]
    E --> F[Model Training]
    
    subgraph Dataset Processing
        C --> G[Train: 4,536 imgs]
        C --> H[Val: 567 imgs]
        C --> I[Test: 567 imgs]
        G --> D
        H --> D
        I --> D
    end
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style F fill:#bbf,stroke:#333,stroke-width:2px
```

### Model Deployment Pipeline

```mermaid
graph LR
    A[Trained Model<br/>best.pt] --> B[Model Export]
    B --> C[HuggingFace Hub<br/>Upload]
    C --> D[Model Card<br/>Creation]
    D --> E[Gradio Interface]
    E --> F[HuggingFace Spaces<br/>Deployment]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#333,stroke-width:2px
    style F fill:#bfb,stroke:#333,stroke-width:2px
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PDT Dataset](https://huggingface.co/datasets/qwer0213/PDT_dataset) by Zhou et al., ECCV 2024
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) framework
- Training performed on Google Colab with NVIDIA A100 GPU
- Model hosted on [Hugging Face](https://huggingface.co/IsmatS/crop_desease_detection)
- Demo hosted on [Hugging Face Spaces](https://huggingface.co/spaces/IsmatS/tree-disease-detector-demo)

## 📚 Citation

If you use this model in your research, please cite:

```bibtex
@software{samadov2024treedisease,
  author = {Ismat Samadov},
  title = {Tree Disease Detection using YOLOv8},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Ismat-Samadov/crop_desease_detection}
}

@inproceedings{zhou2024pdt,
  title={PDT: Uav Target Detection Dataset for Pests and Diseases Tree},
  author={Zhou, Mingle and Xing, Rui and others},
  booktitle={ECCV},
  year={2024}
}
```

## 🔗 Important Links

- 🤗 **Model**: [https://huggingface.co/IsmatS/crop_desease_detection](https://huggingface.co/IsmatS/crop_desease_detection)
- 🚀 **Demo**: [https://huggingface.co/spaces/IsmatS/tree-disease-detector-demo](https://huggingface.co/spaces/IsmatS/tree-disease-detector-demo)
- 💻 **GitHub**: [https://github.com/Ismat-Samadov/crop_desease_detection](https://github.com/Ismat-Samadov/crop_desease_detection)
- 📊 **Dataset**: [https://huggingface.co/datasets/qwer0213/PDT_dataset](https://huggingface.co/datasets/qwer0213/PDT_dataset)
- 🪿 **Notebook**: [https://www.kaggle.com/code/ismetsemedov/crop-desease-detection](https://www.kaggle.com/code/ismetsemedov/crop-desease-detection)
