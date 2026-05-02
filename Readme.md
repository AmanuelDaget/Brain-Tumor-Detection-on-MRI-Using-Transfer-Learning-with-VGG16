# Brain Tumor Detection on MRI Using Transfer Learning with VGG16

A deep learning project for automatic classification of brain MRI images as **tumor** or **no tumor**. The solution leverages transfer learning with the VGG16 architecture using PyTorch. The pipeline features data preparation, augmentation, training, evaluation, and detailed result visualization.

![Brain MRI Example](https://raw.githubusercontent.com/AmanuelDaget/yourrepo/main/example_mri.png) <!-- <- Optional image placeholder -->

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results and Visualizations](#results-and-visualizations)
- [References](#references)

---

## Overview

- **Goal**: Detect presence of brain tumors in MRI images.
- **Model**: VGG16 pretrained on ImageNet, fine-tuned for binary classification.
- **Framework**: PyTorch (with Google Colab/Drive support).
- **Features**:
  - Automatic train/test split
  - Data augmentation (for generalization)
  - Training and test loss/accuracy tracking
  - Confusion matrix and classification report plots
  - Single image prediction utility

---

## Dataset

- Structure: 
  ```
  dataset/
    yes/
      Y1.jpg
      Y2.jpg
      ...
    no/
      N1.jpg
      N2.jpg
      ...
  ```
- Example MRI dataset: [Kaggle Brain Tumor Dataset](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)

_Note: Place your dataset in Google Drive or local directory as above._

---

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/brain-tumor-detection-vgg16.git
    cd brain-tumor-detection-vgg16
    ```

2. **Install Python dependencies:**
    ```bash
    pip install torch torchvision scikit-learn matplotlib pillow
    ```

3. **(If using Colab):**  
   - Mount Google Drive 
   - Unzip your dataset if needed,
   - Adjust `DATA_DIR` in the notebook.

---

## Usage

**Example notebook structure:**

1. **Mount Drive & Prepare Dataset**
2. **Data Preprocessing and Loading**
3. **Model Setup**
4. **Training Loop**
5. **Evaluation (Curves, Confusion Matrix)**
6. **Single Image Prediction**

```python
# Example: Predict on single image
class_names = full_dataset.classes
predict_image('/content/dataset/yes/Y1.jpg', vgg16, class_names)
```

---

## Results and Visualizations

- **Training and Test Loss/Accuracy Curves**  
  ![loss curve](assets/loss_curve.png)
  ![accuracy curve](assets/accuracy_curve.png)

- **Confusion Matrix**  
  ![confusion matrix](assets/confusion_matrix.png)
  
- **Classification Report** printed in notebook output

---

## References

- [VGG16 Paper](https://arxiv.org/abs/1409.1556)
- [Kaggle: MRI Brain Tumor Images](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
- [PyTorch Documentation](https://pytorch.org/)

---

## License

Open for research and educational purposes. Please cite the original datasets if publishing work based on this repository.
