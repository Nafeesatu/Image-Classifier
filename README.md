
# Transfer Learning — Image Classifier

## Goal
Fine-tune a pre-trained Vision Transformer (ViT) model on a custom 5-class dataset to achieve **85%+ validation accuracy**.

## Deliverable
This `Image_Classifier.ipynb` notebook, with visible training curves and predictions, committed to GitHub.

## Setup
1.  Change runtime type: `Runtime` → `Change runtime type` → Select **T4 GPU** → `Save`.
2.  Run all cells from top-to-bottom.

## Project Overview

### Step 1: Create a Synthetic Dataset

Generated 500 synthetic images (5 classes × 100 images each) directly within the notebook. This small dataset is sufficient for transfer learning as the backbone model is already highly pre-trained.

<img src="./shapes/circle/circle_000.png" width="100" height="100"> <img src="./shapes/square/square_000.png" width="100" height="100"> <img src="./shapes/triangle/triangle_000.png" width="100" height="100"> <img src="./shapes/cross/cross_000.png" width="100" height="100"> <img src="./shapes/star/star_000.png" width="100" height="100">

### Step 2: PyTorch Dataset & DataLoaders

A custom `ShapeDataset` was created to handle image loading and transformations. The dataset was split 80/20 into 400 training samples and 100 validation samples. DataLoaders were set up for efficient batch processing.

### Step 3: Load a Pre-trained Model

The Vision Transformer (ViT) model (`google/vit-base-patch16-224`) was loaded from Hugging Face. The final classification layer was replaced to match the 5 custom classes, while the rest of the model's parameters were kept frozen to leverage its pre-trained visual features.

**Note:** A Hugging Face token (`HF_TOKEN`) was used to authenticate the model download from the Hugging Face Hub, ensuring successful loading of the ViT model. In case of failure, a ResNet18 model would be used as a fallback.

### Step 4: Train the Model

The model was trained for 10 epochs using `Adam` optimizer and `CrossEntropyLoss`. A `ReduceLROnPlateau` scheduler was used to adjust the learning rate based on validation loss.

**Results:** The model achieved 100.0% validation accuracy.

**Training Curves:**

![Training Curves](training_curves.png)

### Step 5: Predict on New Unseen Images

The best-performing model (saved as `best_model.pth`) was loaded and used to make predictions on 5 fresh, unseen images. The model successfully classified all 5 new images.

**Predictions:**

![Predictions](predictions.png)

Accuracy on 5 new images: 5/5

## Summary

|                   | Details                                       |
| :---------------- | :-------------------------------------------- |
| **Model**         | ViT-base (pre-trained on ImageNet)            |
| **Training data** | 400 images (5 classes × 80 images)            |
| **Val accuracy**  | 100.0%                                        |
| **Training time** | ~20–30 seconds per epoch on T4 GPU          |

## Key Concepts

*   **Transfer Learning:** Reusing pre-trained features from a model trained on a large dataset.
*   **Fine-tuning:** Gently updating the weights of a pre-trained model on a smaller, custom dataset.
*   **Hugging Face Hub:** A platform for sharing pre-trained AI models and datasets.
*   **ViT (Vision Transformer):** A transformer-based model that treats image patches like sentence tokens for image classification.

```
