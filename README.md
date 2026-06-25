# Image Classifier: Transfer Learning with ViT/ResNet18

This project demonstrates fine-tuning a pre-trained vision model for a custom 5-class image classification task using transfer learning. The goal is to achieve over 85% validation accuracy. The notebook is designed to be run in Google Colab with a T4 GPU.

## Project Goal

To fine-tune a pre-trained vision model (Vision Transformer or ResNet18) on a custom 5-class synthetic dataset and achieve a validation accuracy exceeding 85%.

## Setup

1.  **Runtime:** Ensure you are using a **T4 GPU** runtime in Google Colab (Runtime → Change runtime type → T4 GPU → Save).
2.  **Dependencies:** The notebook will automatically install necessary libraries (`transformers`, `torch`, `torchvision`, `Pillow`) in the first cell.
3.  **Hugging Face Token (Optional but Recommended):** For using the ViT model from Hugging Face, it is recommended to store your Hugging Face API token (`HF_TOKEN`) in Colab secrets. This allows seamless download of the pre-trained model.

Run the cells sequentially from top to bottom. The model download (~350 MB for ViT) happens once and is cached.

## Dataset

A synthetic dataset of 5 unique geometric shapes (circle, square, triangle, cross, star) is generated directly within the notebook. Each class contains 100 images, totaling 500 images. This small dataset size is sufficient due to the power of transfer learning, which reuses features learned from vast datasets like ImageNet.

-   **Classes:** `['circle', 'square', 'triangle', 'cross', 'star']`
-   **Total Images:** 500
-   **Split:** 80% training (400 images), 20% validation (100 images)
-   **Image Size:** 224x224 pixels

## Model Architecture

The project utilizes transfer learning by loading a pre-trained model and replacing only its final classification layer.

-   **Primary Model:** **ViT (Vision Transformer) `google/vit-base-patch16-224`** from Hugging Face. If the download fails, it falls back to ResNet18.
-   **Fallback Model:** **ResNet18** from `torchvision` (pre-trained on ImageNet).
-   **Fine-tuning:** Only the final classification layer is trained, while the vast majority of the pre-trained backbone parameters remain frozen. This dramatically reduces the number of trainable parameters and training time.

## Training

-   **Optimizer:** Adam with a learning rate of `2e-5`.
-   **Loss Function:** Cross-Entropy Loss (`nn.CrossEntropyLoss`).
-   **Learning Rate Scheduler:** `ReduceLROnPlateau` to reduce the learning rate by a factor of 10 if the validation loss doesn't improve for 2 epochs.
-   **Epochs:** 10 epochs.
-   **Batch Size:** 16
-   **Target:** 85%+ validation accuracy.

## Results

The model successfully achieved a high validation accuracy, demonstrating the effectiveness of transfer learning even on a small custom dataset.

-   **Best Validation Accuracy:** 98.0%
-   **Training Time:** Approximately 20-30 seconds per epoch on a T4 GPU.

### Visualizations

-   `training_curves.png`: Plots of training and validation loss and accuracy over epochs.
-   `predictions.png`: Visual examples of predictions on new, unseen images, including true labels, predicted labels, confidence scores, and probability distributions across classes.

## Key Concepts Demonstrated

-   **Transfer Learning:** Reusing pre-trained knowledge from a large model (like ViT or ResNet18) to solve a new, related task with less data and computational resources.
-   **Fine-tuning:** Gently updating a small portion of the pre-trained model's weights to adapt it to the new dataset.
-   **Hugging Face Hub:** A platform for sharing and using pre-trained AI models.
-   **Vision Transformer (ViT):** A transformer-based model that applies the transformer architecture directly to image classification by treating image patches as sequences.

## How to Run

1.  Open the `image_classifier.ipynb` notebook in Google Colab.
2.  Set the runtime to T4 GPU.
3.  (Optional) Add your Hugging Face token to Colab secrets as `HF_TOKEN` for ViT model download.
4.  Run all cells from top to bottom. The notebook will automatically generate data, train the model, and display results and plots.
