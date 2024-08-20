
# transformers-nlp-multi-label-classification

Welcome to the **transformers-nlp-multi-label-classification** repository! 🎉

This repo is all about fine-tuning HuggingFace Transformers for multi-label classification, setting up pipelines, and optimizing models for faster inference. It comes from my experience developing a custom chatbot with intent detection, where multiple topics can be true simultaneously.

I hope these scripts help you fine-tune and deploy your models with ease!

## 🛠️ Repository Structure

Here’s a quick rundown of what you’ll find in this repo:

- **`checkpoints/multi-label-classification/`**: This is where your model checkpoints will be stored during training. Save your progress and pick up where you left off!

- **`data/multi-label-classification/`**: Contains sample data for training, validation, and testing. These samples are here to demonstrate the expected format for multi-label classification problems.

- **`models/multi-label-classification/`**: This is where the fine-tuned and optimized models will be saved. After fine-tuning and optimizing, you'll find your models here, ready for action!

## 📜 Scripts

Here's what each script in the repo does:

1. **`01-multi-label-classification-train.py`**  
   Fine-tunes a HuggingFace model on a multi-label classification problem. If you're looking to train your model, this script is your starting point.

2. **`02-multi-label-classification-pipeline.py`**  
   Builds a pipeline for running inference with your fine-tuned model. This script allows you to run inference on single or multiple samples effortlessly.

3. **`03-multi-label-classification-optimize-model-for-inference.py`**  
   Optimizes your model for faster inference on CPU using ONNX Runtime. Perfect for when you're working on a development server with limited GPU memory.

4. **`04-multi-label-classification-pipeline-inference-optmized-model.py`**  
   Similar to the `02` script, but specifically for inference with the optimized model (using ONNX Runtime). Get faster predictions using a CPU!


## ⚠️ Requirements and Installation Warnings

Before you dive into the scripts, here are a few important notes about the dependencies and installation process:

**Dependency Files**

   - **`requirements-with-inference-optimization.txt`**  
     Includes dependencies for scripts `01-multi-label-classification-train.py` and `02-multi-label-classification-pipeline.py` (excludes ONNX Runtime dependencies).

   - **`requirements-without-inference-optimization.txt`**  
     Includes dependencies for all scripts, including ONNX Runtime dependencies for optimization and inference.

**Note for PyTorch and NVIDIA GPUs**

   If you are using PyTorch with an NVIDIA GPU, it's crucial to ensure you have the correct version of PyTorch installed. Before running the requirements installation, you should install the specific version of PyTorch compatible with your CUDA version:

   ```bash
   pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
