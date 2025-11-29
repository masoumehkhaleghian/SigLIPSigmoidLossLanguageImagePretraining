# SigLIP – Sigmoid Loss for Language–Image Pre-training

This repository contains a minimal, hands-on notebook for running inference with **SigLIP**, a CLIP-style language–image model trained with a **sigmoid loss** instead of the standard softmax/contrastive loss.

The goal is to show how to:

- Load a SigLIP checkpoint from Transformers
- Prepare images and texts with the corresponding processor
- Run a forward pass to obtain image–text similarity scores
- Turn those scores into **independent probabilities** using `sigmoid`

---

## Background

**CLIP** maps images and texts into a shared embedding space and is used in many vision–language applications (image classification, retrieval, generative models, etc.).

**SigLIP** keeps the overall CLIP-like architecture but changes the training objective:

- CLIP: uses a **softmax contrastive loss** over the batch  
- SigLIP: uses a **sigmoid loss** on individual image–text pairs (binary decision: “do these belong together, yes or no?”)

This new loss:

- Scales better to large batch sizes and datasets
- Improves performance on zero-shot image classification and image–text retrieval
- Produces **per-pair probabilities** that do **not** need to sum to 1

The notebook in this repo demonstrates how to work with these probabilities in practice.

---

## Repository Structure

- `Inference_with_(multilingual)_SigLIP_a_better_CLIP_model.ipynb`  
  Jupyter notebook that walks through:
  - Installing the latest Transformers and `sentencepiece`
  - Loading the **shape-optimized SO400M** SigLIP checkpoint:
    - `google/siglip-so400m-patch14-384`
  - Downloading a sample image from COCO
  - Defining several candidate text prompts (e.g. “a photo of 2 cats”, “a photo of 2 hamburgers”, “a photo of 2 dogs”)
  - Using `AutoProcessor` to tokenize text and preprocess the image  
    (note the use of `padding="max_length"` to match the model’s training setup)
  - Running a forward pass with `AutoModel`
  - Reading `logits_per_image` and converting them to probabilities with `torch.sigmoid`
  - Printing and interpreting the probabilities for each image–text pair

---

## Requirements

- Python **3.8+**
- Jupyter Notebook / JupyterLab / VS Code (with Jupyter extension)
- Recommended Python packages:
  - `transformers` (latest, from GitHub)
  - `sentencepiece`
  - `torch`
  - `Pillow`
  - `requests`
