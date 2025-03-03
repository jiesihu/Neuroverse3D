# Universal Model for Neuroimaging

[![Paper Link](link-to-your-paper-if-available)](link-to-your-paper-if-available)

Welcome to the official GitHub repository for **Neuroverse3D**, a In-Context Learning (ICL) universal model for neuroimaging in 3D, as presented in our paper "[*Your Paper Title Here*](link-to-your-paper-if-available)".

This repository provides a demonstration notebook (`Demo.ipynb`) to showcase Neuroverse3D's capabilities across various neuroimaging tasks and to illustrate the flexibility of its context processing settings.

<div align="center">
  <img src="neuroverse3D/framework.pdf"/ width="70%"> <br>
</div>


## Introduction

As a **universal model**, Neuroverse3D demonstrates robust **cross-center generalization** and proficiency across a wide range of neuroimaging tasks without requiring task-specific retraining. This offers a significant advantage for practical applications in diverse clinical and research settings.

Neuroverse3D is a novel deep learning model designed to address the challenges of applying In-Context Learning to 3D neuroimaging data.  It overcomes the significant memory limitations of traditional ICL models when processing volumetric medical images by introducing the **Adaptive Parallel-Sequential Processing (APSP)** approach and the **U-shape fusion strategy**.

## Getting Started

The `Demo.ipynb` notebook provides hands-on demonstrations of Neuroverse3D's capabilities.  Follow the steps below to run the notebook and explore the model's performance on different tasks.

**Running the Demo:**

1.  **Environment Setup:** Ensure you have Python and PyTorch installed, along with the required libraries listed in `requirements.txt` (if applicable). You can install dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Download Checkpoint and Demo Data:** Download the pretrained Neuroverse3D checkpoint (`neuroverse3D.ckpt`) from [GoogleDrive](https://drive.google.com/drive/folders/1NrORQxSKB5jl-cvUJ2eATU1FP3EjtSUc?usp=share_link) and place it in the `./checkpoint/` directory. Download the demo images from [GoogleDrive](https://drive.google.com/drive/folders/1h4x7WtG_GDlckcR4yAI2XZdwnjBOUEt9?usp=share_link) and place it in the `./Demo_data/` directory. 
3.  **Run the Jupyter Notebook:** Open and run the `Demo.ipynb` notebook using Jupyter or JupyterLab.

## Citation

If you find Neuroverse3D useful in your research, please cite our paper: