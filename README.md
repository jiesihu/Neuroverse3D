# Neuroverse3D: Demonstrations and Code

[![Paper Link](link-to-your-paper-if-available)](link-to-your-paper-if-available)

Welcome to the official GitHub repository for **Neuroverse3D**, the first 3D In-Context Learning (ICL) universal model for neuroimaging, as presented in our paper "[*Your Paper Title Here*](link-to-your-paper-if-available)".

This repository provides a demonstration notebook (`Demo.ipynb`) to showcase Neuroverse3D's capabilities across various neuroimaging tasks and to illustrate the flexibility of its context processing settings.

## Introduction

Neuroverse3D is a novel deep learning model designed to address the challenges of applying In-Context Learning to 3D neuroimaging data.  It overcomes the significant memory limitations of traditional ICL models when processing volumetric medical images by introducing the **Adaptive Parallel-Sequential Processing (APSP)** approach and the **U-shape fusion strategy**.  Furthermore, Neuroverse3D incorporates an **optimized loss function** to ensure balanced performance across diverse tasks and enhanced focus on crucial anatomical structures.

As a **universal model**, Neuroverse3D demonstrates robust **cross-center generalization** and proficiency across a wide range of neuroimaging tasks without requiring task-specific retraining. This offers a significant advantage for practical applications in diverse clinical and research settings.

**Key Features of Neuroverse3D:**

*   **First 3D ICL Model for Neuroimaging:**  Specifically designed to process volumetric 3D neuroimages, capturing crucial spatial information that 2D models miss.
*   **Universal Model Capabilities:**  Performs various tasks including segmentation and generation across different neuroimaging modalities and datasets.
*   **Adaptive Parallel-Sequential Processing (APSP):**  Reduces memory footprint, enabling the model to handle **unlimited context sizes** and offering flexible memory-speed trade-offs via mini-context size adjustment.
*   **U-shape Fusion Strategy:**  Efficiently integrates target and context information within a 3D U-Net architecture.
*   **Optimized Loss Function:**  Tailored for 3D neuroimages, addressing class imbalance in segmentation and enhancing anatomical detail in generation.
*   **Strong Generalization:**  Demonstrated robust performance on held-out, cross-center datasets and unseen tasks.
*   **Practical and Efficient:** Eliminates the need for task-specific retraining, offering a highly efficient and versatile solution for neuroimaging analysis.

## Demonstrations

The `Demo.ipynb` notebook provides hands-on demonstrations of Neuroverse3D's capabilities.  Follow the steps below to run the notebook and explore the model's performance on different tasks.

**Running the Demo:**

1.  **Environment Setup:** Ensure you have Python and PyTorch installed, along with the required libraries listed in `requirements.txt` (if applicable). You can install dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Download Checkpoint:** Download the pretrained Neuroverse3D checkpoint (`neuroverse3D.ckpt`) from [link-to-your-checkpoint-download] and place it in the `./checkpoint/` directory.
3.  **Run the Jupyter Notebook:** Open and run the `Demo.ipynb` notebook using Jupyter or JupyterLab.

**Demo Breakdown:**

The notebook is structured to demonstrate Neuroverse3D's performance on various tasks and settings:

*   **Setting:**  Initializes the device and checkpoint path.
*   **Load the Checkpoint:** Loads the pretrained Neuroverse3D model checkpoint and prints model parameter information.
*   **Segmentation Tasks:**
    *   **Specific Anatomical Structure Segmentation:** Demonstrates segmentation of Cerebral White Matter, Cerebral Cortex, and Thalamus.  Foreground indices are provided, referencing `./Demo_data/seg/dataset.json` for class details.  Encourages users to try different segmentation classes.
    *   **Random Brain Region Segmentation:** Shows the model's ability to segment arbitrary brain regions by randomly combining three segmentation targets. Running this section multiple times will showcase performance across diverse regions.
*   **Generation Tasks:** Demonstrates Neuroverse3D's performance on various generation tasks:
    *   **Gaussian Noise Removal:**  Illustrates noise removal from brain images corrupted with Gaussian noise.
    *   **Salt & Pepper Removal:**  Illustrates noise removal from brain images corrupted with Salt & Pepper noise.
    *   **Inpainting:** Demonstrates image inpainting, filling in masked regions of brain images.
    *   **Modality Transform:** Showcases the model's ability to transform one neuroimaging modality (e.g., T1) to another (e.g., T2).
*   **Support Unlimited Context Size:** Demonstrates Neuroverse3D's ability to handle large context sizes (up to 64 and beyond) without code modification, showcasing the benefits of APSP for memory efficiency.  Compares inference time with large vs. small context sizes.
*   **Consistency in Terms of Mini-Context Size:**  Verifies that Neuroverse3D produces consistent predictions even with varying mini-context sizes (`gs` parameter in the code), highlighting the robustness of the APSP approach and its flexibility in resource utilization.

**Adjusting Mini-Context Size (`gs` parameter):**

In the demo code, you will find the `gs` parameter in the `model.forward()` function. This parameter controls the **mini-context size** in APSP.

*   **Smaller `gs` (e.g., `gs=1`):** Reduces memory consumption, allowing for larger context sets or deployment on devices with limited memory. Inference time may increase as processing becomes more sequential.
*   **Larger `gs` (e.g., `gs=2`, `gs=...`):**  Increases memory consumption but reduces inference time as more context is processed in parallel.

Experiment with different `gs` values to find the optimal balance between memory usage and inference speed for your specific application.

## Getting Started

1.  **Clone Repository:**
    ```bash
    git clone [repository-link]
    cd Neuroverse3D
    ```
2.  **Install Dependencies:** (If `requirements.txt` is provided)
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download Pretrained Checkpoint:** Download `neuroverse3D.ckpt` from [link-to-your-checkpoint-download] and place it in the `./checkpoint/` directory.
4.  **Run Demo Notebook:**
    ```bash
    jupyter notebook Demo.ipynb
    ```

## Pretrained Checkpoint

Download the pretrained Neuroverse3D checkpoint (`neuroverse3D.ckpt`) [here](link-to-your-checkpoint-download). Place this file in the `./checkpoint/` directory to run the demonstrations.

## Citation

If you find Neuroverse3D useful in your research, please cite our paper: