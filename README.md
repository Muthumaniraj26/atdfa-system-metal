# ATDFA: AI-Powered Thermal Deformation & Feature Analysis

ATDFA is an intelligent computer vision framework designed to quantify dimensional changes in metal components before and after industrial forging or heat treatment. By leveraging the **Segment Anything Model (SAM)**, the system achieves sub-millimeter precision in identifying expansion or contraction (squeezing) through differential heatmapping.

## 🚀 Live Demo
[View on Hugging Face Spaces](https://huggingface.co/spaces/Muthumaniraj26/atdfa-system-metal)

## 🛠️ Key Techniques
* **Zero-Shot Segmentation:** Uses Meta's SAM (`vit_h`) to isolate metal specimens without pre-training on specific parts.
* **Intelligent Shape Detection:** Calculates a **Circularity Score** ($Area / (\pi \times r^2)$) to dynamically switch UI metrics between Radius and Length/Breadth.
* **Differential Heatmapping:** Performs Boolean mask subtraction to highlight thermal expansion (Red) and squeezing (Blue) in a single overlay.
* **Rotational Invariance:** Uses Minimum Area Bounding Boxes to ensure measurements remain accurate regardless of object orientation.



## 📦 Model Requirements
Due to the large file size (2.6GB), the weights are not included in this repository. The system is configured to fetch them at runtime.

* **Model:** SAM (Huge version)
* **Checkpoint:** `sam_vit_h_4b8939.pth`
* **Official Weights Link:** [Download from Meta AI](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

## 💻 Installation & Local Setup
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/Muthumaniraj26/atdfa-system-metal.git](https://github.com/Muthumaniraj26/atdfa-system-metal.git)
   cd atdfa-system-metal
