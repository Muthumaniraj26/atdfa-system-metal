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
Install Dependencies:

Bash
pip install -r requirements.txt
Download Model Weights: Place the sam_vit_h_4b8939.pth file in the root directory.

Run Application:

Bash
python app.py
📊 Technical Flow
Baseline Setup: Upload the 'Before' image to establish a geometric reference.

Forging Analysis: Upload the 'After' image to detect changes.

Deformation Export: The system generates a PDF report comparing metrics and visualizing material displacement.

Developed by Muthumaniraj S. - Machine Learning Engineer @ RIT


---

### **Project Details Checklist**

| Feature | Implementation |
| :--- | :--- |
| **Model Link** | [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| **Tech Stack** | Python, Flask, PyTorch, OpenCV, Tailwind CSS |
| **Deployment** | Hugging Face Spaces (Cloud-Fetch Mode) |
| **Logic Type** | Pixel-to-Metric Calibration & Mask Subtraction |

### **Next Step**
You can now create a new file named `README.md` in your `D:/mech` folder, paste the content above, and push it to both GitHub and Hugging Face:

```bash
git add README.md
git commit -m "Add professional README with technical documentation"
git push github main
git push hf main
