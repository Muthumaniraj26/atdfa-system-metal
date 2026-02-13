import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

# --- CONFIGURATION ---
CHECKPOINT = r"D:\mech\sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIXELS_TO_MM = 0.5  # Adjust this based on your camera height!

# --- MODEL LOADING ---
print(f"Loading {MODEL_TYPE} on {DEVICE}...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

def capture_and_measure(stage_name):
    cap = cv2.VideoCapture(0)
    print(f"Align {stage_name} metal and press SPACE to capture.")
    
    while True:
        ret, frame = cap.read()
        cv2.imshow(f"Capture {stage_name}", frame)
        if cv2.waitKey(1) & 0xFF == 32: # Spacebar
            captured_img = frame.copy()
            break
    cap.release()
    cv2.destroyAllWindows()

    # Machine Learning Segmentation
    predictor.set_image(captured_img)
    # Define a box for the whole image to help SAM find the main object
    h, w = captured_img.shape[:2]
    input_box = np.array([0, 0, w, h])
    masks, _, _ = predictor.predict(box=input_box, multimask_output=False)
    mask = masks[0].astype(np.uint8) * 255

    # Computer Vision Measurement
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_cnt = max(contours, key=cv2.contourArea)

    # 1. Rectangle Metrics
    rect = cv2.minAreaRect(main_cnt)
    (x,y), (width, height), angle = rect
    
    # 2. Circle Metrics
    (cx, cy), radius = cv2.minEnclosingCircle(main_cnt)
    
    # 3. Shape Decision (Circularity)
    area = cv2.contourArea(main_cnt)
    circularity = area / (np.pi * (radius**2))
    is_circle = circularity > 0.8 # Threshold for circle

    dims = {
        "L": max(width, height) * PIXELS_TO_MM,
        "B": min(width, height) * PIXELS_TO_MM,
        "R": radius * PIXELS_TO_MM,
        "is_circle": is_circle
    }
    return dims

# --- MAIN WORKFLOW ---
def main():
    # Phase 1: BEFORE
    print("--- PHASE 1: BEFORE FORGING ---")
    before = capture_and_measure("BEFORE")
    print(f"Baseline: L:{before['L']:.2f}mm, B:{before['B']:.2f}mm, R:{before['R']:.2f}mm")

    input("\nPerform Forging/Heat Treat. Then press Enter to analyze AFTER image...")

    # Phase 2: AFTER
    print("--- PHASE 2: AFTER FORGING ---")
    after = capture_and_measure("AFTER")
    
    # Phase 3: COMPARISON
    print("\n--- FINAL ANALYSIS ---")
    if before['is_circle']:
        delta = after['R'] - before['R']
        shape_type = "Circular"
    else:
        delta = after['L'] - before['L']
        shape_type = "Rectangular"

    status = "EXPANDED" if delta > 0.1 else "SQUEEZED" if delta < -0.1 else "NO CHANGE"
    
    print(f"Object Type: {shape_type}")
    print(f"Dimension Change: {delta:+.2f} mm")
    print(f"Result: {status}")

if __name__ == "__main__":
    main()