import cv2
import numpy as np
import torch
import base64
import os
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from segment_anything import sam_model_registry, SamPredictor
from fpdf import FPDF
from datetime import datetime

app = Flask(__name__)
CORS(app)

# --- Configuration ---
CHECKPOINT = r"D:\mech\sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIXEL_TO_MM = 0.5 

# --- Model Load ---
print(f"Loading ATDFA Core on {DEVICE}...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

storage = {"before": None, "after": None}

def analyze_specimen(image_bytes):
    nparr = np.frombuffer(image_bytes.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    predictor.set_image(img)
    # --- Improved Segmentation Strategy ---
    # Instead of prompting with the full image box (which often selects the background),
    # we use a central point prompt to target the object in the middle of the frame.
    h, w = img.shape[:2]
    
    # Define a central point for the prompt
    input_point = np.array([[w // 2, h // 2]])
    input_label = np.array([1])  # 1 indicates a foreground point

    # Generate masks using the point prompt
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True, # Allow SAM to return multiple valid masks
    )
    
    # Select the mask with the highest score that is NOT the entire image
    best_mask = None
    max_score = -1
    image_area = w * h
    
    for i, mask_candidate in enumerate(masks):
        mask_area = np.sum(mask_candidate)
        # Filter out masks overly large (>95% image area) or too small (<1%)
        if 0.01 * image_area < mask_area < 0.95 * image_area:
            if scores[i] > max_score:
                best_mask = mask_candidate
                max_score = scores[i]
                
    # Fallback: if no good mask found, take the smallest valid one (assuming object < background)
    if best_mask is None:
         # Sort by size and take the smallest one likely to be the object
         sorted_indices = np.argsort([np.sum(m) for m in masks])
         best_mask = masks[sorted_indices[0]]

    mask = best_mask.astype(np.uint8) * 255
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    cnt = max(contours, key=cv2.contourArea)
    
    # 1. Geometry Calculations
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect); box = np.int64(box) # NumPy 2.x fix
    (cx_rect, cy_rect), (w_box, h_box), angle = rect
    (cx_circ, cy_circ), rad_px = cv2.minEnclosingCircle(cnt)
    
    # 2. Advanced Shape Detection Logic
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    # Polygon Approximation
    # 0.04 factor is a good balance for shape simplification
    approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
    vertices = len(approx)
    
    # Shape Classification
    shape_name = "Unknown"
    is_circ = False
    
    if vertices == 3:
        shape_name = "Triangle"
    elif vertices == 4:
        # Check aspect ratio for Square vs Rectangle
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.95 <= aspect_ratio <= 1.05:
            shape_name = "Square"
        else:
            shape_name = "Rectangle"
    elif vertices == 5:
        shape_name = "Pentagon"
    else:
        # Check circularity for Circle vs Polygon
        # Circularity: 4 * pi * Area / (Perimeter^2)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        if circularity > 0.80:
            shape_name = "Circle"
            is_circ = True
        else:
            shape_name = "Polygon"
    
    # 3. Visual Feedback
    viz = img.copy()
    if is_circ:
        cv2.circle(viz, (int(cx_circ), int(cy_circ)), int(rad_px), (0, 255, 0), 2)
    else:
        cv2.drawContours(viz, [box], 0, (0, 255, 0), 2)
    
    _, buffer = cv2.imencode('.jpg', viz)
    
    # Use max/min of box dimensions for length/breadth to be orientation-agnostic
    length_px = max(w_box, h_box)
    breadth_px = min(w_box, h_box)
    
    return {
        "raw": img,
        "mask": mask,
        "length": round(length_px * PIXEL_TO_MM, 2),
        "breadth": round(breadth_px * PIXEL_TO_MM, 2),
        "radius": round(rad_px * PIXEL_TO_MM, 2),
        "is_circular": bool(is_circ),
        "shape_name": shape_name,
        "visual": base64.b64encode(buffer).decode('utf-8')
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    stage = request.form.get('stage')
    file = request.files['image']
    analysis = analyze_specimen(file)
    storage[stage] = analysis
    
    # Ensure physical images exist for PDF and comparison
    # Use the original image (cv2 reads in BGR)
    # analysis['raw'] is the image
    cv2.imwrite(f"{stage}_temp.jpg", analysis['raw'])
    
    res = {"current": {k: v for k, v in analysis.items() if k not in ['raw', 'mask']}}
    
    if stage == 'after' and storage['before']:
        # --- CENTROID ALIGNMENT ---
        # 1. Get Centroid of Before Mask
        M_before = cv2.moments(storage['before']['mask'])
        if M_before["m00"] != 0:
            cX_b = int(M_before["m10"] / M_before["m00"])
            cY_b = int(M_before["m01"] / M_before["m00"])
        else:
            cX_b, cY_b = 0, 0

        # 2. Get Centroid of After (Current) Mask
        M_after = cv2.moments(analysis['mask'])
        if M_after["m00"] != 0:
            cX_a = int(M_after["m10"] / M_after["m00"])
            cY_a = int(M_after["m01"] / M_after["m00"])
        else:
            cX_a, cY_a = 0, 0
            
        # 3. Calculate Shift
        dX = cX_a - cX_b
        dY = cY_a - cY_b
        
        # 4. Translate Before Mask to align with After Mask
        rows, cols = analysis['mask'].shape
        M_trans = np.float32([[1, 0, dX], [0, 1, dY]])
        aligned_before_mask = cv2.warpAffine(storage['before']['mask'], M_trans, (cols, rows))
        
        # --- DIFFERENTIAL HEATMAP ---
        # Use aligned mask for comparison
        expansion_mask = cv2.subtract(analysis['mask'], aligned_before_mask)
        squeeze_mask = cv2.subtract(aligned_before_mask, analysis['mask'])
        
        heatmap = analysis['raw'].copy() # Use the 'after' raw image as base
        # Red for expansion (After - Aligned Before)
        heatmap[expansion_mask > 0] = [0, 0, 255] 
        # Blue for squeeze (Aligned Before - After)
        heatmap[squeeze_mask > 0] = [255, 0, 0]
        
        # Blend with original image
        cv2.addWeighted(heatmap, 0.5, analysis['raw'], 0.5, 0, heatmap)
        
        _, h_buf = cv2.imencode('.jpg', heatmap)
        res["overlay"] = base64.b64encode(h_buf).decode('utf-8')
        cv2.imwrite("comparison_temp.jpg", heatmap)
        
        # Calculate Delta Difference
        metric = 'radius' if analysis['is_circular'] else 'length'
        diff = round(res['current'][metric] - storage['before'][metric], 2)
        res["difference"] = diff
        res["analysis"] = "EXPANDED" if diff > 0.1 else "SQUEEZED" if diff < -0.1 else "NO CHANGE"
        
    return jsonify(res)

@app.route('/report')
def report():
    if not storage['before'] or not storage['after']:
        return "Process both images first", 400
        
    class PDF(FPDF):
        def header(self):
            # Logo/Title
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'METALLURGICAL DEFORMATION ANALYSIS', 0, 1, 'C')
            self.set_font('Arial', '', 10)
            self.cell(0, 5, f'Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
            self.line(10, 25, 200, 25)
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    
    # --- Comparison Section ---
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '1. VISUAL COMPARISON', 0, 1)
    
    # Image Boxes
    start_y = pdf.get_y()
    
    # Before Image
    pdf.image("before_temp.jpg", x=15, y=start_y+5, w=80, h=60)
    pdf.rect(15, start_y+5, 80, 60)
    pdf.set_xy(15, start_y + 66)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(80, 5, "Baseline Specimen", 0, 0, 'C')
    
    # After Image
    pdf.image("after_temp.jpg", x=115, y=start_y+5, w=80, h=60)
    pdf.rect(115, start_y+5, 80, 60)
    pdf.set_xy(115, start_y + 66)
    pdf.cell(80, 5, "Forged Specimen", 0, 1, 'C')
    
    pdf.ln(10)
    
    # --- Metrics Section ---
    pdf.set_y(start_y + 80)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, '2. METRIC ANALYSIS', 0, 1)
    
    # Table Header
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(60, 8, "Parameter", 1, 0, 'C', 1)
    pdf.cell(65, 8, "Baseline Value", 1, 0, 'C', 1)
    pdf.cell(65, 8, "Forged Value", 1, 1, 'C', 1)
    
    # Data Rows
    b = storage['before']
    a = storage['after']
    
    pdf.set_font('Arial', '', 10)
    
    # Shape Row
    pdf.cell(60, 8, "Detected Shape", 1, 0, 'L')
    pdf.cell(65, 8, b['shape_name'], 1, 0, 'C')
    pdf.cell(65, 8, a['shape_name'], 1, 1, 'C')
    
    # Dimensions Row
    pdf.cell(60, 8, "Primary Dimension", 1, 0, 'L')
    if b['is_circular']:
        pdf.cell(65, 8, f"Radius: {b['radius']} mm", 1, 0, 'C')
        pdf.cell(65, 8, f"Radius: {a['radius']} mm", 1, 1, 'C')
    else:
        pdf.cell(65, 8, f"L: {b['length']} | B: {b['breadth']} mm", 1, 0, 'C')
        pdf.cell(65, 8, f"L: {a['length']} | B: {a['breadth']} mm", 1, 1, 'C')
        
    pdf.ln(5)
    
    # --- Conclusion Section ---
    metric = 'radius' if a['is_circular'] else 'length'
    diff = round(a[metric] - b[metric], 2)
    change_type = "EXPANDED" if diff > 0.1 else "SQUEEZED" if diff < -0.1 else "NO CHANGE"
    
    # Conclusion Box styling
    if change_type == 'EXPANDED':
        pdf.set_fill_color(255, 230, 230) # Red tint
        pdf.set_text_color(200, 0, 0)
        border_col = (200, 0, 0)
    elif change_type == 'SQUEEZED':
        pdf.set_fill_color(230, 230, 255) # Blue tint
        pdf.set_text_color(0, 0, 200)
        border_col = (0, 0, 200)
    else:
        pdf.set_fill_color(240, 240, 240)
        pdf.set_text_color(0, 0, 0)
        border_col = (100, 100, 100)

    pdf.rect(10, pdf.get_y(), 190, 25, 'DF')
    pdf.set_font('Arial', 'B', 14)
    pdf.set_xy(10, pdf.get_y() + 7)
    pdf.cell(190, 10, f"CONCLUSION: {change_type}", 0, 1, 'C')
    
    pdf.set_font('Arial', '', 11)
    pdf.cell(190, 5, f"Deformation Delta: {diff:+.2f} mm ({metric})", 0, 1, 'C')
    
    # Reset Colors
    pdf.set_text_color(0, 0, 0)
    
    # Overlay Image (If exists)
    if os.path.exists("comparison_temp.jpg"):
        pdf.ln(15)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, '3. HEATMAP VISUALIZATION', 0, 1)
        pdf.image("comparison_temp.jpg", x=55, y=pdf.get_y()+5, w=100)
        
    pdf.output("atdfa_report.pdf")
    return send_file("atdfa_report.pdf", as_attachment=True)

if __name__ == '__main__':
    # Run with SSL context to allow camera access over network (HTTPS required)
    # This uses a self-signed certificate, browser will show a warning
    app.run(host='0.0.0.0', port=5050, ssl_context='adhoc')