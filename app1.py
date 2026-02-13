import cv2
import numpy as np
import torch
import base64
import os
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from segment_anything import sam_model_registry, SamPredictor

app = Flask(__name__)
CORS(app)

# --- Configuration ---
# NOTE: Update this path to your actual SAM checkpoint location
CHECKPOINT = r"sam_vit_h_4b8939.pth" 
MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIXEL_TO_MM = 0.5 

# --- Model Load ---
print(f"[ATDFA] Initializing Core on {DEVICE}...")
try:
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    print("[ATDFA] Model Loaded Successfully.")
except Exception as e:
    print(f"[ATDFA] ERROR loading model: {e}")
    print("[ATDFA] Please ensure the checkpoint path is correct.")
    exit()

# In-memory storage for the session
storage = {"before": None, "after": None}

def analyze_specimen(image_bytes):
    """Processes image to detect shape, dimensions, and generate mask."""
    nparr = np.frombuffer(image_bytes.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, "Invalid Image"

    # SAM Segmentation
    predictor.set_image(img)
    h, w = img.shape[:2]
    masks, _, _ = predictor.predict(box=np.array([0, 0, w, h]), multimask_output=False)
    mask = masks[0].astype(np.uint8) * 255
    
    # Contour Analysis
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: 
        return None, "No Object Detected"
        
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    # Geometry
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    (cx_r, cy_r), (w_px, h_px), angle = rect
    (cx_c, cy_c), radius_px = cv2.minEnclosingCircle(cnt)

    # --- SHAPE DETECTION LOGIC (FIXED) ---
    # Circularity: 1.0 is perfect circle. Rectangles are usually < 0.8
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    
    # Aspect Ratio: Helps filter out oval shapes being called circles
    aspect_ratio = max(w_px, h_px) / min(w_px, h_px) if min(w_px, h_px) > 0 else 1
    
    # Thresholds: 
    # Circularity > 0.8 is usually a circle. 
    # We add aspect_ratio < 1.2 to ensure it's not a very rounded rectangle.
    is_circular = (circularity > 0.78) and (aspect_ratio < 1.25)
    shape_name = "CIRCULAR" if is_circular else "RECTANGULAR"

    # Visualization Generation
    viz_img = img.copy()
    if is_circular:
        cv2.circle(viz_img, (int(cx_c), int(cy_c)), int(radius_px), (0, 255, 0), 3)
        cv2.circle(viz_img, (int(cx_c), int(cy_c)), 5, (0, 0, 255), -1)
    else:
        cv2.drawContours(viz_img, [box], 0, (255, 0, 255), 3)
        # Draw center crosshair
        cv2.line(viz_img, (int(cx_r)-10, int(cy_r)), (int(cx_r)+10, int(cy_r)), (0,255,0), 2)
        cv2.line(viz_img, (int(cx_r), int(cy_r)-10), (int(cx_r), int(cy_r)+10), (0,255,0), 2)

    _, buffer = cv2.imencode('.jpg', viz_img)
    img_b64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "raw": img,
        "mask": mask,
        "length": round(max(w_px, h_px) * PIXEL_TO_MM, 2),
        "breadth": round(min(w_px, h_px) * PIXEL_TO_MM, 2),
        "radius": round(radius_px * PIXEL_TO_MM, 2),
        "is_circular": bool(is_circular),
        "shape_name": shape_name,
        "visual": img_b64,
        "metrics": {
            "circularity": round(circularity, 3),
            "aspect_ratio": round(aspect_ratio, 2)
        }
    }, None

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    stage = request.form.get('stage')
    file = request.files.get('image')
    
    if not file:
        return jsonify({"error": "No file provided"}), 400

    analysis, error = analyze_specimen(file)
    if error:
        return jsonify({"error": error}), 400
        
    storage[stage] = analysis
    
    # Prepare response (exclude raw numpy arrays)
    res = {"current": {k: v for k, v in analysis.items() if k not in ['raw', 'mask']}}
    
    # --- DIFFERENTIAL ANALYSIS (FIXED LOGIC) ---
    if stage == 'after' and storage['before']:
        b_data = storage['before']
        a_data = analysis
        
        # Resize masks to match for comparison
        b_mask = b_data['mask']
        a_mask = cv2.resize(a_data['mask'], (b_mask.shape[1], b_mask.shape[0]))
        
        # Calculate Difference Masks
        # Squeezed: Was there before, gone now (Blue)
        squeezed_mask = cv2.subtract(b_mask, a_mask)
        # Expanded: Wasn't there before, is there now (Red)
        expanded_mask = cv2.subtract(a_mask, b_mask)
        
        # Generate Heatmap Overlay
        base_img = cv2.resize(a_data['raw'], (b_mask.shape[1], b_mask.shape[0]))
        overlay = base_img.copy()
        overlay[squeezed_mask > 30] = [255, 100, 0]  # Blue-ish
        overlay[expanded_mask > 30] = [0, 100, 255]  # Red-ish
        heatmap = cv2.addWeighted(base_img, 0.6, overlay, 0.4, 0)
        
        _, h_buf = cv2.imencode('.jpg', heatmap)
        res["heatmap"] = base64.b64encode(h_buf).decode('utf-8')
        
        # Metric Calculation
        # Use Radius for circles, Length for rectangles
        if a_data['is_circular']:
            before_val = b_data['radius']
            after_val = a_data['radius']
            metric_name = "Radius"
        else:
            before_val = b_data['length']
            after_val = a_data['length']
            metric_name = "Length"
            
        # FIXED LOGIC: 
        # Positive Diff = After > Before = EXPANDED
        # Negative Diff = After < Before = SQUEEZED
        diff = round(after_val - before_val, 2)
        
        if abs(diff) < 0.15:
            status = "NO CHANGE"
            status_color = "text-gray-400"
        elif diff > 0:
            status = "EXPANDED"
            status_color = "text-red-500" # Hot/Expansion
        else:
            status = "SQUEEZED"
            status_color = "text-blue-500" # Cold/Compression
            
        res["comparison"] = {
            "status": status,
            "status_color": status_color,
            "difference": diff,
            "metric": metric_name,
            "before_val": before_val,
            "after_val": after_val
        }
        
    return jsonify(res)

# --- HTML TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ATDFA | Thermal Deformation Analysis</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #0f172a; color: #e2e8f0; }
        .mono { font-family: 'JetBrains Mono', monospace; }
        .glass-panel {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(148, 163, 184, 0.1);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        }
        .scan-line {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: #3b82f6;
            box-shadow: 0 0 10px #3b82f6;
            animation: scan 2s linear infinite;
            opacity: 0.5;
            pointer-events: none;
        }
        @keyframes scan {
            0% { top: 0%; opacity: 0; }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { top: 100%; opacity: 0; }
        }
        .metric-card {
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: #3b82f6;
        }
        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #0f172a; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #475569; }
        
        .loader {
            border: 3px solid rgba(255,255,255,0.1);
            border-radius: 50%;
            border-top: 3px solid #3b82f6;
            width: 24px;
            height: 24px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body class="min-h-screen flex flex-col overflow-x-hidden">

    <!-- Header -->
    <header class="border-b border-slate-800 bg-slate-900/80 sticky top-0 z-50 backdrop-blur-md">
        <div class="max-w-7xl mx-auto px-6 py-4 flex justify-between items-center">
            <div class="flex items-center gap-3">
                <div class="w-8 h-8 bg-blue-600 rounded flex items-center justify-center font-bold text-white">A</div>
                <div>
                    <h1 class="text-xl font-bold tracking-tight text-white">ATDFA <span class="text-blue-500">CORE</span></h1>
                    <p class="text-[10px] text-slate-400 uppercase tracking-widest">Thermal Differential Forge Analysis</p>
                </div>
            </div>
            <div class="flex items-center gap-4 text-xs mono text-slate-500">
                <span class="flex items-center gap-2"><span class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span> SYSTEM ONLINE</span>
                <span>v2.4.0</span>
            </div>
        </div>
    </header>

    <main class="flex-grow p-6 max-w-7xl mx-auto w-full grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        <!-- Left Panel: Controls -->
        <div class="lg:col-span-4 space-y-6">
            
            <!-- Upload Module -->
            <div class="glass-panel rounded-2xl p-6 relative overflow-hidden">
                <div class="absolute top-0 right-0 p-4 opacity-10 text-6xl font-black text-white pointer-events-none">01</div>
                <h2 class="text-sm font-bold text-slate-400 uppercase mb-4 tracking-wider">Specimen Input</h2>
                
                <div class="relative group">
                    <input type="file" id="fileInput" accept="image/*" class="hidden" onchange="updateFileName(this)">
                    <label for="fileInput" class="flex flex-col items-center justify-center w-full h-32 border-2 border-dashed border-slate-700 rounded-xl cursor-pointer hover:border-blue-500 hover:bg-slate-800/50 transition-all group-hover:text-blue-400">
                        <div class="flex flex-col items-center justify-center pt-5 pb-6">
                            <svg class="w-8 h-8 mb-3 text-slate-400 group-hover:text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
                            <p class="mb-2 text-sm text-slate-400" id="fileLabel">Click to upload image</p>
                        </div>
                    </label>
                </div>

                <div class="mt-6 space-y-3">
                    <button onclick="analyze('before')" id="btn-before" class="w-full group relative flex items-center justify-between px-6 py-4 bg-slate-800 hover:bg-emerald-900/30 border border-slate-700 hover:border-emerald-500/50 rounded-xl transition-all duration-300">
                        <div class="flex items-center gap-3">
                            <div class="w-8 h-8 rounded-full bg-slate-700 group-hover:bg-emerald-600 flex items-center justify-center text-xs font-bold transition-colors">1</div>
                            <div class="text-left">
                                <div class="text-sm font-bold text-slate-200 group-hover:text-emerald-400">ANALYZE BASELINE</div>
                                <div class="text-[10px] text-slate-500">Pre-forging state</div>
                            </div>
                        </div>
                        <svg class="w-5 h-5 text-slate-600 group-hover:text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
                    </button>

                    <button onclick="analyze('after')" id="btn-after" class="w-full group relative flex items-center justify-between px-6 py-4 bg-slate-800 hover:bg-orange-900/30 border border-slate-700 hover:border-orange-500/50 rounded-xl transition-all duration-300">
                        <div class="flex items-center gap-3">
                            <div class="w-8 h-8 rounded-full bg-slate-700 group-hover:bg-orange-600 flex items-center justify-center text-xs font-bold transition-colors">2</div>
                            <div class="text-left">
                                <div class="text-sm font-bold text-slate-200 group-hover:text-orange-400">ANALYZE FORGED</div>
                                <div class="text-[10px] text-slate-500">Post-forging state</div>
                            </div>
                        </div>
                        <svg class="w-5 h-5 text-slate-600 group-hover:text-orange-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                    </button>
                </div>
            </div>

            <!-- Status Log -->
            <div class="glass-panel rounded-2xl p-6 h-64 flex flex-col">
                <h2 class="text-sm font-bold text-slate-400 uppercase mb-4 tracking-wider">System Log</h2>
                <div id="console-log" class="flex-grow overflow-y-auto mono text-xs space-y-2 text-slate-400 pr-2">
                    <div class="text-emerald-500">> System initialized...</div>
                    <div class="text-slate-500">> Waiting for input...</div>
                </div>
            </div>
        </div>

        <!-- Right Panel: Visualization -->
        <div class="lg:col-span-8 space-y-6">
            
            <!-- Metrics Grid -->
            <div id="metrics-panel" class="hidden grid grid-cols-2 md:grid-cols-4 gap-4">
                <!-- Dynamic Content Here -->
            </div>

            <!-- Main Viewport -->
            <div class="glass-panel rounded-2xl p-1 relative min-h-[500px] flex flex-col">
                <div class="absolute top-4 left-4 z-10 flex gap-2">
                    <span id="shape-badge" class="hidden px-3 py-1 rounded-full bg-blue-500/20 border border-blue-500/50 text-blue-400 text-xs font-bold uppercase tracking-wider backdrop-blur-sm">
                        Detecting...
                    </span>
                    <span id="status-badge" class="hidden px-3 py-1 rounded-full bg-slate-700/50 border border-slate-600 text-slate-300 text-xs font-bold uppercase tracking-wider backdrop-blur-sm">
                        Standby
                    </span>
                </div>

                <!-- Image Container -->
                <div class="relative flex-grow bg-slate-950/50 rounded-xl overflow-hidden flex items-center justify-center group">
                    <div id="scan-line" class="scan-line hidden"></div>
                    
                    <!-- Placeholder -->
                    <div id="placeholder" class="text-center p-10">
                        <div class="w-20 h-20 mx-auto mb-4 rounded-full bg-slate-800 flex items-center justify-center">
                            <svg class="w-10 h-10 text-slate-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                        </div>
                        <h3 class="text-slate-400 font-medium">No Specimen Loaded</h3>
                        <p class="text-slate-600 text-sm mt-1">Upload an image to begin analysis</p>
                    </div>

                    <!-- Result Image -->
                    <img id="result-image" class="hidden max-w-full max-h-[600px] object-contain rounded-lg shadow-2xl" src="" alt="Analysis Result">
                    
                    <!-- Loading Overlay -->
                    <div id="loader" class="hidden absolute inset-0 bg-slate-900/80 backdrop-blur-sm flex flex-col items-center justify-center z-20">
                        <div class="loader mb-4"></div>
                        <div class="mono text-sm text-blue-400 animate-pulse">PROCESSING SEGMENTATION...</div>
                    </div>
                </div>

                <!-- Comparison Bar (Only shows for After) -->
                <div id="comparison-bar" class="hidden border-t border-slate-800 bg-slate-900/80 p-4">
                    <div class="flex items-center justify-between">
                        <div class="flex items-center gap-4">
                            <div class="text-right">
                                <div class="text-[10px] text-slate-500 uppercase">Before</div>
                                <div class="text-lg font-bold mono text-slate-300" id="val-before">0.00</div>
                            </div>
                            <div class="h-8 w-px bg-slate-700"></div>
                            <div>
                                <div class="text-[10px] text-slate-500 uppercase">After</div>
                                <div class="text-lg font-bold mono text-white" id="val-after">0.00</div>
                            </div>
                        </div>
                        
                        <div class="text-center">
                            <div class="text-[10px] text-slate-500 uppercase mb-1">Deformation Analysis</div>
                            <div id="diff-display" class="text-2xl font-black mono tracking-tight">---</div>
                        </div>

                        <div class="flex items-center gap-2">
                            <div class="w-3 h-3 rounded-full bg-blue-500"></div>
                            <span class="text-xs text-slate-400">Squeeze</span>
                            <div class="w-3 h-3 rounded-full bg-red-500 ml-2"></div>
                            <span class="text-xs text-slate-400">Expand</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <script>
        // --- UI Logic ---
        function log(msg, type = 'info') {
            const consoleDiv = document.getElementById('console-log');
            const entry = document.createElement('div');
            const time = new Date().toLocaleTimeString('en-US', {hour12: false, hour: "numeric", minute: "numeric", second: "numeric"});
            
            entry.innerHTML = `<span class="opacity-50">[${time}]</span> ${msg}`;
            if(type === 'error') entry.className = 'text-red-400';
            else if(type === 'success') entry.className = 'text-emerald-400';
            else entry.className = 'text-slate-400';
            
            consoleDiv.appendChild(entry);
            consoleDiv.scrollTop = consoleDiv.scrollHeight;
        }

        function updateFileName(input) {
            if(input.files && input.files[0]) {
                document.getElementById('fileLabel').innerText = input.files[0].name;
                document.getElementById('fileLabel').classList.add('text-blue-400');
                log(`File selected: ${input.files[0].name}`);
            }
        }

        async function analyze(stage) {
            const fileInput = document.getElementById('fileInput');
            if(!fileInput.files[0]) {
                alert("Please select an image first");
                return;
            }

            // UI Loading State
            document.getElementById('loader').classList.remove('hidden');
            document.getElementById('scan-line').classList.remove('hidden');
            document.getElementById('placeholder').classList.add('hidden');
            document.getElementById('result-image').classList.add('hidden');
            
            const formData = new FormData();
            formData.append('image', fileInput.files[0]);
            formData.append('stage', stage);

            try {
                log(`Initiating ${stage.toUpperCase()} analysis sequence...`);
                const response = await fetch('/analyze', { method: 'POST', body: formData });
                const data = await response.json();

                if(data.error) throw new Error(data.error);

                // Render Image
                const img = document.getElementById('result-image');
                img.src = stage === 'after' && data.heatmap ? `data:image/jpeg;base64,${data.heatmap}` : `data:image/jpeg;base64,${data.current.visual}`;
                img.classList.remove('hidden');

                // Update Badges
                const shapeBadge = document.getElementById('shape-badge');
                shapeBadge.innerText = data.current.shape_name;
                shapeBadge.classList.remove('hidden');
                
                const statusBadge = document.getElementById('status-badge');
                statusBadge.innerText = stage.toUpperCase() + " STATE";
                statusBadge.className = `px-3 py-1 rounded-full border text-xs font-bold uppercase tracking-wider backdrop-blur-sm ${stage === 'before' ? 'bg-emerald-500/20 border-emerald-500/50 text-emerald-400' : 'bg-orange-500/20 border-orange-500/50 text-orange-400'}`;
                statusBadge.classList.remove('hidden');

                // Render Metrics
                renderMetrics(data.current);

                // Handle Comparison
                if(stage === 'after' && data.comparison) {
                    const comp = data.comparison;
                    document.getElementById('comparison-bar').classList.remove('hidden');
                    document.getElementById('val-before').innerText = comp.before_val + " mm";
                    document.getElementById('val-after').innerText = comp.after_val + " mm";
                    
                    const diffDisplay = document.getElementById('diff-display');
                    diffDisplay.innerText = comp.status;
                    diffDisplay.className = `text-2xl font-black mono tracking-tight ${comp.status_color}`;
                    
                    log(`Analysis complete: ${comp.status} (${comp.difference > 0 ? '+' : ''}${comp.difference}mm)`, 'success');
                } else {
                    document.getElementById('comparison-bar').classList.add('hidden');
                    log(`Baseline established. Shape: ${data.current.shape_name}, Circularity: ${data.current.metrics.circularity}`, 'success');
                }

            } catch (err) {
                console.error(err);
                log(`Error: ${err.message}`, 'error');
                document.getElementById('placeholder').classList.remove('hidden');
            } finally {
                document.getElementById('loader').classList.add('hidden');
                document.getElementById('scan-line').classList.add('hidden');
            }
        }

        function renderMetrics(data) {
            const container = document.getElementById('metrics-panel');
            container.classList.remove('hidden');
            container.innerHTML = '';

            const metrics = [];
            
            if(data.is_circular) {
                metrics.push({ label: 'Radius', val: data.radius, unit: 'mm', color: 'text-blue-400', icon: '○' });
                metrics.push({ label: 'Circularity', val: data.metrics.circularity, unit: '', color: 'text-slate-300', icon: '◎' });
            } else {
                metrics.push({ label: 'Length', val: data.length, unit: 'mm', color: 'text-purple-400', icon: '▭' });
                metrics.push({ label: 'Breadth', val: data.breadth, unit: 'mm', color: 'text-purple-400', icon: '▭' });
                metrics.push({ label: 'Aspect Ratio', val: data.metrics.aspect_ratio, unit: '', color: 'text-slate-300', icon: '◫' });
            }

            metrics.forEach(m => {
                const div = document.createElement('div');
                div.className = 'metric-card glass-panel p-4 rounded-xl border border-slate-700 bg-slate-800/40';
                div.innerHTML = `
                    <div class="flex justify-between items-start mb-2">
                        <span class="text-[10px] uppercase text-slate-500 font-bold tracking-wider">${m.label}</span>
                        <span class="text-slate-600 text-lg">${m.icon}</span>
                    </div>
                    <div class="flex items-baseline gap-1">
                        <span class="text-2xl font-bold mono ${m.color}">${m.val}</span>
                        <span class="text-xs text-slate-500">${m.unit}</span>
                    </div>
                `;
                container.appendChild(div);
            });
        }
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)