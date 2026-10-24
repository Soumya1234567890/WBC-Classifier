import os
import io
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.services.model_service import WBCModelService, CLASSES, CLASS_INFO

app = FastAPI(
    title="White Blood Cell Differential Classifier",
    description="Deep Learning WBC differential classification with Grad-CAM visual explainability",
    version="1.0.0"
)

# Enable CORS for local testing & sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
SAMPLES_DIR = os.path.join(STATIC_DIR, "samples")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Also mount root paths so relative and absolute links (/css, /js, /samples, /model) work seamlessly
css_dir = os.path.join(STATIC_DIR, "css")
if os.path.exists(css_dir):
    app.mount("/css", StaticFiles(directory=css_dir), name="css")

js_dir = os.path.join(STATIC_DIR, "js")
if os.path.exists(js_dir):
    app.mount("/js", StaticFiles(directory=js_dir), name="js")

samples_dir = os.path.join(STATIC_DIR, "samples")
if os.path.exists(samples_dir):
    app.mount("/samples", StaticFiles(directory=samples_dir), name="samples")

model_dir = os.path.join(STATIC_DIR, "model")
if os.path.exists(model_dir):
    app.mount("/model", StaticFiles(directory=model_dir), name="model")


# In-memory differential counter tally
differential_tally = {cls: 0 for cls in CLASSES}

# Reference ranges for adult peripheral blood
NORMAL_RANGES = {
    'Neutrophil': {'min': 40.0, 'max': 70.0},
    'Lymphocyte': {'min': 20.0, 'max': 40.0},
    'Monocyte': {'min': 2.0, 'max': 8.0},
    'Eosinophil': {'min': 1.0, 'max': 4.0},
    'Basophil': {'min': 0.5, 'max': 1.0},
    'Metamyelocyte': {'min': 0.0, 'max': 0.0},
    'Myelocyte': {'min': 0.0, 'max': 0.0}
}

@app.on_event("startup")
def startup_event():
    # Warm up model on server start
    WBCModelService.get_instance()

@app.get("/", response_class=HTMLResponse)
async def read_root():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>White Blood Cell Differential Classifier</h1>")

@app.get("/api/model-info")
async def get_model_info():
    svc = WBCModelService.get_instance()
    return svc.get_model_info()

@app.get("/api/gallery")
async def get_gallery():
    gallery = {}
    if os.path.exists(SAMPLES_DIR):
        for cat in os.listdir(SAMPLES_DIR):
            cat_path = os.path.join(SAMPLES_DIR, cat)
            if os.path.isdir(cat_path):
                files = [
                    f"/static/samples/{cat}/{fn}"
                    for fn in os.listdir(cat_path)
                    if fn.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]
                gallery[cat] = {
                    'class_name': cat,
                    'count': len(files),
                    'files': files,
                    'info': CLASS_INFO.get(cat, {})
                }
    return gallery

from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Request, Form

@app.post("/api/classify")
async def classify_cell(request: Request):
    svc = WBCModelService.get_instance()
    img_pil = None
    colormap = "JET"

    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        body = await request.json()
        sample_path = body.get("sample_path")
        colormap = body.get("colormap", "JET")
        if not sample_path:
            raise HTTPException(status_code=400, detail="Missing sample_path in JSON")
        clean_path = sample_path.replace("/static/", "").replace("\\", "/").lstrip("/")
        full_path = os.path.join(STATIC_DIR, clean_path)
        if not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail=f"Sample image not found at {full_path}")
        try:
            img_pil = Image.open(full_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading sample image: {str(e)}")
    else:
        # Multipart / form-data
        form = await request.form()
        file = form.get("file")
        sample_path = form.get("sample_path")
        colormap = form.get("colormap", "JET")

        if file and hasattr(file, "read"):
            contents = await file.read()
            if contents:
                img_pil = Image.open(io.BytesIO(contents))
        
        if img_pil is None and sample_path:
            clean_path = str(sample_path).replace("/static/", "").replace("\\", "/").lstrip("/")
            full_path = os.path.join(STATIC_DIR, clean_path)
            if not os.path.exists(full_path):
                raise HTTPException(status_code=404, detail=f"Sample image not found at {full_path}")
            img_pil = Image.open(full_path)

    if img_pil is None:
        raise HTTPException(status_code=400, detail="Either image file or sample_path must be provided")

    result = svc.analyze_image(img_pil, colormap=colormap)
    return result


@app.post("/api/differential/add")
async def add_to_differential(class_name: str = Body(..., embed=True)):
    if class_name not in differential_tally:
        raise HTTPException(status_code=400, detail=f"Unknown WBC class: {class_name}")
    differential_tally[class_name] += 1
    total = sum(differential_tally.values())
    return {
        'added': class_name,
        'current_count': differential_tally[class_name],
        'total_cells': total,
        'summary': get_differential_summary_data()
    }

@app.post("/api/differential/reset")
async def reset_differential():
    global differential_tally
    differential_tally = {cls: 0 for cls in CLASSES}
    return {'message': 'Differential counter reset successfully', 'summary': get_differential_summary_data()}

def get_differential_summary_data():
    total = sum(differential_tally.values())
    rows = []
    clinical_flags = []
    has_left_shift = False

    for cls in CLASSES:
        count = differential_tally[cls]
        pct = round((count / total * 100), 1) if total > 0 else 0.0
        ref = NORMAL_RANGES[cls]
        status = 'Normal'
        if total > 0:
            if pct > ref['max']:
                status = 'Elevated ⚠️'
                if cls in ['Metamyelocyte', 'Myelocyte']:
                    has_left_shift = True
                    clinical_flags.append(f"Left Shift detected: Immature {cls}s circulating ({pct}%).")
                elif cls == 'Neutrophil':
                    clinical_flags.append(f"Neutrophilia ({pct}%): typical of acute bacterial infection.")
                elif cls == 'Eosinophil':
                    clinical_flags.append(f"Eosinophilia ({pct}%): suggestive of allergic or parasitic response.")
                elif cls == 'Lymphocyte':
                    clinical_flags.append(f"Lymphocytosis ({pct}%): suggestive of viral infection or lymphocytic etiology.")
                elif cls == 'Basophil':
                    clinical_flags.append(f"Basophilia ({pct}%): alert for myeloproliferative disorder.")
            elif pct < ref['min']:
                status = 'Decreased 🔻'

        rows.append({
            'class_name': cls,
            'count': count,
            'percentage': pct,
            'reference_range': f"{ref['min']} - {ref['max']}%",
            'status': status
        })

    return {
        'total_counted': total,
        'target_count': 100,
        'progress_percent': min(100, round(total / 100 * 100, 1)),
        'breakdown': rows,
        'has_left_shift': has_left_shift,
        'clinical_flags': clinical_flags
    }

@app.get("/api/differential/summary")
async def get_differential_summary():
    return get_differential_summary_data()
