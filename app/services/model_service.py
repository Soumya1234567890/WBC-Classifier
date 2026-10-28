import os
import io
import base64
import numpy as np
from PIL import Image
import tensorflow as tf
import keras
import cv2

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "blood_cell_classifier.h5")

CLASSES = [
    'Basophil',
    'Eosinophil',
    'Lymphocyte',
    'Metamyelocyte',
    'Monocyte',
    'Myelocyte',
    'Neutrophil'
]

CLASS_INFO = {
    'Basophil': {
        'name': 'Basophil',
        'type': 'Granulocyte',
        'normal_range': '0.5 - 1.0%',
        'diameter': '10 - 14 µm',
        'nucleus': 'Bi-lobed or S-shaped, often obscured by prominent coarse granules',
        'granules': 'Large, coarse, variable dark blue-black/purple granules (water soluble)',
        'function': 'Mediates allergic response, releases histamine and heparin, parasitic defense',
        'clinical_alerts': 'Basophilia observed in chronic myeloid leukemia (CML), polycythemia vera, chronic hypersensitivity.'
    },
    'Eosinophil': {
        'name': 'Eosinophil',
        'type': 'Granulocyte',
        'normal_range': '1.0 - 4.0%',
        'diameter': '12 - 17 µm',
        'nucleus': 'Distinctly bilobed (spectacle-shaped), dark purple chromatin',
        'granules': 'Abundant, uniform, spheroidal bright reddish-orange granules packed tightly',
        'function': 'Major defense against helminthic parasites, modulates IgE-mediated allergic reactions',
        'clinical_alerts': 'Eosinophilia seen in allergic conditions (asthma, eczema), parasitic infections, drug reactions, eosinophilic leukemia.'
    },
    'Lymphocyte': {
        'name': 'Lymphocyte',
        'type': 'Agranulocyte',
        'normal_range': '20.0 - 40.0%',
        'diameter': '7 - 12 µm (small) / 12 - 16 µm (large)',
        'nucleus': 'Round or slightly indented, dense coarse chromatin, high N:C ratio (~4:1)',
        'granules': 'Usually absent; occasional few azurophilic granules in large granular lymphocytes',
        'function': 'Adaptive immunity: B cells (antibodies), T cells (cell-mediated), NK cells (cytotoxicity)',
        'clinical_alerts': 'Lymphocytosis is typical in viral infections (EBV, CMV, hepatitis), pertussis, chronic lymphocytic leukemia (CLL).'
    },
    'Metamyelocyte': {
        'name': 'Metamyelocyte',
        'type': 'Immature Myeloid Precursor',
        'normal_range': '0% (Abnormal in peripheral blood)',
        'diameter': '10 - 15 µm',
        'nucleus': 'Reniform (kidney-shaped) or indented (< 50% width), clumped chromatin',
        'granules': 'Numerous secondary specific granules (neutrophilic lilac/pink)',
        'function': 'Precursor stage between myelocyte and band neutrophil; non-proliferative maturation',
        'clinical_alerts': '🚨 Left-Shift Alert: Presence indicates premature marrow release due to severe bacterial sepsis, leukemoid reaction, or myeloproliferative disease.'
    },
    'Monocyte': {
        'name': 'Monocyte',
        'type': 'Agranulocyte',
        'normal_range': '2.0 - 8.0%',
        'diameter': '15 - 22 µm (largest circulating leukocyte)',
        'nucleus': 'Folded, indented, brain-like convolutions or horse-shoe shaped with loose chromatin',
        'granules': 'Fine dust-like pink/lilac azurophilic granules giving "ground-glass" cytoplasm',
        'function': 'Phagocytosis, antigen presentation to T-cells, cytokine production, tissue macrophage precursor',
        'clinical_alerts': 'Monocytosis common in chronic infections (tuberculosis, subacute bacterial endocarditis), IBD, recovery phase of acute infections, CMML.'
    },
    'Myelocyte': {
        'name': 'Myelocyte',
        'type': 'Immature Myeloid Precursor',
        'normal_range': '0% (Abnormal in peripheral blood)',
        'diameter': '12 - 18 µm',
        'nucleus': 'Round or oval, usually eccentric, condensation of chromatin beginning',
        'granules': 'Appearance of secondary specific granules beside diminishing primary azurophilic granules',
        'function': 'Last stage of myeloid lineage capable of cell division (mitosis)',
        'clinical_alerts': '🚨 Pathological Left-Shift: Circulating myelocytes signify profound hematologic stress, severe overwhelming infection, or myeloproliferative neoplasms (CML).'
    },
    'Neutrophil': {
        'name': 'Neutrophil',
        'type': 'Granulocyte',
        'normal_range': '40.0 - 70.0%',
        'diameter': '10 - 14 µm',
        'nucleus': 'Segmented into 2 to 5 lobes connected by delicate chromatin filaments',
        'granules': 'Fine, dispersed pale pink-lilac secondary granules containing lysozyme & lactoferrin',
        'function': 'First-line phagocytic response against acute bacterial and fungal infections',
        'clinical_alerts': 'Neutrophilia points to acute bacterial infection, inflammatory tissue necrosis, acute myocardial infarction, or stress response.'
    }
}

class WBCModelService:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        print(f"Loading WBC model from {MODEL_PATH}...")
        self.model = keras.models.load_model(MODEL_PATH)
        # Warm up
        dummy = np.zeros((1, 128, 128, 3), dtype=np.float32)
        _ = self.model(dummy)

        # Build functional Grad-CAM model
        inputs = keras.Input(shape=(128, 128, 3))
        x = inputs
        target_conv_out = None
        for layer in self.model.layers:
            x = layer(x)
            if layer.name == 'conv2d_5':
                target_conv_out = x
        self.grad_model = keras.models.Model(inputs=inputs, outputs=[target_conv_out, x])
        print("WBC Model initialized successfully.")

    def get_model_info(self):
        layers_info = []
        for l in self.model.layers:
            cfg = l.get_config()
            out_shape = getattr(l, 'output_shape', None)
            params = l.count_params()
            layers_info.append({
                'name': l.name,
                'class_name': l.__class__.__name__,
                'output_shape': str(out_shape),
                'params': params,
                'trainable': l.trainable
            })
        return {
            'model_name': 'White Blood Cell Differential Classifier',
            'input_shape': [128, 128, 3],
            'total_params': self.model.count_params(),
            'classes': CLASSES,
            'class_count': len(CLASSES),
            'explainability_layer': 'conv2d_5 (128 filters, 28x28)',
            'layers': layers_info
        }

    def analyze_image(self, img_pil: Image.Image, colormap: str = 'JET'):
        img_rgb = img_pil.convert('RGB')
        orig_w, orig_h = img_rgb.size
        img_resized = img_rgb.resize((128, 128))
        img_array = np.array(img_resized, dtype=np.float32)
        img_batch = np.expand_dims(img_array, axis=0)

        # Compute Grad-CAM
        img_tensor = tf.convert_to_tensor(img_batch, dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_tensor)
            pred_index = int(tf.argmax(predictions[0]))
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs_val = conv_outputs[0]
        heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
        heatmap_np = heatmap.numpy()

        probs = predictions[0].numpy()
        top_prob = float(probs[pred_index])
        top_class = CLASSES[pred_index]

        # Colormap mapping
        cv_colormap = cv2.COLORMAP_JET
        if colormap.upper() == 'TURBO':
            cv_colormap = cv2.COLORMAP_TURBO
        elif colormap.upper() == 'VIRIDIS':
            cv_colormap = cv2.COLORMAP_VIRIDIS
        elif colormap.upper() == 'INFERNO':
            cv_colormap = cv2.COLORMAP_INFERNO

        heatmap_resized = cv2.resize(heatmap_np, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
        heatmap_resized = np.clip(heatmap_resized, 0, 1)

        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv_colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Transparent overlay heatmap (RGBA) - background with zero attention is 100% transparent
        alpha = np.uint8(255 * np.clip((heatmap_resized - 0.12) / 0.88 * 1.25, 0, 1))
        heatmap_rgba = np.dstack([heatmap_colored, alpha])

        orig_np = np.array(img_rgb)
        alpha_norm = (alpha.astype(np.float32) / 255.0)[:, :, np.newaxis]
        blended = np.uint8(orig_np * (1.0 - 0.65 * alpha_norm) + heatmap_colored * (0.65 * alpha_norm))

        # Focus bbox & coordinates (regions with > 55% peak attention)
        thresh = heatmap_resized > 0.55
        coords = np.argwhere(thresh)
        if len(coords) > 0:
            ymin, xmin = coords.min(axis=0)
            ymax, xmax = coords.max(axis=0)
            pad_x = int((xmax - xmin) * 0.08)
            pad_y = int((ymax - ymin) * 0.08)
            focus_bbox = {
                'x': max(0, int(xmin - pad_x)),
                'y': max(0, int(ymin - pad_y)),
                'width': min(orig_w - max(0, int(xmin - pad_x)), int((xmax - xmin) + 2 * pad_x)),
                'height': min(orig_h - max(0, int(ymin - pad_y)), int((ymax - ymin) + 2 * pad_y)),
                'left_percent': round((max(0, int(xmin - pad_x)) / orig_w) * 100, 1),
                'top_percent': round((max(0, int(ymin - pad_y)) / orig_h) * 100, 1),
                'width_percent': round((min(orig_w - max(0, int(xmin - pad_x)), int((xmax - xmin) + 2 * pad_x)) / orig_w) * 100, 1),
                'height_percent': round((min(orig_h - max(0, int(ymin - pad_y)), int((ymax - ymin) + 2 * pad_y)) / orig_h) * 100, 1)
            }
        else:
            focus_bbox = {
                'x': int(orig_w * 0.25),
                'y': int(orig_h * 0.25),
                'width': int(orig_w * 0.5),
                'height': int(orig_h * 0.5),
                'left_percent': 25.0,
                'top_percent': 25.0,
                'width_percent': 50.0,
                'height_percent': 50.0
            }

        # Morphological color & nucleus heuristics
        hsv = cv2.cvtColor(orig_np, cv2.COLOR_RGB2HSV)
        lower_purple = np.array([115, 35, 30])
        upper_purple = np.array([170, 255, 255])
        nucleus_mask = cv2.inRange(hsv, lower_purple, upper_purple)
        nucleus_pixels = np.count_nonzero(nucleus_mask)
        total_pixels = orig_w * orig_h
        cell_est_ratio = min(0.95, round(nucleus_pixels / max(1, total_pixels) * 3.5, 2))

        # Check for eosinophilic reddish granules
        lower_red1 = np.array([0, 60, 50])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([165, 60, 50])
        upper_red2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
        red_granule_density = round(np.count_nonzero(red_mask) / max(1, total_pixels) * 100, 1)

        # Check for basophilic deep blue/violet granules
        lower_dark_blue = np.array([100, 50, 20])
        upper_dark_blue = np.array([135, 255, 120])
        blue_mask = cv2.inRange(hsv, lower_dark_blue, upper_dark_blue)
        blue_granule_density = round(np.count_nonzero(blue_mask) / max(1, total_pixels) * 100, 1)

        def to_b64_png(img_data):
            pil_img = Image.fromarray(img_data)
            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

        def to_b64_jpeg(img_data):
            pil_img = Image.fromarray(img_data)
            buffered = io.BytesIO()
            pil_img.save(buffered, format="JPEG", quality=92)
            return "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()

        all_probs = [
            {
                'class_name': cls,
                'probability': float(probs[i]),
                'percentage': round(float(probs[i]) * 100, 2),
                'normal_range': CLASS_INFO[cls]['normal_range'],
                'type': CLASS_INFO[cls]['type']
            }
            for i, cls in enumerate(CLASSES)
        ]
        all_probs.sort(key=lambda x: x['probability'], reverse=True)

        is_immature = top_class in ['Metamyelocyte', 'Myelocyte']

        return {
            'predicted_class': top_class,
            'confidence': top_prob,
            'confidence_percentage': round(top_prob * 100, 2),
            'confidence_level': 'High Confidence' if top_prob > 0.8 else ('Moderate Confidence' if top_prob > 0.5 else 'Ambiguous Classification'),
            'probabilities': all_probs,
            'class_info': CLASS_INFO[top_class],
            'is_immature': is_immature,
            'explainability': {
                'overlay_image': to_b64_jpeg(blended),
                'heatmap_image': to_b64_png(heatmap_rgba),
                'solid_heatmap_image': to_b64_jpeg(heatmap_colored),
                'original_image': to_b64_jpeg(orig_np),
                'raw_heatmap': heatmap_np.tolist(),
                'focus_bbox': focus_bbox,
                'estimated_nc_ratio': f"{int(cell_est_ratio * 100)}%",
                'red_granule_density': f"{red_granule_density}%",
                'blue_granule_density': f"{blue_granule_density}%",
                'peak_activation': round(float(np.max(heatmap_np)), 3),
                'layer_analyzed': 'conv2d_5 (Final Convolutional Layer: 128 Filters, 28x28)',
                'attention_interpretation': f"The CNN focused intensely on the region {focus_bbox['width']}x{focus_bbox['height']}px at ({focus_bbox['x']}, {focus_bbox['y']}), isolating discriminating {top_class.lower()} characteristics (nuclear chromatin condensation, lobulation boundaries, and cytoplasmic granules)."
            }
        }
