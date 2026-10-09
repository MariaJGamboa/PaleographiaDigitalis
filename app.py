import io
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
from flask import Flask, render_template, request, jsonify
from werkzeug.middleware.dispatcher import DispatcherMiddleware

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=PROJECT_ROOT)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 12

CLASSES = [
    "Caroline", "Cursiva", "Half-Uncial", "Humanistic",
    "Humanistic Cursive", "Hybrida", "Praegothica", "Semihybrida",
    "Semitextualis", "Southern Textualis", "Textualis", "Uncial"
]

# Standard transforms for 384x384 input
transform = T.Compose([
    T.Resize((384, 384)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_models():
    """Instancia arquitecturas con torchvision nativo y carga pesos .pth."""
    print("Cargando modelos en memoria...")

    # 1. ConvNeXt-Small
    model_conv = models.convnext_small(weights=None)
    in_features_conv = model_conv.classifier[2].in_features
    
    # Reemplazamos la capa clasificadora adaptándonos a la estructura secuencial de Kaggle
    model_conv.classifier[2] = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(in_features_conv, NUM_CLASSES)
    )

    path_conv = os.path.join(PROJECT_ROOT, "models", "convnext_small_384_clamm.pth")
    state_dict_conv = torch.load(path_conv, map_location=DEVICE)
    if isinstance(state_dict_conv, dict) and "state_dict" in state_dict_conv:
        state_dict_conv = state_dict_conv["state_dict"]
    
    # Usamos strict=False para permitir variaciones menores en los nombres de las capas
    model_conv.load_state_dict(state_dict_conv, strict=False)
    model_conv.to(DEVICE).eval()

    # 2. EfficientNet-V2-S
    model_eff = models.efficientnet_v2_s(weights=None)
    in_features_eff = model_eff.classifier[1].in_features
    model_eff.classifier[1] = nn.Linear(in_features_eff, NUM_CLASSES)

    path_eff = os.path.join(PROJECT_ROOT, "models", "efficientnet_v2_silu_384_clamm.pth")
    state_dict_eff = torch.load(path_eff, map_location=DEVICE)
    if isinstance(state_dict_eff, dict) and "state_dict" in state_dict_eff:
        state_dict_eff = state_dict_eff["state_dict"]
    
    model_eff.load_state_dict(state_dict_eff, strict=False)
    model_eff.to(DEVICE).eval()

    print("Modelos cargados exitosamente.")
    return model_conv, model_eff

model_convnext, model_effnet = load_models()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out_convnext = model_convnext(img_tensor)
            out_effnet = model_effnet(img_tensor)

            prob_convnext = F.softmax(out_convnext, dim=1)
            prob_effnet = F.softmax(out_effnet, dim=1)

            # 60% / 40% weighted ensemble
            ensemble_prob = (0.60 * prob_convnext + 0.40 * prob_effnet).squeeze().cpu().numpy()

        predictions = [
            {"class": CLASSES[i], "probability": float(ensemble_prob[i])}
            for i in range(len(CLASSES))
        ]
        predictions.sort(key=lambda x: x["probability"], reverse=True)

        return jsonify({"success": True, "predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# In production, mount the Flask app under a subpath for reverse proxying via nginx.
# This ensures all routes are served under /understory instead of /.
production = DispatcherMiddleware(None, {
    '/PaleographiaDigitalis': app
})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
