from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
from core.model import get_model
import os
import io

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pth")

app = FastAPI()

# cors middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class_names = ["tomato_early_blight", "tomato_healthy", "tomato_late_blight"]

# Load model once
model = get_model(num_classes=len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.to(DEVICE)
model.eval()

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)


@app.get("/")
def root():
    return {"status": "Agri-Attention API is running!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)[0]

        # get top 2 predictions
        top_probs, top_indices = torch.topk(probs, 2)

        results = []
        for i in range(2):
            results.append(
                {
                    "label": class_names[top_indices[i].item()],
                    "confidence": float(top_probs[i].item()),
                }
            )

    return {"predictions": results}
