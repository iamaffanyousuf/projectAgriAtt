import torch.nn.functional as F

import torch
from PIL import Image
from torchvision import transforms

from core.model import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load class names
class_names = ["tomato_early_blight", "tomato_healthy", "tomato_late_blight"]

# Load model
model = get_model(num_classes=len(class_names))
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.to(DEVICE)
model.eval()

# Transform
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ]
)

# CLI input
img_path = input("Enter image path: ")

image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)

# Predict
with torch.no_grad():
    outputs = model(image)
    probs = F.softmax(outputs, dim=1)[0]

# Top 2 predictions
top_probs, top_indices = torch.topk(probs, 2)

print("Predictions:")

for i in range(2):
    label = class_names[top_indices[i].item()]
    conf = top_probs[i].item() * 100

    if i == 0:
        print(f"Primary: {label} ({conf:.2f}%)")
    else:
        print(f"Secondary: {label} ({conf:.2f}%)")
