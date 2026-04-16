import torch.nn.functional as F

import torch
from PIL import Image
from torchvision import transforms

from model import get_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load class names
class_names = [
    "wheat_rust",
    "mung_leaf_spot"
]

# Load model
model = get_model(num_classes=len(class_names))
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# CLI input
img_path = input("Enter image path: ")

image = Image.open(img_path).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)

# Predict
with torch.no_grad():
    outputs = model(image)
    probs = F.softmax(outputs, dim=1)
    confidence, pred = torch.max(probs, 1)

print(f"Prediction: {class_names[pred.item()]}")
print(f"Confidence: {confidence.item()*100:.2f}%")