import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
import os

# Set device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
class CropDiseaseModel(torch.nn.Module):
    def __init__(self):
        super(CropDiseaseModel, self).__init__()
        self.model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=38)  # Adjust num_classes

    def forward(self, x):
        return self.model(x)

# Load the model
model = CropDiseaseModel()
model_path = "C:/onedrive/Desktop/Crop/CropCare/models/efficientnet_b3_disease_model.pth"  # Explicit model path
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.to(device)
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Class labels (update based on your dataset)
class_labels = ["Healthy", "Bacterial Spot", "Leaf Mold", "Powdery Mildew", "Downy Mildew", "Rust", "Scab", "Black Rot"]

def predict_disease(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()
    
    return class_labels[predicted_class]
