from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the CIFAR-10 CNN model architecture
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x8x8
            nn.Flatten(),
            nn.Linear(64*8*8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

# Initialize FastAPI app
app = FastAPI(title="CIFAR-10 Classifier API",
             description="API for classifying images using a CIFAR-10 trained CNN model",
             version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define model path
MODEL_PATH = os.getenv('MODEL_PATH', 'cifar10_model.pth')

try:
    # Load the model
    logger.info(f"Loading model from {MODEL_PATH}...")
    model = CIFAR10CNN()
    
    # Check if model file exists
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR-10 class names
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

@app.get("/")
async def root():
    try:
        logger.info("Root endpoint called")
        return {
            "status": "success",
            "message": "CIFAR-10 Image Classification API is running",
            "endpoints": {
                "root": "/",
                "predict": "/predict",
                "docs": "/docs"
            }
        }
    except Exception as e:
        logger.error(f"Error in root endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        logger.info("Predict endpoint called")
        
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and process the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if image is not in RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transformations
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()
        
        logger.info(f"Prediction successful: class {predicted_class}")
        return {
            "status": "success",
            "predicted_class": predicted_class,
            "class_name": classes[predicted_class],
            "confidence": float(torch.softmax(outputs, dim=1)[0][predicted_class].item())
        }
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"} 