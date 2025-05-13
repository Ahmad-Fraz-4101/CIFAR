from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import logging

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
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

try:
    # Load the model
    logger.info("Loading model...")
    model = CIFAR10CNN()
    model.load_state_dict(torch.load('cifar10_model.pth', map_location=torch.device('cpu')))
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # CIFAR-10 images are 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for CIFAR-10
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
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        logger.info("Predict endpoint called")
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
            "predicted_class": predicted_class,
            "class_name": classes[predicted_class]
        }
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        } 