from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from google.cloud import storage
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch
from io import BytesIO
from ecg_model.grad import ecg_grad

def load_model():
    latest_model_path_to_save = "ecg_model/api/model_20230614-124525.pt"
    latest_model = torch.load(latest_model_path_to_save)
    return latest_model

def transformation(image):
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return transform(image)

app = FastAPI()
app.state.model = load_model()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/predict")
async def predict(file: UploadFile):
    class_names = ['Abnormal', 'Normal']

    file_request = await file.read()
    X_img = Image.open(BytesIO(file_request)).convert('RGB')
    img_processed = transformation(X_img)
    img_processed = img_processed.unsqueeze(0)

    # Set the model to evaluation mode
    app.state.model.eval()

    # Forward pass
    with torch.no_grad():
        output = app.state.model(img_processed)

    # Get the predicted class label
    _, predicted = torch.max(output, 1)
    predicted_label = predicted.item()
    predicted_class = class_names[predicted_label]
    print(predicted_class)
    grad_image = ecg_grad(app.state.model, img_processed, X_img)
    print(type(grad_image))
    response = FileResponse("03001_hr.jpg")
    response.headers["prediction"] = predicted_class
    return response
