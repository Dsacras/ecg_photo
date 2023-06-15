from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch
from io import BytesIO
from ecg_model.grad import ecg_grad
from ecg_model.save_load_model import load_model_url


def transformation(image):
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image)

app = FastAPI()


# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/predict")
async def predict(file: UploadFile, model_url):
    class_names = ['Abnormal', 'Normal']

    file_request = await file.read()
    app.state.model = load_model_url(model_url)
    X_img = Image.open(BytesIO(file_request)).convert('RGB')
    img_processed = transformation(X_img)
    img_processed = img_processed.unsqueeze(0)

    # Set the model to evaluation mode
    app.state.model.eval()

    # Forward pass
    with torch.no_grad():
        output = app.state.model(img_processed)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_prob, predicted_label = torch.max(probabilities, 1)
        predicted_label = predicted_label.item()
        predicted_class = class_names[predicted_label]
        predicted_prob = predicted_prob.item()

    ecg_grad(app.state.model, img_processed, X_img)

    response = FileResponse("ecg_model/api/grad_cam.jpg")
    response.headers["prediction"] = predicted_class
    response.headers["confidence"] = str(predicted_prob)
    return response
