from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ecg_model.save_load_model import load_model
from ecg_model.model import transformation
from PIL import Image

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

@app.get("/predict")
def predict():
    assert app.state.model is not None
    img_path = "../raw_data/02002_hr.jpg"
    X_img = Image.open(img_path).convert('RGB')

    X_processed = transformation(X_img)

    y_pred = app.state.model(X_processed)

    print("\nPrediction done: ", y_pred, y_pred.shape, "\n")
    return y_pred


if __name__ == '__main__':
    predict()
