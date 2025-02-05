from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from keras.utils import load_img, img_to_array
import numpy as np
import tensorflow as tf
from io import BytesIO


app = FastAPI()

model = tf.keras.models.load_model('./model-Model3.h5')
print("Model loaded")

# Labels
labels = ["Dermatitis Atopik", "Dermatitis Kontak Alergi", "Dermatitis Perioral", "Dermatitis Seboroik", "Neurodermatitis", "Normal"]


origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(file: UploadFile):
    img_content = await file.read()

    image_bytes = BytesIO(img_content)

    image = load_img(image_bytes, target_size=(224, 224))
    
    # Preprocess the image
    image_array = img_to_array(image)  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Add a batch dimension
    

    # Predict
    predicted = model.predict(image_array)

     # Select the label with highest probability
    index = np.argmax(predicted)
    prediction_label = labels[index]
    

    # Return the prediction value and corresponding label
    return {"result": prediction_label, "percentage": float(predicted[0][index])}

