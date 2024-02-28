import os
from io import BytesIO
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import logging
logging.basicConfig(level=logging.DEBUG)
model_posco = tf.keras.models.load_model("models/poscogen/PCOSGen-train.h5")
posco_pred = ["Unhealthy", "Healthy"]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get('/')
def check_status():
    return ["The server is Running"]

@app.post('/predict_pcos')
async def predict_posco(image: UploadFile = File(...)):
    contents = await image.read()
    with open(image.filename, 'wb') as f:
        f.write(contents)
    contents = tf.io.read_file(image.filename)
    os.remove(image.filename)
    contents = tf.io.decode_image(contents)
    contents = tf.expand_dims(contents, axis = 0)
    contents = tf.image.resize(contents, (180, 180))
    prediction = tf.math.argmax(model_posco.predict(contents), 1)
    label = posco_pred[prediction[0]]
    logging.info(label)
    return {"prediction": label}