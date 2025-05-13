import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
MODEL_PATH = "best_model.keras"
model = load_model(MODEL_PATH)
class_names = ["Early Blight", "Late Blight", "Healthy"]

# Preprocess image
def preprocess(img):
    img = img.resize((256, 256))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])
    return class_names[class_index]

# Gradio Interface
interface = gr.Interface(
    fn=preprocess,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    title="Potato Leaf Disease Detection",
    description="Upload a potato leaf image to classify as Early Blight, Late Blight, or Healthy."
)

interface.launch()
