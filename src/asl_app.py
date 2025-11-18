import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

MODEL_PATH = "/Users/dianhaoli/asl_fingerspell/asl_model_v1.keras"   
IMG_SIZE = (64, 64)                

try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully")
    print("Model input shape:", model.input_shape)
    print("Model output shape:", model.output_shape)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class_labels = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
    'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]
def predict_asl(img):
    if model is None:
        return {"Error: model not loaded": 1.0}
    if img is None:
        return {"Error: no image": 1.0}

    try:
        expected_channels = model.input_shape[-1]
        if expected_channels == 1:
            img = img.convert("L")   
        else:
            img = img.convert("RGB") 
        img = img.resize(IMG_SIZE)
        arr = image.img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = arr.astype("float32")
        if arr.max() > 1.0:
            arr /= 255.0  
        preds = model.predict(arr, verbose=0)
        probs = preds[0]
        if probs.sum() < 0.99 or probs.sum() > 1.01:
            exp = np.exp(probs - np.max(probs))
            probs = exp / np.sum(exp)
        top_indices = np.argsort(probs)[-3:][::-1]
        results = {class_labels[i]: float(probs[i]) for i in top_indices}
        return results
    except Exception as e:
        print("Prediction error:", e)
        return {"Error": 1.0, "Message": str(e)}
demo = gr.Interface(
    fn=predict_asl,
    inputs=gr.Image(
        sources=["webcam", "upload"],
        type="pil",
        label="Take or upload an ASL hand sign"
    ),
    outputs=gr.Label(
        num_top_classes=3,
        label="Predicted ASL Letter"
    ),
    title="ASL Hand Sign Classifier",
    description="Take or upload a photo of an ASL hand sign to predict the letter.",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(
        share=True,         
        inbrowser=True,   
        debug=True,         
        show_error=True
    )
