import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
import cv2

# -- 1) Preprocessing function ------------------------------------------------
def preprocess_img(img):
    """Process canvas image to match MNIST format"""
    # Convert RGBA to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    
    # Threshold to clean up drawing
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours and crop to digit
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        cropped = thresh[y:y+h, x:x+w]
    else:
        cropped = thresh

    # Resize and pad to 28x28
    ratio = 20 / max(cropped.shape)
    resized = cv2.resize(cropped, 
                        (int(cropped.shape[1]*ratio), 
                         int(cropped.shape[0]*ratio)),
                        interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((28, 28), dtype=np.uint8)
    dx = (28 - resized.shape[1]) // 2
    dy = (28 - resized.shape[0]) // 2
    canvas[dy:dy+resized.shape[0], dx:dx+resized.shape[1]] = resized
    
    # Normalize and reshape for model
    return (canvas.astype(np.float32) / 255.0).reshape(1, 28, 28, 1)

# -- 2) Load model -----------------------------------------------------------
model = tf.keras.models.load_model("digit_recognition_model.keras",compile=False)  # Remove compile=False

# -- 3) Streamlit UI ---------------------------------------------------------
st.title("✏️ Draw a Digit")
st.write("Draw a digit below (0–9) and I'll try to guess it!")

canvas = st_canvas(
    fill_color="#000000",
    stroke_width=18,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# -- 4) Prediction logic -----------------------------------------------------
if canvas.image_data is not None:
    # Process image
    processed = preprocess_img(canvas.image_data)
    
    # Show processed image
    st.image(processed[0, :, :, 0], width=140, 
            caption="Processed Input")
    
    # Make prediction
    pred = model.predict(processed, verbose=0)[0]
    digit = np.argmax(pred)
    confidence = pred[digit]
    
    st.subheader(f"Prediction: {digit} ({(confidence*100):.1f}% confidence)")
    st.bar_chart(pred)