
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

SIGN_NAMES = ['Speed limit 20','Speed limit 30','Speed limit 50','Speed limit 60',
 'Speed limit 70','Speed limit 80','End speed limit 80','Speed limit 100',
 'Speed limit 120','No passing','No passing >3.5t','Right-of-way at intersection',
 'Priority road','Yield','Stop','No vehicles','Vehicles >3.5t prohibited',
 'No entry','General caution','Dangerous curve left','Dangerous curve right',
 'Double curve','Bumpy road','Slippery road','Road narrows right',
 'Road work','Traffic signals','Pedestrians','Children crossing','Bicycles crossing',
 'Beware ice/snow','Wild animals crossing','End speed + passing limits',
 'Turn right ahead','Turn left ahead','Ahead only','Go straight or right',
 'Go straight or left','Keep right','Keep left','Roundabout mandatory',
 'End of no passing','End no passing >3.5t']

IMG_SIZE = 32

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('traffic_sign_cnn.keras')

model = load_model()

st.set_page_config(page_title="Traffic Sign Recognition")
st.title("Traffic Sign Recognition")
st.write("Upload a traffic sign image and the model will predict its class.")

uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])

if uploaded is not None:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Uploaded image', use_container_width=True)
    arr = np.array(img.resize((IMG_SIZE, IMG_SIZE)), dtype='float32') / 255.0
    preds = model.predict(np.expand_dims(arr, 0))[0]
    cls = int(np.argmax(preds))
    conf = float(preds[cls])
    st.subheader("Prediction")
    st.success(f"{SIGN_NAMES[cls]} (ID {cls})")
    st.write(f"Confidence: {conf:.4f}")
