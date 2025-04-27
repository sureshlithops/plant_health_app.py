import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ========== Set Page Configuration First (Must be the First Streamlit Command) ========== #
st.set_page_config(page_title="🌱 Plant Disease Detector", layout="centered")

# Rest of your code remains the same...