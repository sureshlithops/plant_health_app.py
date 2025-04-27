import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# ========== Set Page Configuration First (Must be the First Streamlit Command) ========== #
st.set_page_config(page_title="🌱 Plant Disease Detector", layout="centered")

# ========== Set Background and Theme ========== #
def set_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://images.unsplash.com/photo-1552031823-1949c5db4b6d?ixlib=rb-4.0.3&auto=format&fit=crop&w=1950&q=80');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }

        .block-container {
            background-color: rgba(255, 255, 255, 0.88);
            padding: 2rem;
            border-radius: 12px;
        }

        h1, h2, h3 {
            color: #2e7d32;
        }

        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
            border: none;
            transition: background-color 0.3s ease;
        }

        .stButton>button:hover {
            background-color: #388e3c;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

# Apply background style
set_background()

# ========== Load Your Trained Model ========== #
@st.cache_resource
def load_plant_model():
    # Replace with the path to your trained model
    model_path = 'path_to_your_model.h5'
    model = load_model(model_path)
    return model

model = load_plant_model()

# ========== Define Class Labels ========== #
# Replace with your actual class labels
CLASS_NAMES = ['Healthy', 'Disease1', 'Disease2', 'Disease3']  # Example labels

# ========== Image Preprocessing ========== #
def preprocess_image(img, target_size=(224, 224)):
    """Preprocess the image for model prediction"""
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize if your model expects this
    return img_array

# ========== UI Setup ========== #
st.title("🌿 Plant Disease Detection App")

st.markdown("""
Upload a plant leaf image, and we'll evaluate it for any potential diseases.  
The results will include possible diagnoses and suggested actions.  
""")

uploaded_file = st.file_uploader("📤 Upload a leaf image", type=["jpg", "jpeg", "png"])

# ========== Image Display and Processing ========== #
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)
    st.success("Image uploaded successfully!")

    if st.button("🔍 Evaluate for Disease"):
        with st.spinner("Analyzing image... Please wait."):
            try:
                # Preprocess the image
                processed_image = preprocess_image(image)
                
                # Make prediction
                predictions = model.predict(processed_image)
                predicted_class = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                
                # Get class name
                disease_name = CLASS_NAMES[predicted_class]
                
                st.success("✅ Disease Evaluation Complete!")
                
                # Display results
                with st.expander("📋 See Diagnosis Result"):
                    st.subheader(f"Diagnosis: {disease_name}")
                    st.write(f"Confidence: {confidence:.2%}")
                    
                    # Add disease-specific information (customize this based on your classes)
                    if disease_name == 'Healthy':
                        st.markdown("""
                        **Healthy Plant Leaf**
                        - No signs of disease detected
                        - Continue with current care routine
                        """)
                    else:
                        st.markdown(f"""
                        **Possible Disease: {disease_name}**
                        
                        **Description:**  
                        [Add description of this disease here]
                        
                        **Recommended Actions:**  
                        - [Add treatment recommendation 1]
                        - [Add treatment recommendation 2]
                        - [Add prevention tips]
                        """)
                        
                    # Show probabilities for all classes
                    st.subheader("Detailed Probabilities:")
                    for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, predictions[0])):
                        st.progress(float(prob), text=f"{class_name}: {prob:.2%}")
                        
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
else:
    st.info("👆 Please upload a plant leaf image to begin.")
