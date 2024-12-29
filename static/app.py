# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

# Load the dataset
@st.cache_data
def load_data():
    file_path = "Crop_recommendation.csv"  # Ensure this file exists
    data = pd.read_csv(file_path)
    return data

# Train the model
@st.cache_resource
def train_model(data):
    features = data[['temperature', 'humidity', 'rainfall']]
    target = data['label']
    model = RandomForestClassifier(random_state=42)
    model.fit(features, target)
    return model

# Fetch the crop details
@st.cache_data
def get_crop_details(crop_name, data):
    crop_info = data[data['label'] == crop_name].iloc[0]
    return crop_info

# Get crop image path
@st.cache_data
def get_crop_image(crop_name):
    crop_images = {
        "rice": "images/rice.jpg",
        "maize": "images/maize.jpg",
        "chickpea": "images/chickpea.jpg",
        "kidneybeans": "images/kidneybeans.jpg",
        "pigeonpeas": "images/pigeonpeas.jpg",
        "mothbeans": "images/mothbeans.jpg",
        "mungbean": "images/mungbean.jpg",
        "blackgram": "images/blackgram.jpg",
        "lentil": "images/lentil.jpg",
        "pomegranate": "images/pomegranate.jpg",
        "banana": "images/banana.jpg",
        "mango": "images/mango.jpg",
        "grapes": "images/grapes.jpg",
        "watermelon": "images/watermelon.jpg",
        "muskmelon": "images/muskmelon.jpg",
        "apple": "images/apple.jpg",
        "orange": "images/orange.jpg",
        "papaya": "images/papaya.jpg",
        "coconut": "images/coconut.jpg",
        "cotton": "images/cotton.jpg",
        "jute": "images/jute.jpg",
        "coffee": "images/coffee.jpg"
    }
    return crop_images.get(crop_name, "images/default.jpg")

# Load data and train the model
crop_data = load_data()
model = train_model(crop_data)

# Inject custom CSS for background image
def add_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add background image
add_background("https://tse1.mm.bing.net/th?id=OIP.CIExTFRR0rH_G5n1SNkDswHaE6&pid=Api")

# Streamlit app UI
st.title("Crop Recommendation System 🌾")
st.write("Enter the environmental conditions to get a crop recommendation.")

# Input fields for user
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=200.0)

# Predict and display result
if st.button("Recommend Crop"):
    input_data = np.array([[temperature, humidity, rainfall]])
    prediction = model.predict(input_data)
    crop_name = prediction[0]

    # Fetch crop details
    crop_details = get_crop_details(crop_name, crop_data)

    # Display all inputs
    st.subheader("Crop Details")
    st.write(f"**Nitrogen (N):** {crop_details['N']}")
    st.write(f"**Phosphorus (P):** {crop_details['P']}")
    st.write(f"**Potassium (K):** {crop_details['K']}")
    st.write(f"**Temperature (°C):** {crop_details['temperature']}")
    st.write(f"**Humidity (%):** {crop_details['humidity']}")
    st.write(f"**pH value:** {crop_details['ph']}")
    st.write(f"**Rainfall (mm):** {crop_details['rainfall']}")

    # Highlight the crop recommendation
    st.subheader("Recommended Crop 🌱")
    st.success(f"The recommended crop is: **{crop_name}**.")

    # Display crop image
    crop_image_path = get_crop_image(crop_name)
    if os.path.exists(crop_image_path):
        st.image(crop_image_path, caption=crop_name.capitalize(), use_column_width=True)
    else:
        st.warning("No image available for this crop.")
