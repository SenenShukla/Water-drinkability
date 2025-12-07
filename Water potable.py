import streamlit as st
import joblib
import numpy as np

# Set page configuration for a better look
st.set_page_config(
    page_title="Water Potability Predictor",
    page_icon="💧",
    layout="centered"
)

# 1. Load the trained model
@st.cache_resource
def load_model():
    try:
        # We use joblib because the error '\x0e' indicates it was saved with joblib
        model = joblib.load('best_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model once
model = load_model()

# 2. App Title and Description
st.title("Water Potability Predictor 💧")
st.markdown("### Predict if water is safe for human consumption")
st.write("Enter the water quality metrics below to get a prediction.")

# 3. User Input Fields
# Group inputs into three columns for a cleaner interface
col1, col2, col3 = st.columns(3)

with col1:
    ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1, help="Approximation of the pH metric (0-14).")
    hardness = st.number_input("Hardness (mg/L)", min_value=0.0, value=200.0, step=1.0)
    solids = st.number_input("Solids (ppm)", min_value=0.0, value=20000.0, step=10.0, help="Total dissolved solids.")

with col2:
    chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, value=7.0, step=0.1)
    sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, value=300.0, step=1.0)
    conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, value=400.0, step=1.0)

with col3:
    organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, value=15.0, step=0.1)
    trihalomethanes = st.number_input("Trihalomethanes (μg/L)", min_value=0.0, value=60.0, step=1.0)
    turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=4.0, step=0.1)

# 4. Prediction Logic
if st.button("Predict Potability", type="primary"):
    if model is not None:
        # Prepare the input array in the correct order
        # The model expects a 2D array: [[feature1, feature2, ...]]
        input_data = np.array([[
            ph, hardness, solids, chloramines, sulfate, 
            conductivity, organic_carbon, trihalomethanes, turbidity
        ]])
        
        try:
            # Make prediction
            prediction = model.predict(input_data)
            result = prediction[0]  # Get the single result (0 or 1)
            
            st.divider()
            
            # Display Result
            if result == 1:
                st.success("💧 **Prediction: Potable (1)**")
                st.write("This water sample is predicted to be **safe** for consumption.")
            else:
                st.error("💀 **Prediction: Not Potable (0)**")
                st.write("This water sample is predicted to be **unsafe** for consumption.")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Model is not loaded. Please check that 'best_model.pkl' is in the directory.")