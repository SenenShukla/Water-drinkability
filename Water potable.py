import streamlit as st
import pickle
import numpy as np

# 1. Load the trained model
# We use @st.cache_resource to load the model only once and keep it in memory
@st.cache_resource
def load_model():
    try:
        # Ensure 'best_model.pkl' is in the same directory as this script
        with open('best_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# 2. App Title and Description
st.title("Water Potability Predictor 💧")
st.markdown("""
This app predicts if water is **Potable (1)** or **Not Potable (0)** based on its quality metrics.
Please enter the values for the following parameters:
""")

# 3. User Input Fields
# We organize inputs into columns for a cleaner layout
col1, col2, col3 = st.columns(3)

with col1:
    ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1, help="Acid-base balance (0-14)")
    hardness = st.number_input("Hardness", min_value=0.0, value=200.0, step=1.0)
    solids = st.number_input("Solids (TDS)", min_value=0.0, value=20000.0, step=10.0)

with col2:
    chloramines = st.number_input("Chloramines", min_value=0.0, value=7.0, step=0.1)
    sulfate = st.number_input("Sulfate", min_value=0.0, value=300.0, step=1.0)
    conductivity = st.number_input("Conductivity", min_value=0.0, value=400.0, step=1.0)

with col3:
    organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=15.0, step=0.1)
    trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=60.0, step=1.0)
    turbidity = st.number_input("Turbidity", min_value=0.0, value=4.0, step=0.1)

# 4. Prediction Logic
if st.button("Predict Potability"):
    if model is not None:
        # Arrange inputs in the exact order the model expects
        # Standard order: ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity
        input_data = np.array([[
            ph, hardness, solids, chloramines, sulfate, 
            conductivity, organic_carbon, trihalomethanes, turbidity
        ]])
        
        # Make the prediction
        prediction = model.predict(input_data)
        result = prediction[0] # This will be 0 or 1
        
        # 5. Output Display
        st.divider()
        st.subheader("Prediction Result:")
        
        if result == 1:
            st.success(f"Output: {result} (Potable)")
            st.write("The water is predicted to be **safe for consumption**.")
        else:
            st.error(f"Output: {result} (Not Potable)")
            st.write("The water is predicted to be **unsafe for consumption**.")
    else:
        st.warning("Model could not be loaded. Please check the file path.")