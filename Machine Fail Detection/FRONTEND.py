import streamlit as st
import joblib
import numpy as np

model = joblib.load('logistic_model.pkl')

st.title("Machine Failure Prediction")

st.header("Enter Sensor Data")
footfall = st.number_input("Footfall", min_value=0.0, step=1.0)
temp_mode = st.number_input("Temperature Mode", min_value=0.0, step=1.0)
aq = st.number_input("Air Quality Index (AQ)", min_value=0.0, step=1.0)
uss = st.number_input("Ultrasonic Sensor (USS)", min_value=0.0, step=1.0)
cs = st.number_input("Current Sensor (CS)", min_value=0.0, step=1.0)
voc = st.number_input("VOC Level", min_value=0.0, step=1.0)
rp = st.number_input("Rotational Position (RP)", min_value=0.0, step=1.0)
ip = st.number_input("Input Pressure (IP)", min_value=0.0, step=1.0)
temperature = st.number_input("Temperature", min_value=0.0, step=1.0)


if st.button("Predict"):
    
    input_data = np.array([
        footfall, temp_mode, aq, uss, cs, voc, rp, ip, temperature
    ]).reshape(1, -1)
    
  
    prediction = model.predict(input_data)[0]
    result = "Failure Predicted" if prediction == 1 else "No Failure"
    
   
    st.subheader("Prediction Result:")
    st.write(result)
