#Check README.md for description of the new features added.
#Imports
import json
import requests
import streamlit as st
from pathlib import Path
from streamlit.logger import get_logger
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# If you start the fast api server on a different port
# make sure to change the port below
FASTAPI_BACKEND_ENDPOINT = "http://localhost:8000"

# Make sure you have iris_model.pkl file in FastAPI_Labs/src folder.
# If it's missing run train.py in FastAPI_Labs/src folder
# If your FastAPI_Labs folder name is different, update accordingly in the following path
FASTAPI_IRIS_MODEL_LOCATION = Path(__file__).resolve().parents[2] / 'FastAPI_Labs' / 'model' / 'iris_model.pkl'

# streamlit logger
LOGGER = get_logger(__name__)

def run():
    # Set the main dashboard page browser tab title and icon
    st.set_page_config(
        page_title="Iris Flower Prediction Demo",
        page_icon="🪻",
    )

    # Build the sidebar first
    with st.sidebar:
        # Check the status of backend
        try:
            backend_request = requests.get(FASTAPI_BACKEND_ENDPOINT)
            if backend_request.status_code == 200:
                st.success("Backend online ✅")
            else:
                st.warning("Problem connecting 😭")
        except requests.ConnectionError as ce:
            LOGGER.error(ce)
            st.error("Backend offline 😱")

        st.info("Configure parameters")
        
        # Sliders for manual input
        sepal_length = st.slider("Sepal Length",4.3, 7.9, 5.1, 0.1)
        sepal_width = st.slider("Sepal Width",2.0, 4.4, 3.5, 0.1)
        petal_length = st.slider("Petal Length",1.0, 6.9, 1.4, 0.1)
        petal_width = st.slider("Petal Width",0.1, 2.5, 0.2, 0.1)
        
        # Take JSON file as input
        test_input_file = st.file_uploader('Upload test prediction file',type=['json'])

        if test_input_file:
            st.write('Preview file')
            test_input_data = json.load(test_input_file)
            st.json(test_input_data)
            st.session_state["IS_JSON_FILE_AVAILABLE"] = True
        else:
            st.session_state["IS_JSON_FILE_AVAILABLE"] = False
            
        # Predict buttons
        predict_button_file = st.button('Predict from JSON File')
        predict_button_sliders = st.button('Predict from Sliders')

    # Dashboard body
    st.write("# Iris Flower Prediction! 🪻")

    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Tabs for Prediction / History / Info
    tab1, tab2, tab3 = st.tabs(["📊 Predict", "📜 History", "ℹ️ Info"])

    #Tab1:Prediction
    with tab1:
        st.subheader("Make a Prediction")

        def handle_prediction(client_input, input_type="sliders"):
            try:
                result_container = st.empty()
                with st.spinner('Predicting...'):
                    predict_iris_response = requests.post(f'{FASTAPI_BACKEND_ENDPOINT}/predict', data=client_input)
                if predict_iris_response.status_code == 200:
                    iris_content = json.loads(predict_iris_response.content)
                    
                    target_names = ["setosa", "versicolor", "virginica"]
                    prediction_value = iris_content["response"]
                    if prediction_value in [0, 1, 2, "0", "1", "2"]:
                        predicted = target_names[int(prediction_value)]
                    else:
                        predicted = prediction_value
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("### Input Provided")
                        st.json(json.loads(client_input))
                    with col2:
                        st.write("### Prediction")
                        result_container.success(f"The flower predicted is: {predicted}")

                    probs = [1.0 if name == predicted else 0.0 for name in target_names]
                    probs_df = pd.DataFrame([probs], columns=target_names)
                    st.subheader("Prediction Probabilities")
                    st.bar_chart(probs_df.T)

                    st.session_state["history"].append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "prediction": predicted,
                        "input_type": input_type
                    })

                else:
                    st.toast(f':red[Status {predict_iris_response.status_code}]', icon="🔴")
            except Exception as e:
                st.toast(':red[Problem with backend]', icon="🔴")
                LOGGER.error(e)

        if predict_button_file and st.session_state["IS_JSON_FILE_AVAILABLE"]:
            client_input = json.dumps(test_input_data['input_test'])
            handle_prediction(client_input, input_type="file")

        if predict_button_sliders:
            client_input = json.dumps({
                "petal_length": petal_length,
                "sepal_length": sepal_length,
                "petal_width": petal_width,
                "sepal_width": sepal_width
            })
            handle_prediction(client_input, input_type="sliders")

    #Tab2:History 
    with tab2:
        st.subheader("Prediction History")
        if st.session_state["history"]:
            history_df = pd.DataFrame(st.session_state["history"])

            # Clear history button
            if st.button("Clear History"):
                st.session_state["history"] = []
                st.success("Prediction history cleared ✅")
                st.rerun()  

            st.table(history_df)
            st.download_button("Download History as CSV", history_df.to_csv(index=False), "history.csv")
        else:
            st.info("No predictions made yet.")

    #Tab3: Info 
    with tab3:
        st.subheader("Model & Usage Info")
        st.write("This app uses a FastAPI backend to serve predictions on the Iris dataset.")
        st.write("You can either upload a JSON file with input values or use sliders in the sidebar.")
        example_input = {
            "input_test": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }
        st.download_button("Download Example JSON", json.dumps(example_input), "example.json")

        #Scatter plot of slider inputs
        fig, ax = plt.subplots()
        ax.scatter([sepal_length], [petal_length], color="red", s=100, label="Your Input")
        ax.set_xlabel("Sepal Length")
        ax.set_ylabel("Petal Length")
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    run()
