import logging
import gradio as gr
import pandas as pd
from autogluon.tabular import TabularPredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the AutoGluon model from the Hugging Face Hub
MODEL_ID = "aai540-group3/diabetes-readmission"
predictor = TabularPredictor.load(MODEL_ID)

# Define constants for the Gradio interface
AGE_RANGE = (0, 100)
TIME_IN_HOSPITAL_RANGE = (1, 14)
NUM_PROCEDURES_RANGE = (0, 10)
NUM_MEDICATIONS_RANGE = (0, 20)
NUMBER_DIAGNOSES_RANGE = (1, 10)
READMITTED_CHOICES = ["<30", ">30", "NO"]

# Define the inference function
def predict(
    age, time_in_hospital, num_procedures, num_medications, number_diagnoses,
    metformin, repaglinide, nateglinide, chlorpropamide, glimepiride,
    glipizide, glyburide, pioglitazone, rosiglitazone, acarbose, insulin,
    readmitted
):
    # Create a DataFrame from the input features
    input_data = pd.DataFrame([{
        "age": age,
        "time_in_hospital": time_in_hospital,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_diagnoses": number_diagnoses,
        "metformin": int(metformin),
        "repaglinide": int(repaginide),
        "nateglinide": int(nateglinide),
        "chlorpropamide": int(chlorpropamide),
        "glimepiride": int(glimepiride),
        "glipizide": int(glipizide),
        "glyburide": int(glyburide),
        "pioglitazone": int(pioglitazone),
        "rosiglitazone": int(rosiglitazone),
        "acarbose": int(acarbose),
        "insulin": int(insulin),
        "readmitted": readmitted
    }])

    try:
        # Make a prediction using the AutoGluon predictor
        prediction = predictor.predict(input_data)
        logger.info(f"Prediction received: {prediction}")

        return f"<h1 style='font-size: 48px; color: green;'>Prediction: {prediction.iloc[0]}</h1>"

    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return "<h1 style='font-size: 48px; color: red;'>Error in prediction</h1>"

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(minimum=AGE_RANGE[0], maximum=AGE_RANGE[1], label="Age"),
        gr.Slider(minimum=TIME_IN_HOSPITAL_RANGE[0], maximum=TIME_IN_HOSPITAL_RANGE[1], label="Time in Hospital (days)"),
        gr.Slider(minimum=NUM_PROCEDURES_RANGE[0], maximum=NUM_PROCEDURES_RANGE[1], label="Number of Procedures"),
        gr.Slider(minimum=NUM_MEDICATIONS_RANGE[0], maximum=NUM_MEDICATIONS_RANGE[1], label="Number of Medications"),
        gr.Slider(minimum=NUMBER_DIAGNOSES_RANGE[0], maximum=NUMBER_DIAGNOSES_RANGE[1], label="Number of Diagnoses"),
        gr.Checkbox(label="Metformin"),
        gr.Checkbox(label="Repaglinide"),
        gr.Checkbox(label="Nateglinide"),
        gr.Checkbox(label="Chlorpropamide"),
        gr.Checkbox(label="Glimepiride"),
        gr.Checkbox(label="Glipizide"),
        gr.Checkbox(label="Glyburide"),
        gr.Checkbox(label="Pioglitazone"),
        gr.Checkbox(label="Rosiglitazone"),
        gr.Checkbox(label="Acarbose"),
        gr.Checkbox(label="Insulin"),
        gr.Radio(choices=READMITTED_CHOICES, label="Readmitted")
    ],
    outputs=gr.HTML(label="Prediction"),
    title="Diabetes Readmission Prediction",
    description="Enter patient data to predict the likelihood of readmission."
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()