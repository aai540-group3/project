import logging
import requests
import gradio as gr

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for the model
API_URL = "https://api-inference.huggingface.co/models/aai540-group3/diabetes-readmission"

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
    # Create a dictionary from the input features
    input_data = {
        "age": age,
        "time_in_hospital": time_in_hospital,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_diagnoses": number_diagnoses,
        "metformin": int(metformin),
        "repaglinide": int(repaglinide),
        "nateglinide": int(nateglinide),
        "chlorpropamide": int(chlorpropamide),
        "glimepiride": int(glimepiride),
        "glipizide": int(glipizide),
        "glyburide": int(glyburide),
        "pioglitazone": int(pioglitazone),
        "rosiglitazone": int(rosiglitazone),
        "acarbose": int(acarbose),
        "insulin": int(insulin),
        "readmitted": readmitted  # Ensure this is correctly mapped
    }

    try:
        # Make a request to the Hugging Face inference API
        response = requests.post(API_URL, json={"inputs": input_data})
        response.raise_for_status()  # Raise an error for bad responses

        prediction = response.json()
        logger.info(f"Prediction received: {prediction}")
        return f"<h1 style='font-size: 48px; color: green;'>Prediction: {prediction}</h1>"

    except requests.exceptions.RequestException as e:
        logger.error(f"Error in API request: {e}")
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
