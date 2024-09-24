import gradio as gr
import numpy as np
import joblib
model=joblib.load('insurance_model2.pkl')
scaler =joblib.load('insurance_scaler.pkl')


# Function to encode inputs and make predictions
def get_user_input(age, bmi, sex, smoker, region, children):
    # Encoding sex
    sex_female = 1 if sex == 'female' else 0
    sex_male = 1 if sex == 'male' else 0

    # Encoding smoker
    smoker_yes = 1 if smoker == 'yes' else 0
    smoker_no = 1 if smoker == 'no' else 0

    # Encoding region
    region_northeast = 1 if region == 'northeast' else 0
    region_northwest = 1 if region == 'northwest' else 0
    region_southeast = 1 if region == 'southeast' else 0
    region_southwest = 1 if region == 'southwest' else 0

    # Encoding children
    children_0 = 1 if children == 0 else 0
    children_1 = 1 if children == 1 else 0
    children_2 = 1 if children == 2 else 0
    children_3 = 1 if children == 3 else 0
    children_4 = 1 if children == 4 else 0
    children_5 = 1 if children == 5 else 0

    # Combine the inputs into a single list (total 16 features)
    user_input = [age, bmi, sex_female, sex_male, smoker_no, smoker_yes,
                  region_northeast, region_northwest, region_southeast, region_southwest,
                  children_0, children_1, children_2, children_3, children_4, children_5]

    # Convert the input to a NumPy array and reshape it for the model
    user_input = np.array(user_input).reshape(1, -1)

    # Scale the numerical features (age and bmi)
    user_input[:, 0:2] = scaler.transform(user_input[:, 0:2])

    # Predict using the loaded model
    prediction = model.predict(user_input)

    # Return the prediction
    return f"The predicted insurance charges are: ${prediction[0]:.2f}"

# Create the Gradio interface with updated components
interface = gr.Interface(
    fn=get_user_input,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="BMI"),
        gr.Radio(choices=["male", "female"], label="Sex"),
        gr.Radio(choices=["yes", "no"], label="Smoker"),
        gr.Dropdown(choices=["northeast", "northwest", "southeast", "southwest"], label="Region"),
        gr.Slider(0, 5, step=1, label="Number of Children")
    ],
    outputs="text"
)

# Launch the interface
interface.launch()
