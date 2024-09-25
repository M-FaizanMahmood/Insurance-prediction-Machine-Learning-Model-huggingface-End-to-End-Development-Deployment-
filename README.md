This project involves building and deploying a Random Forest model for house insurance prediction, achieving 96% accuracy & R-squared of 0.8499. The model is trained on a dataset containing seven key features: age, sex, BMI, children, smoker, region, and charges, charges as the target variable, A Gradio interface was developed for users to input details and receive real-time predictions. The model is deployed on Hugging Face Spaces for public access.


# **File Title : Insurance_Data_Cleaning.ipynb**

**Key Tasks:**
**Data Visualization:** In-depth analysis using pair plots and box plots to understand data distribution and relationships.
**Data Cleaning:** Outliers were capped instead of removed to retain data integrity.
**Feature Engineering & Correlation Analysis:** A correlation matrix was created to assess feature importance.
**Encoding & Scaling:** Data was one-hot encoded using get_dummies, and standardized using a scaler saved as **"insurance_scaler2.pkl"** for future predictions.
The final preprocessed dataset was saved as **"insurance_processed.csv"** for use in the machine learning pipeline.

# **File Title : Insurance_Model_Building.ipynb**

**key Tasks:**
**Machine Learning Model Creation:** The cleaned dataset ("insurance_processed.csv") was used to build a Random Forest Regressor model for predicting insurance charges.

**Model Selection:** A Random Forest Regressor with 100 estimators was chosen due to its ability to handle non-linearity, capture feature interactions, measure feature importance, resist overfitting, and efficiently scale with data.

**Model Evaluation:**
Mean Squared Error: 17,538,480.31
R-squared: 0.8499

**Model Validation:** The model was cross-validated with user inputs for prediction accuracy.
**User Interface & Deployment:** A Gradio interface was developed for real-time predictions, and the model was saved as **insurance_model2.pkl** for future use

# **File Title : Insurance_Model_Gradio.ipynb**

This file of the project contains a user interface which was built using the Gradio library, that allows users to input their details—such as age, sex, BMI, number of children, smoker status, and region. These inputs are then scaled and encoded using predefined parameters and the saved scaler "insurance_scaler2.pkl".

The system processes the inputs through the Random Forest Regressor model and provides an estimated prediction of the user's insurance charges. The interface allows for seamless interaction and real-time estimation of insurance costs based on the provided information.

# **File Title : app.py**

This file contains the deployment script for the Random Forest Regressor model, which is hosted on Hugging Face Spaces. The script runs the Gradio-based user interface, allowing users to input their details (age, sex, BMI, children, smoker, and region) and receive an estimated insurance charge. The app uses the saved model **insurance_model2.pkl** and the scaler **insurance_scaler2.pkl** to process the inputs and generate real-time predictions. The model is deployed for public access, offering seamless interaction for predicting insurance costs.

















