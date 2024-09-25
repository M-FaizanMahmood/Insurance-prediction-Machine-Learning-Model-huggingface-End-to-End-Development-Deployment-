This project involves building and deploying a Random Forest model for house insurance prediction, achieving 96% accuracy. The model is trained on a dataset containing seven key features: age, sex, BMI, children, smoker, region, and charges, charges as the target variable 

**File Title : insurance_Data_Cleaning.ipynb**

**Key Tasks:**
**Data Visualization:** In-depth analysis using pair plots and box plots to understand data distribution and relationships.
**Data Cleaning:** Outliers were capped instead of removed to retain data integrity.
**Feature Engineering & Correlation Analysis:** A correlation matrix was created to assess feature importance.
**Encoding & Scaling:** Data was one-hot encoded using get_dummies, and standardized using a scaler saved as "insurance_scaler2.pkl" for future predictions.
The final preprocessed dataset was saved as "insurance_processed.csv" for use in the machine learning pipeline.
