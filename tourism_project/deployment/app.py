import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="Harsha1001/Tourism-Package-Prediction", filename="best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a the tourism package taken by the customer based on various parameters.
Please enter the customer details to get a prediction.
""")

# User input
Age = st.number_input("Age", min_value=18, max_value=100, value=20, step=1)
TypeOfContact = st.selectbox("Type Of Contact", ['Self Enquiry','Company Invited'])
CityTier = st.selectbox("City Tier", ['1','2','3'])
DurationOfPitch = st.number_input("Duration Of Pitch", min_value=0.0, max_value=200.0, value=5.0, step=1.0)
Occupation = st.selectbox("Occupation", ['Free Lancer','Small Business','Large Business','Salaried'])
Gender = st.selectbox("Gender", ['Female','Male'])
NumberofPersonVisiting = st.number_input("Number Of Person Visiting", min_value=0, max_value=5, value=5, step=1)
NumberOfFollowups = st.number_input("Number Of Followups", min_value=0, max_value=10, value=5, step=1)
ProductPitched = st.selectbox("Product Pitched", ['Deluxe','Basic','Standard','Super Deluxe','King'])
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1, max_value=10, value=5, step=1)
MaritalStatus = st.selectbox("Marital Status", ['Married','Unmarried','Single','Divorced','King'])
NumberofTrips = st.number_input("Number Of Trips", min_value=1, max_value=50, value=1, step=1)
Passport = st.selectbox("Passport", ['Yes','No'])
PitchSatisfaction = st.number_input("Pitch Satisfaction", min_value=1, max_value=5, value=5, step=1)
OwnCar = st.selectbox("Own Car", ['Yes','No'])
NumberOfChildrenVisiting = st.number_input(" Number Of Children Visiting", min_value=0, max_value=10, value=0, step=1)
Designation = st.selectbox("Designation", ['AVP','Executive','Manager','Senior Manager','VP'])
MonthlyIncome = st.number_input("Monthly Income", min_value=0, max_value=1000000, value=0, step=1)

# Convert 'Yes'/'No' to 1/0 for Passport and OwnCar
Passport_numeric = 1 if Passport == 'Yes' else 0
OwnCar_numeric = 1 if OwnCar == 'Yes' else 0

# Assemble input into DataFrame
input_data = pd.DataFrame([{
'Age': Age,
'TypeofContact': TypeOfContact,
'CityTier': CityTier,
'DurationOfPitch': DurationOfPitch,
'Occupation': Occupation,
'Gender':Gender,
'NumberOfPersonVisiting':NumberofPersonVisiting,
'NumberOfFollowups':NumberOfFollowups,
'ProductPitched':ProductPitched,
'PreferredPropertyStar':PreferredPropertyStar,
'MaritalStatus':MaritalStatus,
'NumberOfTrips':NumberofTrips,
'Passport':Passport_numeric, # Use numeric value
'PitchSatisfactionScore':PitchSatisfaction, # Corrected column name to match the dataset schema
'OwnCar':OwnCar_numeric, # Use numeric value
'NumberOfChildrenVisiting':NumberOfChildrenVisiting,
'Designation':Designation,
'MonthlyIncome':MonthlyIncome
}])


if st.button("Predict Package Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Customer will purchase the package" if prediction == 1 else "Customer will not purchase the package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
