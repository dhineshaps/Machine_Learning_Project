import pandas as pd 
import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://Dhinesh:Test123@fetquestprojects.fpaoxah.mongodb.net/?retryWrites=true&w=majority&appName=FETQuestProjects"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['student']
collection = db["student_pred"]


def load_model():
    print("here in load_model")
    with open("student_lr_final_model.pkl", "rb") as file:
        model,scaler,le = pickle.load(file) #this return these 3 which we loaded whiule writting refer collab
    return model,scaler,le

def preprocessing_input_data(data, scaler,le):
    print("here in preprocessing_input_data")
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])[0]
    print("here in preprocessing_input_data 1")
    df = pd.DataFrame([data])
    print("here in preprocessing_input_data 2")
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
     print("here in predict_data")
     model,scaler,le = load_model()
     processed_data = preprocessing_input_data(data,scaler,le)
     prediction = model.predict(processed_data)
     return prediction
     
def main():
    st.title("Student Perfromance Prediction")
    st.write("Enter your data to get a prediction for your performance")

    hr_stud = st.number_input("Hours Studied", min_value=1,max_value=10,value=5)
    pre_scr =st.number_input("Previous Scores", min_value=40,max_value=100,value=70)
    ext_act =st.selectbox("Extracurricular Activities",["Yes","No"])
    slp_hr =st.number_input("Sleep Hours", min_value=4,max_value=10,value=7)
    sam_prac =st.number_input("Sample Question Papers Practiced", min_value=0,max_value=10,value=5)

    if st.button("Predict your score"):
        user_data = {
            "Hours Studied":hr_stud,
            "Previous Scores":pre_scr,
            "Extracurricular Activities":ext_act,
            "Sleep Hours":slp_hr,
            "Sample Question Papers Practiced":sam_prac
        }
        prediction = predict_data(user_data)
        st.success(f"your prediction result is {prediction}")
        user_data = {k: v.item() if isinstance(v, (np.integer, np.floating)) else v for k, v in user_data.items()}
        user_data['prediction'] = round(float(prediction[0]), 2)
        collection.insert_one(user_data)

if __name__ == "__main__":
    main()