# -*- coding: utf-8 -*-

import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

st.title("""  Health Insurance Cost Prediction (College AIML project) """)
st.write(""" # By """)
st.write("""     ## Pavan R and Prajwal G""")
st.write(""" # Predicted Insurence Price """)


st.image("""bg-insurance.jpg""")


class Preprocessing_OHE():

    def __init__(self, new_data):
        self.new_data = new_data

    def binary(self):

        self.new_data["sex"] = self.new_data["sex"].apply(
            lambda x: 1 if x == "Male" else 0)
        self.new_data["smoker"] = self.new_data["smoker"].apply(
            lambda x: 1 if x == "Yes" else 0)
        self.new_data["Medical Problem"] = self.new_data["Medical Problem"].apply(
            lambda x: 1 if x == "Severe" else 0)

        return self.new_data

    def multinomial(self):

        self.new_data["region_southeast"] = self.new_data["Region"].apply(
            lambda x: 1 if x == "Southeast" else 0)

        self.new_data["region_northeast"] = self.new_data["Region"].apply(
            lambda x: 1 if x == "Northeast" else 0)

        self.new_data["region_southwest"] = self.new_data["Region"].apply(
            lambda x: 1 if x == "Southwest" else 0)

        self.new_data["region_northwest"] = self.new_data["Region"].apply(
            lambda x: 1 if x == "Northwest" else 0)

        return self.new_data.drop(["Region"], axis="columns")


def input_data():

    age = st.slider(label="age", min_value=18, max_value=64, step=1),

    sex = st.selectbox("sex", ("Male", "Female")),

    bmi = st.number_input(label="bmi", min_value=18.0,
                          max_value=47.0, step=0.1),

    children = st.slider(label="children", min_value=0, max_value=5, step=1),

    smoker = st.selectbox("smoker", ("No", "Yes")),

    medical_problem = st.selectbox("Medical Problem", ("Light", "Severe")),

    region = st.radio(
        "Region", ('South India', 'North India', 'East India', 'West India'))

    return age, sex, bmi, children, smoker, medical_problem, region


def create_dataframe():

    age, sex, bmi, children, smoker, medical_problem, region = input_data()

    features_dict = {"age": age, "sex": sex,
                     "bmi": bmi, "children": children,
                     "smoker": smoker,
                     "Medical Problem": medical_problem,
                     "Region": region}

    new_data = pd.DataFrame(features_dict)

    return new_data


def preprocess(new_data):

    ohe_preproccesing = Preprocessing_OHE(new_data)

    new_data = ohe_preproccesing.binary()
    new_data = ohe_preproccesing.multinomial()

    return new_data


def predict(new_data):

    model = XGBRegressor()
    model.load_model("xgb_insurence.json")

    return model.predict(new_data)


def main():

    new_data = create_dataframe()

    st.subheader("User Input")
    st.table(new_data)

    new_data = preprocess(new_data)

    # Realizamos las predicciones
    if st.button(label='Predict'):

        charges = predict(new_data) * 70
        st.success(
            f'The estimated health insurance charge is: ₹{charges} ')


if __name__ == "__main__":
    main()
