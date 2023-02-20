import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import seaborn as sns
import matplotlib.pyplot as plt


def file_download(df: pd.DataFrame) -> str:
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'''<a href="data:file/csv;base64, {b64}" download="churn_data.csv">
        Download CSV file
    </a>'''

    return href

def user_input_features() -> pd.DataFrame:
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    PaymentMethod = st.sidebar.selectbox('Payment Method', (
        'Bank transfer (automatic)', 'Credit card (automatic)', 'Mailed check', 'Electronic check'
    ))

    MonthlyCharges = st.sidebar.slider('Monthly Charges', 18.0, 118.0, 18.0)
    tenure = st.sidebar.slider('tenure', 0.0, 72.0, 0.0)

    data = {
        'gender': [gender], 
        'PaymentMethod': [PaymentMethod],
        'MonthlyCharges': [MonthlyCharges],
        'tenure': [tenure]
    }

    return pd.DataFrame(data)


if __name__ == '__main__':
    st.set_page_config(page_title='Churn Predictor')
    st.markdown('''
<h1> Churn Prediction App </h1>

<p align="right">Built with ☕ by <a href="https://github.com/Kunal-Kumar-Sahoo/" target="_blank">Kunal Kumar Sahoo</a></p>

<p>
Customer churn is defined as the loss of customers after a certain period of time. Companies are interested in targeting customers who are likely to churn. They can target these customers with special deals and promotions to influence them to stay with the company. <br>

This app predicts the probability of a customer churning using Telco Customer data. Here

customer churn means the customer does not make another purchase after a period of time. </p>
    ''', unsafe_allow_html=True)

    df_selected = pd.read_csv('Dataset/data.csv')
    df_selected_all = df_selected[[
        'gender', 'Partner', 'Dependents', 'PhoneService', 'tenure', 'MonthlyCharges'
    ]].copy()

    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown(file_download(df_selected_all), unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader('Upload your input CSV file', type=['csv'])

    input_df = user_input_features()


    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    # else:
    #     input_df = user_input_features()

    churn_raw = pd.read_csv('Dataset/data.csv')
    churn_raw.fillna(0, inplace=True)
    churn = churn_raw.drop(columns=['Churn'])
    df = pd.concat([input_df, churn], axis=0)

    encode = ['gender', 'PaymentMethod']
    for col in encode:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df, dummy], axis=1)
        del df[col]

    df = df[:1]
    df.fillna(0, inplace=True)
    
    features = [
        'MonthlyCharges', 'tenure', 'gender_Female', 'gender_Male', 
        'PaymentMethod_Bank transfer (automatic)', 
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]

    df = df[features]

    st.subheader('User Input Features')
    print(df.columns)

    if uploaded_file is not None:
        st.write(df)
    else:
        st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
        st.write(df)

    load_clf = pickle.load(open('model/churn_clf.pkl', 'rb'))
    prediction = load_clf.predict(df)
    prediction_proba = load_clf.predict_proba(df)

    churn_labels = np.array(['No', 'Yes'])

    st.write(churn_labels[prediction])

    st.subheader('Prediction Probability')

    st.write(prediction_proba)
    