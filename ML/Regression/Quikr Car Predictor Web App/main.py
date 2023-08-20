import numpy as np
import streamlit as st
import joblib
print(np.array([2.1,2]))
st.title("Car Price Predictor")
# import the model
pipe = joblib.load(open('pipe.pkl','rb'))

df = joblib.load(open('df.pkl','rb'))
names = joblib.load(open('names.pkl','rb'))
# Name
nm = st.selectbox('Name',df['name'])
# Company
comp = st.selectbox('Company',df['company'])

# Year size
year = st.number_input('Year')

# Km Driven
km = st.number_input('kms_driven')

# Fuel Type
ft = st.selectbox('Fuel Type',df['fuel_type'])
if st.button('Predict Price'):
    pred = np.exp(pipe.predict([[nm, comp, year, km, ft]])[0])
    st.title("The predicted price is PKR " + str(round(pred,1)))

