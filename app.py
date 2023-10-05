import streamlit as st        
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from PIL import Image

image_directory = "winepredict-master/360_F_135879274_4CgDcLaAwtfhIgLiSmqarh2H8krigOia.webp"
image = Image.open(image_directory)

PAGE_CONFIG = {"page_title":"Wine Quality App", 
               "page_icon":image, 
               "layout":"centered", 
               "initial_sidebar_state":"auto"}

st.set_page_config(**PAGE_CONFIG)

st.markdown(""" <style> .font {
font-size:45px ; font-family: 'Cooper Black'; color: #F14747;} 
</style> """, unsafe_allow_html=True)
st.markdown('<p class="font">🍷Wine Quality Prediction App</p>', unsafe_allow_html=True)
st.subheader(' This app predicts the Wine Quality type')

st.sidebar.header('User Input Parameters')
  
def user_input_features():
        fixed_acidity = st.sidebar.slider('fixed acidity', 4.6, 15.9, 8.31)
        volatile_acidity = st.sidebar.slider('volatile acidity', 0.12,1.58 , 0.52)
        citric_acid = st.sidebar.slider('citric acid', 0.0,1.0 , 0.5)
        chlorides = st.sidebar.slider('chlorides', 0.01,0.6 , 0.08)
        total_sulfur_dioxide=st.sidebar.slider('total sulfur dioxide', 6.0,289.0 , 46.0)
        alcohol=st.sidebar.slider('alcohol', 8.4,14.9, 10.4)
        sulphates=st.sidebar.slider('sulphates', 0.33,2.0,0.65 )
        data = {'fixed_acidity': fixed_acidity,
                'volatile_acidity': volatile_acidity,
                'citric_acid': citric_acid,
                'chlorides': chlorides,
              'total_sulfur_dioxide':total_sulfur_dioxide,
              'alcohol':alcohol,
                'sulphates':sulphates}
        features = pd.DataFrame(data, index=[0])
        return features
df = user_input_features()
df.columns = ['Fixed Acidity','Volatile Acidity','Citric Acid','Chlorides','Total Sulfur Dioxide','Alcohol','Sulphates']
st.subheader('User Input parameters')
st.table(df)

data=pd.read_csv("winepredict-master/winequality-red.csv")
X =np.array(data[['fixed acidity', 'volatile acidity' , 'citric acid' , 'chlorides' , 'total sulfur dioxide' , 'alcohol' , 'sulphates']])
Y = np.array(data['quality'])
clf = RandomForestClassifier()
clf.fit(X, Y)
st.subheader('Class labels and their corresponding index number')


col1, col2=st.columns(2)
with col1:
	st.table(pd.DataFrame({'wine quality': [3, 4, 5, 6, 7, 8 ]}))
with col2:
	st.image("winepredict-master/wines-1761613_1920.jpg")

prediction = clf.predict(df)
prediction_probability = clf.predict_proba(df)
st.subheader('Prediction Probability')
st.table(prediction_probability)

x=prediction
col1, col2=st.columns(2)
with col1:
	st.markdown('<p class="font">Predicted Quality:</p>', unsafe_allow_html=True)
with col2:
	if x==3:
		st.markdown('<p class="font">🍷🍷🍷</p>', unsafe_allow_html=True)
	if x==4:
		st.markdown('<p class="font">🍷🍷🍷🍷</p>', unsafe_allow_html=True)
	if x==5:
		st.markdown('<p class="font">🍷🍷🍷🍷🍷</p>', unsafe_allow_html=True)
	if x==6:
		st.markdown('<p class="font">🍷🍷🍷🍷🍷🍷</p>', unsafe_allow_html=True)
	if x==7:
		st.markdown('<p class="font">🍷🍷🍷🍷🍷🍷🍷</p>', unsafe_allow_html=True)
	if x==8:
		st.markdown('<p class="font">🍷🍷🍷🍷🍷🍷🍷🍷</p>', unsafe_allow_html=True)