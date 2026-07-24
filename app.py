import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
st.title('Passenger Survived Chance in The Titanic Journey ')

pclass=st.slider('Enter The passenger class for user',1,3)
sex=st.selectbox('Enter the passenger Gender',['male','female'])
sibsp=st.slider('Enter the passenger total number of Sibling and Spouse',1,8)
parch=st.slider('Enter the passenger total number of  Parent and child',0,8)
Fare=st.number_input('Enter the Fare of the Passenger')
embarked=st.selectbox('Enter the passenger station from where they started the journey',['Southampton','Chebourg','Queenstown'])


data=pd.DataFrame([{'Pclass':pclass,'Sex':sex,'SibSp':sibsp,	'Parch':parch,'Fare':Fare,'Embarked':embarked}])

model=load_model('model.h5')

with open('label_encoder.pkl','rb') as file:
    label=pickle.load(file)

with open('onehot_encoder.pkl','rb') as file:
    onehot=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


data['Sex']=label.transform(data['Sex'])
embarked=onehot.transform(data[['Embarked']])

embarked=pd.DataFrame(embarked,columns=onehot.get_feature_names_out())

data=pd.concat([data.drop(columns=['Embarked']),embarked],axis=1)\

data[['Pclass','SibSp',	'Parch'	,'Fare']]=scaler.transform(data[['Pclass','SibSp',	'Parch'	,'Fare']])

y=model.predict(data)

y=y[0][0]
def Chance(y):
    if y>0.5:
        return 'The passenger will survived the journey'
    else:
        return ' The passenger wont durvived the journey'

if st.button('Predict The Survival Chance'):
    st.write('Probability of Passenger Survival chance',y)
    st.write(Chance(y))