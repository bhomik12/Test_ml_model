import streamlit as st 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



st.sidebar.header(" Cost Prediction")
result = st.sidebar.radio(label ='Select an Option',options=['Data info', 'Correlation heatmap','Prediction','Scatterplot','Customer Acquisition'])

if result == 'Data info':
    st.header('Data Information')
    data = pd.read_csv('Retention CP prediction.csv', index_col=0) 
    st.write(data)

elif result == 'Correlation heatmap':
    st.header('Correlation Heatmap')
    data = pd.read_csv('Retention CP prediction.csv', index_col=0)
    fig, ax = plt.subplots(figsize=(20,18))
    sns.heatmap(data=data.corr() ,annot = True)
    st.write(fig)

elif result == 'Prediction':
    st.header('Table of Predicted Values')
    data = pd.read_csv('Retention CP prediction.csv',index_col=0)
    encoder = OrdinalEncoder()
    final_data = encoder.fit_transform(data.drop(columns='cost'))
    X = final_data
    y = data['cost']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = DecisionTreeRegressor(random_state=44)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    st.table(predictions)    

elif result == 'Scatterplot':
    st.header('Scatter Plot Of Predicted Values')
    data = pd.read_csv('Retention CP prediction.csv',index_col=0)
    encoder = OrdinalEncoder()
    final_data = encoder.fit_transform(data.drop(columns='cost'))
    X = final_data
    y = data['cost']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = DecisionTreeRegressor(random_state=44)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    plt.scatter(y_test,predictions)

    st.write(fig)
 
else:
    st.header("Customer Acquisition Value")
    df = pd.read_csv('Retention CP prediction.csv', index_col=0)
    cat_cols = [cols for cols in df.columns if df[cols].dtypes == "O"]
    len(cat_cols)

    X = df.drop('cost', axis=1, inplace=False)
    Y = df['cost']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    ct = ColumnTransformer([
        ('Step1', OrdinalEncoder(), cat_cols)
    ])
    pipeline = Pipeline([
        ('Coltf_step', ct),
        ('Decision tree', DecisionTreeRegressor())
    ])
    pipeline.fit(X_train, Y_train)
    pipeline.score(X_train,Y_train)
    L = pipeline.score(X_train, Y_train)
    st.write("Customer Acquisition :" ,L)
