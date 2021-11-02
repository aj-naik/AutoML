import streamlit as st
import numpy as np 
import pandas as pd 
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
import os
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("AutoML Studio")
    st.sidebar.title("AutoML Studio")
    st.markdown("This webapp is designed to generate data insights, train and test ML model for any dataset")
    st.sidebar.markdown("This webapp is designed to generate data insights, train and test ML model for any dataset")

    st.sidebar.subheader("Select the type of problem")
    problem_type = st.sidebar.selectbox("Problem Type", ["Classification", "Regression"])

    def save_uploadedfile(uploadedfile):
        with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
            f.write(uploadedfile.getbuffer())
        return 
    
    def load_data(file_data):
        if file_data is not None:
            filepaths = ["/temDir/{}".format(file_data.name)]             
            for fp in filepaths:
                # Split the extension from the path and normalise it to lowercase.
                ext = os.path.splitext(fp)[-1].lower()
            if ext == ".json":
                data = pd.read_json(file_data)
            elif ext == ".txt":
                data = pd.read_csv(file_data)
            elif ext == ".xlsx":
                data = pd.read_excel(file_data)
            elif ext == ".xls":
                data = pd.read_excel(file_data)
            elif ext == ".csv":
                data = pd.read_csv(file_data)
            else:
                st.error("Please upload a valid file")

        return data

    @st.cache(persist=True)
    def split(df):
        y = df['Chance of Admit']
        x = df.drop(columns=['Chance of Admit'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    def viz_data(plot_list,df):
        if 'Correlation Matrix' in plot_list:
            st.subheader("Correlation Matrix")
            corr_matrix = df.corr()
            plt.figure(figsize = (12, 12))
            st.write(sns.heatmap(corr_matrix, annot = True))
            st.pyplot()
            
        if 'Histogram' in plot_list:
            st.subheader("Histogram")
            fig, ax = plt.subplots()
            ax = sns.histplot(df,bins=10)
            st.pyplot(fig)

        if 'Pairplot' in plot_list:
            st.subheader("Pairplot")
            fig = sns.pairplot(df)
            st.pyplot(fig)

    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("AUC Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()

    if problem_type == "Classification":
        file_data = st.sidebar.file_uploader("Upload Data file", type=["csv", "txt", "xlsx", "xls", "json"])
        save_uploadedfile(file_data)
        # st.write(filepaths)
        data = load_data(file_data)
        if st.sidebar.checkbox("Display Dataset", False):
            st.dataframe(data)
        
        st.sidebar.subheader("Data Engineering")
        feature_selection = st.sidebar.multiselect("Select Features", data.columns)
        df = data[feature_selection]
        st.dataframe(df)      

        if st.sidebar.checkbox("Data Plots", False):
            plot_list = st.sidebar.multiselect("Choose Plots", ('Correlation Matrix', 'Histogram', 'Pairplot'))
            st.write(df)
            viz_data(plot_list,df)   


    


if __name__ == '__main__':
    main()