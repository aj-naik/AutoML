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
from io import BytesIO
# from pyxlsb import open_workbook as open_xlsb
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():

    def save_uploadedfile(uploadedfile):
        with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
            f.write(uploadedfile.getbuffer())
        return 

    st.title("AutoML Studio")
    st.sidebar.title("AutoML Studio")
    st.markdown("This webapp is designed to generate data insights, train and test ML model for any dataset")
    st.sidebar.markdown("This webapp is designed to generate data insights, train and test ML model for any dataset")
    
    file_data = st.sidebar.file_uploader("Upload Data file", type=["csv", "txt", "xlsx", "xls", "json"])
    save_uploadedfile(file_data)

    st.sidebar.subheader("Select the type of problem")
    problem_type = st.sidebar.selectbox("Problem Type", ["Classification", "Regression"])
    
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
    def split(x,y,test_size):
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

    # def to_excel(df):
        output = BytesIO()
        writer = pd.ExcelWriter(output, engine='xlsxwriter')
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']
        format1 = workbook.add_format({'num_format': '0.00'}) 
        worksheet.set_column('A:A', None, format1)  
        writer.save()
        processed_data = output.getvalue()
        return processed_data

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
        # st.write(filepaths)
        data = load_data(file_data)
        if st.sidebar.checkbox("Display Dataset", False):
            st.dataframe(data)
        
        st.sidebar.subheader("Feature Engineering")
        st.sidebar.markdown("Select columns for training")
        feature_selection = st.sidebar.multiselect("Select Features", data.columns, key='1')
        df = data[feature_selection]
        label = st.sidebar.selectbox("Select Label", data.columns)
        y = data[label]
        if st.sidebar.checkbox("Show Selected Columns", False):
            st.markdown("Selected columns for training")
            st.write(df.describe())
            st.dataframe(df)
            st.markdown("Select Label Column")
            st.dataframe(y)

        st.sidebar.markdown("Create new Features")
        st.sidebar.markdown("Only One operation can be stored as of now")
        feature_creation1 = st.sidebar.selectbox("Select Feature 1", data.columns, key='2')
        feature_creation2 = st.sidebar.selectbox("Select Feature 2", data.columns, key='3')
        # st.write(data[feature_creation1] * data[feature_creation2])
        op = st.sidebar.selectbox("Select Operation", ["+", "-", "*", "/", "log of feature 1","log of feature 2", "exp of feature 1", "exp of feature 2"])
        feature_name = st.sidebar.text_input("Feature Name", "Enter Name")
        if st.sidebar.button("Create"):
            if op == "+":
                data[feature_name] = data[feature_creation1] + data[feature_creation2]
            elif op == "-":
                data[feature_name] = data[feature_creation1] - data[feature_creation2]
            elif op == "*":
                df[feature_name] = data[feature_creation1] * data[feature_creation2]
            elif op == "/":
                data[feature_name] = data[feature_creation1] / data[feature_creation2]
            elif op == "log of feature 1":
                data[feature_name] = np.log(data[feature_creation1])
            elif op == "log of feature 2":
                data[feature_name] = np.log(data[feature_creation2]) 
            elif op == "exp of feature 1":
                data[feature_name] = np.exp(data[feature_creation1])
            elif op == "exp of feature 2":
                data[feature_name] = np.exp(data[feature_creation2])
            st.dataframe(data)
            df_xlsx = to_excel(data)
            st.download_button(label='📥 Download Current Result',
                                data=df_xlsx ,
                                file_name= 'df_test.xlsx')
        
        


        st.sidebar.subheader("Data Visualization")
        st.sidebar.markdown("Data Visualization will only work for features selected which are of numerical type.")
        if st.sidebar.checkbox("Data Plots", False):
            plot_list = st.sidebar.multiselect("Choose Plots", ('Correlation Matrix', 'Histogram', 'Pairplot'))
            viz_data(plot_list,df)   


    


if __name__ == '__main__':
    main()