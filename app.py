import streamlit as st
import numpy as np 
import pandas as pd 
import base64
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import pickle
import os
from io import BytesIO
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    
    def save_uploadedfile(uploadedfile):
        '''Helper function to put uploaded file into temp directory'''
        if uploadedfile is not None:
            with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
                f.write(uploadedfile.getbuffer())
        return 

    st.title("AutoML Studio")
    st.sidebar.title("AutoML Studio")
    st.markdown("This webapp is designed to generate data insights, train and test ML model for many dataset")
    st.sidebar.markdown("This webapp is designed to generate data insights, train and test ML model for many dataset")
    
    file_data = st.sidebar.file_uploader("Upload Data file", type=["csv", "txt", "xlsx", "xls", "json"])
    save_uploadedfile(file_data)

    st.sidebar.subheader("Select the type of problem")
    problem_type = st.sidebar.selectbox("Problem Type", ["Classification", "Regression"])
    
    def load_data(file_data):
        '''Function to load dataset onto the app'''
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

    def viz_data(plot_list,df):
        '''Function to display plots for dataset'''
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
            
    
    def split(df, y, test_size):
        '''Function to split datset into train test'''
        x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=test_size, random_state=0)
        return x_train, x_test, y_train, y_test

    def get_table_download_link(df):
        """Generates a link allowing the data in a given panda dataframe to be downloaded
        in:  dataframe
        out: href string
        """
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}">Download CSV file</a> (Right-click and save as &lt;some_name&gt;.csv or any other format)'
        return href
    
    def download_model(model):
        """Generates a link allowing the data to be downloaded
        in:  model
        out: pkl file
        """
        output_model = pickle.dumps(model)
        b64 = base64.b64encode(output_model).decode()
        href = f'<a href="data:file/output_model;base64,{b64}">Download Trained Model .pkl File</a> (right-click and save as &lt;some_name&gt;.pkl)'
        return href

    def classify_select(): 
        '''Function to load trained classification models for testing'''
        if classify == 'Logistic Regression':
            load_weights = open('LR.pkl', 'rb') 
            classifier = pickle.load(load_weights)
        elif classify == 'Support Vector Machine (SVM)':
            load_weights = open('SVM.pkl', 'rb')
        elif classify == 'K-Nearest Neighbor':
            load_weights = open('KNN.pkl', 'rb')  
            classifier = pickle.load(load_weights)
        elif classify == 'Decision Tree':
            load_weights = open('DT.pkl', 'rb') 
            classifier = pickle.load(load_weights)
        elif classify == 'Random Forest':
            load_weights = open('RF.pkl', 'rb') 
            classifier = pickle.load(load_weights)
        elif classify == 'XGBoost':
            load_weights = open('XGB.pkl', 'rb') 
            classifier = pickle.load(load_weights)
        return classifier
        
    def predictor_select():
        '''Function to load trained regression models for testing'''
        if classify == 'Linear Regression':
            load_weights = open('LR.pkl', 'rb') 
            classifier = pickle.load(load_weights)
        elif classify == 'Support Vector Regressor (SVR)':
            load_weights = open('SVM.pkl', 'rb') 
            classifier = pickle.load(load_weights)
        elif classify == 'K-Nearest Regressor':
            load_weights = open('KNR.pkl', 'rb')
            classifier = pickle.load(load_weights)
        elif classify == 'Decision Tree':
            load_weights = open('DT.pkl', 'rb') 
            classifier = pickle.load(load_weights)
        elif classify == 'Random Forest':
            load_weights = open('RF.pkl', 'rb') 
            classifier = pickle.load(load_weights)
        elif classify == 'XGBoost':
            load_weights = open('XGB.pkl', 'rb') 
            classifier = pickle.load(load_weights)
        return classifier

    def plot_metrics(metrics_list):
        '''Function to plot mertics for dataset'''
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

    '''Code block for Classification and Regression problems'''
    
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
            st.markdown("Statistics of data")
            st.write(df.describe())
            st.markdown("Selected columns for training")
            st.dataframe(df)
            st.markdown("Select Label Column")
            st.dataframe(y)
            st.markdown(get_table_download_link(df), unsafe_allow_html=True)

        st.sidebar.markdown("Create new Features")
        st.sidebar.markdown("Only One operation can be stored as of now")

        feature_creation1 = st.sidebar.selectbox("Select Feature 1", data.columns, key='2')
        feature_creation2 = st.sidebar.selectbox("Select Feature 2", data.columns, key='3')
        # st.write(data[feature_creation1] * data[feature_creation2])

        op = st.sidebar.selectbox("Select Operation", ["+", "-", "*", "/", "log of feature 1","log of feature 2", "exp of feature 1", "exp of feature 2"])
        feature_name = st.sidebar.text_input("Feature Name", "Enter Name")
        
        if st.sidebar.button("Generate"):

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

            st.markdown(get_table_download_link(data), unsafe_allow_html=True)
        
        st.sidebar.subheader("Data Visualization")
        st.sidebar.markdown("Data Visualization will only work for features selected which are of numerical type.")
        if st.sidebar.checkbox("Data Plots", False):
            plot_list = st.sidebar.multiselect("Choose Plots", ('Correlation Matrix', 'Histogram', 'Pairplot'))
            viz_data(plot_list,df)  

        st.sidebar.subheader("Modelling Data") 
        st.sidebar.markdown("Select ML algorithm and train, test model on the dataset")
        
        testsize = st.sidebar.slider("Select Test Size", 0.1, 0.9, 0.5)
        x_train, x_test, y_train, y_test = split(df, y, testsize)
        
        if st.sidebar.checkbox("Train a model",False):
            st.sidebar.subheader("Choose Classifier")
            classifier = st.sidebar.selectbox("Classifier", ("Logistic Regression","Support Vector Machine (SVM)","K-Nearest Neighbor","Decision Tree", "Random Forest", "XGBoost"))

            if classifier == 'Logistic Regression':
                st.sidebar.subheader("Model Hyperparameters")
                C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
                max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

                metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

                if st.sidebar.button("Classify", key='classify'):
                    st.subheader("Logistic Regression Results")
                    model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
                    model.fit(x_train, y_train)
                    accuracy = model.score(x_test, y_test)
                    y_pred = model.predict(x_test)
                    st.write('Model Accuracy is ',accuracy.round(3)*100)
                    st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                    st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                    plot_metrics(metrics)
                
                if st.button("Save Model"):
                    model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
                    model.fit(x_train, y_train)
                    weights =open("LR.pkl",mode = "wb")
                    pickle.dump(model,weights)
                    weights.close()
                    st.success('Your model saved sucessfully')
                    st.markdown(download_model(model), unsafe_allow_html=True)

            if classifier == 'Support Vector Machine (SVM)':
                st.sidebar.subheader("Model Hyperparameters")
                C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
                kernel = st.sidebar.radio("Kernel", ('linear', 'poly', 'rbf', 'sigmoid'), key='kernel')
                gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

                metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
                
                if st.sidebar.button("Classify", key='classify'):
                    st.subheader("Support Vector Machine (SVM) Results")
                    model = SVC(C=C, kernel=kernel, gamma=gamma)
                    model.fit(x_train, y_train)
                    accuracy = model.score(x_test, y_test)
                    y_pred = model.predict(x_test)
                    st.write('Model Accuracy is ',accuracy.round(3)*100)
                    st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                    st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                    plot_metrics(metrics)

                if st.button("Save Model"):
                    model = SVC(C=C, kernel=kernel, gamma=gamma)
                    model.fit(x_train, y_train)
                    weights =open("SVM.pkl",mode = "wb")
                    pickle.dump(model,weights)
                    weights.close()                
                    st.success('Your model saved sucessfully')
                    st.markdown(download_model(model), unsafe_allow_html=True)

            if classifier == 'K-Nearest Neighbor':
                st.sidebar.subheader("Model Hyperparameters")
                neigh = st.sidebar.number_input("No. of Neighbours")
                algo = st.sidebar.radio("Algorithm", ('auto', 'ball_tree', 'kd_tree', 'brute'), key='algo')

                metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
            
                if st.sidebar.button("Predict", key='classify'):
                    st.subheader("K-Nearest Regressor Results")
                    model = KNeighborsClassifier(n_neighbors = int(neigh), algorithm=algo)
                    model.fit(x_train, y_train)
                    accuracy = model.score(x_test, y_test)
                    y_pred = model.predict(x_test)
                    st.write('Model Accuracy is ',accuracy.round(3)*100)
                    st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                    st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                    plot_metrics(metrics)

                if st.button("Save Model"):
                    model = KNeighborsClassifier(n_neighbors = int(neigh), algorithm=algo)
                    model.fit(x_train, y_train)
                    weights =open("KNN.pkl",mode = "wb")
                    pickle.dump(model,weights)
                    weights.close()                
                    st.success('Your model saved sucessfully')
                    st.markdown(download_model(model), unsafe_allow_html=True)

            if classifier == 'Random Forest':
                st.sidebar.subheader("Model Hyperparameters")
                n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
                max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='n_estimators')
                bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
                metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

                if st.sidebar.button("Classify", key='classify'):
                    st.subheader("Random Forest Results")
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
                    model.fit(x_train, y_train)
                    accuracy = model.score(x_test, y_test)
                    y_pred = model.predict(x_test)
                    st.write('Model Accuracy is ',accuracy.round(3)*100)
                    st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                    st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                    plot_metrics(metrics)

                if st.button("Save Model"):   
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
                    model.fit(x_train, y_train)
                    weights =open("RF.pkl",mode = "wb")
                    pickle.dump(model,weights)
                    weights.close()
                    st.success('Your model saved sucessfully')
                    st.markdown(download_model(model), unsafe_allow_html=True)

            if classifier == 'Decision Tree':
                st.sidebar.subheader("Model Hyperparameters")
                max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 30, step=1, key='n_estimators')
                metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

                if st.sidebar.button("Classify", key='classify'):
                    st.subheader("Decision Tree Results")
                    model = DecisionTreeClassifier(max_depth=max_depth)
                    model.fit(x_train, y_train)
                    accuracy = model.score(x_test, y_test)
                    y_pred = model.predict(x_test)              
                    st.write('Model Accuracy is ',accuracy.round(3)*100)
                    st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                    st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                    plot_metrics(metrics)

                if st.button("Save Model"):
                    model = DecisionTreeClassifier(max_depth=max_depth)
                    model.fit(x_train, y_train)
                    weights =open("DT.pkl",mode = "wb")
                    pickle.dump(model,weights)
                    weights.close()
                    st.success('Your model saved sucessfully')
                    st.markdown(download_model(model), unsafe_allow_html=True)

            if classifier == 'XGBoost':
                st.sidebar.subheader("Model Hyperparameters")
                n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
                max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 30, step=1, key='n_estimators')
                C = st.sidebar.number_input("C (Learning Rate parameter)", 0.01, 10.0, step=0.01, key='C_LR')
                metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

                if st.sidebar.button("Classify", key='classify'):
                    st.subheader("XGBoost Results")
                    model = xgb.XGBRegressor(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = C,
                        max_depth = max_depth, n_estimators = n_estimators)
                    model.fit(x_train, y_train)
                    accuracy = model.score(x_test, y_test)
                    y_pred = model.predict(x_test)
                    st.success('Your model saved sucessfully')                
                    st.write('Model Accuracy is ',accuracy.round(3)*100)
                    st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
                    st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
                    plot_metrics(metrics)

                if st.button("Save Model"):
                    model = xgb.XGBRegressor(objective ='reg:logistic', colsample_bytree = 0.3, learning_rate = C,
                        max_depth = max_depth, n_estimators = n_estimators)
                    model.fit(x_train, y_train)
                    weights =open("XGB.pkl",mode = "wb")
                    pickle.dump(model,weights)
                    weights.close()
                    st.success('Your model saved sucessfully')
                    st.markdown(download_model(model), unsafe_allow_html=True)
       
    
                
    elif problem_type == "Regression":
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
            st.markdown("Statistics of data")
            st.write(df.describe())
            st.markdown("Selected columns for training")
            st.dataframe(df)
            st.markdown("Select Label Column")
            st.dataframe(y)
            st.markdown(get_table_download_link(df), unsafe_allow_html=True)

        st.sidebar.markdown("Create new Features")
        st.sidebar.markdown("Only One operation can be stored as of now")

        feature_creation1 = st.sidebar.selectbox("Select Feature 1", data.columns, key='2')
        feature_creation2 = st.sidebar.selectbox("Select Feature 2", data.columns, key='3')
        # st.write(data[feature_creation1] * data[feature_creation2])

        op = st.sidebar.selectbox("Select Operation", ["+", "-", "*", "/", "log of feature 1","log of feature 2", "exp of feature 1", "exp of feature 2"])
        feature_name = st.sidebar.text_input("Feature Name", "Enter Name")
        
        if st.sidebar.button("Generate"):

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

            st.markdown(get_table_download_link(data), unsafe_allow_html=True)
        
        
        st.sidebar.subheader("Data Visualization")
        st.sidebar.markdown("Data Visualization will only work for features selected which are of numerical type.")
        if st.sidebar.checkbox("Data Plots", False):
            plot_list = st.sidebar.multiselect("Choose Plots", ('Correlation Matrix', 'Histogram', 'Pairplot'))
            viz_data(plot_list,df)  

        st.sidebar.subheader("Modelling Data") 
        st.sidebar.markdown("Select ML algorithm and train, test model on the dataset")
        
        testsize = st.sidebar.slider("Select Test Size", 0.1, 0.9, 0.5)
        x_train, x_test, y_train, y_test = split(df, y, testsize)
        
        
        if st.sidebar.checkbox("Train the Model",False):
            st.sidebar.subheader("Choose Model")
            classifier = st.sidebar.selectbox("Model", ("Linear Regression","Support Vector Regressor (SVR)","K-Nearest Regressor","Decision Tree", "Random Forest", "XGBoost"))
            # scaler = st.sidebar.selectbox("Choose Scaler", ("Standard Scaler","Min-Max Scaler","No Scaler"))

            if classifier == 'Linear Regression':
                if st.sidebar.button("Train", key='classify'):
                    st.subheader("Linear Regression Results")
                    model = LinearRegression()
                    model.fit(x_train, y_train)
                    accuracy = model.score(x_test, y_test)
                    y_pred = model.predict(x_test)
                    st.write('Model Accuracy is ',accuracy.round(3)*100)
                
                if st.button("Save Model"):
                    model = LinearRegression()
                    model.fit(x_train, y_train)
                    weights =open("LR.pkl",mode = "wb")
                    pickle.dump(model,weights)
                    weights.close()
                    st.success('Your model saved sucessfully')
                    st.markdown(download_model(model), unsafe_allow_html=True)

            if classifier == 'Support Vector Regressor (SVR)':
                st.sidebar.subheader("Model Hyperparameters")
                C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
                kernel = st.sidebar.radio("Kernel", ('linear', 'poly', 'rbf', 'sigmoid'), key='kernel')
                gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')
                
                if st.sidebar.button("Train", key='classify'):
                    st.subheader("Support Vector Regressor (SVR) Results")
                    model = SVR(C=C, kernel=kernel, gamma=gamma)
                    model.fit(x_train, y_train)
                    accuracy = model.score(x_test, y_test)
                    y_pred = model.predict(x_test)
                    st.write('Model Accuracy is ',accuracy.round(3)*100)

                if st.button("Save Model"):
                    model = SVR(C=C, kernel=kernel, gamma=gamma)
                    model.fit(x_train, y_train)
                    weights =open("SVM.pkl",mode = "wb")
                    pickle.dump(model,weights)
                    weights.close()                
                    st.success('Your model saved sucessfully')
                    st.markdown(download_model(model), unsafe_allow_html=True)

            if classifier == 'K-Nearest Regressor':
                st.sidebar.subheader("Model Hyperparameters")
                neigh = st.sidebar.number_input("No. of Neighbours")
                algo = st.sidebar.radio("Algorithm", ('auto', 'ball_tree', 'kd_tree', 'brute'), key='algo')
            
                if st.sidebar.button("Train", key='classify'):
                    st.subheader("K-Nearest Regressor Results")
                    model = KNeighborsRegressor(n_neighbors = int(neigh), algorithm=algo)
                    model.fit(x_train, y_train)
                    accuracy = model.score(x_test, y_test)
                    y_pred = model.predict(x_test)
                    st.write('Model Accuracy is ',accuracy.round(3)*100)

                if st.button("Save Model"):
                    model = KNeighborsRegressor(n_neighbors = int(neigh), algorithm=algo)
                    model.fit(x_train, y_train)
                    weights =open("KNR.pkl",mode = "wb")
                    pickle.dump(model,weights)
                    weights.close()                
                    st.success('Your model saved sucessfully')
                    st.markdown(download_model(model), unsafe_allow_html=True)

            if classifier == 'Random Forest':
                st.sidebar.subheader("Model Hyperparameters")
                n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
                max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='n_estimators')
                bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')

                if st.sidebar.button("Train", key='classify'):
                    st.subheader("Random Forest Results")
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
                    model.fit(x_train, y_train)
                    accuracy = model.score(x_test, y_test)
                    y_pred = model.predict(x_test)
                    st.write('Model Accuracy is ',accuracy.round(3)*100)

                if st.button("Save Model"):   
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
                    model.fit(x_train, y_train)
                    weights =open("RF.pkl",mode = "wb")
                    pickle.dump(model,weights)
                    weights.close()
                    st.success('Your model saved sucessfully')
                    st.markdown(download_model(model), unsafe_allow_html=True)

            if classifier == 'Decision Tree':
                st.sidebar.subheader("Model Hyperparameters")
                max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 30, step=1, key='n_estimators')

                if st.sidebar.button("Train", key='classify'):
                    st.subheader("Decision Tree Results")
                    model = DecisionTreeRegressor(max_depth=max_depth)
                    model.fit(x_train, y_train)
                    accuracy = model.score(x_test, y_test)
                    y_pred = model.predict(x_test)              
                    st.write('Model Accuracy is ',accuracy.round(3)*100)

                if st.button("Save Model"):
                    model = DecisionTreeRegressor(max_depth=max_depth)
                    model.fit(x_train, y_train)
                    weights =open("DT.pkl",mode = "wb")
                    pickle.dump(model,weights)
                    weights.close()
                    st.success('Your model saved sucessfully')
                    st.markdown(download_model(model), unsafe_allow_html=True)

            if classifier == 'XGBoost':
                st.sidebar.subheader("Model Hyperparameters")
                n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
                max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 30, step=1, key='max_depth')
                C = st.sidebar.number_input("C (Learning Rate parameter)", 0.01, 10.0, step=0.01, key='C_LR')

                if st.sidebar.button("Train", key='classify'):
                    st.subheader("XGBoost Results")
                    model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = C,
                        max_depth = max_depth, n_estimators = n_estimators)
                    model.fit(x_train, y_train)
                    accuracy = model.score(x_test, y_test)
                    y_pred = model.predict(x_test)             
                    st.write('Model Accuracy is ',accuracy.round(3)*100)

                if st.button("Save Model"):
                    model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = C,
                        max_depth = max_depth, n_estimators = n_estimators)
                    model.fit(x_train, y_train)
                    weights =open("XGB.pkl",mode = "wb")
                    pickle.dump(model,weights)
                    weights.close()
                    st.success('Your model has been saved sucessfully')
                    st.markdown(download_model(model), unsafe_allow_html=True)

    

if __name__ == '__main__':
    main()