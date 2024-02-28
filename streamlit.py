import streamlit as st
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import base64

# Feature Engineering
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from scipy.stats import boxcox
import numpy as np

# Model classification and Regreesion
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

# Model Clustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering

# metrics for valuation model 
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from sklearn.metrics import calinski_harabasz_score,davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA


# set pages of dashboard
st.set_page_config(
    page_title="UI Machine Learning",
    page_icon="✅",
    layout="wide",
)

# # Add content to the sidebar
# # Obtain image from URL
# url = 'https://drive.google.com/uc?id=1agUzQ6K1LTYAw4JdRKlsuo39fknqS5gx'
# response = requests.get(url)
# image = Image.open(BytesIO(response.content))

# # Show image
# st.sidebar.image(image, use_column_width=True, width=50)

st.sidebar.markdown('**Guideline: [*click here!*](https://www.google.com/search?sca_esv=682fb458cb082d73&rlz=1C1CHBF_enKR1058ID1058&sxsrf=ACQVn0-tyTd2f481K-TV5nlom58dn6NFZg:1708680383706&q=apel&tbm=isch&source=lnms&sa=X&ved=2ahUKEwjMuri6ksGEAxXI1TgGHZwWDKYQ0pQJegQIDBAB&biw=1422&bih=612&dpr=1.35#imgrc=88-1RY50Ek_bsM)**')

st.sidebar.title('Load Data')
uploaded_file = st.sidebar.file_uploader("Choose CSV or Excel file", type=['csv', 'xlsx', 'xls'])

with open('D:/Project_Data/project/Project Pribadi/Template Model/style.css') as f:
    css = f.read()

st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# select encoding
encodings = ['utf-8', 'ISO-8859-1', 'latin1']

# If the file has been uploaded
if uploaded_file is not None:

    # Check the type of the uploaded file
    file_ext = uploaded_file.name.split('.')[-1]

    # Initialize delimiter option
    delimiter_option = st.sidebar.radio("Delimiter Option:", ("With Delimiter", "Without Delimiter", "Open CSV"))

    # Read the CSV or Excel file into a DataFrame
    if file_ext == 'csv':
        for encoding in encodings:
            try:
                if delimiter_option == "With Delimiter":
                   df = pd.read_csv(uploaded_file, encoding=encoding, delimiter=';')
                   df_copy = df.copy()

                elif delimiter_option == "Without Delimiter":
                     df = pd.read_csv(uploaded_file, encoding=encoding)
                     df_copy = df.copy()
                
                elif delimiter_option == 'Open CSV':
                     df = pd.read_csv(uploaded_file)
                     df_copy = df.copy()

                if not df.empty:
                    break  # Break the loop if reading succeeds and dataframe is not empty

                else:
                    st.warning("Empty CSV file detected.")
                    st.stop()  # Stop further execution

            except UnicodeDecodeError:
                st.error(f"Error reading CSV file with encoding {encoding}")
                continue  # Try the next encoding

            except pd.errors.ParserError as e:
                # Try reading CSV without specifying delimiter
                try:
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    df_copy = df.copy()
                    st.warning("Delimiter not specified. Attempting to read without delimiter.")
                    if not df.empty:
                        break
                    else:
                        st.warning("Empty CSV file detected.")
                        st.stop()  # Stop further execution

                except Exception as e:
                    st.error(f"Error reading CSV file: {e}")
                    st.stop()  # Stop further execution

            except Exception as e:
                st.error(f"Error reading CSV file: {e}")
                st.stop()  # Stop further execution

    elif file_ext in ['xls', 'xlsx']:
        df = pd.read_excel(uploaded_file)
        df_copy = df.copy()
        
    else:
        st.error('Your file must be in Excel or CSV format!')
        st.stop()  # Stop further execution
    
    # Display the DataFrame
    st.sidebar.write('Columns Name:')
    st.sidebar.write(df.columns.tolist())
    st.sidebar.write('Uploaded Data:')
    st.sidebar.write(df.head())

    st.title('Predict Field')
    st.write("Hello, welcome to this website. You can use it to predict your data. Enjoy using this website." 
              "Don't forget to read **guidline** that provide from us, thankyou 😊.")

    # Choose model
    model_option = st.sidebar.radio("Model Option:", ("Classification", "Regression", "Clustering"))
    
    try:
        global target_column
        # Proceed if model_option is either Classification or Regression
        if model_option in ['Classification', 'Regression']:

            # Select target column
            target_column = st.sidebar.selectbox("Select target column:", df.columns)
            
            if model_option == 'Classification':

                # Ensure target column is binary
                if len(df[target_column].unique()) != 2:
                    st.warning("For classification, the target column should have only two unique values.")
                    
                else:
                    # Proceed with train-test or train-test-validation split
                    split_option = st.sidebar.radio("Split Option:", ("Train-Test", "Train-Test-Validation")) 

                    if split_option == "Train-Test":
                        # Split data for train-test
                        train_size = st.sidebar.number_input("Train set proportion:", min_value=0.1, max_value=1.0, value=0.8, step=0.05)
                        test_size = 1 - train_size
                        
                        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[target_column]), 
                                                                            df[target_column], 
                                                                            test_size=test_size, 
                                                                            train_size=train_size, 
                                                                            random_state=42, 
                                                                            stratify=df[target_column])

                    elif split_option == "Train-Test-Validation":
                        # Split data for train-test-validation
                        train_size = st.sidebar.number_input("Train set proportion:", min_value=0.1, max_value=1.0, value=0.6, step=0.05)
                        test_size = st.sidebar.number_input("Test set proportion:", min_value=0.1, max_value=1.0, value=0.2, step=0.05)
                        valid_size = 1 - train_size - test_size
                        
                        X_train, temp, y_train, temp_y = train_test_split(df.drop(columns=[target_column]), 
                                                                        df[target_column], 
                                                                        test_size=(test_size + valid_size), 
                                                                        train_size=train_size, 
                                                                        random_state=42, 
                                                                        stratify=df[target_column])
                        
                        X_test, X_valid, y_test, y_valid = train_test_split(temp, 
                                                                            temp_y, 
                                                                            test_size=(valid_size / (test_size + valid_size)), 
                                                                            train_size=(test_size / (test_size + valid_size)), 
                                                                            random_state=42, 
                                                                            stratify=temp_y)
                        
            elif model_option == 'Regression':
                # Ensure target column is numeric or float
                if df[target_column].dtype not in ['int64', 'float64']:
                    st.warning("For regression, the target column should be numeric or float type.")

                else:
                    # Proceed with train-test or train-test-validation split
                    split_option = st.sidebar.radio("Split Option:", ("Train-Test", "Train-Test-Validation")) 

                    if split_option == "Train-Test":
                        # Split data for train-test
                        train_size = st.sidebar.number_input("Train set proportion:", min_value=0.1, max_value=1.0, value=0.8, step=0.05)
                        test_size = 1 - train_size
                        
                        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[target_column]), 
                                                                            df[target_column], 
                                                                            test_size=test_size, 
                                                                            train_size=train_size, 
                                                                            random_state=42)

                    elif split_option == "Train-Test-Validation":

                        # Split data for train-test-validation
                        train_size = st.sidebar.number_input("Train set proportion:", min_value=0.1, max_value=1.0, value=0.6, step=0.05)
                        test_size = st.sidebar.number_input("Test set proportion:", min_value=0.1, max_value=1.0, value=0.2, step=0.05)
                        valid_size = 1 - train_size - test_size
                        
                        X_train, temp, y_train, temp_y = train_test_split(df.drop(columns=[target_column]), 
                                                                          df[target_column], 
                                                                          test_size=(test_size + valid_size), 
                                                                          train_size=train_size, 
                                                                          random_state=42)
                        
                        X_test, X_valid, y_test, y_valid = train_test_split(temp, 
                                                                            temp_y, 
                                                                            test_size=(valid_size / (test_size + valid_size)), 
                                                                            train_size=(test_size / (test_size + valid_size)), 
                                                                            random_state=42)
        else:
            pass

    except:
        st.warning('Select appropriate data types for the target column.')

        if model_option == 'Classification' or model_option == 'Regression':
            try:
                # Display the splits
                st.subheader("Split Data Summary:")
                
                if split_option == "Train-Test":
                    st.text(f"X_train set size: {X_train.shape}, X_Test set size: {X_test.shape}")
                    st.text(f"y_train set size: {y_train.shape}, y_test set size: {y_test.shape}")

                    split_option_train_test = st.selectbox("Select Train-Test Data:", ("X_train", "y_train", 
                                                                                        "X_test", "y_test"),
                                                                                        key = 'split')

                    if split_option_train_test == 'X_train':
                        st.write(X_train.head())
                    elif split_option_train_test == 'X_test':
                        st.write(X_test.head())
                    elif split_option_train_test == 'y_train':
                        st.write(y_train.head())
                    elif split_option_train_test == 'y_test':
                        st.write(y_test.head())

                elif split_option == "Train-Test-Validation":
                    st.text(f"X_Train set size: {X_train.shape}, X_Test set size: {X_test.shape}, X_Valid set size: {X_valid.shape}")
                    st.text(f"y_train set size: {y_train.shape}, y_test set size: {y_test.shape}, y_valid set size: {y_valid.shape}")

                    split_option_valid = st.selectbox("Select Train-Test-Valid Data:", ("X_train", "y_train", 
                                                                                        "X_test", "y_test", 
                                                                                        "X_valid", "y_valid"),
                                                                                        key = 'split')
                    if split_option_valid == 'X_train':
                        st.write(X_train.head())
                    elif split_option_valid == 'X_test':
                        st.write(X_test.head())
                    elif split_option_valid == 'X_valid':
                        st.write(X_valid.head())
                    elif split_option_valid == 'y_train':
                        st.write(y_train.head())
                    elif split_option_valid == 'y_test':
                        st.write(y_test.head())
                    elif split_option_valid == 'y_valid':
                        st.write(y_valid.head())
            except:
                st.info('Wait until you are done selecting target data')

    # create One Hot Encoding or Label Encoding
    st.subheader("Feature Engineering:")
    st.markdown("**Encoding Data:**")
    st.write("This process will change your category predictor to a numeric. Note: Select Label Encoding if your data has values such as small, medium, or large. please preprocess it accordingly.")

    select_encoding = st.radio("Select One Hot Encoding or Label Encoding:", 
                              ('One-Hot-Encoding', 'Label Encoder'))
    
    # Get list of available object type columns
    object_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    # model classification or regression
    if model_option == 'Classification' or model_option == 'Regression':
        try:
            if select_encoding == 'One-Hot-Encoding':

                if split_option == "Train-Test":
                    X_train = pd.get_dummies(data=X_train)
                    X_test = pd.get_dummies(data=X_test)

                elif split_option == "Train-Test-Validation":
                    X_train = pd.get_dummies(data=X_train)
                    X_test = pd.get_dummies(data=X_test)
                    X_valid = pd.get_dummies(data=X_valid)

            elif select_encoding == 'Label Encoder':

                # Select predictors
                st.markdown("**Select columns to Label Encode:**")

                # Multiselect widget for columns to label encode
                columns_to_encode = st.multiselect("Select Columns Label Encode:", object_columns)

                if split_option == "Train-Test":

                    # Label encoding for selected columns
                    label_encoder = LabelEncoder()
                    X_train_encoded = X_train[columns_to_encode].apply(label_encoder.fit_transform)
                    X_test_encoded = X_test[columns_to_encode].apply(label_encoder.transform)
                    
                    # One-hot encoding for non-selected columns
                    X_train_ohe = pd.get_dummies(X_train.drop(columns=columns_to_encode))
                    X_test_ohe = pd.get_dummies(X_test.drop(columns=columns_to_encode))

                    # Concatenate label encoded and one-hot encoded columns
                    X_train = pd.concat([X_train_encoded, X_train_ohe], axis=1)
                    X_test = pd.concat([X_test_encoded, X_test_ohe], axis=1)

                elif split_option == "Train-Test-Validation":

                    # Label encoding for selected columns
                    label_encoder = LabelEncoder()
                    X_train_encoded = X_train[columns_to_encode].apply(label_encoder.fit_transform)
                    X_test_encoded = X_test[columns_to_encode].apply(label_encoder.transform)
                    X_valid_encoded = X_valid[columns_to_encode].apply(label_encoder.transform)

                    # One-hot encoding for non-selected columns
                    X_train_ohe = pd.get_dummies(X_train.drop(columns=columns_to_encode))
                    X_test_ohe = pd.get_dummies(X_test.drop(columns=columns_to_encode))
                    X_valid_ohe = pd.get_dummies(X_valid.drop(columns=columns_to_encode))

                    # Concatenate label encoded and one-hot encoded columns
                    X_train = pd.concat([X_train_encoded, X_train_ohe], axis=1)
                    X_test = pd.concat([X_test_encoded, X_test_ohe], axis=1)
                    X_valid = pd.concat([X_valid_encoded, X_valid_ohe], axis=1)
            
        except:
            st.info('Wait until you are done selecting target data')

    elif model_option == 'Clustering':
        
        # Get list of available object type columns
        object_columns = df.select_dtypes(include=['object']).columns.tolist()

        if select_encoding == 'One-Hot-Encoding':
            
            # check data shape
            st.text('Data Input Shape Before Encoding: {}'.format(df.shape))

            # change category to numeric
            df = pd.get_dummies(df)
            
            # check data shape
            st.text('Data Input Shape After Encoding: {}'.format(df.shape))

        elif select_encoding == 'Label Encoder':

            # Select predictors
            st.markdown("**Select columns to Label Encode:**")

            # Multiselect widget for columns to label encode
            columns_to_encode = st.multiselect("Select Columns Label Encode:", options=object_columns)
            
            # check data shape
            st.text('Data Input Shape Before Encoding: {}'.format(df.shape))

            # Label encoding for selected columns
            label_encoder = LabelEncoder()

            # Transform category to numeric
            df_encode = df[columns_to_encode].apply(label_encoder.fit_transform)

            # One-hot encoding for non-selected columns
            df_ohe = pd.get_dummies(df.drop(columns=columns_to_encode))

            # Concatenate label encoded and one-hot encoded columns
            df = pd.concat([df_encode, df_ohe], axis=1)
        
            # check data shape
            st.text('Data Input Shape After Encoding: {}'.format(df.shape))

    if model_option == 'Classification' or model_option == 'Regression':

        # Display the splits
        st.markdown("**Data After Encoding:**")

        try:
            if split_option == "Train-Test":

                st.text(f"X_train set size: {X_train.shape}, X_Test set size: {X_test.shape}")
                st.text(f"y_train set size: {y_train.shape}, y_test set size: {y_test.shape}")

                split_option_train_test = st.selectbox("Select Train-Test Data:", ("X_train", "y_train", 
                                                                                    "X_test", "y_test"), 
                                                                                    key = 'encoding')

                if split_option_train_test == 'X_train':
                    st.write(X_train.head())
                elif split_option_train_test == 'X_test':
                    st.write(X_test.head())
                elif split_option_train_test == 'y_train':
                    st.write(y_train.head())
                elif split_option_train_test == 'y_test':
                    st.write(y_test.head())

            elif split_option == "Train-Test-Validation":

                st.text(f"X_Train set size: {X_train.shape}, X_Test set size: {X_test.shape}, X_Valid set size: {X_valid.shape}")
                st.text(f"y_train set size: {y_train.shape}, y_test set size: {y_test.shape}, y_valid set size: {y_valid.shape}")

                split_option_valid = st.selectbox("Select Train-Test-Valid Data:", ("X_train", "y_train", 
                                                                                    "X_test", "y_test", 
                                                                                    "X_valid", "y_valid"), 
                                                                                        key = 'encoding')
                if split_option_valid == 'X_train':
                    st.write(X_train.head())
                elif split_option_valid == 'X_test':
                    st.write(X_test.head())
                elif split_option_valid == 'X_valid':
                    st.write(X_valid.head())
                elif split_option_valid == 'y_train':
                    st.write(y_train.head())
                elif split_option_valid == 'y_test':
                    st.write(y_test.head())
                elif split_option_valid == 'y_valid':
                    st.write(y_valid.head())
        
        except:
                st.info('Wait until you are done selecting target data')

    elif model_option == 'Clustering':
        
        # Display data after encoding
        st.markdown("**Data After Encoding:**")
        st.write(df)

    st.subheader('Data Scaling')

    class Scaling:
    
        def __init__(self):
            pass
        
        @staticmethod
        def standarize_data(data, scaler=None):
            """
            This function is used to standarize data using the standarize from scikit-learn.

            Parameters:
            ----------
            data : pd.DataFrame
                Input data in the form of a pd.DataFrame (e.g., X_train, X_test, X_valid).
            
            scaler : standarize, optional
                A pre-existing scaler if you have one, default is None.

            Returns:
            -------
            standarize_data : pd.DataFrame
                The normalized data.
            
            scaler : standarize
                The scaler used for normalization.
            """
            # If a scaler is not provided, create a new one
            if scaler is None:
                # Fit scaler during initialization
                scaler = StandardScaler()
                scaler.fit(data)

            # Normalize the data (transform)
            standarize_data = scaler.transform(data)
            standarize_data = pd.DataFrame(standarize_data,
                                            index=data.index,
                                            columns=data.columns)

            return standarize_data, scaler
        
        @staticmethod
        def minmax_data(data, scaler=None):
            """
            This function is used to MinMaxScaler data using the MinMaxScaler from scikit-learn.

            Parameters:
            ----------
            data : pd.DataFrame
                Input data in the form of a pd.DataFrame (e.g., X_train, X_test, X_valid).

            scaler : MinMaxScaler, optional
                A pre-existing scaler if you have one, default is None.

            Returns:
            -------
            minmax_data : pd.DataFrame
                The MinMaxScaler data.

            scaler : MinMaxScaler
                The scaler used for MinMaxScaler.
            """
            # If a scaler is not provided, create a new one
            if scaler is None:
                # Fit scaler during initialization
                scaler = MinMaxScaler()
                scaler.fit(data)

            # Normalize the data (transform)
            minmax_data = scaler.transform(data)
            minmax_data = pd.DataFrame(minmax_data,
                                        index=data.index,
                                        columns=data.columns)

            return minmax_data, scaler
        
        @staticmethod
        def robust_data(data, scaler=None):
            """
            This function is used to RobustScaler data using the RobustScaler from scikit-learn.

            Parameters:
            ----------
            data : pd.DataFrame
                Input data in the form of a pd.DataFrame (e.g., X_train, X_test, X_valid).

            scaler : RobustScaler, optional
                A pre-existing scaler if you have one, default is None.

            Returns:
            -------
            minmax_data : pd.DataFrame
                The RobustScaler data.

            scaler : RobustScaler
                The scaler used for RobustScaler.
            """
            # If a scaler is not provided, create a new one
            if scaler is None:
                # Fit scaler during initialization
                scaler = RobustScaler()
                scaler.fit(data)

            # Normalize the data (transform)
            robust_data = scaler.transform(data)
            robust_data = pd.DataFrame(robust_data,
                                        index=data.index,
                                        columns=data.columns)

            return robust_data, scaler
        
        @staticmethod
        def log_transform(data):
            """
            This function is used to log-transform data.

            Parameters:
            ----------
            data : pd.DataFrame
                Input data in the form of a pd.DataFrame (e.g., X_train, X_test, X_valid).

            Returns:
            -------
            log_data : pd.DataFrame
                The log-transformed data.
            """
            log_data = np.log1p(data)  # Use np.log1p for better handling of zeros
            return log_data
        
        @staticmethod
        def boxcox_transform(data):
            """
            This function is used to perform Box-Cox transformation on data.

            Parameters:
            ----------
            data : pd.DataFrame or 2D array
                Input data to be transformed. Each column will be transformed separately.

            Returns:
            -------
            transformed_data : pd.DataFrame or 2D array
                The transformed data.

            lambda_values : list of floats
                The lambda values used for the transformation of each column.
            """
            transformed_data = pd.DataFrame()
            lambda_values = []

            # Iterate over each column in the DataFrame
            for column in data.columns:
                # Shift the data if it contains non-positive values
                min_val = data[column].min()
                if min_val <= 0:
                    shifted_data = data[column] - min_val + 1e-6  # Add a small epsilon to avoid zero values
                else:
                    shifted_data = data[column]

                # Perform Box-Cox transformation on each column
                transformed_column, lambda_value = boxcox(shifted_data)
                transformed_data[column] = transformed_column
                lambda_values.append(lambda_value)

            return transformed_data, lambda_values

    # scaling data
    select_scaling = st.selectbox("Scaling Option:", ('---', "Standarize-Scaling", 'Robust-Scaling', 
                                                      "MinMax-Scaling", "BoxCox-Transform", 'Log-Transform'))

    # initialize scaling data
    scaling_instance = Scaling()
    
    if model_option == 'Classification' or model_option == 'Regression':
        try:
            # standarize scaling
            if select_scaling == 'Standarize-Scaling':
                if split_option == 'Train-Test':
                    # scaling data train and test
                    X_train, scaler = scaling_instance.standarize_data(X_train)
                    X_test, _ = scaling_instance.standarize_data(X_test, scaler)

                elif split_option == "Train-Test-Validation":
                    # scaling data train, test, and validation
                    X_train, scaler = scaling_instance.standarize_data(X_train)
                    X_test, _ = scaling_instance.standarize_data(X_test, scaler)
                    X_valid, _ = scaling_instance.standarize_data(X_valid, scaler)
            
            # Minmax scaling
            elif select_scaling == 'MinMax-Scaling':
                if split_option == 'Train-Test':
                    # scaling data train and test
                    X_train, scaler = scaling_instance.minmax_data(X_train)
                    X_test, scaler = scaling_instance.minmax_data(X_test, scaler)

                elif split_option == "Train-Test-Validation":
                    # scaling data train, test, and validation
                    X_train, scaler = scaling_instance.minmax_data(X_train)
                    X_test, _ = scaling_instance.minmax_data(X_test, scaler)
                    X_valid, _= scaling_instance.minmax_data(X_valid. scaler) 
            
            # Robust scaling
            elif select_scaling == 'Robust-Scaling':
                if split_option == 'Train-Test':
                    # scaling data train and test
                    X_train, scaler = scaling_instance.robust_data(X_train)
                    X_test, _ = scaling_instance.robust_data(X_test, scaler)

                elif split_option == "Train-Test-Validation":
                    # scaling data train, test, and validation
                    X_train, scaler = scaling_instance.robust_data(X_train)
                    X_test, _ = scaling_instance.robust_data(X_test, scaler)
                    X_valid, _ = scaling_instance.robust_data(X_valid, scaler) 

            # log Transform
            elif select_scaling == 'Log-Transform':
                if split_option == 'Train-Test':
                    # transform data train and test
                    X_train = scaling_instance.log_transform(X_train)
                    X_test = scaling_instance.log_transform(X_test)

                elif split_option == "Train-Test-Validation":
                    # transform data train, test, and validation
                    X_train = scaling_instance.log_transform(X_train)
                    X_test = scaling_instance.log_transform(X_test)
                    X_valid = scaling_instance.log_transform(X_valid) 
            
                # BoxCox Transform
                elif select_scaling == 'BoxCox-Transform':
                    if split_option == 'Train-Test':
                        # Transform data train and test
                        X_train, lambda_values_train = scaling_instance.boxcox_transform(X_train)
                        X_test, _ = scaling_instance.boxcox_transform(X_test, lambda_values_train)

                    elif split_option == "Train-Test-Validation":
                        # Transform data train, test, and validation
                        X_train, lambda_values_train = scaling_instance.boxcox_transform(X_train)
                        X_test, _ = scaling_instance.boxcox_transform(X_test, lambda_values_train)
                        X_valid, _ = scaling_instance.boxcox_transform(X_valid, lambda_values_train)
        
        except:
            st.info('Wait until you are done selecting target data')

    elif model_option == 'Clustering':

        # standarize scaling
        if select_scaling == 'Standarize-Scaling':
           df, scaler = scaling_instance.standarize_data(df)
        
        # Minmax scaling
        elif select_scaling == 'MinMax-Scaling': 
             df, scaler = scaling_instance.minmax_data(df)
        
        # Robust scaling
        elif select_scaling == 'Robust-Scaling':
             df, scaler = scaling_instance.robust_data(df)

        # log Transform
        elif select_scaling == 'Log-Transform':
             df = scaling_instance.log_transform(df)

        # BoxCox Transform
        elif select_scaling == 'BoxCox-Transform':
             df, scaler = scaling_instance.boxcox_transform(df)

    if model_option == 'Classification' or model_option == 'Regression':

        # Display the splits
        st.markdown("**Data After Scaling:**")
        
        try:
            if split_option == "Train-Test":

                st.text(f"X_train set size: {X_train.shape}, X_Test set size: {X_test.shape}")
                st.text(f"y_train set size: {y_train.shape}, y_test set size: {y_test.shape}")

                split_option_train_test = st.selectbox("Select Train-Test Data:", ("X_train", "y_train", 
                                                                                    "X_test", "y_test"), 
                                                                                    key = 'scaling')

                if split_option_train_test == 'X_train':
                    st.write(X_train.head())
                elif split_option_train_test == 'X_test':
                    st.write(X_test.head())
                elif split_option_train_test == 'y_train':
                    st.write(y_train.head())
                elif split_option_train_test == 'y_test':
                    st.write(y_test.head())
                
            elif split_option == "Train-Test-Validation":

                st.text(f"X_Train set size: {X_train.shape}, X_Test set size: {X_test.shape}, X_Valid set size: {X_valid.shape}")
                st.text(f"y_train set size: {y_train.shape}, y_test set size: {y_test.shape}, y_valid set size: {y_valid.shape}")

                split_option_valid = st.selectbox("Select Train-Test-Valid Data:", ("X_train", "y_train", 
                                                                                    "X_test", "y_test", 
                                                                                    "X_valid", "y_valid"), 
                                                                                    key = 'scaling')
                if split_option_valid == 'X_train':
                    st.write(X_train.head())
                elif split_option_valid == 'X_test':
                    st.write(X_test.head())
                elif split_option_valid == 'X_valid':
                    st.write(X_valid.head())
                elif split_option_valid == 'y_train':
                    st.write(y_train.head())
                elif split_option_valid == 'y_test':
                    st.write(y_test.head())
                elif split_option_valid == 'y_valid':
                    st.write(y_valid.head()) 
        
        except:
            st.info('Wait until you are done selecting target data')
    
    elif model_option == 'Clustering':
        
        # check data shape
        st.text('Data Shape After Scaling: {}'.format(df.shape))
        st.write(df)
    
    if model_option in ['Classification', 'Regression']:
        
        global selected_predictors

        # reduction predictors
        select_predictors = st.radio('**Select Predictors Reduction:**', ('None', 'Manual Select'))
        st.write('*Manual Select: if you want to just use a few predictors!*')

        # Predictors selection
        if select_predictors == 'None':
            pass
        
        # Manual select (Choose a few predictors) 
        elif select_predictors == 'Manual Select':

            if split_option == 'Train-Test' and (model_option == 'Classification' or model_option == 'Regression'):
                
                # Select predictors
                selected_predictors = st.multiselect('Select Predictors', X_train.columns)
                X_train = X_train[selected_predictors]

            elif split_option == 'Train-Test-Validation' and (model_option == 'Classification' or model_option == 'Regression'):
                
                # Select predictors
                selected_predictors = st.multiselect('Select Predictors', X_valid.columns)
                X_valid = X_valid[selected_predictors]

    else :
        pass

    def train_predict():
        """
        Function for evaluate model using data train
        """
        # Classification
        if (split_option == "Train-Test" or split_option == 'Train-Test-Validation') and model_option == 'Classification':

            try:
                # Metric score
                metrics = {'model name': [], 'Accuracy': [], 'Recall': [],
                        'Precision': [], 'F1-Score': [], 'ROC-AUC': []}
            
                # Create model baseline
                dummy_model = DummyClassifier(strategy='most_frequent')  

                # Fitting model baseline
                dummy_model.fit(X_train, y_train)

                # Predict with model baseline
                baseline_predictions = dummy_model.predict(X_train)

                # Calculate evaluatuin metrics 
                baseline_accuracy = accuracy_score(y_train, baseline_predictions) # Accuracy
                baseline_recall = recall_score(y_train, baseline_predictions) # recall
                baseline_precision = precision_score(y_train, baseline_predictions) # precision
                baseline_f1_score = f1_score(y_train, baseline_predictions) # f1_score
                baseline_roc_auc = roc_auc_score(y_train, baseline_predictions) # roc-auc

                # add metrics to dictionary
                metrics['Accuracy'].append(baseline_accuracy)
                metrics['Recall'].append(baseline_recall)
                metrics['Precision'].append(baseline_precision)
                metrics['F1-Score'].append(baseline_f1_score)
                metrics['ROC-AUC'].append(baseline_roc_auc)
                metrics['model name'].append('Baseline Model')

                # Select model
                select_models = st.multiselect('Select Model', ('Logistic Regression','K-NN', 'Decision Tree', 
                                                                'SVM','XG-Boost', 'Random Forest', 'AdaBoost'),
                                                                 key = 'train_predict')

                # classification model                                                       
                for model_name in select_models:
                    
                    # Select model K-NN
                    if 'K-NN' in model_name:
                        # Model initialization
                        model_knn = KNeighborsClassifier()

                        # fitting model
                        model_knn.fit(X_train, y_train)

                        # predict model
                        predictions = model_knn.predict(X_train)

                        # Model name
                        metrics['model name'].append('K-NN')

                        # Calculate metrics evaluation
                        accuracy = accuracy_score(y_train, predictions) # Accuarcy           
                        recall = recall_score(y_train, predictions) # Recall
                        f1_score_value = f1_score(y_train, predictions) # F1-Score
                        precision = precision_score(y_train, predictions) # Precision
                        roc_auc = roc_auc_score(y_train, predictions) # roc-auc

                        # add metrics to dictionary
                        metrics['Accuracy'].append(accuracy)
                        metrics['Recall'].append(recall)
                        metrics['F1-Score'].append(f1_score_value)
                        metrics['Precision'].append(precision)
                        metrics['ROC-AUC'].append(roc_auc)
                    
                    elif 'Decision Tree' in model_name:

                        # Model initialization
                        model_tree = DecisionTreeClassifier(max_depth=5)

                        # fitting model
                        model_tree.fit(X_train, y_train)

                        # predict model
                        predictions = model_tree.predict(X_train)

                        # Model name
                        metrics['model name'].append('Desicion-Tree')

                        # Calculate metrics evaluation
                        accuracy = accuracy_score(y_train, predictions) # Accuarcy           
                        recall = recall_score(y_train, predictions) # Recall
                        f1_score_value = f1_score(y_train, predictions) # F1-Score
                        precision = precision_score(y_train, predictions) # Precision
                        roc_auc = roc_auc_score(y_train, predictions) # roc_auc

                        # add metrics to dictionary
                        metrics['Accuracy'].append(accuracy)
                        metrics['Recall'].append(recall)
                        metrics['F1-Score'].append(f1_score_value)
                        metrics['Precision'].append(precision)
                        metrics['ROC-AUC'].append(roc_auc)  
                    
                    elif 'SVM' in model_name:

                        # Model initialization
                        model_svm = SVC(C=1.0, kernel='rbf', gamma='scale')

                        # fitting model
                        model_svm.fit(X_train, y_train)

                        # predict model
                        predictions = model_svm.predict(X_train)

                        # Model name
                        metrics['model name'].append('SVM')

                        # Ca# Calculate metrics evaluation
                        accuracy = accuracy_score(y_train, predictions) # Accuarcy           
                        recall = recall_score(y_train, predictions) # Recall
                        f1_score_value = f1_score(y_train, predictions) # F1-Score
                        precision = precision_score(y_train, predictions) # Precision
                        roc_auc = roc_auc_score(y_train, predictions) # roc_auc

                        # add metrics to dictionary
                        metrics['Accuracy'].append(accuracy)
                        metrics['Recall'].append(recall)
                        metrics['F1-Score'].append(f1_score_value)
                        metrics['Precision'].append(precision)
                        metrics['ROC-AUC'].append(roc_auc)  
                    
                    elif 'XG-Boost' in model_name:

                        # Model initialization
                        model_xg = XGBClassifier()

                        # fitting model
                        model_xg.fit(X_train, y_train)

                        # predict model
                        predictions = model_svm.predict(X_train)

                        # Model name
                        metrics['model name'].append('XG-Boost')

                        # Calculate metrics evaluation
                        accuracy = accuracy_score(y_train, predictions) # Accuarcy           
                        recall = recall_score(y_train, predictions) # Recall
                        f1_score_value = f1_score(y_train, predictions) # F1-Score
                        precision = precision_score(y_train, predictions) # Precision
                        roc_auc = roc_auc_score(y_train, predictions) # roc_auc

                        # add metrics to dictionary
                        metrics['Accuracy'].append(accuracy)
                        metrics['Recall'].append(recall)
                        metrics['F1-Score'].append(f1_score_value)
                        metrics['Precision'].append(precision)
                        metrics['ROC-AUC'].append(roc_auc)  

                    elif 'AdaBoost' in model_name:

                        # Model initialization
                        model_boost = AdaBoostClassifier()

                        # fitting model
                        model_boost.fit(X_train, y_train)

                        # predict model
                        predictions = model_boost.predict(X_train)

                        # Model name
                        metrics['model name'].append('AdaBoost')

                        # Calculate metrics evaluation
                        accuracy = accuracy_score(y_train, predictions) # Accuarcy           
                        recall = recall_score(y_train, predictions) # Recall
                        f1_score_value = f1_score(y_train, predictions) # F1-Score
                        precision = precision_score(y_train, predictions) # Precision
                        roc_auc = roc_auc_score(y_train, predictions) # roc_auc

                        # add metrics to dictionary
                        metrics['Accuracy'].append(accuracy)
                        metrics['Recall'].append(recall)
                        metrics['F1-Score'].append(f1_score_value)
                        metrics['Precision'].append(precision)
                        metrics['ROC-AUC'].append(roc_auc)    

                    elif 'Random Forest' in model_name:

                        # Model initialization
                        model_forest = RandomForestClassifier(max_depth=5)

                        # fitting model
                        model_forest.fit(X_train, y_train)

                        # predict model
                        predictions = model_forest.predict(X_train)

                        # Model name
                        metrics['model name'].append('Random-Forest')

                        # Calculate metrics evaluation
                        accuracy = accuracy_score(y_train, predictions) # Accuarcy           
                        recall = recall_score(y_train, predictions) # Recall
                        f1_score_value = f1_score(y_train, predictions) # F1-Score
                        precision = precision_score(y_train, predictions) # Precision
                        roc_auc = roc_auc_score(y_train, predictions) # roc_auc

                        # add metrics to dictionary
                        metrics['Accuracy'].append(accuracy)
                        metrics['Recall'].append(recall)
                        metrics['F1-Score'].append(f1_score_value)
                        metrics['Precision'].append(precision)
                        metrics['ROC-AUC'].append(roc_auc)  

                    elif 'Logistic Regression' in model_name:

                        # Model initialization
                        model_logit = LogisticRegression()

                        # fitting model
                        model_logit.fit(X_train, y_train)

                        # predict model
                        predictions = model_logit.predict(X_train)

                        # Model name
                        metrics['model name'].append('Logistic-Regression')

                        # Calculate metrics evaluation
                        accuracy = accuracy_score(y_train, predictions) # Accuarcy           
                        recall = recall_score(y_train, predictions) # Recall
                        f1_score_value = f1_score(y_train, predictions) # F1-Score
                        precision = precision_score(y_train, predictions) # Precision
                        roc_auc = roc_auc_score(y_train, predictions) # roc_auc

                        # add metrics to dictionary
                        metrics['Accuracy'].append(accuracy)
                        metrics['Recall'].append(recall)
                        metrics['F1-Score'].append(f1_score_value)
                        metrics['Precision'].append(precision)
                        metrics['ROC-AUC'].append(roc_auc)  

                # Change to dataframe
                metric = pd.DataFrame(metrics)
                st.write(metric)
        
            except:
                st.info('Wait until you are done selecting target data')
        
        elif (split_option == "Train-Test" or split_option == 'Train-Test-Validation') and model_option == 'Regression':
            
            try:
                metrics_regression = {'model name': [], 'MSE': [], 'MAE': [],
                                'R2': [], 'RMSE': []}
                                                        
                # Create model baseline
                dummy_model = DummyRegressor(strategy='mean') 

                # Fitting model baseline
                dummy_model.fit(X_train, y_train)

                # Predict with model baseline
                baseline_predictions = dummy_model.predict(X_train)

                # Calculate evaluatuin metrics 
                baseline_mse = mean_squared_error(y_train, baseline_predictions) # MSE
                baseline_mae = mean_absolute_error(y_train, baseline_predictions) # MAE
                baseline_rmse = np.sqrt(baseline_mse) # RMSE
                baseline_r2 = r2_score(y_train, baseline_predictions) # R2

                # add metrics to dictionary
                metrics_regression['MSE'].append(baseline_mse)
                metrics_regression['MAE'].append(baseline_mae)
                metrics_regression['RMSE'].append(baseline_rmse)
                metrics_regression['R2'].append(baseline_r2)
                metrics_regression['model name'].append('Baseline Model')

                # Select model
                select_models_regriss = st.multiselect('Select Model', ('Linear Regression','K-NN', 'Decision Tree', 
                                                                        'SVM','XG-Boost', 'Random Forest', 'AdaBoost',
                                                                        'Ridge', 'Lasso'),
                                                                         key = 'train_predict')
                # Select model Linear Regressions
                for model_name in select_models_regriss:
                    
                    if 'Linear Regression' in model_name:
                        # Model initialization
                        model_linear = LinearRegression()

                        # fitting model
                        model_linear.fit(X_train, y_train)

                        # predict model
                        predictions = model_linear.predict(X_train)
                        
                        # Calculate evaluatuin metrics 
                        mse = mean_squared_error(y_train, predictions) # MSE
                        mae = mean_absolute_error(y_train, predictions) # MAE
                        rmse = np.sqrt(mse) # RMSE
                        r2 = r2_score(y_train, predictions) # R2

                        # add metrics to dictionary
                        metrics_regression['MSE'].append(mse)
                        metrics_regression['MAE'].append(mae)
                        metrics_regression['RMSE'].append(rmse)
                        metrics_regression['R2'].append(r2)
                        metrics_regression['model name'].append('Linear Regression')

                    elif 'K-NN' in model_name:

                        # Model initialization
                        model_knn = KNeighborsRegressor()

                        # fitting model
                        model_knn.fit(X_train, y_train)

                        # predict model
                        predictions = model_knn.predict(X_train)

                        # Calculate evaluatuin metrics 
                        mse = mean_squared_error(y_train, predictions) # MSE
                        mae = mean_absolute_error(y_train, predictions) # MAE
                        rmse = np.sqrt(mse) # RMSE
                        r2 = r2_score(y_train, predictions) # R2

                        # add metrics to dictionary
                        metrics_regression['MSE'].append(mse)
                        metrics_regression['MAE'].append(mae)
                        metrics_regression['RMSE'].append(rmse)
                        metrics_regression['R2'].append(r2)
                        metrics_regression['model name'].append('K-NN')

                    elif 'Decision Tree' in model_name:

                        # Model initialization
                        model_tree = DecisionTreeRegressor()

                        # fitting model
                        model_tree.fit(X_train, y_train)

                        # predict model
                        predictions = model_tree.predict(X_train)

                        # Calculate evaluatuin metrics 
                        mse = mean_squared_error(y_train, predictions) # MSE
                        mae = mean_absolute_error(y_train, predictions) # MAE
                        rmse = np.sqrt(mse) # RMSE
                        r2 = r2_score(y_train, predictions) # R2

                        # add metrics to dictionary
                        metrics_regression['MSE'].append(mse)
                        metrics_regression['MAE'].append(mae)
                        metrics_regression['RMSE'].append(rmse)
                        metrics_regression['R2'].append(r2)
                        metrics_regression['model name'].append('Decision Tree')

                    elif 'SVM' in model_name:

                        # Model initialization
                        model_svm = SVR(C=1.0, kernel='rbf', gamma='scale')

                        # fitting model
                        model_svm.fit(X_train, y_train)

                        # predict model
                        predictions = model_svm.predict(X_train)

                        # Calculate evaluatuin metrics 
                        mse = mean_squared_error(y_train, predictions) # MSE
                        mae = mean_absolute_error(y_train, predictions) # MAE
                        rmse = np.sqrt(mse) # RMSE
                        r2 = r2_score(y_train, predictions) # R2

                        # add metrics to dictionary
                        metrics_regression['MSE'].append(mse)
                        metrics_regression['MAE'].append(mae)
                        metrics_regression['RMSE'].append(rmse)
                        metrics_regression['R2'].append(r2)
                        metrics_regression['model name'].append('SVM')

                    elif 'XG-Boost' in model_name:

                        # Model initialization
                        model_xg = XGBRegressor()

                        # fitting model
                        model_xg.fit(X_train, y_train)

                        # predict model
                        predictions = model_svm.predict(X_train)

                        # Calculate evaluatuin metrics 
                        mse = mean_squared_error(y_train, predictions) # MSE
                        mae = mean_absolute_error(y_train, predictions) # MAE
                        rmse = np.sqrt(mse) # RMSE
                        r2 = r2_score(y_train, predictions) # R2

                        # add metrics to dictionary
                        metrics_regression['MSE'].append(mse)
                        metrics_regression['MAE'].append(mae)
                        metrics_regression['RMSE'].append(rmse)
                        metrics_regression['R2'].append(r2)
                        metrics_regression['model name'].append('XG-Boost')

                    elif 'AdaBoost' in model_name:

                        # Model initialization
                        model_boost = AdaBoostRegressor()

                        # fitting model
                        model_boost.fit(X_train, y_train)

                        # predict model
                        predictions = model_boost.predict(X_train)

                        # Calculate evaluatuin metrics 
                        mse = mean_squared_error(y_train, predictions) # MSE
                        mae = mean_absolute_error(y_train, predictions) # MAE
                        rmse = np.sqrt(mse) # RMSE
                        r2 = r2_score(y_train, predictions) # R2

                        # add metrics to dictionary
                        metrics_regression['MSE'].append(mse)
                        metrics_regression['MAE'].append(mae)
                        metrics_regression['RMSE'].append(rmse)
                        metrics_regression['R2'].append(r2)
                        metrics_regression['model name'].append('AdaBoost')

                    elif 'Random Forest' in model_name:

                        # Model initialization
                        model_forest = RandomForestRegressor(max_depth=5)

                        # fitting model
                        model_forest.fit(X_train, y_train)

                        # predict model
                        predictions = model_forest.predict(X_train)

                        # Calculate evaluatuin metrics 
                        mse = mean_squared_error(y_train, predictions) # MSE
                        mae = mean_absolute_error(y_train, predictions) # MAE
                        rmse = np.sqrt(mse) # RMSE
                        r2 = r2_score(y_train, predictions) # R2

                        # add metrics to dictionary
                        metrics_regression['MSE'].append(mse)
                        metrics_regression['MAE'].append(mae)
                        metrics_regression['RMSE'].append(rmse)
                        metrics_regression['R2'].append(r2)
                        metrics_regression['model name'].append('Random Forest')

                    elif 'Ridge' in model_name:

                        # Model initialization with alpha (penalty)
                        model_ridge = Ridge()  

                        # Fitting model
                        model_ridge.fit(X_train, y_train)

                        # Predict model
                        predictions = model_ridge.predict(X_train)

                        # Calculate evaluation metrics 
                        mse = mean_squared_error(y_train, predictions) # MSE
                        mae = mean_absolute_error(y_train, predictions) # MAE
                        rmse = np.sqrt(mse) # RMSE
                        r2 = r2_score(y_train, predictions) # R2

                        # Add metrics to dictionary
                        metrics_regression['MSE'].append(mse)
                        metrics_regression['MAE'].append(mae)
                        metrics_regression['RMSE'].append(rmse)
                        metrics_regression['R2'].append(r2)
                        metrics_regression['model name'].append('Ridge Regression')

                    # Lasso Regression
                    if 'Lasso' in model_name:
                        # Model initialization with alpha (penalty)
                        model_lasso = Lasso()  # Anda dapat menyesuaikan alpha sesuai kebutuhan

                        # Fitting model
                        model_lasso.fit(X_train, y_train)

                        # Predict model
                        predictions = model_lasso.predict(X_train)

                        # Calculate evaluation metrics 
                        mse = mean_squared_error(y_train, predictions) # MSE
                        mae = mean_absolute_error(y_train, predictions) # MAE
                        rmse = np.sqrt(mse) # RMSE
                        r2 = r2_score(y_train, predictions) # R2

                        # Add metrics to dictionary
                        metrics_regression['MSE'].append(mse)
                        metrics_regression['MAE'].append(mae)
                        metrics_regression['RMSE'].append(rmse)
                        metrics_regression['R2'].append(r2)
                        metrics_regression['model name'].append('Lasso Regression')

                # Change to dataframe
                metrics_regression = pd.DataFrame(metrics_regression)
                st.write(metrics_regression)
            
            except:
                st.info('Wait until you are done selecting target data')
    
    def valid_predict():
        """
        Function for evaluate model using data train and valid
        """
        global select_models
        global select_models_regriss
        global best_params
        global best_params_regriss

        def selection_permutation(X_train, y_train, model):
            """
            Permutation for obtain contribution in each predictors

            Parameters:
            -----------
            X_train : pd.DataFrame
                Data train predictors

            y_train : Series
                Data train target
            
            Return:
            -------
            None
            """
            # Fit the data to the model
            model.fit(X_train, y_train)

            # Perform permutation feature importance
            perm_importance = permutation_importance(model, X_train, y_train)

            # Get the feature importances and indices
            feature_importances = perm_importance.importances_mean
            feature_indices = np.argsort(feature_importances)[::-1]

            # Sum of predictors columns
            k = len(X_train.columns)

            # Get the indices of the top k features
            top_k_indices = feature_indices[:k]

            # Get the column names of the top k features
            selected_feature_names = X_train.columns[top_k_indices]

            # Get the sorted feature importances
            sorted_feature_importances = feature_importances[top_k_indices]

            # Plot bar chart using st.pyplot()
            fig, ax = plt.subplots(figsize=(10,6))
            ax.bar(selected_feature_names, sorted_feature_importances, color='skyblue')
            ax.set_xlabel('')
            ax.set_ylabel('Importance Score')
            ax.set_title('Contribution Predictors to the Model')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
                
            # Display plot using st.pyplot()
            st.pyplot(fig)
        
        def selection_permutation_regressor(X_train, y_train, model):
            """
            Permutation for obtaining contribution of each predictor based on MSE

            Parameters:
            -----------
            X_train : pd.DataFrame
                Data train predictors

            y_train : Series
                Data train target
            
            model : sklearn model object
                Trained regression model
            
            Return:
            -------
            None
            """
            # Fit the data to the model
            model.fit(X_train, y_train)

            # Get predictions
            y_pred = model.predict(X_train)

            # Calculate MSE with original predictions
            mse_original = mean_squared_error(y_train, y_pred)

            # Initialize dictionary to store feature importances
            feature_importances = {}

            # Iterate through each predictor
            for col in X_train.columns:
                # Copy the original data
                X_train_permuted = X_train.copy()

                # Permute the predictor column
                X_train_permuted[col] = np.random.permutation(X_train_permuted[col])

                # Get predictions with permuted data
                y_pred_permuted = model.predict(X_train_permuted)

                # Calculate MSE with permuted predictions
                mse_permuted = mean_squared_error(y_train, y_pred_permuted)

                # Calculate feature importance based on MSE difference
                feature_importances[col] = mse_permuted - mse_original

            # Sort feature importances
            sorted_feature_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)

            # Extract top features and their importance scores
            selected_feature_names, sorted_scores = zip(*sorted_feature_importances)

            # Plot bar chart using st.pyplot()
            fig, ax = plt.subplots(figsize=(10,6))
            ax.bar(selected_feature_names, sorted_scores, color='skyblue')
            ax.set_xlabel('')
            ax.set_ylabel('Importance Score')
            ax.set_title('Contribution Predictors to the Model')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Display plot using st.pyplot()
            st.pyplot(fig)

        def cross_valid(X, y, cross_validation, model):
            """
            Retrain a model base on select model using specified or cross-validated best parameters.

            Parameters :
            ------------
            X: pd.DataFrame 
               Training data or valid data

            y : pd.Series 
              Target values.

            cross_validation : object 
                Cross-validation object to find the best parameters.
            
            model : str
                Model name from library

            Returns :
            ---------    
            best_params: list
                best_params model

            error : float
                Accuracy of the model.
            """
            # copy data to avoid data leakage
            X_copy = X.copy()

            # Find the best parameters using cross-validation
            best_rand = cross_validation.fit(X_copy, y)
            best_params = best_rand.best_params_

            # Initialize the model with the best parameters
            model_retrain = model(**best_params)

            # Retrain the model using training data
            model_retrain.fit(X, y)

            # Model prediction
            y_pred = model_retrain.predict(X)

            # Calculate accuracy 
            accuracy = accuracy_score(y, y_pred)

            return accuracy, best_params

        def re_train(X, y, best_params, model, model_name):
            """
                Retrain a model with the specified parameters and 
            calculate evaluation metrics and feature importance.

            Parameters:
            -----------
            X : array-like or pandas DataFrame
                Features dataset for training.

            y : array-like or pandas Series
                Target variable for training.

            best_params : dict
                Best parameters for the model.

            model : class
                Model class to be used for training.

            model_name : str
                Name of the model.

            Returns:
            --------
            metrics_eval : pd.DataFrame
                DataFrame containing evaluation metrics (accuracy, recall, precision, F1-score).
            
            """
            # Initialize dictionary to store evaluation metrics
            re_train_metrics = {
                'Model Name': [],
                'Accuracy': [],
                'Recall': [],
                'Precision': [],
                'F1-score': [],
                'ROC-AUC': []
            }

            # Initialize and train the model with the best parameters
            model_instance = model(**best_params)
            model_instance.fit(X, y)

            # Predictions using the trained model
            predictions = model_instance.predict(X)
            
            # Feature Importance
            selection_permutation(X, y, model_instance)

            # Calculate evaluation metrics
            accuracy = accuracy_score(y, predictions) # Accuracy
            recall = recall_score(y, predictions) # Recall
            precision = precision_score(y, predictions) # Precision
            f1_score_value = f1_score(y, predictions) # F1-Score
            roc_auc = f1_score(y, predictions) # Roc-auc

            # Add metrics to dictionary
            re_train_metrics['Model Name'].append(model_name)
            re_train_metrics['Accuracy'].append(accuracy)
            re_train_metrics['Recall'].append(recall)
            re_train_metrics['Precision'].append(precision)
            re_train_metrics['F1-score'].append(f1_score_value)
            re_train_metrics['ROC-AUC'].append(roc_auc)

            # Convert metrics dictionary to DataFrame
            metrics_eval = pd.DataFrame(re_train_metrics)
            
            return metrics_eval

        def cross_valid_regressor(X, y, cross_validation, model):
            """
            Retrain a model based on selected model using specified or cross-validated best parameters.

            Parameters :
            ------------
            X: pd.DataFrame 
            Training data or validation data

            y : pd.Series 
            Target values.

            cross_validation : object 
                Cross-validation object to find the best parameters.
            
            model : class
                Model class from library

            Returns :
            ---------    
            best_params : dict
                Best parameters of the model.

            r2 : float
                R-squared value indicating the model performance.
            """
            # copy data to avoid data leakage
            X_copy = X.copy()

            # Find the best parameters using cross-validation
            cross_validation.fit(X_copy, y)
            best_params = cross_validation.best_params_

            # Initialize the model with the best parameters
            model_retrain = model(**best_params)

            # Retrain the model using training data
            model_retrain.fit(X, y)

            # Model prediction
            y_pred = model_retrain.predict(X)

            # Calculate R-squared
            r2 = r2_score(y, y_pred)

            return r2, best_params

        def re_train_regressor(X, y, best_params, model, model_name):
            """
            Retrain a model with the specified parameters and calculate evaluation metrics.

            Parameters:
            -----------
            X : array-like or pandas DataFrame
                Features dataset for training.

            y : array-like or pandas Series
                Target variable for training.

            best_params : dict
                Best parameters for the model.

            model : class
                Model class to be used for training.

            model_name : str
                Name of the model.

            Returns:
            --------
            metrics_eval : pd.DataFrame
                DataFrame containing evaluation metrics (MSE, MAE, RMSE, R2).
            """
            # Initialize dictionary to store evaluation metrics
            re_train_metrics = {
                'Model Name': [],
                'MSE': [],
                'MAE': [],
                'RMSE': [],
                'R2': []
            }

            # Initialize and train the model with the best parameters
            model_instance = model(**best_params)
            model_instance.fit(X, y)

            # Predictions using the trained model
            predictions = model_instance.predict(X)
            
            # Feature Importance
            selection_permutation_regressor(X, y, model_instance)

            # Calculate evaluation metrics
            mse = mean_squared_error(y, predictions)
            mae = mean_absolute_error(y, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, predictions)

            # Add metrics to dictionary
            re_train_metrics['Model Name'].append(model_name)
            re_train_metrics['MSE'].append(mse)
            re_train_metrics['MAE'].append(mae)
            re_train_metrics['RMSE'].append(rmse)
            re_train_metrics['R2'].append(r2)

            # Convert metrics dictionary to DataFrame
            metrics_eval = pd.DataFrame(re_train_metrics)
            
            return metrics_eval

        class Model_Classification:

            def __init__(self, X, y):
                self.X = X
                self.y = y

            def Decision_Tree(self):
                """
                Re-train Decision Tree Model

                Parameters:
                -----------
                    None

                Returns:
                --------
                best_params : list
                    Best params Model
                """
                # cross validation
                skf = KFold(n_splits=5)

                # model
                model_tree = DecisionTreeClassifier() 

                # Define the parameter grid for Decision Tree
                params_grid = {
                                'criterion': ['gini', 'entropy'],  
                                'splitter': ['best', 'random'],  
                                'max_depth': [None, 10, 20, 30],  
                                'min_samples_split': [2, 5, 10],  
                                'min_samples_leaf': [1, 2, 4],  
                                'max_features': ['auto', 'sqrt', 'log2', None]  
                                }

                random_search = RandomizedSearchCV(
                                                    model_tree,
                                                    param_distributions=params_grid,
                                                    n_iter=10,
                                                    cv=skf,
                                                    scoring='accuracy',
                                                    random_state=42
                                                  )
                
                accuracy_valid, best_params = cross_valid(X=self.X, 
                                                          y=self.y, 
                                                          cross_validation=random_search, 
                                                          model=DecisionTreeClassifier)

                # Valid cross validation
                st.write('**Accuracy Model from Cross-Validation:**', accuracy_valid)
            
                # re-train model
                metrics_eval = re_train(self.X, 
                                        self.y, 
                                        best_params, 
                                        model=DecisionTreeClassifier, 
                                        model_name='Decision-Tree')
                
                st.write('**Re-Train Evaluation**', metrics_eval)

                return best_params
        
            def knn_model(self):
                """
                Re-train K-NN Model

                Parameters:
                -----------
                    None

                Returns:
                --------
                best_params : list
                    Best params Model
                """
                # cross validation
                skf = KFold(n_splits=5)
                
                # model
                model_knn = KNeighborsClassifier() 

                # Determine parameter grid for K-NN
                params_grid = {
                    'n_neighbors': [3, 5, 7],  
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  
                    'leaf_size': [10, 20],
                    'p': [1, 2]
                }

                random_search = RandomizedSearchCV(
                                                    model_knn,
                                                    param_distributions = params_grid,
                                                    n_iter = 10,
                                                    cv = skf,
                                                    scoring = 'accuracy',
                                                    random_state = 42
                                                )
                
                accuracy_valid, best_params = cross_valid(X= self.X, 
                                                          y= self.y,   
                                                          cross_validation= random_search, 
                                                          model= KNeighborsClassifier)

                # Valid cross validation
                st.write('**Accuracy Model from Cross-Validation:**', accuracy_valid)
                
                # re-train model
                metrics_eval = re_train(self.X, 
                                        self.y, 
                                        best_params, 
                                        model = KNeighborsClassifier, 
                                        model_name = 'K-NN')
                
                st.write('**Re-Train Evaluation:**', metrics_eval)
                
                return best_params
            
            def svm(self):
                """
                Re-train SVM Model

                Parameters:
                -----------
                    None

                Returns:
                --------
                best_params : list
                    Best params Model
                """
                # cross validation
                skf = KFold(n_splits=5)
                
                # Model
                model_svm = SVC() 

                # Define parameter grid for SVM
                params_grid = {
                    'C': [0.1, 1, 10],  
                    'kernel': ['linear', 'rbf'],  
                    'gamma': ['scale', 'auto'],  
                    'degree': [2, 3, 4]  
                }

                random_search = RandomizedSearchCV(
                                                    model_svm,
                                                    param_distributions = params_grid,
                                                    n_iter = 10,
                                                    cv = skf,
                                                    scoring = 'accuracy',
                                                    random_state = 42
                                                )
                
                accuracy_valid, best_params = cross_valid(X= self.X, 
                                                          y= self.y,   
                                                          cross_validation= random_search, 
                                                          model= SVC)

                # Valid cross validation
                st.write('**Accuracy Model from Cross-Validation:**', accuracy_valid)
                
                # re-train model
                metrics_eval = re_train(self.X, 
                                        self.y, 
                                        best_params, 
                                        model = SVC, 
                                        model_name = 'SVM')
                
                st.write('**Re-Train Evaluation:**', metrics_eval)
                
                return best_params

            def random_forest(self):
                """
                Re-train Random Forest Model

                Parameters:
                -----------
                    None

                Returns:
                --------
                best_params : list
                    Best params Model
                """
                # cross validation
                skf = KFold(n_splits=5)

                # Model
                model_rf = RandomForestClassifier() 

                # Define parameter grid for Random Forest
                params_grid = {
                    'n_estimators': [50, 100, 200], 
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20],  
                    'min_samples_split': [2, 5],  
                    'min_samples_leaf': [1, 2], 
                    'max_features': ['auto', 'sqrt']  
                }

                random_search = RandomizedSearchCV(
                                                    model_rf,
                                                    param_distributions=params_grid,
                                                    n_iter=5,  
                                                    cv=skf,
                                                    scoring='accuracy',
                                                    n_jobs=-1,  
                                                    random_state=42
                                                )

                accuracy_valid, best_params = cross_valid(X=self.X, 
                                                          y=self.y, 
                                                          cross_validation=random_search, 
                                                          model=RandomForestClassifier)

                # Validasi cross
                st.write('**Accuracy Model from Cross-Validation:**', accuracy_valid)
                
                # Re-train model
                metrics_eval = re_train(self.X, 
                                        self.y, 
                                        best_params, 
                                        model=RandomForestClassifier, 
                                        model_name='Random Forest')
                
                st.write('**Re-Train Evaluation**', metrics_eval)

                return best_params
            
            def xg_boost(self):
                """
                Re-train Xg-Boost Model

                Parameters:
                -----------
                    None

                Returns:
                --------
                best_params : list
                    Best params Model
                """
                # cross validation
                skf = KFold(n_splits=5)

                # Model
                model_xgboost = XGBClassifier() 

                # Define parameter grid for XGBoost
                params_grid = {
                    'n_estimators': [50, 100, 200],  
                    'learning_rate': [0.01, 0.1, 1.0],  
                    'max_depth': [3, 5, 7],  
                    'subsample': [0.5, 0.8, 1.0],  
                    'colsample_bytree': [0.5, 0.8, 1.0],  
                    'gamma': [0, 0.1, 0.2],  
                    'reg_alpha': [0, 0.1, 0.5],  
                    'reg_lambda': [0, 0.1, 0.5]  
                }

                random_search = RandomizedSearchCV(
                                                    model_xgboost,
                                                    param_distributions=params_grid,
                                                    n_iter=5,  
                                                    cv=skf,
                                                    scoring='accuracy',
                                                    n_jobs=-1,  
                                                    random_state=42
                                                )

                accuracy_valid, best_params = cross_valid(X=self.X, 
                                                          y=self.y, 
                                                          cross_validation=random_search, 
                                                          model=XGBClassifier)

                # Validasi cross
                st.write('**Accuracy Model from Cross-Validation:**', accuracy_valid)
                
                # Re-train model
                metrics_eval = re_train(self.X, 
                                        self.y, 
                                        best_params, 
                                        model=XGBClassifier, 
                                        model_name='XGBoost')
                
                st.write('**Re-Train Evaluation**', metrics_eval)

                return best_params
            
            def adaboost(self):
                """
                Re-train Adaboost Model

                Parameters:
                -----------
                    None

                Returns:
                --------
                best_params : list
                    Best params Model
                """
                # cross validation
                skf = KFold(n_splits=5)

                # Model
                model_adaboost = AdaBoostClassifier() 

                # Define parameter grid for AdaBoost
                params_grid = {
                    'n_estimators': [50, 100, 200],  
                    'learning_rate': [0.01, 0.1, 1.0],  
                    'algorithm': ['SAMME', 'SAMME.R']  
                }

                random_search = RandomizedSearchCV(
                    model_adaboost,
                    param_distributions=params_grid,
                    n_iter=5,  
                    cv=skf,
                    scoring='accuracy',
                    n_jobs=-1,  
                    random_state=42
                )

                accuracy_valid, best_params = cross_valid(X=self.X, 
                                                          y=self.y, 
                                                          cross_validation=random_search, 
                                                          model=AdaBoostClassifier)

                # Validasi cross
                st.write('**Accuracy Model from Cross-Validation:**', accuracy_valid)
                
                # Re-train model
                metrics_eval = re_train(self.X, 
                                        self.y, 
                                        best_params, 
                                        model=AdaBoostClassifier, 
                                        model_name='AdaBoost')
                
                st.write('**Re-Train Evaluation**', metrics_eval)

                return best_params
            
            def logit(self):
                """
                Re-train Logistic Regression Model

                Parameters:
                -----------
                    None

                Returns:
                --------
                best_params : list
                    Best params Model
                """
                # cross validation
                skf = KFold(n_splits=5)

                # Define the logistic regression model
                model_logistic = LogisticRegression()

                # Define the parameter grid for logistic regression
                params_grid = {
                    'penalty': ['l1', 'l2', 'elasticnet'],
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                    'max_iter': [100, 200, 300, 400, 500]
                } 

                # Initialize RandomizedSearchCV
                random_search = RandomizedSearchCV(
                                                    model_logistic,
                                                    param_distributions=params_grid,
                                                    n_iter=10,
                                                    cv=skf,
                                                    scoring='accuracy',
                                                    random_state=42
                                                )

                # Perform cross-validation
                accuracy_valid, best_params = cross_valid(X=self.X,
                                                          y=self.y,
                                                          cross_validation=random_search,
                                                          model=LogisticRegression)

                # Print cross-validation accuracy
                st.write('**Accuracy Model from Cross-Validation:**', accuracy_valid)

                # Retrain the model
                metrics_eval = re_train(self.X,
                                        self.y,
                                        best_params,
                                        model=LogisticRegression,
                                        model_name='Logistic-Regression')

                # Print accuracy after retraining
                st.write('**Re-Train Evaluation**', metrics_eval)

                return best_params
        
        class Model_Regressor(Model_Classification):
            def __init__(self, X, y):
                super().__init__(X, y)

            def decision_tree_regressor(self):
                """
                Re-train Decision Tree Model

                Parameters:
                -----------
                    None

                Returns:
                --------
                best_params : list
                    Best params Model
                """
                # cross validation
                skf = KFold(n_splits=5)

                # model
                model_tree = DecisionTreeRegressor() 

                # Define the parameter grid for Decision Tree
                params_grid = {
                    'criterion': ['mse', 'friedman_mse', 'mae'], 
                    'splitter': ['best', 'random'],  
                    'max_depth': [None, 10, 20, 30],  
                    'min_samples_split': [2, 5, 10],  
                    'min_samples_leaf': [1, 2, 4],  
                    'max_features': ['auto', 'sqrt', 'log2', None]  
                }

                random_search = RandomizedSearchCV(
                                                    model_tree,
                                                    param_distributions=params_grid,
                                                    n_iter=10,
                                                    cv=skf,
                                                    scoring='r2',
                                                    random_state=42
                                                  )
                
                r2_valid, best_params = cross_valid_regressor(X=self.X, 
                                                              y=self.y, 
                                                              cross_validation=random_search, 
                                                              model=DecisionTreeRegressor)

                # Valid cross validation
                st.write('**R2 Model from Cross-Validation:**', r2_valid)
            
                # re-train model
                metrics_eval = re_train_regressor(self.X, 
                                                  self.y, 
                                                  best_params, 
                                                  model=DecisionTreeRegressor, 
                                                  model_name='Decision-Tree')
                
                st.write('**Re-Train Evaluation**', metrics_eval)

                return best_params
            
            def knn_model_regressor(self):
                """
                Re-train K-NN Model

                Parameters:
                -----------
                    None

                Returns:
                --------
                best_params : list
                    Best params Model
                """
                # cross validation
                skf = KFold(n_splits=5)
                
                # model
                model_knn = KNeighborsRegressor() 

                # Determine parameter grid for K-NN
                params_grid = {
                    'n_neighbors': [3, 5, 7],  
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  
                    'leaf_size': [10, 20],
                    'p': [1, 2]
                }

                random_search = RandomizedSearchCV(
                                                    model_knn,
                                                    param_distributions = params_grid,
                                                    n_iter = 10,
                                                    cv = skf,
                                                    scoring = 'r2',
                                                    random_state = 42
                                                )
                
                r2_valid, best_params = cross_valid_regressor(X= self.X, 
                                                              y= self.y,   
                                                              cross_validation= random_search, 
                                                              model= KNeighborsRegressor)

                # Valid cross validation
                st.write('**R2 Model from Cross-Validation:**', r2_valid)
                
                # re-train model
                metrics_eval = re_train_regressor(self.X, 
                                                  self.y, 
                                                  best_params, 
                                                  model = KNeighborsClassifier, 
                                                  model_name = 'K-NN')
                            
                st.write('**Re-Train Evaluation:**', metrics_eval)
                
                return best_params
            
            def linear_regression(self):
                """
                Re-train Linear Regression Model

                Parameters:
                -----------
                    None

                Returns:
                --------
                best_params : list
                    Best params Model
                """
                # No hyperparameters to tune for Linear Regression
                
                # Initialize and train the model
                model_lr = LinearRegression()
                model_lr.fit(self.X, self.y)

                # Predictions using the trained model
                predictions = model_lr.predict(self.X)

                # Calculate evaluation metrics
                mse = mean_squared_error(self.y, predictions)
                mae = mean_absolute_error(self.y, predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(self.y, predictions)

                # Convert metrics to DataFrame
                metrics_eval = pd.DataFrame({
                                            'Model Name': ['Linear Regression'],
                                            'MSE': [mse],
                                            'MAE': [mae],
                                            'RMSE': [rmse],
                                            'R2': [r2]
                                        })
                
                st.write('**Re-Train Evaluation:**', metrics_eval)

                selection_permutation_regressor(X_train, y_train, model = model_lr)
                
                return None 
        
            def svm_regressor(self):
                """
                Re-train Support Vector Machine (SVM) Model

                Parameters:
                -----------
                    None

                Returns:
                --------
                best_params : list
                    Best params Model
                """
                # cross validation
                skf = KFold(n_splits=5)
                
                # model
                model_svm = SVR() 

                # Define the parameter grid for SVM
                params_grid = {
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto']
                }

                random_search = RandomizedSearchCV(
                                                    model_svm,
                                                    param_distributions=params_grid,
                                                    n_iter=10,
                                                    cv=skf,
                                                    scoring='r2',
                                                    random_state=42
                                                )
                
                r2_valid, best_params = cross_valid_regressor(X=self.X, 
                                                              y=self.y, 
                                                              cross_validation=random_search, 
                                                              model=SVR)

                # Valid cross validation
                st.write('**R2 Model from Cross-Validation:**', r2_valid)
                
                # re-train model
                metrics_eval = re_train_regressor(self.X, 
                                                 self.y, 
                                                 best_params, 
                                                 model=SVR, 
                                                 model_name='SVM')
                            
                st.write('**Re-Train Evaluation:**', metrics_eval)
                
                return best_params

            def xgboost_regressor(self):
                """
                Re-train XGBoost Model

                Parameters:
                -----------
                    None

                Returns:
                --------
                best_params : list
                    Best params Model
                """
                # cross validation
                skf = KFold(n_splits=5)
                
                # model
                model_xgb = XGBRegressor() 

                # Define the parameter grid for XGBoost
                params_grid = {
                    'n_estimators': [100, 500, 1000],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.5, 0.7, 1.0],
                    'colsample_bytree': [0.5, 0.7, 1.0],
                    'gamma': [0, 0.1, 0.2],
                    'reg_alpha': [0, 0.1, 0.5],
                    'reg_lambda': [0, 0.1, 0.5]
                }

                random_search = RandomizedSearchCV(
                                                    model_xgb,
                                                    param_distributions=params_grid,
                                                    n_iter=10,
                                                    cv=skf,
                                                    scoring='r2',
                                                    random_state=42
                                                )
                
                r2_valid, best_params = cross_valid_regressor(X=self.X, 
                                                            y=self.y, 
                                                            cross_validation=random_search, 
                                                            model=XGBRegressor)

                # Valid cross validation
                st.write('**R2 Model from Cross-Validation:**', r2_valid)
                
                # re-train model
                metrics_eval = re_train_regressor(self.X, 
                                                  self.y, 
                                                  best_params, 
                                                  model=XGBRegressor, 
                                                  model_name='XGBoost')
                            
                st.write('**Re-Train Evaluation:**', metrics_eval)
                
                return best_params

            def random_forest_regressor(self):
                """
                Re-train Random Forest Model

                Parameters:
                -----------
                    None

                Returns:
                --------
                best_params : list
                    Best params Model
                """
                # cross validation
                skf = KFold(n_splits=5)
                
                # model
                model_rf = RandomForestRegressor() 

                # Define the parameter grid for Random Forest
                params_grid = {
                    'n_estimators': [100, 500, 1000],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['auto', 'sqrt', 'log2']
                }

                random_search = RandomizedSearchCV(
                                                    model_rf,
                                                    param_distributions=params_grid,
                                                    n_iter=10,
                                                    cv=skf,
                                                    scoring='r2',
                                                    random_state=42
                                                )
                
                r2_valid, best_params = cross_valid_regressor(X=self.X, 
                                                              y=self.y, 
                                                              cross_validation=random_search, 
                                                              model=RandomForestRegressor)

                # Valid cross validation
                st.write('**R2 Model from Cross-Validation:**', r2_valid)
                
                # re-train model
                metrics_eval = re_train_regressor(self.X, 
                                                  self.y, 
                                                  best_params, 
                                                  model=RandomForestRegressor, 
                                                  model_name='Random Forest')
                            
                st.write('**Re-Train Evaluation:**', metrics_eval)
                
                return best_params

            def adaboost_regressor(self):
                """
                Re-train AdaBoost Model

                Parameters:
                -----------
                    None

                Returns:
                --------
                best_params : list
                    Best params Model
                """
                # cross validation
                skf = KFold(n_splits=5)
                
                # model
                model_adaboost = AdaBoostRegressor() 

                # Define the parameter grid for AdaBoost
                params_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.5, 1.0]
                }

                random_search = RandomizedSearchCV(
                                                    model_adaboost,
                                                    param_distributions=params_grid,
                                                    n_iter=10,
                                                    cv=skf,
                                                    scoring='r2',
                                                    random_state=42
                                                )
                
                r2_valid, best_params = cross_valid_regressor(X=self.X, 
                                                              y=self.y, 
                                                              cross_validation=random_search, 
                                                              model=AdaBoostRegressor)

                # Valid cross validation
                st.write('**R2 Model from Cross-Validation:**', r2_valid)
                
                # re-train model
                metrics_eval = re_train_regressor(self.X, 
                                                  self.y, 
                                                  best_params, 
                                                  model=AdaBoostRegressor, 
                                                  model_name='AdaBoost')
                            
                st.write('**Re-Train Evaluation:**', metrics_eval)
                
                return best_params

            def ridge_regressor(self):
                """
                Re-train Ridge Model

                Parameters:
                -----------
                    None

                Returns:
                --------
                best_params : list
                    Best params Model
                """
                # cross validation
                skf = KFold(n_splits=5)
                
                # model
                model_ridge = Ridge() 

                # Define the parameter grid for Ridge
                params_grid = {
                    'alpha': [0.1, 1, 10, 100]
                }

                random_search = RandomizedSearchCV(
                                                    model_ridge,
                                                    param_distributions=params_grid,
                                                    n_iter=10,
                                                    cv=skf,
                                                    scoring='r2',
                                                    random_state=42
                                                )
                
                r2_valid, best_params = cross_valid_regressor(X=self.X, 
                                                              y=self.y, 
                                                              cross_validation=random_search, 
                                                              model=Ridge)

                # Valid cross validation
                st.write('**R2 Model from Cross-Validation:**', r2_valid)
                
                # re-train model
                metrics_eval = re_train_regressor(self.X, 
                                                  self.y, 
                                                  best_params, 
                                                  model=Ridge, 
                                                  model_name='Ridge')
                            
                st.write('**Re-Train Evaluation:**', metrics_eval)
                
                return best_params

            def lasso_regressor(self):
                """
                Re-train Lasso Model

                Parameters:
                -----------
                    None

                Returns:
                --------
                best_params : list
                    Best params Model
                """
                # cross validation
                skf = KFold(n_splits=5)
                
                # model
                model_lasso = Lasso() 

                # Define the parameter grid for Lasso
                params_grid = {
                    'alpha': [0.1, 1, 10, 100]
                }

                random_search = RandomizedSearchCV(
                                                    model_lasso,
                                                    param_distributions=params_grid,
                                                    n_iter=10,
                                                    cv=skf,
                                                    scoring='r2',
                                                    random_state=42
                                                )
                
                r2_valid, best_params = cross_valid_regressor(X=self.X, 
                                                             y=self.y, 
                                                             cross_validation=random_search, 
                                                             model=Lasso)

                # Valid cross validation
                st.write('**R2 Model from Cross-Validation:**', r2_valid)
                
                # re-train model
                metrics_eval = re_train_regressor(self.X, 
                                                  self.y, 
                                                  best_params, 
                                                  model=Lasso, 
                                                  model_name='Lasso')
                            
                st.write('**Re-Train Evaluation:**', metrics_eval)
                
                return best_params

        # select model
        if model_option == 'Classification':
            try:
                # Select model classification
                select_models = st.selectbox('Select Model', ('---', 'Logistic Regression', 'K-NN', 'Decision Tree', 
                                                              'SVM', 'XG-Boost', 'Random Forest', 'AdaBoost'),
                                                               key='valid_predict')

                if split_option == 'Train-Test': 
                
                    if 'Decision Tree' in select_models: 

                        # Instance model classification                    
                        model = Model_Classification(X_train, y_train)

                        # Best params
                        best_params = model.Decision_Tree()
                
                    elif 'K-NN' in select_models: 
                        
                        # Instance model classification  
                        model = Model_Classification(X_train, y_train)

                        # Best params
                        best_params = model.knn_model()
                    
                    elif 'SVM' in select_models: 
                        
                        # Instance model classification  
                        model = Model_Classification(X_train, y_train)

                        # Best params
                        best_params = model.svm()

                    elif 'Random Forest' in select_models: 
                        
                        # Instance model classification  
                        model = Model_Classification(X_train, y_train)

                        # Best params
                        best_params = model.random_forest()
                        
                    elif 'XG-Boost' in select_models: 
                        
                        # Instance model classification  
                        model = Model_Classification(X_train, y_train)

                        # Best params
                        best_params = model.xg_boost()
                        
                    elif 'AdaBoost' in select_models: 
                        
                        # Instance model classification  
                        model = Model_Classification(X_train, y_train)

                        # Best params
                        best_params = model.adaboost()
                        
                    elif 'Logistic Regression' in select_models: 
                        
                        # Instance model classification  
                        model = Model_Classification(X_train, y_train)

                        # Best params
                        best_params = model.logit()
                
                # select type split data and model
                if split_option == 'Train-Test-Validation': 
                    
                    if 'Decision Tree' in select_models: 

                        # Instance model classification                    
                        model = Model_Classification(X_train, y_train)

                        # Best params
                        best_params = model.Decision_Tree()
                
                    elif 'K-NN' in select_models: 
                        
                        # Instance model classification  
                        model = Model_Classification(X_train, y_train)

                        # Best params
                        best_params = model.knn_model()
                    
                    elif 'SVM' in select_models: 
                        
                        # Instance model classification  
                        model = Model_Classification(X_train, y_train)

                        # Best params
                        best_params = model.svm()

                    elif 'Random Forest' in select_models: 
                        
                        # Instance model classification  
                        model = Model_Classification(X_train, y_train)

                        # Best params
                        best_params = model.random_forest()
                        
                    elif 'XG-Boost' in select_models: 
                        
                        # Instance model classification  
                        model = Model_Classification(X_train, y_train)

                        # Best params
                        best_params = model.xg_boost()
                        
                    elif 'AdaBoost' in select_models: 
                        
                        # Instance model classification  
                        model = Model_Classification(X_train, y_train)

                        # Best params
                        best_params = model.adaboost()
                        
                    elif 'Logistic Regression' in select_models: 
                        
                        # Instance model classification  
                        model = Model_Classification(X_train, y_train)

                        # Best params
                        best_params = model.logit()
            
            except:
                st.info('Wait until you are done selecting target data')
        
        elif model_option == 'Regression':
            
            try:

                # Select model regression
                select_models_regriss = st.selectbox('Select Model', ('---', 'Linear Regression','K-NN', 'Decision Tree', 
                                                                      'SVM','XG-Boost', 'Random Forest', 'AdaBoost',
                                                                      'Ridge', 'Lasso'),
                                                                       key = 'valid_predict_regressor')
                
                if split_option == 'Train-Test': 
                
                    if 'Decision Tree' in select_models_regriss: 

                        # Instance model Regression                    
                        model = Model_Regressor(X_train, y_train)

                        # Best params
                        best_params_regriss = model.decision_tree_regressor()
                    
                    elif 'Linear Regression' in select_models_regriss: 

                        # Instance model Regression                    
                        model = Model_Regressor(X_train, y_train)

                        # Best params
                        best_params_regriss = model.linear_regression()
                
                    elif 'K-NN' in select_models_regriss: 
                        
                        # Instance model Regression  
                        model = Model_Regressor(X_train, y_train)

                        # Best params
                        best_params_regriss = model.knn_model_regressor()
                    
                    elif 'SVM' in select_models_regriss: 
                        
                        # Instance model Regression  
                        model = Model_Regressor(X_train, y_train)

                        # Best params
                        best_params_regriss = model.svm_regressor()

                    elif 'Random Forest' in select_models_regriss: 
                        
                        # Instance model Regression  
                        model = Model_Regressor(X_train, y_train)

                        # Best params
                        best_params_regriss = model.random_forest_regressor()
                        
                    elif 'XG-Boost' in select_models_regriss: 
                        
                        # Instance model Regression  
                        model = Model_Regressor(X_train, y_train)

                        # Best params
                        best_params_regriss = model.xgboost_regressor()
                        
                    elif 'AdaBoost' in select_models_regriss: 
                        
                        # Instance model Regression  
                        model = Model_Regressor(X_train, y_train)

                        # Best params
                        best_params_regriss = model.adaboost_regressor()
                        
                    elif 'Lasso' in select_models_regriss: 
                        
                        # Instance model Regression  
                        model = Model_Regressor(X_train, y_train)

                        # Best params
                        best_params_regriss = model.lasso_regressor()
                    
                    elif 'Ridge' in select_models_regriss: 
                        
                        # Instance model Regression  
                        model = Model_Regressor(X_train, y_train)

                        # Best params
                        best_params_regriss = model.ridge_regressor()
                
                elif split_option == 'Train-Test-Validation': 
                
                    if 'Decision Tree' in select_models_regriss: 

                        # Instance model Regression                    
                        model = Model_Regressor(X_valid, y_valid)

                        # Best params
                        best_params_regriss = model.decision_tree_regressor()

                    elif 'Linear Regression' in select_models_regriss: 

                        # Instance model Regression                    
                        model = Model_Regressor(X_valid, y_valid)

                        # Best params
                        best_params_regriss = model.linear_regression()

                    elif 'K-NN' in select_models_regriss: 
                        
                        # Instance model Regression  
                        model = Model_Regressor(X_valid, y_valid)

                        # Best params
                        best_params_regriss = model.knn_model_regressor()
                    
                    elif 'SVM' in select_models_regriss: 
                        
                        # Instance model Regression  
                        model = Model_Regressor(X_valid, y_valid)

                        # Best params
                        best_params_regriss = model.svm_regressor()

                    elif 'Random Forest' in select_models_regriss: 
                        
                        # Instance model Regression  
                        model = Model_Regressor(X_valid, y_valid)

                        # Best params
                        best_params_regriss = model.random_forest_regressor()
                        
                    elif 'XG-Boost' in select_models_regriss: 
                        
                        # Instance model Regression  
                        model = Model_Regressor(X_valid, y_valid)

                        # Best params
                        best_params_regriss = model.xgboost_regressor()
                        
                    elif 'AdaBoost' in select_models_regriss: 
                        
                        # Instance model Regression  
                        model = Model_Regressor(X_valid, y_valid)

                        # Best params
                        best_params_regriss = model.adaboost_regressor()
                        
                    elif 'Lasso' in select_models_regriss: 
                        
                        # Instance model Regression  
                        model = Model_Regressor(X_valid, y_valid)

                        # Best params
                        best_params_regriss = model.lasso_regressor()
                    
                    elif 'Ridge' in select_models_regriss: 
                        
                        # Instance model Regression  
                        model = Model_Regressor(X_valid, y_valid)

                        # Best params
                        best_params_regriss = model.ridge_regressor()
            
            except:
                st.info('Wait until you are done selecting target data')
        
    def testing():
        """Function for evaluation of the test model"""

        # dicst for metrics
        test_metrics = {
                        'Model Name': [],
                        'Accuracy': [],
                        'Recall': [],
                        'Precision': [],
                        'F1-score': [],
                        'ROC-AUC': []
                    }
                
        # dicst for metrics
        test_metrics_regris = {
                               'Model Name': [],
                               'MSE': [],
                               'MAE': [],
                               'RMSE': [],
                               'R2': []
                        }
        
        if (split_option == 'Train-Test' and model_option == 'Classification') or \
           (split_option == 'Train-Test-Validation' and  model_option == 'Classification'):
            
            try:
                if 'Logistic Regression' in select_models:

                    # Initialize logistic regression model with best_params
                    model_logistic = LogisticRegression(**best_params) 

                    # Fit the model
                    model_logistic.fit(X_test, y_test)

                    # Make predictions
                    predictions = model_logistic.predict(X_test)

                    # Calculate evaluation metrics 
                    logit_accuracy = accuracy_score(y_test, predictions)
                    logit_recall = recall_score(y_test, predictions)
                    logit_precision = precision_score(y_test, predictions)
                    logit_f1_score = f1_score(y_test, predictions)
                    logit_roc_auc = roc_auc_score(y_test, predictions)

                    # Add metrics to dictionary
                    test_metrics['Model Name'].append('Logistic Regression')
                    test_metrics['Accuracy'].append(logit_accuracy)
                    test_metrics['Recall'].append(logit_recall)
                    test_metrics['Precision'].append(logit_precision)
                    test_metrics['F1-score'].append(logit_f1_score)
                    test_metrics['ROC-AUC'].append(logit_roc_auc)

                    # Interpretation model
                    summary = pd.DataFrame({'features': X_test.columns.tolist() + ['constant'],
                                            'weights': model_logistic.coef_[0].tolist() + model_logistic.intercept_.tolist()})
                    
                    summary = summary.sort_values(by='weights', ascending=False)

                elif 'K-NN' in select_models:

                    # Initialize K-NN model with best_params
                    model_knn = KNeighborsClassifier(**best_params) 

                    # Fit the model
                    model_knn.fit(X_test, y_test)

                    # Make predictions
                    predictions = model_knn.predict(X_test)

                    # Calculate evaluation metrics 
                    knn_accuracy = accuracy_score(y_test, predictions)
                    knn_recall = recall_score(y_test, predictions)
                    knn_precision = precision_score(y_test, predictions)
                    knn_f1_score = f1_score(y_test, predictions)
                    knn_roc_auc = roc_auc_score(y_test, predictions)

                    # Add metrics to dictionary
                    test_metrics['Model Name'].append('K-NN')
                    test_metrics['Accuracy'].append(knn_accuracy)
                    test_metrics['Recall'].append(knn_recall)
                    test_metrics['Precision'].append(knn_precision)
                    test_metrics['F1-score'].append(knn_f1_score)
                    test_metrics['ROC-AUC'].append(knn_roc_auc)

                elif 'Decision Tree' in select_models:

                    # Initialize Decision Tree model with best_params
                    model_tree = DecisionTreeClassifier(**best_params) 

                    # Fit the model
                    model_tree.fit(X_test, y_test)

                    # Make predictions
                    predictions = model_tree.predict(X_test)

                    # Calculate evaluation metrics 
                    tree_accuracy = accuracy_score(y_test, predictions)
                    tree_recall = recall_score(y_test, predictions)
                    tree_precision = precision_score(y_test, predictions)
                    tree_f1_score = f1_score(y_test, predictions)
                    tree_roc_auc = roc_auc_score(y_test, predictions)

                    # Add metrics to dictionary
                    test_metrics['Model Name'].append('Decision-Tree')
                    test_metrics['Accuracy'].append(tree_accuracy)
                    test_metrics['Recall'].append(tree_recall)
                    test_metrics['Precision'].append(tree_precision)
                    test_metrics['F1-score'].append(tree_f1_score)
                    test_metrics['ROC-AUC'].append(tree_roc_auc)       
                
                elif 'SVM' in select_models:

                    # Initialize SVM model with best_params
                    model_svm = SVC(**best_params) 

                    # Fit the model
                    model_svm.fit(X_test, y_test)

                    # Make predictions
                    predictions = model_svm.predict(X_test)

                    # Calculate evaluation metrics 
                    svm_accuracy = accuracy_score(y_test, predictions)
                    svm_recall = recall_score(y_test, predictions)
                    svm_precision = precision_score(y_test, predictions)
                    svm_f1_score = f1_score(y_test, predictions)
                    svm_roc_auc = roc_auc_score(y_test, predictions)

                    # Add metrics to dictionary
                    test_metrics['Model Name'].append('SVM')
                    test_metrics['Accuracy'].append(svm_accuracy)
                    test_metrics['Recall'].append(svm_recall)
                    test_metrics['Precision'].append(svm_precision)
                    test_metrics['F1-score'].append(svm_f1_score)
                    test_metrics['ROC-AUC'].append(svm_roc_auc)

                elif 'XG-Boost' in select_models:

                    # Initialize XG-Boost model with best_params
                    model_xg = XGBClassifier(**best_params) 

                    # Fit the model
                    model_xg.fit(X_test, y_test)

                    # Make predictions
                    predictions = model_xg.predict(X_test)

                    # Calculate evaluation metrics 
                    xg_accuracy = accuracy_score(y_test, predictions)
                    xg_recall = recall_score(y_test, predictions)
                    xg_precision = precision_score(y_test, predictions)
                    xg_f1_score = f1_score(y_test, predictions)
                    xg_roc_auc = roc_auc_score(y_test, predictions)

                    # Add metrics to dictionary
                    test_metrics['Model Name'].append('XG-Boost')
                    test_metrics['Accuracy'].append(xg_accuracy)
                    test_metrics['Recall'].append(xg_recall)
                    test_metrics['Precision'].append(xg_precision)
                    test_metrics['F1-score'].append(xg_f1_score)
                    test_metrics['ROC-AUC'].append(xg_roc_auc)
                
                elif 'Random Forest' in select_models:

                    # Initialize Random Forest model with best_params
                    model_random = RandomForestClassifier(**best_params) 

                    # Fit the model
                    model_random.fit(X_test, y_test)

                    # Make predictions
                    predictions = model_random.predict(X_test)

                    # Calculate evaluation metrics 
                    random_accuracy = accuracy_score(y_test, predictions)
                    random_recall = recall_score(y_test, predictions)
                    random_precision = precision_score(y_test, predictions)
                    random_f1_score = f1_score(y_test, predictions)
                    random_roc_auc = roc_auc_score(y_test, predictions)

                    # Add metrics to dictionary
                    test_metrics['Model Name'].append('Random Forest')
                    test_metrics['Accuracy'].append(random_accuracy)
                    test_metrics['Recall'].append(random_recall)
                    test_metrics['Precision'].append(random_precision)
                    test_metrics['F1-score'].append(random_f1_score)
                    test_metrics['ROC-AUC'].append(random_roc_auc)
                
                elif 'AdaBoost' in select_models:

                    # Initialize AdaBoost model with best_params
                    model_adb = AdaBoostClassifier(**best_params) 

                    # Fit the model
                    model_adb.fit(X_test, y_test)

                    # Make predictions
                    predictions = model_adb.predict(X_test)

                    # Calculate evaluation metrics 
                    ada_accuracy = accuracy_score(y_test, predictions)
                    ada_recall = recall_score(y_test, predictions)
                    ada_precision = precision_score(y_test, predictions)
                    ada_f1_score = f1_score(y_test, predictions)
                    ada_roc_auc = roc_auc_score(y_test, predictions)

                    # Add metrics to dictionary
                    test_metrics['Model Name'].append('AdaBoost')
                    test_metrics['Accuracy'].append(ada_accuracy)
                    test_metrics['Recall'].append(ada_recall)
                    test_metrics['Precision'].append(ada_precision)
                    test_metrics['F1-score'].append(ada_f1_score)
                    test_metrics['ROC-AUC'].append(ada_roc_auc)
            
            except:
                st.info('Wait until you are done selecting target data')

        elif (split_option == 'Train-Test' and model_option == 'Regression') or\
             (split_option == 'Train-Test-Validation' and  model_option == 'Regression'):
            
            try:
            
                if select_models_regriss == 'Linear Regression':

                    # Initialize Linear Regression model with best_params
                    model_regriss = LinearRegression() 

                    # Fit the model
                    model_regriss.fit(X_test, y_test)

                    # Make predictions
                    predictions = model_regriss.predict(X_test)

                    # Calculate evaluation metrics 
                    linear_mse = mean_squared_error(y_test, predictions)
                    linear_mae = mean_absolute_error(y_test, predictions)
                    linear_rmse = np.sqrt(linear_mse)
                    linear_r2 = r2_score(y_test, predictions)

                    # Add metrics to dictionary
                    test_metrics_regris['Model Name'].append('Linear Regression')
                    test_metrics_regris['MSE'].append(linear_mse)
                    test_metrics_regris['MAE'].append(linear_mae)
                    test_metrics_regris['RMSE'].append(linear_rmse)
                    test_metrics_regris['R2'].append(linear_r2)

                    # Interpretation model
                    summary = pd.DataFrame({'features': X_test.columns.tolist() + ['constant'],
                                            'weights': list(model_regriss.coef_) + [model_regriss.intercept_]})

                    
                    summary = summary.sort_values(by='weights', ascending=False)
                
                elif select_models_regriss == 'K-NN':

                    # Initialize K-NN model with best_params
                    model_knn = KNeighborsRegressor(**best_params_regriss) 

                    # Fit the model
                    model_knn.fit(X_test, y_test)

                    # Make predictions
                    predictions = model_knn.predict(X_test)

                    # Calculate evaluation metrics 
                    knn_mse = mean_squared_error(y_test, predictions)
                    knn_mae = mean_absolute_error(y_test, predictions)
                    knn_rmse = np.sqrt(knn_mse)
                    knn_r2 = r2_score(y_test, predictions)

                    # Add metrics to dictionary
                    test_metrics_regris['Model Name'].append('K-NN')
                    test_metrics_regris['MSE'].append(knn_mse)
                    test_metrics_regris['MAE'].append(knn_mae)
                    test_metrics_regris['RMSE'].append(knn_rmse)
                    test_metrics_regris['R2'].append(knn_r2)
            
                elif select_models_regriss == 'Decision Tree':

                    # Initialize Decision Tree model with best_params
                    model_tree = DecisionTreeRegressor(**best_params_regriss) 

                    # Fit the model
                    model_tree.fit(X_test, y_test)

                    # Make predictions
                    predictions = model_tree.predict(X_test)

                    # Calculate evaluation metrics 
                    tree_mse = mean_squared_error(y_test, predictions)
                    tree_mae = mean_absolute_error(y_test, predictions)
                    tree_rmse = np.sqrt(tree_mse)
                    tree_r2 = r2_score(y_test, predictions)

                    # Add metrics to dictionary
                    test_metrics_regris['Model Name'].append('Decision Tree')
                    test_metrics_regris['MSE'].append(tree_mse)
                    test_metrics_regris['MAE'].append(tree_mae)
                    test_metrics_regris['RMSE'].append(tree_rmse)
                    test_metrics_regris['R2'].append(tree_r2)
                
                elif select_models_regriss == 'SVM':

                    # Initialize Decision Tree model with best_params
                    model_svm = SVR(**best_params_regriss) 

                    # Fit the model
                    model_svm.fit(X_test, y_test)

                    # Make predictions
                    predictions = model_svm.predict(X_test)

                    # Calculate evaluation metrics 
                    svm_mse = mean_squared_error(y_test, predictions)
                    svm_mae = mean_absolute_error(y_test, predictions)
                    svm_rmse = np.sqrt(svm_mse)
                    svm_r2 = r2_score(y_test, predictions)

                    # Add metrics to dictionary
                    test_metrics_regris['Model Name'].append('SVM')
                    test_metrics_regris['MSE'].append(svm_mse)
                    test_metrics_regris['MAE'].append(svm_mae)
                    test_metrics_regris['RMSE'].append(svm_rmse)
                    test_metrics_regris['R2'].append(svm_r2)
                
                elif select_models_regriss == 'XG-Boost':

                    # Initialize XG-Boost model with best_params
                    model_xg = XGBRegressor(**best_params_regriss) 

                    # Fit the model
                    model_xg.fit(X_test, y_test)

                    # Make predictions
                    predictions = model_xg.predict(X_test)

                    # Calculate evaluation metrics 
                    xg_mse = mean_squared_error(y_test, predictions)
                    xg_mae = mean_absolute_error(y_test, predictions)
                    xg_rmse = np.sqrt(xg_mse)
                    xg_r2 = r2_score(y_test, predictions)

                    # Add metrics to dictionary
                    test_metrics_regris['Model Name'].append('XG-Boost')
                    test_metrics_regris['MSE'].append(xg_mse)
                    test_metrics_regris['MAE'].append(xg_mae)
                    test_metrics_regris['RMSE'].append(xg_rmse)
                    test_metrics_regris['R2'].append(xg_r2)
                
                elif select_models_regriss == 'Random Forest':

                    # Initialize Random Forest model with best_params
                    model_random = RandomForestRegressor(**best_params_regriss) 

                    # Fit the model
                    model_random.fit(X_test, y_test)

                    # Make predictions
                    predictions = model_random.predict(X_test)

                    # Calculate evaluation metrics 
                    random_mse = mean_squared_error(y_test, predictions)
                    random_mae = mean_absolute_error(y_test, predictions)
                    random_rmse = np.sqrt(random_mse)
                    random_r2 = r2_score(y_test, predictions)

                    # Add metrics to dictionary
                    test_metrics_regris['Model Name'].append('Random Forest')
                    test_metrics_regris['MSE'].append(random_mse)
                    test_metrics_regris['MAE'].append(random_mae)
                    test_metrics_regris['RMSE'].append(random_rmse)
                    test_metrics_regris['R2'].append(random_r2)

                elif select_models_regriss == 'AdaBoost':

                    # Initialize AdaBoost model with best_params
                    model_ada = AdaBoostRegressor(**best_params_regriss) 

                    # Fit the model
                    model_ada.fit(X_test, y_test)

                    # Make predictions
                    predictions = model_ada.predict(X_test)

                    # Calculate evaluation metrics 
                    ada_mse = mean_squared_error(y_test, predictions)
                    ada_mae = mean_absolute_error(y_test, predictions)
                    ada_rmse = np.sqrt(ada_mse)
                    ada_r2 = r2_score(y_test, predictions)

                    # Add metrics to dictionary
                    test_metrics_regris['Model Name'].append('AdaBoost')
                    test_metrics_regris['MSE'].append(ada_mae)
                    test_metrics_regris['MAE'].append(ada_rmse)
                    test_metrics_regris['RMSE'].append(ada_rmse)
                    test_metrics_regris['R2'].append(ada_r2)
                
                elif select_models_regriss == 'Ridge':

                    # Initialize Ridge model with best_params
                    model_ridge = Ridge(**best_params_regriss) 

                    # Fit the model
                    model_ridge.fit(X_test, y_test)

                    # Make predictions
                    predictions = model_ridge.predict(X_test)

                    # Calculate evaluation metrics 
                    ridge_mse = mean_squared_error(y_test, predictions)
                    ridge_mae = mean_absolute_error(y_test, predictions)
                    ridge_rmse = np.sqrt(ridge_mse)
                    ridge_r2 = r2_score(y_test, predictions)

                    # Add metrics to dictionary
                    test_metrics_regris['Model Name'].append('Ridge')
                    test_metrics_regris['MSE'].append(ridge_mse)
                    test_metrics_regris['MAE'].append(ridge_mae)
                    test_metrics_regris['RMSE'].append(ridge_rmse)
                    test_metrics_regris['R2'].append(ridge_r2)

                elif select_models_regriss == 'Lasso':

                    # Initialize Lasso model with best_params
                    model_lasso = Lasso(**best_params_regriss) 

                    # Fit the model
                    model_lasso.fit(X_test, y_test)

                    # Make predictions
                    predictions = model_lasso.predict(X_test)

                    # Calculate evaluation metrics 
                    lasso_mse = mean_squared_error(y_test, predictions)
                    lasso_mae = mean_absolute_error(y_test, predictions)
                    lasso_rmse = np.sqrt(lasso_mse)
                    lasso_r2 = r2_score(y_test, predictions)

                    # Add metrics to dictionary
                    test_metrics_regris['Model Name'].append('Lasso')
                    test_metrics_regris['MSE'].append(lasso_mse)
                    test_metrics_regris['MAE'].append(lasso_mae)
                    test_metrics_regris['RMSE'].append(lasso_rmse)
                    test_metrics_regris['R2'].append(lasso_r2)
            except:
                st.info('Wait until you are done selecting target data')
        
        if (split_option == 'Train-Test' and model_option == 'Classification') or \
           (split_option == 'Train-Test-Validation' and  model_option == 'Classification'):
            
            try:
                # change to dataframe
                test_metrics = pd.DataFrame(test_metrics)
                st.write(test_metrics)

                if select_models == 'Logistic Regression': 
                    # interpretation model
                    st.write('**Interpretation Model:**')
                    st.write(summary.T)
                
                else:
                    pass

            except:
                st.info('Wait until you are done selecting target data')
        
        if (split_option == 'Train-Test' and model_option == 'Regression') or \
           (split_option == 'Train-Test-Validation' and  model_option == 'Regression'):
            
            try:
                # change to dataframe
                metrics_regriss = pd.DataFrame(test_metrics_regris)
                st.write(metrics_regriss)

                if select_models_regriss == 'Linear Regression':

                    # interpretation model
                    st.write('**Interpretation Model:**')
                    st.write(summary.T)
            
            except:
                st.info('Wait until you are done selecting target data')
        
    def clustering():
        """Function for evaluation metrics for each model selected"""

        try:
            # metrics model
            metric_clustering = {'Model Name': [], 'Best Cluster': [], 'Silhouette': [], 'Calinski-Harabasz': [], 'Davies-Bouldin': [],
                                'adjusted_mutual_info_score': [], 'adjusted_rand_score': [], 'normalized_mutual_info_score': []}
            
            # Select model
            model_select_clust = st.selectbox('Select Model', ('----', 'K-Means','DBSCAN',
                                                                        'GMM', 'Agglomerative'),
                                                                        key='clustering')
            

            if model_option == 'Clustering':

                global labels
                
                if model_select_clust in 'K-NN' :
                    
                    # Parameters to search
                    param_dist = {
                        'n_clusters': np.random.randint(2, 10),
                        'init': ['k-means++', 'random'],
                        'n_init': np.random.randint(10, 31)
                    }

                    # Initialize KMeans
                    kmeans = KMeans()

                    # Perform RandomizedSearch
                    random_search = RandomizedSearchCV(estimator=kmeans,
                                                        param_distributions=param_dist,
                                                        scoring=make_scorer(silhouette_score),
                                                        n_jobs=-1)

                    random_search.fit(df)

                    # Best parameters and best silhouette score
                    best_params = random_search.best_params_

                    # Best number of clusters
                    best_n_components = best_params['n_clusters']

                    # Assuming you have obtained labels from the clustering algorithm
                    labels = random_search.best_estimator_.labels_
                    
                    # Add best cluster to the dictionary
                    metric_clustering['Best Cluster'].append(best_n_components)
                    
                    # Calculate Silhouette score
                    silhouette = silhouette_score(df, labels)
                    metric_clustering['Silhouette'].append(silhouette)

                    # Calculate Calinski-Harabasz score
                    calinski_harabasz = calinski_harabasz_score(df, labels)
                    metric_clustering['Calinski-Harabasz'].append(calinski_harabasz)

                    # Calculate Davies-Bouldin score
                    davies_bouldin = davies_bouldin_score(df, labels)
                    metric_clustering['Davies-Bouldin'].append(davies_bouldin)

                    # Calculate adjusted mutual information score
                    adj_mutual_info = adjusted_mutual_info_score(labels, labels)
                    metric_clustering['adjusted_mutual_info_score'].append(adj_mutual_info)

                    # Calculate adjusted rand score
                    adj_rand = adjusted_rand_score(labels, labels)
                    metric_clustering['adjusted_rand_score'].append(adj_rand)

                    # Calculate normalized mutual information score
                    norm_mutual_info = normalized_mutual_info_score(labels, labels)
                    metric_clustering['normalized_mutual_info_score'].append(norm_mutual_info)
                    
                    # Add model name to the dictionary
                    metric_clustering['Model Name'].append('K-Means')
                    
                elif model_select_clust == 'GMM':

                    # Parameters to search for GMM
                    param_dist = {
                        'n_components': np.random.randint(2, 10),  
                        'init_params': ['kmeans', 'random'],  
                        'covariance_type': ['full', 'tied', 'diag', 'spherical']  
                    }

                    # Initialize GMM
                    gmm = GaussianMixture()

                    # Perform RandomizedSearch
                    random_search = RandomizedSearchCV(estimator=gmm,
                                                    param_distributions=param_dist,
                                                    n_iter=10,  # Jumlah iterasi pencarian acak
                                                    scoring=make_scorer(silhouette_score),
                                                    n_jobs=-1)

                    random_search.fit(df)

                    # Best parameters
                    best_params = random_search.best_params_

                    # Best number of clusters
                    best_n_components = best_params['n_components']

                    # Assuming you have obtained labels from the clustering algorithm
                    labels = random_search.best_estimator_.predict(df)
                    
                    # Add best cluster to the dictionary
                    metric_clustering['Best Cluster'].append(best_n_components)
                    
                    # Calculate Silhouette score
                    silhouette = silhouette_score(df, labels)
                    metric_clustering['Silhouette'].append(silhouette)

                    # Calculate Calinski-Harabasz score
                    calinski_harabasz = calinski_harabasz_score(df, labels)
                    metric_clustering['Calinski-Harabasz'].append(calinski_harabasz)

                    # Calculate Davies-Bouldin score
                    davies_bouldin = davies_bouldin_score(df, labels)
                    metric_clustering['Davies-Bouldin'].append(davies_bouldin)

                    # Calculate adjusted mutual information score
                    adj_mutual_info = adjusted_mutual_info_score(labels, labels)
                    metric_clustering['adjusted_mutual_info_score'].append(adj_mutual_info)

                    # Calculate adjusted rand score
                    adj_rand = adjusted_rand_score(labels, labels)
                    metric_clustering['adjusted_rand_score'].append(adj_rand)

                    # Calculate normalized mutual information score
                    norm_mutual_info = normalized_mutual_info_score(labels, labels)
                    metric_clustering['normalized_mutual_info_score'].append(norm_mutual_info)
                    
                    # Add model name to the dictionary
                    metric_clustering['Model Name'].append('GMM')

                elif model_select_clust == 'Agglomerative' :

                    # Parameters to search for Agglomerative Clustering
                    param_dist = {
                        'n_clusters': np.random.randint(2, 10),  # Number of clusters
                        'linkage': ['ward', 'complete', 'average', 'single']  # Linkage method
                    }

                    # Initialize Agglomerative Clustering
                    clustering = AgglomerativeClustering()

                    # Perform RandomizedSearch
                    random_search = RandomizedSearchCV(estimator=clustering,
                                                        param_distributions=param_dist,
                                                        n_iter=10,  # Number of random parameter settings that are sampled
                                                        scoring=make_scorer(silhouette_score),
                                                        n_jobs=-1)

                    random_search.fit(df)  # Assuming df is your data

                    # Best parameters
                    best_params = random_search.best_params_

                    # Best number of clusters
                    best_n_clusters = best_params['n_clusters']

                    # Assuming you have obtained labels from the clustering algorithm
                    labels = random_search.best_estimator_.labels_
                    
                    # Add best cluster to the dictionary
                    metric_clustering['Best Cluster'].append(best_n_clusters)
                    
                    # Calculate Silhouette score
                    silhouette = silhouette_score(df, labels)
                    metric_clustering['Silhouette'].append(silhouette)

                    # Calculate Calinski-Harabasz score
                    calinski_harabasz = calinski_harabasz_score(df, labels)
                    metric_clustering['Calinski-Harabasz'].append(calinski_harabasz)

                    # Calculate Davies-Bouldin score
                    davies_bouldin = davies_bouldin_score(df, labels)
                    metric_clustering['Davies-Bouldin'].append(davies_bouldin)

                    # Calculate adjusted mutual information score
                    adj_mutual_info = adjusted_mutual_info_score(labels, labels)
                    metric_clustering['adjusted_mutual_info_score'].append(adj_mutual_info)

                    # Calculate adjusted rand score
                    adj_rand = adjusted_rand_score(labels, labels)
                    metric_clustering['adjusted_rand_score'].append(adj_rand)

                    # Calculate normalized mutual information score
                    norm_mutual_info = normalized_mutual_info_score(labels, labels)
                    metric_clustering['normalized_mutual_info_score'].append(norm_mutual_info)
                    
                    # Add model name to the dictionary
                    metric_clustering['Model Name'].append('Agglomerative')

                elif model_select_clust == 'DBSCAN':

                    # Parameters to search for DBSCAN Clustering
                    param_dist = {
                        'eps': np.random.uniform(0.01),  
                        'min_samples': np.random.randint(2, 11),  
                    }

                    # Initialize DBSCAN Clustering
                    clustering = DBSCAN()

                    # Perform RandomizedSearch
                    random_search = RandomizedSearchCV(estimator=clustering,
                                                       param_distributions=param_dist,
                                                       n_iter=10, 
                                                       scoring=make_scorer(silhouette_score),
                                                       n_jobs=-1)

                    random_search.fit(df)  # Assuming X is your data

                    # Assuming you have obtained labels from the clustering algorithm
                    labels = random_search.best_estimator_.labels_

                    # Calculate Silhouette score
                    silhouette = silhouette_score(df, labels)
                    metric_clustering['Silhouette'].append(silhouette)

                    # Count the number of clusters (excluding noise points)
                    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                                        
                    metric_clustering['Best Cluster'].append(num_clusters)

                    # For metrics that don't depend on the length of labels, you can append placeholders
                    metric_clustering['Calinski-Harabasz'].append(None)
                    metric_clustering['Davies-Bouldin'].append(None)
                    metric_clustering['adjusted_mutual_info_score'].append(None)
                    metric_clustering['adjusted_rand_score'].append(None)
                    metric_clustering['normalized_mutual_info_score'].append(None)

                    # Add model name to the dictionary
                    metric_clustering['Model Name'].append('DBSCAN')

                # change metrics
                metric_clustering = pd.DataFrame(metric_clustering)
                st.write(metric_clustering)

                st.write('**Data with Cluster**')
                df_copy['Cluster'] = labels

                # Noise cluster
                filter_noise = df_copy.loc[df_copy['Cluster'] != -1]                    
                st.write(filter_noise)

                # Download data as CSV 
                csv = df_copy.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # Convert dataframe to bytes
                href = f'<a href="data:file/csv;base64,{b64}" download="{model_select_clust}_clustered_data.csv">Download {model_select_clust} Clustered Data as CSV</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        except:
            st.info('To obtain a data cluster, you must select the model evaluation first!')
    
    def user():
        """
        Function for input data from user
        """
        # Initialization input data
        global input_data

        # Model option
        if model_option == 'Classification' or model_option == 'Regression':

            # Condition original predictors
            if select_predictors == 'None':

                # Dict initialization
                data = {}

                # Columns list
                data_columns = X_test.columns.tolist()

                # Input data each columns
                for col in data_columns:
                    value = st.text_input(f"Input Values'{col}': ")
                    
                    # Numeric col and category
                    if value.replace('.', '', 1).isdigit():
                        data[col] = float(value)
                    else:
                        data[col] = value
            
            # Condition with select a few predictors
            elif select_predictors == 'Manual Select':

                # Predictor name
                selected_data = X_test[selected_predictors]
                
                # list data
                data = {}
                data_columns = selected_data.columns.tolist()
                
                # Input data each columns
                for col in data_columns:
                    value = st.text_input(f"Input values '{col}': ")
                    
                    # Numeric col and category
                    if value.replace('.', '', 1).isdigit():
                        data[col] = float(value)
                    else:
                        data[col] = value
        
        input_data = pd.DataFrame([data])
        st.write(input_data)

    def predict_user():
        """
        Function for result prediction 
        """
        # Button for predict
        if st.button('Predict'):
            
            # predict with model 
            if model_option == 'Classification' and (split_option == 'Train-Test' or split_option == 'Train-Test-Validation'):
                
                if select_models == 'Decision Tree':
                    
                    # Model predict
                    model_tree = DecisionTreeClassifier(**best_params)
                    
                    # fitting model
                    model_tree.fit(X_train, y_train)
                    
                    # Predict model
                    predict = model_tree.predict(input_data)        
                    
                    # Probability result
                    probability = model_tree.predict_proba(input_data)  

                    # predict result
                    if predict == 0:
                        st.success(f"No, {select_models} Probability target {target_column}: {probability[0][0]:.2f}")
                    
                    else:
                        st.success(f"Yes, {select_models} Probability target {target_column}: {probability[0][1]:.2f}")

                elif select_models == 'K-NN':
                    
                    # Model predict
                    model_knn = KNeighborsClassifier(**best_params)
                    
                    # fitting model
                    model_knn.fit(X_train, y_train)
                    
                    # Predict model
                    predict = model_knn.predict(input_data)        
                    
                    # Probability result
                    probability = model_knn.predict_proba(input_data)  

                    # predict result
                    if predict == 0:
                        st.success(f"No, {select_models} Probability target {target_column}: {probability[0][0]:.2f}")
                    
                    else:
                        st.success(f"Yes, {select_models} Probability target {target_column}: {probability[0][1]:.2f}")
                
                elif select_models == 'Logistic Regression':
                    
                    # Model predict
                    model_logit = LogisticRegression(**best_params)
                    
                    # fitting model
                    model_logit.fit(X_train, y_train)
                    
                    # Predict model
                    predict = model_logit.predict(input_data)        
                    
                    # Probability result
                    probability = model_logit.predict_proba(input_data)  

                    # predict result
                    if predict == 0:
                        st.success(f"No, {select_models} Probability target {target_column}: {probability[0][0]:.2f}")
                    
                    else:
                        st.success(f"Yes, {select_models} Probability target {target_column}: {probability[0][1]:.2f}")

                elif select_models == 'SVM':
                    
                    # Model predict
                    model_svm = SVC(**best_params)
                    
                    # fitting model
                    model_svm.fit(X_train, y_train)
                    
                    # Predict model
                    predict = model_svm.predict(input_data)        
                    
                    # Probability result
                    probability = model_svm.predict_proba(input_data)  

                    # predict result
                    if predict == 0:
                        st.success(f"No, {select_models} Probability target {target_column}: {probability[0][0]:.2f}")
                    
                    else:
                        st.success(f"Yes, {select_models} Probability target {target_column}: {probability[0][1]:.2f}")
                
                elif select_models == 'XG-Boost':
                    
                    # Model predict
                    model_xg = XGBClassifier(**best_params)
                    
                    # fitting model
                    model_xg.fit(X_train, y_train)
                    
                    # Predict model
                    predict = model_xg.predict(input_data)        
                    
                    # Probability result
                    probability = model_xg.predict_proba(input_data)  

                    # predict result
                    if predict == 0:
                        st.success(f"No, {select_models} Probability target {target_column}: {probability[0][0]:.2f}")
                    
                    else:
                        st.success(f"Yes, {select_models} Probability target {target_column}: {probability[0][1]:.2f}")

                elif select_models == 'Random Forest':
                    
                    # Model predict
                    model_random = RandomForestClassifier(**best_params)
                    
                    # fitting model
                    model_random.fit(X_train, y_train)
                    
                    # Predict model
                    predict = model_random.predict(input_data)        
                    
                    # Probability result
                    probability = model_random.predict_proba(input_data)  

                    # predict result
                    if predict == 0:
                        st.success(f"No, {select_models} Probability target {target_column}: {probability[0][0]:.2f}")
                    
                    else:
                        st.success(f"Yes, {select_models} Probability target {target_column}: {probability[0][1]:.2f}")
                
                elif select_models == 'AdaBoost':
                    
                    # Model predict
                    model_ada = AdaBoostClassifier(**best_params)
                    
                    # fitting model
                    model_ada.fit(X_train, y_train)
                    
                    # Predict model
                    predict = model_ada.predict(input_data)        
                    
                    # Probability result
                    probability = model_ada.predict_proba(input_data)  

                    # predict result
                    if predict == 0:
                        st.success(f"No, {select_models} Probability target {target_column}: {probability[0][0]:.2f}")
                    
                    else:
                        st.success(f"Yes, {select_models} Probability target {target_column}: {probability[0][1]:.2f}")

            # predict with model 
            if model_option == 'Regression' and (split_option == 'Train-Test' or split_option == 'Train-Test-Validation'):

                if select_models_regriss == 'Decision Tree':
                   
                   # Model predict 
                   model_tree = DecisionTreeRegressor(**best_params_regriss)

                   # fitting model
                   model_tree.fit(X_train, y_train)

                   # Predict model
                   predict = model_tree.predict(input_data)  
                    
                   # Result
                   st.success(f'{select_models_regriss}: The predicted {target_column} is {predict[0]}')
                
                elif select_models_regriss == 'K-NN':
                   
                   # Model predict 
                   model_knn = KNeighborsRegressor(**best_params_regriss)

                   # fitting model
                   model_knn.fit(X_train, y_train)

                   # Predict model
                   predict = model_knn.predict(input_data)  
                    
                   # Result
                   st.success(f'{select_models_regriss}: The predicted {target_column} is {predict[0]}')
                
                elif select_models_regriss == 'SVM':
                   
                   # Model predict 
                   model_svm = SVR(**best_params_regriss)

                   # fitting model
                   model_svm.fit(X_train, y_train)

                   # Predict model
                   predict = model_svm.predict(input_data)  
                    
                   # Result
                   st.success(f'{select_models_regriss}: The predicted {target_column} is {predict[0]}')
                
                elif select_models_regriss == 'XG-Boost':
                   
                   # Model predict 
                   model_xg = XGBRegressor(**best_params_regriss)

                   # fitting model
                   model_xg.fit(X_train, y_train)

                   # Predict model
                   predict = model_xg.predict(input_data)  
                    
                   # Result
                   st.success(f'{select_models_regriss}: The predicted {target_column} is {predict[0]}')
                
                elif select_models_regriss == 'Random Forest':
                   
                   # Model predict 
                   model_random = RandomForestRegressor(**best_params_regriss)

                   # fitting model
                   model_random.fit(X_train, y_train)

                   # Predict model
                   predict = model_random.predict(input_data)  
                    
                   # Result
                   st.success(f'{select_models_regriss}: The predicted {target_column} is {predict[0]}')
                
                elif select_models_regriss == 'AdaBoost':
                   
                   # Model predict 
                   model_ada = AdaBoostRegressor(**best_params_regriss)

                   # fitting model
                   model_ada.fit(X_train, y_train)

                   # Predict model
                   predict = model_ada.predict(input_data)  
                    
                   # Result
                   st.success(f'{select_models_regriss}: The predicted {target_column} is {predict[0]}')
                
                elif select_models_regriss == 'Ridge':
                   
                   # Model predict 
                   model_ridge = Ridge(**best_params_regriss)

                   # fitting model
                   model_ridge.fit(X_train, y_train)

                   # Predict model
                   predict = model_ridge.predict(input_data)  
                    
                   # Result
                   st.success(f'{select_models_regriss}: The predicted {target_column} is {predict[0]}')
                
                elif select_models_regriss == 'Lasso':
                   
                   # Model predict 
                   model_lasso = Lasso(**best_params_regriss)

                   # fitting model
                   model_lasso.fit(X_train, y_train)

                   # Predict model
                   predict = model_lasso.predict(input_data)  
                    
                   # Result
                   st.success(f'{select_models_regriss}: The predicted {target_column} is {predict[0]}')

    def main():
        """
        Function for running each previous function 
        """
        try:
            if (model_option == 'Regression' or model_option == 'Classification') and \
                (split_option == 'Train-Test' or split_option == 'Train-Test-Validation'):
            
                # run function 
                st.subheader('Prediction')
                st.markdown('**Train Model**')
                train_predict() # Predict with train data

                st.markdown('**Re-Train Model**')
                st.markdown('*Note: Using the Best Model Before!*')
                    
                valid_predict() # Predict with valid and re-train data

                st.markdown('**Test Evaluation**')
                testing() # Predict with test data

                st.subheader('Form Input Data to Prediction')
                user()
                predict_user()

            elif (model_option == 'Clustering'):

                # run function
                st.subheader('Clustering')
                st.markdown('**Evaluation Metrics**')
                clustering()

        except NameError:
            st.info("Wait until you are done selecting target data")

    if __name__ == "__main__":
        main()

# Note belum :  
                # buat deep learningnya semua model
                # Belum ada simpan data setelah predict
                # belum ada multiclass untuk classification

                
