
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load dataset (assuming it's available locally as 'enhanced_dataset.csv')
#@st.cache_data
#def load_data():
    #return pd.read_csv(/content/drive/MyDrive/assessment.st20313571/combined_output.csv)

# Page 1: Data Overview
def data_overview(data):
    st.title("Data Overview")
    st.write("Here you can view basic information about the dataset.")

    # Display data shape
    st.write("### Dataset Shape")
    st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

    # Display sample data
    st.write("### Sample Data")
    st.write(data.head())

    # Display summary statistics
    st.write("### Summary Statistics")
    st.write(data.describe())

    # Display data types
    st.write("### Data Types")
    st.write(data.dtypes)

# Page 2: Exploratory Data Analysis (EDA)
def eda(data):
    st.title("Exploratory Data Analysis (EDA)")
    st.write("This section provides visual insights into the dataset.")

    # Correlation heatmap
    st.write("### Correlation Heatmap")
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    corr = numeric_data.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot(plt)

    # Histogram
    st.write("### Histogram")
    column = st.selectbox("Select a column for histogram", numeric_data.columns)
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True, bins=30)
    st.pyplot(plt)

    # Scatter plot
    st.write("### Scatter Plot")
    col1 = st.selectbox("Select X-axis column", numeric_data.columns)
    col2 = st.selectbox("Select Y-axis column", numeric_data.columns)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data[col1], y=data[col2])
    st.pyplot(plt)

# Page 3: Modeling and Prediction
def modeling_and_prediction(data):
    st.title("Modeling and Prediction")
    st.write("Train a model and make predictions.")

    # Select features and target
    target = st.selectbox("Select Target Variable", data.columns)
    features = st.multiselect("Select Feature Variables", [col for col in data.columns if col != target])

    if target and features:
        X = data[features]
        y = data[target]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train a Random Forest model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Display metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        st.write(f"### Model Performance")
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"R²: {r2:.2f}")

        # Feature importance
        st.write("### Feature Importance")
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)
        st.write(feature_importance)

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        st.pyplot(plt)

# Main application logic
def main():
    # Load data
    data = load_data()

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Overview", "EDA", "Modeling and Prediction"])

    # Page routing
    if page == "Data Overview":
        data_overview(data)
    elif page == "EDA":
        eda(data)
    elif page == "Modeling and Prediction":
        modeling_and_prediction(data)

if __name__ == "__main__":
    main()
