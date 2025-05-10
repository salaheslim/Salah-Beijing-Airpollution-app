import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load data with caching
@st.cache_data
def load_data():
    return pd.read_csv("analyzed_data.csv", on_bad_lines='skip')

# Page 1: Data Overview
def data_overview(data):
    st.title("📊 Data Overview")
    st.write("This page provides a structured overview of the dataset you are working with.")

    # Sidebar option to select number of rows
    num_rows = st.sidebar.slider("Number of rows to preview", min_value=5, max_value=50, value=5, step=5)

    # Use columns to display shape and data types side-by-side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📐 Dataset Shape")
        st.metric(label="Rows", value=data.shape[0])
        st.metric(label="Columns", value=data.shape[1])

    with col2:
        st.subheader("🧬 Data Types")
        st.dataframe(data.dtypes.rename("Type").reset_index().rename(columns={"index": "Column"}))

    # Expanders for better layout
    with st.expander("🔍 Preview Sample Data"):
        st.dataframe(data.head(num_rows))

    with st.expander("📈 Summary Statistics"):
        st.dataframe(data.describe())

    with st.expander("📑 Raw Data (Optional)"):
        st.dataframe(data)

    # Optional: Show categorical column distribution
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        st.subheader("🧮 Categorical Column Distribution")
        selected_cat_col = st.selectbox("Select a categorical column to visualize", categorical_cols)
        if selected_cat_col:
            cat_counts = data[selected_cat_col].value_counts().reset_index()
            cat_counts.columns = [selected_cat_col, 'Count']

            plt.figure(figsize=(10, 5))
            sns.barplot(data=cat_counts, x=selected_cat_col, y='Count')
            plt.xticks(rotation=45)
            st.pyplot(plt)
    else:
        st.info("No categorical columns available for distribution plot.")

# Page 2: Exploratory Data Analysis
def eda(data):
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.write("This section provides visual insights into the dataset.")

    numeric_data = data.select_dtypes(include=['float64', 'int64'])

    if numeric_data.empty:
        st.warning("No numeric columns available for EDA.")
        return

    tab1, tab2, tab3 = st.tabs(["📉 Correlation Heatmap", "📊 Histogram", "🔁 Scatter Plot"])

    with tab1:
        st.subheader("Correlation Heatmap")
        st.caption("Shows pairwise correlation between numeric features.")

        corr = numeric_data.corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        st.pyplot(plt)

    with tab2:
        st.subheader("Histogram")
        st.caption("View distribution of a selected numeric column.")

        hist_col = st.selectbox("Select a column for histogram", numeric_data.columns, key="hist")
        bins = st.slider("Number of bins", min_value=5, max_value=100, value=30, step=5)

        plt.figure(figsize=(10, 6))
        sns.histplot(data[hist_col], kde=True, bins=bins, color='skyblue')
        plt.title(f"Histogram of {hist_col}")
        st.pyplot(plt)

    with tab3:
        st.subheader("Scatter Plot")
        st.caption("Visualize relationships between two numeric variables.")

        col1 = st.selectbox("Select X-axis column", numeric_data.columns, key="x_col")
        col2 = st.selectbox("Select Y-axis column", numeric_data.columns, key="y_col")

        if col1 != col2:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=data[col1], y=data[col2], alpha=0.7)
            plt.title(f"{col1} vs {col2}")
            st.pyplot(plt)
        else:
            st.info("Please select different columns for X and Y axes.")


# Page 3: Modeling and Prediction
def modeling_and_prediction(data):
    st.title("Modeling and Prediction")
    st.write("Train a model and make predictions.")

    target = st.selectbox("Select Target Variable", data.columns)
    features = st.multiselect("Select Feature Variables", [col for col in data.columns if col != target])

    if target and features:
        try:
            X = data[features].dropna()
            y = data[target].loc[X.index]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            st.write("### Model Performance")
            st.write(f"RMSE: {rmse:.2f}")
            st.write(f"R²: {r2:.2f}")

            st.write("### Feature Importance")
            importance = model.feature_importances_
            feature_importance = pd.DataFrame({'Feature': features, 'Importance': importance}).sort_values(by='Importance', ascending=False)
            st.write(feature_importance)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance)
            st.pyplot(plt.gcf())
            plt.clf()
        except Exception as e:
            st.error(f"Error during modeling: {e}")

# Main app logic
def main():
    st.set_page_config(page_title="Air Pollution Analysis App", layout="wide")
    data = load_data()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Overview", "EDA", "Modeling and Prediction"])

    if page == "Data Overview":
        data_overview(data)
    elif page == "EDA":
        eda(data)
    elif page == "Modeling and Prediction":
        modeling_and_prediction(data)

if __name__ == "__main__":
    main()
