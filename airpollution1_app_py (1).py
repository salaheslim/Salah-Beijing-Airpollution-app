import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import numpy as np
import io
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
import io

def eda(data):
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.write("This section provides visual insights into the dataset.")

    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    categorical_data = data.select_dtypes(include=['object', 'category'])

    if numeric_data.empty:
        st.warning("No numeric columns available for EDA.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📉 Correlation Heatmap", 
        "📊 Histogram", 
        "🔁 Scatter Plot", 
        "📦 Box Plot", 
        "📈 Line Chart"
    ])

    # Tab 1: Correlation Heatmap
    with tab1:
        st.subheader("Correlation Heatmap")
        st.caption("Shows pairwise correlation between numeric features.")

        corr = numeric_data.corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        st.pyplot(plt)

    # Tab 2: Histogram
    with tab2:
        st.subheader("Histogram")
        hist_col = st.selectbox("Select a column for histogram", numeric_data.columns, key="hist")
        bins = st.slider("Number of bins", min_value=5, max_value=100, value=30)

        plt.figure(figsize=(10, 6))
        sns.histplot(data[hist_col], kde=True, bins=bins, color='orange')
        plt.title(f"Histogram of {hist_col}")
        st.pyplot(plt)

        # Save and download
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        st.download_button("Download Histogram as PNG", buf.getvalue(), file_name="histogram.png", mime="image/png")

    # Tab 3: Scatter Plot
    with tab3:
        st.subheader("Scatter Plot")
        x = st.selectbox("X-axis", numeric_data.columns, key="scatter_x")
        y = st.selectbox("Y-axis", numeric_data.columns, key="scatter_y")

        if x != y:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=data[x], y=data[y], alpha=0.7)
            plt.title(f"{x} vs {y}")
            st.pyplot(plt)

            # Save and download
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            st.download_button("Download Scatter Plot as PNG", buf.getvalue(), file_name="scatterplot.png", mime="image/png")
        else:
            st.info("Please select different columns for X and Y axes.")

    # Tab 4: Box Plot
    with tab4:
        st.subheader("Box Plot")
        if not categorical_data.empty:
            cat_col = st.selectbox("Select Categorical Column", categorical_data.columns)
            num_col = st.selectbox("Select Numeric Column", numeric_data.columns, key="boxplot")

            plt.figure(figsize=(10, 6))
            sns.boxplot(x=data[cat_col], y=data[num_col])
            plt.xticks(rotation=45)
            plt.title(f"{num_col} Distribution across {cat_col}")
            st.pyplot(plt)

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            st.download_button("Download Box Plot as PNG", buf.getvalue(), file_name="boxplot.png", mime="image/png")
        else:
            st.info("No categorical columns found for box plotting.")

    # Tab 5: Line Chart (for time series)
    with tab5:
        st.subheader("Line Chart")
        if 'date' in data.columns or 'datetime' in data.columns:
            date_col = 'date' if 'date' in data.columns else 'datetime'
            data[date_col] = pd.to_datetime(data[date_col])
            line_col = st.selectbox("Select Numeric Column to Plot", numeric_data.columns, key="linechart")

            plt.figure(figsize=(10, 6))
            sns.lineplot(x=data[date_col], y=data[line_col])
            plt.title(f"{line_col} Over Time")
            plt.xticks(rotation=45)
            st.pyplot(plt)

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            st.download_button("Download Line Chart as PNG", buf.getvalue(), file_name="linechart.png", mime="image/png")
        else:
            st.info("No datetime column found. Please ensure your dataset has a 'date' or 'datetime' column.")



# Page 3: Modeling and Prediction
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import io

def modeling_and_prediction(data):
    st.title("🤖 Modeling and Prediction")
    st.write("Train a regression model and evaluate its performance.")

    target = st.selectbox("🎯 Select Target Variable", data.columns)
    features = st.multiselect("🧮 Select Feature Variables", [col for col in data.columns if col != target])

    if target and features:
        X = data[features]
        y = data[target]

        # Model selection
        st.subheader("🔧 Model Configuration")
        model_type = st.selectbox("Choose a Regression Model", ["Random Forest", "Linear Regression", "Support Vector Regressor (SVR)"])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        if model_type == "Random Forest":
            model = RandomForestRegressor(random_state=42)
        elif model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "Support Vector Regressor (SVR)":
            model = SVR()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        st.subheader("📈 Model Performance")
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        st.metric("RMSE", f"{rmse:.2f}")
        st.metric("R² Score", f"{r2:.2f}")

        # Predicted vs Actual Plot
        st.subheader("🔍 Predicted vs Actual Plot")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        st.pyplot(plt)

        # Download button for predictions
        results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        csv_data = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions as CSV", csv_data, file_name="predictions.csv", mime="text/csv")

        # Feature Importance for tree-based models
        if model_type == "Random Forest":
            st.subheader("📊 Feature Importance")
            importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importance
            }).sort_values(by='Importance', ascending=False)
            st.write(feature_importance_df)

            # Plot importance
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
            plt.title("Feature Importance")
            st.pyplot(plt)

            # Download feature importance
            csv_imp = feature_importance_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Feature Importance as CSV", csv_imp, file_name="feature_importance.csv", mime="text/csv")


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
