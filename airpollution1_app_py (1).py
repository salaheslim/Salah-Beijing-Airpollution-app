
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import numpy as np
import io
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


# Load data with caching
@st.cache_data
def load_data():
    return pd.read_csv("analyzed_data.csv", on_bad_lines='skip')

# Page 1: Data Overview

def data_overview(data):
    st.title("📊 Data Overview")
    st.write("This page provides a structured overview of the dataset you are working with.")

    # Sidebar slider for preview row count
    num_rows = st.sidebar.slider("Number of rows to preview", min_value=5, max_value=50, value=5, step=5)

    # ---- Dataset Overview: Shape and Data Types ----
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📐 Dataset Shape")
        st.metric(label="Rows", value=data.shape[0])
        st.metric(label="Columns", value=data.shape[1])

    with col2:
        st.subheader("🧬 Data Types")
        dtypes_df = data.dtypes.rename("Type").reset_index().rename(columns={"index": "Column"})
        st.dataframe(dtypes_df)

    # ---- Expanders: Data Preview, Summary, Raw Data ----
    with st.expander("🔍 Preview Sample Data"):
        st.dataframe(data.head(num_rows))

    with st.expander("📈 Summary Statistics"):
        st.dataframe(data.describe(include='all'))

    with st.expander("📑 Raw Data"):
        st.dataframe(data)

    # ---- Categorical Column Distribution ----
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns

    if len(categorical_cols) > 0:
        st.subheader("🧮 Categorical Column Distribution")
        selected_cat_col = st.selectbox("Select a categorical column to visualize", categorical_cols)

        # Plot selected categorical column
        cat_counts = data[selected_cat_col].value_counts().reset_index()
        cat_counts.columns = [selected_cat_col, 'Count']

        plt.figure(figsize=(10, 5))
        sns.barplot(data=cat_counts, x=selected_cat_col, y='Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.clf()  # Clear plot to prevent overlap
    else:
        st.info("No categorical columns available for distribution plot.")

# Page 2: Exploratory Data Analysis

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
        pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
        weather = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']

        corr = numeric_data[pollutants+weather].corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        st.pyplot(plt)
        plt.clf()

    # Tab 2: Histogram
    with tab2:
        st.subheader("Histogram")
        hist_col = st.selectbox("Select a column for histogram", numeric_data.columns, key="hist")
        bins = st.slider("Number of bins", min_value=5, max_value=100, value=30)

        plt.figure(figsize=(10, 6))
        sns.histplot(data[hist_col], kde=True, bins=bins, color='orange')
        plt.title(f"Histogram of {hist_col}")
        st.pyplot(plt)

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        st.download_button("Download Histogram as PNG", buf.getvalue(), file_name="histogram.png", mime="image/png")
        plt.clf()

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

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            st.download_button("Download Scatter Plot as PNG", buf.getvalue(), file_name="scatterplot.png", mime="image/png")
            plt.clf()
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
            plt.clf()
        else:
            st.info("No categorical columns found for box plotting.")

    # Tab 5: Line Chart (for time series)
    with tab5:
        st.subheader("Line Chart")
        date_cols = [col for col in data.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            date_col = st.selectbox("Select datetime column", date_cols)
            data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
            line_col = st.selectbox("Select Numeric Column to Plot", numeric_data.columns, key="linechart")

            plt.figure(figsize=(10, 6))
            sns.lineplot(x=data[date_col], y=data[line_col])
            plt.title(f"{line_col} Over Time")
            plt.xticks(rotation=45)
            st.pyplot(plt)

            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            st.download_button("Download Line Chart as PNG", buf.getvalue(), file_name="linechart.png", mime="image/png")
            plt.clf()
        else:
            st.info("No datetime column found. Please ensure your dataset has a 'date' or 'datetime' column.")




# Page 3: Modeling and Prediction
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
        model_type = st.selectbox("Choose a Regression Model", ["Random Forest", "K-Nearest Neighbors", "Linear Regression", "AdaBoost"])

        # Feature Scaling (for KNN and AdaBoost)
        if model_type in ["K-Nearest Neighbors", "AdaBoost"]:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        if model_type == "Random Forest":
            model = RandomForestRegressor(random_state=42)
        elif model_type == "Linear Regression":
            model = LinearRegression()
        elif model_type == "K-Nearest Neighbors":
            model = KNeighborsRegressor()
        elif model_type == "AdaBoost":
            model = AdaBoostRegressor(random_state=42)

        # Hyperparameter tuning for Random Forest and KNN
        if model_type == "Random Forest":
            n_estimators = st.slider("Number of Trees", min_value=10, max_value=500, value=100)
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        elif model_type == "K-Nearest Neighbors":
            n_neighbors = st.slider("Number of Neighbors", min_value=1, max_value=50, value=5)
            model = KNeighborsRegressor(n_neighbors=n_neighbors)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        st.subheader("📈 Model Performance")
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.metric("RMSE", f"{rmse:.2f}")
        st.metric("MAE", f"{mae:.2f}")
        st.metric("R² Score", f"{r2:.2f}")

        # Predicted vs Actual Plot
        st.subheader("🔍 Predicted vs Actual Plot")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        plt.grid(True)
        st.pyplot(plt)

        # Regression Plot (optional enhancement)
        st.subheader("📉 Regression Line Plot")
        plt.figure(figsize=(10, 6))
        sns.regplot(x=y_test, y=y_pred, scatter_kws={'alpha':0.6})
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Regression Line (Actual vs Predicted)")
        plt.grid(True)
        st.pyplot(plt)

        # Download button for predictions
        results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        csv_data = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions as CSV", csv_data, file_name="predictions.csv", mime="text/csv")

        # Feature Importance for Random Forest
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

        # Save Model
        st.subheader("💾 Save Trained Model")
        save_button = st.button("Save Model")
        if save_button:
            model_filename = f"{model_type}_model.pkl"
            joblib.dump(model, model_filename)
            st.success(f"Model saved as {model_filename}")
            with open(model_filename, 'rb') as f:
                st.download_button("📥 Download Model", f, file_name=model_filename)

# Main app logic
def main():
    st.set_page_config(page_title="Air Pollution Analysis App", layout="wide")
    data = load_data()  # Ensure you have the load_data function correctly implemented

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
