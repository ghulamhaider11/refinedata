import streamlit as st
import pandas as pd 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve, RocCurveDisplay, PrecisionRecallDisplay

# Function to save a trained model
def save_model(model, filename):
    joblib.dump(model, filename)
    st.success(f"Model saved as {filename}")

# Function to load a saved model
def load_model(filename):
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        st.error("Model file not found.")
        return None

# Function to plot feature importance
def plot_feature_importance(model, feature_columns):
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    # Plotting the feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title("Feature Importance")
    st.pyplot(plt)

def machine_learning(df):
    st.subheader("🧠 Machine Learning Operations")
    
    if df is not None:
        # Let users select the target column and feature columns
        target_column = st.selectbox("Select target column", df.columns)
        feature_columns = st.multiselect("Select feature columns", df.columns)
        
        # Let users select a machine learning model from a dropdown
        model_choice = st.selectbox("Select Machine Learning Model", [
            "Random Forest", 
            "Logistic Regression", 
            "Support Vector Machine (SVM)", 
            "Decision Tree", 
            "K-Nearest Neighbors (KNN)"
        ])
        
        # Hyperparameter tuning based on model choice
        if model_choice == "Random Forest":
            n_estimators = st.slider("Number of Trees (n_estimators)", min_value=10, max_value=200, value=100, step=10)
        elif model_choice == "Decision Tree":
            max_depth = st.slider("Maximum Depth (max_depth)", min_value=1, max_value=20, value=5, step=1)
        elif model_choice == "K-Nearest Neighbors (KNN)":
            n_neighbors = st.slider("Number of Neighbors (n_neighbors)", min_value=1, max_value=20, value=5, step=1)
        
        # Cross-validation option
        cross_validate = st.checkbox("Enable Cross-Validation")
        if cross_validate:
            n_folds = st.slider("Number of Cross-Validation Folds", min_value=2, max_value=10, value=5, step=1)

        # Ensure that the user has selected both target and feature columns
        if feature_columns and target_column:
            model_filename = "trained_model.pkl"  # Default filename for the saved model

            if st.button("Train Model"):
                # Prepare the features (X) and target (y)
                X = df[feature_columns]
                y = df[target_column]
                
                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train the selected model with the corresponding hyperparameters
                if model_choice == "Random Forest":
                    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                elif model_choice == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                elif model_choice == "Support Vector Machine (SVM)":
                    model = SVC(kernel='linear', random_state=42)
                elif model_choice == "Decision Tree":
                    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                elif model_choice == "K-Nearest Neighbors (KNN)":
                    model = KNeighborsClassifier(n_neighbors=n_neighbors)
                
                # Cross-validation or regular training
                if cross_validate:
                    # Perform cross-validation
                    cv_scores = cross_val_score(model, X, y, cv=n_folds)
                    st.write(f"Cross-Validation Scores: {cv_scores}")
                    st.write(f"Mean CV Score: {np.mean(cv_scores):.2f}")
                else:
                    # Train the model on the training set and evaluate on the test set
                    model.fit(X_train, y_train)
                    predictions = model.predict(X_test)
                    accuracy = accuracy_score(y_test, predictions)
                    report = classification_report(y_test, predictions)
                    
                    # Display the results
                    st.write(f"Model Accuracy: {accuracy:.2f}")
                    st.text(report)
                    
                    # Feature importance for models that support it
                    if model_choice in ["Random Forest", "Decision Tree"]:
                        st.subheader("Feature Importance")
                        plot_feature_importance(model, feature_columns)
                    
                    # Option to save the trained model
                    if st.checkbox("Save Model"):
                        save_model(model, model_filename)
            
            # Option to load and use a saved model
            if st.checkbox("Load Saved Model"):
                loaded_model = load_model(model_filename)
                if loaded_model is not None:
                    st.success(f"Loaded model from {model_filename}")
                    predictions = loaded_model.predict(X_test)
                    accuracy = accuracy_score(y_test, predictions)
                    st.write(f"Accuracy of Loaded Model: {accuracy:.2f}")

        else:
            st.warning("Please select both target and feature columns.")
