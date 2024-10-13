from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


    
    return accuracy, report

# In the main function:
if st.sidebar.checkbox("Enable Machine Learning"):
    target_column = st.selectbox("Select target column", df.columns)
    feature_columns = st.multiselect("Select feature columns", df.columns)
    if st.button("Train Model"):
        accuracy, report = train_model(df, target_column, feature_columns)
        st.write(f"Model Accuracy: {accuracy}")
        st.text(report)
        
        
        
        



#time series 

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

def perform_time_series_analysis(df, date_column, value_column):
    df[date_column] = pd.to_datetime(df[date_column])
    df.set_index(date_column, inplace=True)
    
    result = seasonal_decompose(df[value_column], model='additive', period=30)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 16))
    result.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    result.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    result.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    result.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    
    return fig

# In the main function:
if st.sidebar.checkbox("Enable Time Series Analysis"):
    date_column = st.selectbox("Select date column", df.columns)
    value_column = st.selectbox("Select value column", df.columns)
    if st.button("Perform Time Series Analysis"):
        fig = perform_time_series_analysis(df, date_column, value_column)
        st.pyplot(fig)
        
        
        
        
#data version control 

import dvc
import os

def initialize_dvc():
    if not os.path.exists('.dvc'):
        dvc.api.init()
        st.success("DVC initialized in the current directory.")
    else:
        st.info("DVC is already initialized.")

def add_to_dvc(df, filename):
    df.to_csv(filename, index=False)
    os.system(f"dvc add {filename}")
    os.system("dvc push")
    st.success(f"File {filename} added to DVC and pushed to remote storage.")

# In the main function:
if st.sidebar.checkbox("Enable Data Version Control"):
    if st.button("Initialize DVC"):
        initialize_dvc()
    
    if st.button("Add current dataframe to DVC"):
        filename = st.text_input("Enter filename to save (e.g., data.csv)")
        add_to_dvc(df, filename)
        
        
        
#data quality assesment 

def assess_data_quality(df):
    quality_report = {
        "completeness": df.notna().mean().to_dict(),
        "uniqueness": df.nunique().to_dict(),
        "consistency": {},
        "validity": {}
    }
    
    # Check for consistency in categorical columns
    for col in df.select_dtypes(include=['object']):
        quality_report["consistency"][col] = df[col].value_counts(normalize=True).to_dict()
    
    # Check for validity in numeric columns
    for col in df.select_dtypes(include=['int64', 'float64']):
        quality_report["validity"][col] = {
            "min": df[col].min(),
            "max": df[col].max(),
            "mean": df[col].mean(),
            "std": df[col].std()
        }
    
    return quality_report

# In the main function:
if st.sidebar.checkbox("Assess Data Quality"):
    quality_report = assess_data_quality(df)
    st.json(quality_report)