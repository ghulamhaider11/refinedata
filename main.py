import streamlit as st
import pandas as pd
from io import BytesIO
from data_operations import data_operations
from visualization import visualization
from utils import load_data, display_preview_option
from styles import inject_custom_css
from ydata_profiling import ProfileReport
import numpy as np
from pydantic_settings import BaseSettings
from ml import machine_learning

def main():
    # Set a wide layout and page title
    st.set_page_config(page_title='Data Exploration Tool', layout='wide')

    # Inject custom CSS
    inject_custom_css()

    # Title and description
    st.markdown("""
    <h1 style="text-align: center;">📊 DATA REFINERY </h1>
    <p style="text-align: center; font-size: 18px; color: white; background-color: #004080; padding: 10px; border-radius: 5px;">
    Welcome to Data Refinery Tool! This app helps you to perform Basic preprocessing & visualize data from CSV or Excel files.<br>
    </p>
    <hr style="border: 1px solid #004080;">
    """, unsafe_allow_html=True)

    # Sidebar Section - File Upload
    st.sidebar.title("📁 Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])

    if 'df' not in st.session_state:
        st.session_state.df = None  # Initialize df in session state

    if uploaded_file:
        st.session_state.df = load_data(uploaded_file)
        if st.session_state.df is not None:
            st.success("File successfully uploaded!")
            display_preview_option(st.session_state.df)

            # Sidebar menu with buttons for sections
            st.sidebar.title("🛠️ Menu")
            if 'data_operations' not in st.session_state:
                st.session_state['data_operations'] = False
            if 'visualization' not in st.session_state:
                st.session_state['visualization'] = False
            if 'machine_learning' not in st.session_state:
                st.session_state['machine_learning'] = False

            # Button to toggle sections
            if st.sidebar.button("🛠️ Data Operations"):
                st.session_state['data_operations'] = True
                st.session_state['visualization'] = False
                st.session_state['machine_learning'] = False

            if st.sidebar.button("📊 Visualization"):
                st.session_state['data_operations'] = False
                st.session_state['visualization'] = True
                st.session_state['machine_learning'] = False

            if st.sidebar.button("🧠 Machine Learning"):
                st.session_state['data_operations'] = False
                st.session_state['visualization'] = False
                st.session_state['machine_learning'] = True

            # Call the corresponding functions based on the button pressed
            if st.session_state['data_operations']:
                st.session_state.df = data_operations(st.session_state.df)  # Ensure updated DataFrame is saved
                st.write(st.session_state.df)  # Debugging output to see DataFrame after operations

            if st.session_state['visualization']:
                visualization(st.session_state.df)

            if st.session_state['machine_learning']:
                machine_learning(st.session_state.df)  # Call the machine_learning function

            # Download updated DataFrame
            st.sidebar.subheader("Download Updated Data")
            file_format = st.sidebar.radio("Choose file format", ["CSV", "Excel"])

            # Check if df is available for download
            if st.session_state.df is not None and not st.session_state.df.empty:  # Check if df is not None and not empty
                if file_format == "CSV":
                    csv = st.session_state.df.to_csv(index=False).encode('utf-8')
                    st.sidebar.download_button("Download Updated CSV", data=csv, file_name='updated_data.csv', mime='text/csv')
                else:
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        st.session_state.df.to_excel(writer, index=False, sheet_name='Sheet1')
                    excel_data = output.getvalue()
                    st.sidebar.download_button("Download Updated Excel", data=excel_data, file_name='updated_data.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
            else:
                st.sidebar.warning("No data available for download. Please perform an operation first.")
        else:
            st.error("Failed to load the data. Please check your file and try again.")
    else:
        st.info("Please upload a CSV or Excel file to get started.")

if __name__ == "__main__":
    main()
