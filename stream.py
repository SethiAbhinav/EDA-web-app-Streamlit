import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from pandas_profiling import profile_report
import base64
from streamlit.elements.legacy_altair import generate_chart

from streamlit.type_util import OptionSequence

st.set_page_config(page_title = "EzDA", 
                    page_icon = ":bar_chart:",
                    layout = "wide")

main_bg = "background.jpg"
main_bg_ext = "jpg"

side_bg = "sidebar.jpg"
side_bg_ext = "jpg"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# add this -->  
# hide_streamlit_style = """
#             <style>
#             #MainMenu {visibility: hidden;}
#             header {visibility: hidden;}
#             footer {visibility: hidden;}
#             </style>
#             """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.title("EzDA")
st.text("A data analytics tool to make EDA simpler than ever!")

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download ="cleaned_{uploaded_file.name.split(".")[0]}.csv">Download csv file</a>'
    return href

@st.cache
def load_data():
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine = "openpyxl")
        elif uploaded_file.name.endswith('.pkl'):
            df = pd.read_pickle(uploaded_file)
        return df

if st.sidebar.checkbox("Begin the EDA-venture"):
    # with st.expander("Upload file"):
    st.subheader("Upload a file:")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx","pkl"])
    st.markdown("**Note:** Only .csv, .xlsx and .pkl files are supported.")
    df = load_data()
else:
    df = None

if df is not None:
    st.sidebar.header("Choose your task")
    task = st.sidebar.selectbox("", ["Data Exploration", "Data Cleaning", "Data Visualization", "Data Profiling"])
    if task == "Data Exploration":
        if st.button("Show Data"):
            st.dataframe(df)
        st.subheader("Visualise a column:")
        cols = ['None']
        cols.extend(df.columns)
        plot_col = st.selectbox("Select a column", cols)
        if plot_col != 'None':
            st.markdown(f"**Plotting the distribution of : {plot_col}**")
            st.altair_chart(alt.Chart(df).mark_bar().encode(
        x=alt.X(plot_col, bin=alt.Bin(maxbins=20)),
        y='count()'))
        else:
            st.markdown("**No column selected.**")
    elif task == "Data Cleaning":
            # multiselect box to chose the columns to remove
            st.subheader("Select the columns to remove")
            st.markdown("A correlation matrix (for all applicable columns) has been provided for reference : ")
            matrix = df.corr()
            plt.figure(figsize=(16,12))
            # Create a custom diverging palette
            cmap = sns.diverging_palette(250, 15, s=75, l=40,
                                        n=9, center="light", as_cmap=True)
            _ = sns.heatmap(matrix, center=0, annot=True, 
                            fmt='.2f', square=True, cmap=cmap)
            # show the corr 
            st.pyplot(plt)
            cols = df.columns
            columns = []
            for col in cols:
                columns.append(col)
            # print(columns)
            
            cols_to_use = st.multiselect(label = "Select the columns you wish to use", options = df.columns, default = columns)
            if st.button("Filter columns"):
                df = df[cols_to_use]
                st.dataframe(df)
                st.markdown(get_table_download_link(df), unsafe_allow_html=True)
                
            # choose target column
            # target = st.selectbox("Choose target column", df.columns)

    elif task == "Data Visualization":
        st.set_option('deprecation.showPyplotGlobalUse', False)
        if st.button("Show Data"):
            st.dataframe(df)
        st.text("The plots of all columns with numerical entries: \n")
        with st.spinner("Generating plots..."):
            df.hist(bins=30, figsize=(20,20))
            st.pyplot()
        st.balloons()
        

    elif task == "Data Profiling":
        with st.spinner("Creating Profile. May take a while..."):
            profile = df.profile_report(title="Data Profile")
            profile.config.html.minify_html = False
            profile.to_file(output_file="data_profile.html")
            st.markdown("<a href='http://127.0.0.1:5500/data_profile.html'>Data Profile</a>", unsafe_allow_html=True)
        
