import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import profile_report
import base64

st.set_page_config(page_title = "EzDA", 
                    page_icon = ":bar_chart:",
                    layout = "wide",
                    initial_sidebar_state= "collapsed",)

main_bg = "data_background.jpg"
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

if not st.sidebar.checkbox("Begin the EDA-venture"):
    df =  None
    if st.button("What is EDA?"):
        st.info("Exploratory Data Analysis refers to the critical process of performing initial investigations on data so as to:\n- Discover patterns\n- Spot anomalies\n- Test hypothesis\n- Check assumptions with the help of summary statistics and graphical representations.")
else:
    with st.expander("Upload a file"):
        uploaded_file = st.file_uploader("", type=["csv", "xlsx","pkl"])
        st.markdown("**Note:** Only .csv, .xlsx and .pkl files are supported.")
        df = load_data()

if df is not None:
    st.sidebar.header("Choose your task")
    task = st.sidebar.selectbox("", ["Data Exploration", "Data Cleaning", "Data Visualization", "Data Profiling"])
    if task == "Data Exploration":
        with st.expander("Show Data"):
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
        choice = st.sidebar.radio("",["Feature Selection", "Filter Data"])
        if choice == "Feature Selection":
            # multiselect box to chose the columns to remove
            st.subheader("Feature Selection")
            with st.expander("Show correlation matrix"):
                st.info("How does correlation help in feature selection?\n- Features with high correlation are more linearly dependent.\n- Hence have almost the same effect on the dependent variable.\n- When two features have high correlation, we can drop one of the two features.")
                st.markdown("A __*correlation matrix*__ (for all applicable columns) has been provided for reference : ")
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
            
            cols_to_use = st.multiselect(label = "Select the columns you wish to use for your analysis:", options = df.columns, default = columns)
            if st.button("Filter columns"):
                df = df[cols_to_use]
                st.dataframe(df)
                st.markdown(get_table_download_link(df), unsafe_allow_html=True)
        elif choice == "Filter Data":
            st.subheader("Filter Data")
            st.markdown("__Note :__ Upload the cleaned dataset and proceed.")
            df2 = pd.DataFrame(df.isna().sum(),columns = ['Count of missing values'])
            st.dataframe(df2)
            # choose target column
            # target = st.selectbox("Choose target column", df.columns)

    elif task == "Data Visualization":
        st.set_option('deprecation.showPyplotGlobalUse', False)
        with st.expander("Show Data"):
            st.dataframe(df)
        st.text("The plots of all columns with numerical entries: \n")
        with st.spinner("Generating plots..."):
            # with st.expander("Plots"):
            df.hist(bins=30, figsize=(20,20))
            st.pyplot()
        st.balloons()
        

    elif task == "Data Profiling":
        with st.spinner("Creating Profile. May take a while..."):
            profile = df.profile_report(title="Data Profile")
            profile.config.html.minify_html = False
            profile.to_file(output_file="data_profile.html")
            st.markdown("<a href='http://127.0.0.1:5500/data_profile.html'>Data Profile</a>", unsafe_allow_html=True)
        
