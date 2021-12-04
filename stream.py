import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from pandas_profiling import profile_report
import base64
import sklearn
import sklearn
from sklearn.model_selection import train_test_split

st.set_page_config(page_title = "EzDA", 
                    page_icon = ":bar_chart:",
                    layout = "wide",
                    initial_sidebar_state= "collapsed",)

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

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.title("EzDA")
st.text("A data analytics tool to make EDA simpler than ever!")


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    filename = uploaded_file.name.split(".")[0]
    if filename.startswith("cleaned"):
        filename = filename.split("cleaned_")[1]
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download ="cleaned_{filename}.csv">Download csv file</a>'
    return href

@st.cache(allow_output_mutation=True)
def load_data():
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, na_values=" ", keep_default_na=True)
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
                df.to_csv('file.csv', na_rep='NULL')
                st.markdown(get_table_download_link(df), unsafe_allow_html=True)
                
        elif choice == "Filter Data":
            st.subheader("Filter Data")
            st.markdown("__Note :__ Upload the cleaned dataset and proceed.")
            with st.expander("Show filtered data"):
                st.dataframe(df)
                
            missing_df = pd.DataFrame(df.isna().sum(),columns = ['Count of missing values'])
            with st.expander("Show missing values"):
                st.dataframe(missing_df)
                st.write("Total values in the dataframe:", df.shape[0])

            missing_df = missing_df[missing_df['Count of missing values'] > 0]
            cols = missing_df.index
            # print(cols.values)

            columns_missing = ['None']
            for col in cols.values:
                columns_missing.append(col)

            column_selected = st.selectbox("Select columns to filter", columns_missing)

            if column_selected != 'None':
                st.markdown(f"**Filtering the data for : {column_selected}**")
                task = st.radio("Do what with NaN values?", [ "Fill with mean", "Fill with median"])
                if task == "Fill with mean":

                    # fill nan values in a column with the columns mean
                    mean = df[column_selected].mean()
                    st.write("Mean value:", int(mean))
                    
                    df2 = df[column_selected].fillna(int(mean))
                    st.dataframe(df2)

                    if st.checkbox("Update dataset"):
                        df[column_selected] = df[column_selected].fillna(int(mean))
                        st.write("Dataset updated")

                elif task == "Fill with median":
                    # fill nan values in a column with the columns median
                    median = int(df[column_selected].median())
                    st.write("Median value:", median)

                    df2 = df[column_selected].fillna(int(median))
                    st.dataframe(df2)
                    
                    if st.checkbox("Update dataset"):
                        df[column_selected] = df[column_selected].fillna(median)
                        st.write("Dataset updated")
            elif column_selected == 'None':
                # the current dataframe
                st.markdown("The current dataset is:")
                st.dataframe(df)
                
                st.markdown("**No column selected. If you are done, you can download the file from the link below:**")
                st.markdown(get_table_download_link(df), unsafe_allow_html=True)
            
    elif task == "Data Visualization":
        st.set_option('deprecation.showPyplotGlobalUse', False)
        with st.expander("Show Data"):
            st.dataframe(df)
        st.text("The plots of all columns with numerical entries: \n")

        if st.button("Generate plots"):
            with st.spinner("Generating plots..."):
                df.hist(bins=30, figsize=(20,20))
                st.pyplot()
            st.balloons()

    elif task == "Data Profiling":
        st.subheader("Data Profiling")
        st.markdown("[Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling) is an exceptional tool for Exploratory Data Analysis.")
        st.info("The report serves as this excellent EDA tool that can offer the following benefits:\n- Overview\n- Variables\n- Interactions\n- Correlations\n- Missing values\n- A sample of your data.\n")
        if st.button("Generate report"):
            with st.spinner("Creating Profile. May take a while..."):
                profile = df.profile_report(title="Data Profile")
                profile.config.html.minify_html = False
                profile.to_file(output_file="data_profile.html")
                st.markdown("View your data profile from the link below:\n- <a href='http://127.0.0.1:5500/data_profile.html'>Data Profile</a>", unsafe_allow_html=True)


# 
# TODO
# Add a machine learning component

# # import lazypredict
# # from lazypredict.Supervised import LazyClassifier

#     elif task == "Trivial Machine Learning":
#         task_choice = st.sidebar.radio("Select task: ",["Classification", "Regression"])
#         st.subheader("Trivial Machine Learning")
#         st.write("Upload cleaned dataset & proceed.")
#         with st.expander("Show data"):
#             st.dataframe(df)
        
#         # select target column
#         target = st.selectbox("Choose target column", df.columns)

#         # select columns to use
#         cols = df.columns

#         # train test split
#         X_train, X_test, y_train, y_test = train_test_split(df[cols], df[target], test_size=0.2, random_state=42)

#         st.info("The following table provides you with the best(initial, without any advanced data manipulation applied) choice of model for your task.")

#         if task_choice == "Classification":
#             st.subheader("Basic Machine Learning Classifiers")
#             clf = LazyClassifier(verbose=0,ignore_warnings=True)
#             models, predictions = clf.fit(X_train, X_test, y_train, y_test)
#             st.dataframe(models, high_memory=True)

#         elif task_choice == "Regression":
#             st.subheader("Basic Machine Learning Regressors")
#             reg = LazyRegressor(verbose=0,ignore_warnings=True)
#             models, predictions = reg.fit(X_train, X_test, y_train, y_test)
#             st.dataframe(models, high_memory=True)
# 
