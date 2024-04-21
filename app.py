import pickle
import streamlit as st
import pandas as pd

# Load the KMeans model
model = pickle.load(open('D:\customer_mall\kmeans.pkl', 'rb'))

# Sidebar with option menu
with st.sidebar:
    selected = st.selectbox('Mall Segmentation tool', ['Cluster customers'], index=0, format_func=lambda x: 'Cluster customers')

if selected == 'Cluster customers':
    st.title('Cluster customers Using ML')

# File uploader for CSV data
uploaded_file = st.file_uploader("Choose a .CSV file with columns: Age, Annual Income, Spending Score")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)

    # Cluster the data when the button is clicked
    if st.button('Cluster The data'):
        clusters = model.predict(data)
        data['Clusters'] = clusters
        st.write('Data clustered successfully!')
        
        # Function to convert DataFrame to CSV bytes
        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(data)

        # Download button for the clustered data
        st.download_button("Press to Download", csv, "clustered_data.csv", "text/csv", key='download-csv')
