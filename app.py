import streamlit as st
import pandas as pd
import helper_functions as hf

# add title
st.set_page_config(page_title="Unveiling hidden insights from raw text data",
                   page_icon=":paper:", layout="wide")
st.title('Analyze your text easily')
st.write('Understanding what your raw text data is trying to tell you can be a daunting task. '
         'This app is designed to help you uncover hidden insights from your text data effortlessly. '
         'Simply upload your CSV or Excel file, and let the app do the rest!')
st.info('This app works really well for Indonesian text, as the preprocessing method was designed for Indonesian text. '
         'However, it can also be used for other languages with some limitations.')

# adjust the layout of the app
adjust_top_pad = """
    <style>
        div.block-container {padding-top:1rem;}
    </style>
    """
st.markdown(adjust_top_pad, unsafe_allow_html=True)

# display the file uploader
uploaded_file = st.file_uploader("Upload a CSV or Excel containing a column named 'text' for analysis", type=["csv", "xlsx", "xls"])

df = None
MAX_ROWS = 20000

# check the file type and read accordingly, then check the column
if uploaded_file is not None:
    try:
        # Read based on file type
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, on_bad_lines='skip')
        elif uploaded_file.name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)
        else:
            st.warning("Unsupported file format. Please upload a .csv or .xlsx/.xls file.")

        # Check if required column exists
        if df is not None and 'text' not in df.columns:
            st.warning("The uploaded file must contain a 'text' column.")
            df = None
            
        elif df is not None:
            # Check row count limit
            if df is not None and len(df) > MAX_ROWS:
                st.warning(f"File has {len(df):,} rows. Please upload a file with no more than {MAX_ROWS:,} rows.")
                df = None
            else:
                st.success(f"File has {len(df):,} rows. It has been uploaded and validated successfully!")

                st.write("Preview of the uploaded data:")
                st.dataframe(df.head())

                st.success("Text column found, processing...")
                # initial cleaning
                df['text'] = df['text'].dropna()
                df['text'] = df['text'].drop_duplicates()

                # text cleaning
                df['text-clean'] = df['text'].apply(hf.clean_review)
                df['text-clean'] = df['text-clean'].apply(hf.remove_stop_words)
                df['text-clean'] = df['text-clean'].apply(hf.normalize_review)

                # show the clean text
                st.write("Preview of the cleaned text:")
                st.write(df['text-clean'].head())

                # display the bigram
                st.success("Generating bigram barplot...")
                fig_bigram = hf.create_bigram_barplot(df, 'text-clean')
                st.pyplot(fig_bigram)

                # display the trigram
                st.success("Generating trigram barplot...")
                fig_trigram = hf.create_trigram_barplot(df, 'text-clean')
                st.pyplot(fig_trigram)

                # display the wordcloud
                st.success("Generating wordcloud...")
                fig_wordcloud = hf.create_wordcloud(df, 'text-clean')
                st.pyplot(fig_wordcloud)

    except Exception as e:
        st.error(f"Error reading file: {e}")

else:
    st.info("Please upload a CSV or Excel file to begin analysis")