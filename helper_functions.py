# import library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import cv2
from collections import Counter
from wordcloud import WordCloud
import re
import emoji
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from openai import OpenAI
import streamlit as st

import io
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import inch
from datetime import datetime

###############################################################################
#### Clean the text from tag, punctuation, emoji, hashtag, redundant space ####
###############################################################################

character = ['.',',',';',':','-,','...','?','!','(',')','[',']','{','}','<','>','"','/','\'','#','-','@',
             'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
             'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# function to delete redundant character
def repeatcharClean(text):
    for i in range(len(character)):
        charac_long = 5
        while charac_long > 2:
            char = character[i]*charac_long
            text = text.replace(char,character[i])
            charac_long -= 1
    return text

# function to clean review
def clean_review(text):
    # handle potential float values or NaN
    if isinstance(text, float):
      text = str(text) if not pd.isna(text) else ''
    # lowercase text
    text = text.lower()
    # change enter to space
    text = re.sub(r'\n', ' ', text)
    # delete emoji
    text = emoji.demojize(text)
    text = re.sub(':[A-Za-z_-]+:', ' ', text)
    # delete emoticon
    text = re.sub(r"([xX;:]'?[dDpPvVoO3)(])", ' ', text)
    # delete link
    text = re.sub(r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})", "", text)
    # delete username
    text = re.sub(r"@[^\s]+[\s]?", ' ', text) #@[\w\.]+ #@([a-zA-Z0-9._]{1,30})
    # delete hashtag
    text = re.sub(r'#(\S+)', r'\1', text)
    # remove html
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    text = re.sub(html, "", text)
    # delete number and some symbol
    text = re.sub('[^a-zA-Z,.?!]+',' ',text)
    # remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7f]',r'', text)
    # remove punctuation
    text = re.sub(r'[]!"$%&\'()*+,./:;=#@?[\\^_`{|}~-]+', "", text)
    # delete redundant char
    text = repeatcharClean(text)
    # clear spasi
    text = re.sub('[ ]+',' ',text)
    return text

##################################################
#### Clean the text from Indonesian stopwords ####
##################################################

# default stopwords
default_stopwords = list(stopwords.words('indonesian'))

# additional stopwords
more_stopwords = ["yg", "utk", 'cuman', 'deh', 'Btw', 'btw', 'tapi', 'tp', 'gua', 'gue', 'gw', 'lo', 'lu', 'kalo',
                  'kl', 'trs', 'jd', 'nih', 'nich', 'ntar', "nya", '1g', 'gk', 'ecusli', 'dpt', 'dr', 'kpn', 'kok',
                  'kyk', 'donk', 'dong', 'yah', 'u', 'ya', 'ga', 'km', 'eh', 'sih',
                  'en', 'bang', 'br', 'kyk', 'rp', 'jt', 'kan', 'gpp', ' ,', ', ',
                  'sm', "usah", 'mas', 'sob', 'thx', 'ato', 'jg', 'g', 'kuk', 'mak', 'haha', 'ly' 'tp', 'haha', 'dg', 'dri', 'udh',
                  'duh', 'ye', 'wkwkwk', 'syg', 'btw', 'nerjemahin', 'gaes', 'guys', "moga", 'kmrn', 'nemu', "yukkk",
                  'wkwkw', 'klas', 'iw', 'ew', 'lho', 'loh', 'sbnry', 'org', 'gtu', 'bwt',
                  'klrga', 'clau', 'lbh', 'cpet', 'ku', 'wke', 'mba', 'mas', 'sdh', 'kann', 'ol', 'spt', 'dim', 'bs', 'krn', 'jgn',
                  'sapa', 'spt', 'sh', "wakakaka", 'sihhh', 'hehe', 'ini', 'dgn', 'la', "kl", 'ttg', 'mana', 'kmna', 'kmn', 'tdk',
                  'tuh', 'dah', 'kek', 'ko', 'pls', 'bbrp', 'pd', 'bruh', 'bro', 'bre', 'ok', 'okay', 'okei', 'okok',
                  'mah', 'dhhh', 'kpd', 'tuh', 'kzl', 'byar', 'si', 'sii', 'begitu', 'bgt',
                  'cm', 'sy', 'hahahaha', 'weh', 'dlu', 'tuhh', 'tgl', 'aja', 'karena', 'krn', 'bcs', 'cz',
                  'yg', 'bgt', 'blm', 'dah', 'gak', 'po', 'klo', 'dah', 'deh']

# combine all stopword
list_stopwords = []
list_stopwords = default_stopwords + more_stopwords

def remove_stop_words(text):
    # split the text into words
    words = text.split()
    # filter out words with less than 2 characters
    words = [word for word in words if len(word) > 2]
    tokens_without_stopword = [word for word in words if not word in list_stopwords]
    # join the remaining words into a string
    text = ' '.join(tokens_without_stopword)
    return text

###############################################
#### Replace slang words to formal words ####
###############################################

# handling slang words
kamus_alay = pd.read_csv('kamus_alay.csv')

normalize_word_dict = {}
for index, row in kamus_alay.iterrows():
    if row[0] not in normalize_word_dict:
        normalize_word_dict[row[0]] = row[1]

def normalize_review(text):
    # tokenize
    list_text = word_tokenize(text)
    # change slang words
    list_text = [normalize_word_dict[term] if term in normalize_word_dict else term for term in list_text]
    # re-unite words into sentence
    text = " ".join(list_text)
    return text

#############################
#### Bigram plot creation ####
#############################

def visualize_bigram(df, text_column):
    """
    Generate and save a bigram horizontal barplot from a specified column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the text data.
    text_column (str): The name of the column to generate the bigram from.
    """

    # Tokenization
    df['tokens'] = df[text_column].apply(lambda x: x.split())

    # Create Bigrams
    df['bigrams'] = df['tokens'].apply(lambda x: [x[i] + " " + x[i+1] for i in range(len(x)-1)])

    # Counting bigram frequency
    bigram_freq = Counter([item for sublist in df['bigrams'] for item in sublist])

    # Creating DataFrame for top 15 bigrams
    bigrams_freq_df = pd.DataFrame(bigram_freq.most_common(15), columns=['Bigram', 'Frequency'])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(bigrams_freq_df['Bigram'], bigrams_freq_df['Frequency'], color="#1399BB")
    ax.set_title(f'Top 15 Bigrams in Text', fontsize=18)
    ax.set_xlabel('Frequency', fontsize=16)
    ax.set_ylabel('Bigram', fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    fig.tight_layout()

    return fig

##############################
#### Trigram plot creation ####
##############################

def visualize_trigram(df, text_column):
    """
    Generate and save a trigram horizontal barplot from a specified DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the text data.
    text_column (str): The name of the column to generate the trigram from.
    """

    # Tokenization
    df['tokens'] = df[text_column].apply(lambda x: x.split())

    # Create Trigrams
    df['trigrams'] = df['tokens'].apply(lambda x: [x[i] + " " + x[i+1] + " " + x[i+2] for i in range(len(x)-2)])

    # Counting trigram frequency
    trigram_freq = Counter([item for sublist in df['trigrams'] for item in sublist])

    # Creating DataFrame for top 15 trigrams
    trigrams_freq_df = pd.DataFrame(trigram_freq.most_common(15), columns=['Trigram', 'Frequency'])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(trigrams_freq_df['Trigram'], trigrams_freq_df['Frequency'], color="#A5F043")
    ax.set_title(f'Top 15 Trigrams in Text', fontsize=18)
    ax.set_xlabel('Frequency', fontsize=16)
    ax.set_ylabel('Trigram', fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    fig.tight_layout()

    return fig

###################################################
#### Unigram / Most frequent words plot creation ####
###################################################

def read_and_convert_image(image_path):
    """
    Read an image, convert it to black and white if it's in RGB color, and convert it to a numpy array.

    Parameters:
    image_path (str): The path to the image file.

    Returns:
    np.ndarray: The processed image as a numpy array.
    """

    # Read the image
    image = cv2.imread(image_path)

    # Check if the image is in RGB color
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convert the image to black and white
        image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # If the image is already in grayscale, use it as is
        image_bw = image

    # Convert the image to a numpy array
    image_array = np.array(image_bw)

    return image_array


def create_wordcloud_with_mask(df, text_column):
    """
    Generate and save a word cloud from a specified column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the text data.
    text_column (str): The name of the column to generate the word cloud from.
    """
    # Choose dataset column
    text_data = df[text_column]

    # mask used
    my_array = read_and_convert_image('mask-card-diamond.png')

    # Generate word cloud
    all_text = ' '.join(text_data.tolist())
    word_cloud = WordCloud(max_words=100, background_color='white',
                           random_state=100, mask=my_array,
                           contour_width=2, contour_color='white', colormap='hsv'
                           ).generate(all_text)

    # Plot word cloud
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(word_cloud, interpolation='bilinear')
    ax.set_title(f'WordCloud of Frequently Used Words in Text', fontsize=10)
    ax.axis("off")
    
    return fig

################################################################
#### Get top most frequent words, bigrams, trigrams in list ####
################################################################

# function to get most frequent words in list
def get_most_frequent_words(df, text_column, top_n=15):
    """
    Get the most frequent words from a specified column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the text data.
    text_column (str): The name of the column to analyze.
    top_n (int): The number of top frequent words to return.

    Returns:
    pd.DataFrame: A DataFrame containing the most frequent words and their counts.
    """
    # Tokenization
    df['tokens'] = df[text_column].apply(lambda x: x.split())

    # Flatten the list of tokens and count frequency
    all_words = [word for sublist in df['tokens'] for word in sublist]
    word_freq = Counter(all_words)

    # Creating DataFrame for top N words
    most_frequent_words_df = pd.DataFrame(word_freq.most_common(top_n), columns=['Word', 'Frequency'])
    most_frequent_words_df = most_frequent_words_df.sort_values(by='Frequency', ascending=False)

    # Save the top N words into an ordered list (from most to least frequent)
    most_frequent_words_list = most_frequent_words_df['Word'].tolist()
    
    return most_frequent_words_list

# function to get top bigram in list
def get_bigram_list(df, text_column):
    """
    Generate and save a bigram from a specified column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the text data.
    text_column (str): The name of the column to generate the bigram from.

    Returns:
    pd.DataFrame: A DataFrame containing the top 15 bigrams and their frequencies.
    """
    # Tokenization
    df['tokens'] = df[text_column].apply(lambda x: x.split())

    # Create Bigrams
    df['bigrams'] = df['tokens'].apply(lambda x: [x[i] + " " + x[i+1] for i in range(len(x)-1)])

    # Counting bigram frequency
    bigram_freq = Counter([item for sublist in df['bigrams'] for item in sublist])

    # Creating DataFrame for top 15 bigrams
    bigrams_freq_df = pd.DataFrame(bigram_freq.most_common(15), columns=['Bigram', 'Frequency'])
    bigrams_freq_df = bigrams_freq_df.sort_values(by='Frequency', ascending=False)

    # Save the top 15 bigrams into an ordered list (from most to least frequent)
    bigrams_list = bigrams_freq_df['Bigram'].tolist()

    return bigrams_list

# function to get top trigram in list
def get_trigram_list(df, text_column):
    """
    Generate and save a trigram from a specified column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the text data.
    text_column (str): The name of the column to generate the trigram from.

    Returns:
    pd.DataFrame: A DataFrame containing the top 15 trigrams and their frequencies.
    """
    # Tokenization
    df['tokens'] = df[text_column].apply(lambda x: x.split())

    # Create Trigrams
    df['trigrams'] = df['tokens'].apply(lambda x: [x[i] + " " + x[i+1] + " " + x[i+2] for i in range(len(x)-2)])

    # Counting trigram frequency
    trigram_freq = Counter([item for sublist in df['trigrams'] for item in sublist])

    # Creating DataFrame for top 15 trigrams
    trigrams_freq_df = pd.DataFrame(trigram_freq.most_common(15), columns=['Trigram', 'Frequency'])
    trigrams_freq_df = trigrams_freq_df.sort_values(by='Frequency', ascending=False)

    # Save the top 15 trigrams into an ordered list (from most to least frequent)
    trigrams_list = trigrams_freq_df['Trigram'].tolist()
    
    return trigrams_list


############################################
#### Prompt Engineering to get insights ####
############################################

# Set up the OpenAI client
client = OpenAI(api_key=st.secrets["api"]["OPENAI_API_KEY"])

# Setting up the recommended model
model = "gpt-4o-mini"

def generate_insights(most_frequent_words, bigrams, trigrams):
    """
    Generate insights based on the provided linguistic features.

    Parameters:
    most_frequent_words (list): List of most frequent words from the text.
    bigrams (list): List of bigrams from the text.
    trigrams (list): List of trigrams from the text.

    Returns:
    str: Generated insights in Indonesian language.
    """
    
    prompt = f"""
    Analisis teks berikut dalam Bahasa Indonesia:
    Kata-kata yang paling sering muncul: {', '.join(most_frequent_words)}
    Bigrams: {', '.join(bigrams)}
    Trigrams: {', '.join(trigrams)}
    Berikan wawasan mendalam dalam format bullet points tentang tema, sentimen, emosi, dan pola yang muncul dari teks ini.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700
    )
    
    return response.choices[0].message.content.strip()


############################################
#### Report generation ####
############################################

# Convert Matplotlib figure to BytesIO object for saving as image
def matplotlib_fig_to_bytesio(fig):
    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format='png', bbox_inches='tight')
    img_bytes.seek(0)
    return img_bytes

# Generate a PDF report with the bigram, trigram, word cloud images, and AI-generated insights
def draw_wrapped_text(c, text, x, y, max_width, line_height=14, font_name="Helvetica", font_size=10):
    from reportlab.pdfbase.pdfmetrics import stringWidth

    c.setFont(font_name, font_size)
    words = text.split()
    line = ""
    for word in words:
        test_line = line + word + " "
        if stringWidth(test_line, font_name, font_size) <= max_width:
            line = test_line
        else:
            c.drawString(x, y, line.strip())
            y -= line_height
            line = word + " "
    if line:
        c.drawString(x, y, line.strip())
        y -= line_height
    return y

def create_pdf_report(bigram_img, trigram_img, wordcloud_img, insights_text, filename):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    margin = 50
    max_text_width = width - 2 * margin
    y = height - margin
    page_num = 1

    def footer():
        c.setFont("Helvetica", 9)
        c.drawCentredString(width / 2.0, 25, f"Page {page_num}")

    def next_page():
        nonlocal y, page_num
        footer()
        c.showPage()
        page_num += 1
        y = height - margin

    def check_page_break(required_space):
        if y < margin + required_space:
            next_page()

    # --- Title & Metadata ---
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Text Analysis Report")
    y -= 20

    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Filename: {filename}")
    y -= 15
    c.drawString(margin, y, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 30

    # --- Bigrams ---
    check_page_break(220)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Top Bigrams")
    y -= 10
    c.drawImage(ImageReader(bigram_img), margin, y - 200, width=5.5*inch, height=2.5*inch)
    y -= 220

    # --- Trigrams ---
    check_page_break(220)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Top Trigrams")
    y -= 10
    c.drawImage(ImageReader(trigram_img), margin, y - 200, width=5.5*inch, height=2.5*inch)
    y -= 220

    # --- Word Cloud ---
    check_page_break(220)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "Word Cloud")
    y -= 10
    c.drawImage(ImageReader(wordcloud_img), margin, y - 200, width=6*inch, height=3.5*inch)
    y -= 220

    # --- Insights (Markdown as raw text with wrapping) ---
    check_page_break(100)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "AI-Generated Insights")
    y -= 20

    for line in insights_text.strip().split('\n'):
        if line.strip():
            check_page_break(30)
            y = draw_wrapped_text(c, "• " + line.strip(), margin, y, max_text_width)

    # Final footer on last page
    footer()
    c.save()
    buffer.seek(0)
    return buffer