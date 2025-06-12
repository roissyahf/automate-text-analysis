# import library
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud
import re
import emoji
import html
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

################################
#### preprocessing the text ####
################################

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


################################
#### analyzing the text by drawing charts ####
################################


# function to draw bigrams
def create_bigram_barplot(df, text_column):
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

    # Creating bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(bigrams_freq_df['Bigram'], bigrams_freq_df['Frequency'], color="#1798E2")
    ax.set_title(f'Top 15 Bigrams pada Text', fontsize=18)
    ax.set_xlabel('Frequency', fontsize=16)
    ax.set_ylabel('Bigram', fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    fig.tight_layout()
    return fig

# function to draw trigrams
def create_trigram_barplot(df, text_column):
  """
    Generate and save a trigram horizontal barplot from a specified column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the text data.
    text_column (str): The name of the column to generate the bigram from.
  """

  # Tokenization
  df['tokens'] = df[text_column].apply(lambda x: x.split())

  # Trigram (change)
  df['trigrams'] = df['tokens'].apply(lambda x: [x[i] + " " + x[i+1] + " " + x[i+2] for i in range(len(x)-2)])

  # Counting trigram frequency
  trigram_freq = Counter([item for sublist in df['trigrams'] for item in sublist])

  # Creating DataFrame for top bigrams
  trigrams_freq_df = pd.DataFrame(trigram_freq.most_common(15), columns=['Trigram', 'Frequency'])

  # Creating barplot
  fig, ax = plt.subplots(figsize=(12, 8))
  ax.barh(trigrams_freq_df['Trigram'], trigrams_freq_df['Frequency'], color="#F0E546")
  ax.set_title(f'Top 15 Trigrams pada Text', fontsize=18)
  ax.set_xlabel('Frequency', fontsize=16)
  ax.set_ylabel('Trigram', fontsize=16)
  ax.tick_params(axis='x', labelsize=14)
  ax.tick_params(axis='y', labelsize=14)
  fig.tight_layout()  
  return fig

# function to draw word clouds
def create_wordcloud(df, text_column):
    """
    Generate and save a word cloud from a specified column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the text data.
    text_column (str): The name of the column to generate the word cloud from.

    Returns:
    figure
    """
    # choose dataset column
    text_data = df[text_column]

    # generate word cloud
    all_text = ' '.join(text_data.tolist())
    word_cloud = WordCloud(max_words=100, background_color='white',
                           random_state=100, colormap='cool').generate(all_text)
    fig = plt.figure(figsize=(32, 16))
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.title(f'WordCloud of Frequently Used Words', fontsize=20)
    plt.axis("off")

    return fig