import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load data
df = pd.read_csv("ESG_daily_news.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.dropna(subset=['Date'], inplace=True)

# Clean text fallback
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import string

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|\@\w+|\#\w+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word.isalpha() and word not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(clean_text)
df['clean_headline'] = df['headline'].apply(clean_text)

# Sentiment scoring
from textblob import TextBlob

df['sentiment_score'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['sentiment_label'] = df['sentiment_score'].apply(
    lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral')
)

# Keyword match
keywords = ['climate', 'carbon', 'supply']
def match_keywords(text):
    matches = [kw for kw in keywords if kw in text]
    return ", ".join(matches) if matches else None

df['matched_keywords'] = df['clean_text'].apply(match_keywords)
df_keywords = df[df['matched_keywords'].notnull()]

# ---- Streamlit UI ----
st.set_page_config(layout="wide", page_title="🌍 ESG Sentiment Analyzer")

st.title("🌿 ESG News Sentiment Dashboard")
st.markdown("Analyze sentiment trends in ESG-related news headlines with keyword filtering and visual analytics.")

# Filters
with st.sidebar:
    st.header("🔍 Filters")
    selected_sentiments = st.multiselect("Select Sentiment(s)", options=df['sentiment_label'].unique(), default=df['sentiment_label'].unique())
    keyword_filter = st.text_input("Filter by keyword (e.g. climate)", "")
    show_keyword_only = st.checkbox("Show only articles with ESG keywords", value=False)

# Filtered Data
filtered = df[df['sentiment_label'].isin(selected_sentiments)]

if keyword_filter.strip():
    filtered = filtered[filtered['clean_text'].str.contains(keyword_filter.strip().lower())]

if show_keyword_only:
    filtered = filtered[filtered['matched_keywords'].notnull()]

# Time-Series Plot
st.subheader("📈 Sentiment Over Time")
sentiment_over_time = filtered.groupby([pd.Grouper(key='Date', freq='W'), 'sentiment_label']).size().unstack().fillna(0)

fig, ax = plt.subplots(figsize=(12, 5))
sentiment_over_time.plot(ax=ax)
ax.set_title("Sentiment Trend by Week")
ax.set_ylabel("Article Count")
st.pyplot(fig)

# Word Clouds
st.subheader("☁️ Word Clouds by Sentiment")
cols = st.columns(len(df['sentiment_label'].unique()))
for i, sentiment in enumerate(df['sentiment_label'].unique()):
    text = " ".join(filtered[filtered['sentiment_label'] == sentiment]['clean_text'])
    wordcloud = WordCloud(width=400, height=300, background_color='white').generate(text)
    with cols[i]:
        st.markdown(f"**{sentiment}**")
        st.image(wordcloud.to_array(), use_column_width=True)

# Keyword Matches
st.subheader("📰 Articles Matching ESG Keywords")
st.dataframe(filtered[['Date', 'headline', 'matched_keywords', 'sentiment_label']].sort_values('Date', ascending=False), use_container_width=True)
