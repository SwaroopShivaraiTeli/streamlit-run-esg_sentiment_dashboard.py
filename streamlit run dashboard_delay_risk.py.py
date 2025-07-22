#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# In[24]:


import nltk

# Download necessary NLTK data once
nltk.download('punkt')       # Sentence tokenizer
nltk.download('stopwords')   # Stopword list


# In[16]:


nltk.download('punkt')
nltk.download('stopwords')


# In[17]:


df = pd.read_csv(r"D:\NEW DATA ANALYTICS\New folder\ESG_daily_news.csv")
print(f" Loaded {len(df)} rows.")


# In[22]:


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\@\w+|\#","", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned = [word for word in tokens if word not in stop_words and word.isalpha()]
    return " ".join(cleaned)


# In[26]:


import re
import string

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_text_fallback(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|\@\w+|\#\w+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word.isalpha() and word not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)

# Apply fallback cleaner
df['clean_text'] = df['text'].apply(clean_text_fallback)
df['clean_headline'] = df['headline'].apply(clean_text_fallback)

print("Cleaned using fallback method (no NLTK).")


# In[30]:


from textblob import TextBlob


# In[31]:


def get_sentiment(text):
    return TextBlob(text).sentiment.polarity


# In[32]:


df['sentiment_score'] = df['clean_text'].apply(get_sentiment)


# In[33]:


df['sentiment_label'] = df['sentiment_score'].apply(
    lambda x: 'Positive' if x > 0.1 else ('Negative' if x < -0.1 else 'Neutral')
)


# In[34]:


df[['clean_text', 'sentiment_score', 'sentiment_label']].head()


# In[35]:


#Create a Word Cloud for Each Sentiment Category


# In[36]:


from wordcloud import WordCloud


# In[37]:


# Function to generate word cloud
def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.show()


# In[38]:


# Word clouds per sentiment type
for sentiment in ['Positive', 'Negative', 'Neutral']:
    subset = df[df['sentiment_label'] == sentiment]
    generate_wordcloud(subset['clean_text'], f"WordCloud - {sentiment} Sentiment")


# In[39]:


#Filter Texts by Keywords (e.g., “climate”, “carbon”, “supply”)


# In[40]:


keywords = ['climate', 'carbon', 'supply']


# In[41]:


# Filter rows where any keyword appears in clean_text
filtered_df = df[df['clean_text'].apply(lambda x: any(keyword in x for keyword in keywords))]


# In[42]:


print(f" Found {len(filtered_df)} rows with keywords {keywords}")
filtered_df[['headline', 'sentiment_label']].head()


# In[43]:


# Create a Time-Series Trend of ESG Sentiment


# In[45]:


print("Columns in df:", df.columns.tolist())


# In[48]:


df['date'] = pd.to_datetime(df['Date'], errors='coerce')


# In[49]:


df.dropna(subset=['date'], inplace=True)


# In[50]:


# Group by date and sentiment
sentiment_over_time = df.groupby([df['date'].dt.date, 'sentiment_label']).size().unstack().fillna(0)


# In[51]:


# Plot
plt.figure(figsize=(12, 6))
sentiment_over_time.plot(kind='line', marker='o', figsize=(14, 6))
plt.title('📈 ESG Sentiment Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Articles')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:


##Filter by Specific Keywords


# In[52]:


# Define keywords to search
keywords = ['climate', 'carbon', 'supply']


# In[54]:


# Create a new column to tag which keyword matched (if any)
def match_keywords(text):
    matches = [kw for kw in keywords if kw in text]
    return ", ".join(matches) if matches else None


# In[55]:


# Apply on clean text
df['matched_keywords'] = df['clean_text'].apply(match_keywords)


# In[56]:


# Filter rows where keywords are matched
df_keywords = df[df['matched_keywords'].notnull()]


# In[57]:


print(f"🔍 Found {len(df_keywords)} articles mentioning your keywords.")
df_keywords[['Date', 'headline', 'matched_keywords', 'sentiment_label']].head()


# In[58]:


##Word Clouds per Sentiment Type


# In[59]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

def plot_wordcloud(data, sentiment):
    text = " ".join(data[data['sentiment_label'] == sentiment]['clean_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for {sentiment} Sentiment')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()


# In[60]:


# Generate word clouds for each sentiment
for sentiment in df['sentiment_label'].unique():
    plot_wordcloud(df, sentiment)


# In[ ]:




