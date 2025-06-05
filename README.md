# Yelp Review Extraction and Sentiment Analysis Pipeline

An advanced, robust framework engineered to programmatically harvest, preprocess, and scrutinize consumer evaluations from Yelp. Curated by SD7Campeon, this repository harnesses formidable Python libraries—`requests`, `BeautifulSoup`, `pandas`, `nltk`, `textblob`, `matplotlib`, and `seaborn`—to orchestrate meticulous data procurement, rigorous textual normalization, lexical refinement, sentiment quantification, and perspicuous visualization for profound analytical insights.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [System Architecture](#system-architecture)
- [Web Scraping Methodology](#web-scraping-methodology)
- [Data Ingestion and Structuring](#data-ingestion-and-structuring)
- [Textual Data Preprocessing](#textual-data-preprocessing)
- [Lexical Normalization via Lemmatization](#lexical-normalization-via-lemmatization)
- [Sentiment Quantification](#sentiment-quantification)
- [Exploratory Data Analysis and Visualization](#exploratory-data-analysis-and-visualization)
- [Error Handling and Robustness](#error-handling-and-robustness)
- [Limitations and Ethical Considerations](#limitations-and-ethical-considerations)
- [Future Enhancements](#future-enhancements)
- [License and Contribution](#license-and-contribution)

---

## Project Overview

Authored by SD7Campeon, this pipeline exemplifies a sophisticated, end-to-end workflow for extracting Yelp reviews, ameliorating textual data quality, and elucidating customer sentiment. It amalgamates web scraping, data structuring, lexical preprocessing, and advanced analytics to empower researchers, data scientists, and business analysts with actionable insights into consumer feedback.

---

## Prerequisites

To operationalize this pipeline, ensure your environment is equipped with requisite dependencies. Execute these commands in your terminal or notebook:

```bash
!pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install nltk textblob pandas numpy requests beautifulsoup4 matplotlib seaborn
!apt-get update && apt-get install -y chromium-chromedriver
```

### Imports for Pipeline Execution
```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from textblob import Word, TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from requests.exceptions import RequestException
import time

# Download linguistic corpora for NLTK
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## System Architecture

This pipeline is architecturally stratified into discrete, modular strata:

1. **Web Scraping Layer**: Executes HTTP requests to procure raw HTML from Yelp, employing user-agent obfuscation to evade rudimentary anti-scraping defenses.
2. **Data Structuring Layer**: Parses HTML via BeautifulSoup, distilling review text into a structured Pandas DataFrame.
3. **Text Preprocessing Layer**: Normalizes text through case conversion, punctuation ablation, and exclusion of semantically vacuous tokens.
4. **Lexical Normalization Layer**: Leverages lemmatization to reconcile morphological variants, bolstering semantic coherence.
5. **Analytical Layer**: Quantifies textual metrics and sentiment, yielding polarity and subjectivity indices.
6. **Visualization Layer**: Renders graphical depictions of sentiment distributions and textual statistics.

---

## Web Scraping Methodology

Yelp’s formidable anti-scraping fortifications necessitate a circumspect approach. HTTP GET requests are dispatched with browser-emulating headers to mitigate detection:

```python
url = 'https://www.yelp.com/biz/tesla-san-francisco?osq=Tesla+Dealership'

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/114.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9"
}

try:
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()  # Validate successful response
    print(f"HTTP Status: {response.status_code}")
except RequestException as e:
    raise Exception(f"Failed to retrieve page: {e}")

soup = BeautifulSoup(response.text, 'html.parser')

# Extract reviews, filtering out empty elements
review_elements = soup.findAll(class_='lemon--span__373c0__3997G raw__373c0__3rKqk', attrs={'lang': 'en'})
reviews = [elem.text.strip() for elem in review_elements if elem.text.strip()]
```

---

## Data Ingestion and Structuring

Raw reviews are ingested into a Pandas DataFrame, augmented with foundational metrics for textual analysis:

```python
df = pd.DataFrame(reviews, columns=['review'])

# Compute descriptive statistics
df['word_count'] = df['review'].apply(lambda x: len(x.split()))
df['char_count'] = df['review'].str.len()

def average_word_length(text):
    words = text.split()
    return np.mean([len(word) for word in words]) if words else 0

df['avg_word_length'] = df['review'].apply(average_word_length)

# Import and apply stopwords corpus
stop_words = set(stopwords.words('english'))
df['stopword_count'] = df['review'].apply(
    lambda x: len([word for word in x.lower().split() if word in stop_words])
)

# Display initial DataFrame snapshot
df.head()
```

---

## Textual Data Preprocessing

A rigorous preprocessing regimen enhances text quality:

- **Case Normalization**: Converts text to lowercase for uniformity.
- **Punctuation Excision**: Ablates non-alphanumeric characters via regex.
- **Stopword Elimination**: Excises common English stopwords.
- **Custom Token Filtration**: Removes high-frequency, low-value lexemes.

```python
# Convert to lowercase
df['review_lower'] = df['review'].str.lower()

# Remove punctuation and non-alphanumeric characters
df['review_no_punct'] = df['review_lower'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

# Eliminate stopwords
df['review_no_stopwords'] = df['review_no_punct'].apply(
    lambda x: ' '.join(word for word in x.split() if word not in stop_words)
)

# Define and apply custom stopwords
additional_stopwords = {
    'get', 'us', 'see', 'use', 'said', 'asked', 'day', 'go',
    'even', 'ive', 'right', 'left', 'always', 'would', 'told',
    'one', 'also', 'ever', 'x', 'take', 'let', 'got', 'made',
    'make', 'time', 'back', 'know'
}

df['review_cleaned'] = df['review_no_stopwords'].apply(
    lambda x: ' '.join(word for word in x.split() if word not in additional_stopwords)
)

# Display processed DataFrame
df.head()
```

---

## Lexical Normalization via Lemmatization

Lemmatization reconciles morphological variants, distilling words to their canonical forms to augment semantic fidelity:

```python
df['lemmatized_review'] = df['review_cleaned'].apply(
    lambda text: ' '.join(Word(word).lemmatize() for word in text.split())
)

# Juxtapose original and normalized reviews
print("Original Review:\n", df['review'].iloc[0])
print("\nLemmatized Review:\n", df['lemmatized_review'].iloc[0])
```

---

## Sentiment Quantification

TextBlob’s sentiment analyzer quantifies polarity (positive/negative valence) and subjectivity (opinion intensity) for nuanced opinion mining:

```python
df['polarity'] = df['lemmatized_review'].apply(lambda text: TextBlob(text).sentiment.polarity)
df['subjectivity'] = df['lemmatized_review'].apply(lambda text: TextBlob(text).sentiment.subjectivity)

# Display sentiment metrics
df[['lemmatized_review', 'polarity', 'subjectivity']].head()
```

---

## Exploratory Data Analysis and Visualization

Graphical representations elucidate sentiment and textual distributions, leveraging `matplotlib` and `seaborn` for perspicuous insights:

```python
# Configure plot aesthetics
plt.style.use('seaborn')

# Polarity distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['polarity'], bins=30, kde=True, color='skyblue', stat='density')
plt.title('Distribution of Review Polarity', fontsize=14, pad=15)
plt.xlabel('Polarity Score', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

# Subjectivity distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['subjectivity'], bins=30, kde=True, color='salmon', stat='density')
plt.title('Distribution of Review Subjectivity', fontsize=14, pad=15)
plt.xlabel('Subjectivity Score', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

# Word count distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['word_count'], bins=30, kde=True, color='green', stat='density')
plt.title('Distribution of Review Word Counts', fontsize=14, pad=15)
plt.xlabel('Word Count', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Error Handling and Robustness

To fortify the pipeline against network volatility and scraping impediments, robust error handling is integrated:

```python
def fetch_yelp_reviews(url, headers, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response
        except RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt + 1 == max_retries:
                raise Exception("Max retries exceeded for URL fetch")
            time.sleep(2 ** attempt)  # Exponential backoff
    return None

# Example usage
response = fetch_yelp_reviews(url, headers)
if response:
    soup = BeautifulSoup(response.text, 'html.parser')
    review_elements = soup.findAll(class_='lemon--span__373c0__3997G raw__373c0__3rKqk', attrs={'lang': 'en'})
    reviews = [elem.text.strip() for elem in review_elements if elem.text.strip()]
```

---

## Limitations and Ethical Considerations

- **Anti-Scraping Defenses**: Yelp deploys sophisticated countermeasures. Persistent 403 errors may necessitate headless browsers (e.g., Selenium, Playwright) or proxy rotation.
- **Ethical Data Usage**: This pipeline, crafted by SD7Campeon, is intended for academic and research purposes. Scraping without consent may contravene Yelp’s terms of service or robots.txt.
- **Analytical Bias**: Sentiment models may misinterpret context, sarcasm, or cultural nuances, attenuating accuracy.
- **Performance Constraints**: Large-scale scraping demands rate limiting, robust error handling, and resource optimization.

---

## Future Enhancements

- Integrate advanced NLP models (e.g., BERT) for superior sentiment classification.
- Implement pagination to harvest reviews across multiple pages.
- Augment with real-time data streaming for dynamic analysis.
- Develop a dashboard for interactive visualization and user engagement.

---

## License and Contribution

Distributed under the MIT License by SD7Campeon. Contributions, bug reports, and feature proposals are heartily welcomed.

Adhere to ethical scraping practices, respecting Yelp’s terms of use and robots.txt.

For inquiries, collaboration, or enhancements, please initiate an issue or submit a pull request.

Crafted with precision for the data science community by SD7Campeon, June 05, 2025.
