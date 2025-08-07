import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re
from youtube_comment_downloader import YoutubeCommentDownloader
import gdown
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from arabert.preprocess import ArabertPreprocessor
from pyarabic.araby import strip_tashkeel, strip_diacritics
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import emoji
import numpy as np
from collections import Counter

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')

# Page config
st.set_page_config(page_title="YouTube Comments Sentiment Analysis", layout="wide")
st.title("🎥 YouTube Comments Sentiment Analysis")
st.markdown("---")

# Language settings in sidebar
st.sidebar.header("🌍 Language Settings")
language = st.sidebar.radio(
    "Select Comment Language:",
    ("Arabic", "English"),
    index=0
)

# Initialize Arabert preprocessor
arabert_prep = ArabertPreprocessor(model_name="aubmindlab/bert-base-arabertv02")

def download_model_files(language):
    """Download model files based on language"""
    lang_code = "ar" if language == "Arabic" else "en"
    model_dir = f"models/{lang_code}"
    os.makedirs(model_dir, exist_ok=True)
    
    # For Arabic, we'll use the fine-tuned model from notebook
    if language == "Arabic":
        model_path = "./ar_model"
        if not os.path.exists(model_path):
            st.error("Arabic model not found. Please ensure the fine-tuned model is in ./ar_model directory")
            return False
        return True
    
    # For English, download standard model files
    light_files = ["config.json", "vocab.txt", "special_tokens_map.json", "tokenizer_config.json"]
    
    for filename in light_files:
        src = f"{lang_code}/{filename}"
        dst = f"{model_dir}/{filename}"
        
        if not os.path.exists(dst):
            try:
                with open(src, 'rb') as f_src, open(dst, 'wb') as f_dst:
                    f_dst.write(f_src.read())
            except Exception as e:
                st.error(f"Error copying {filename}: {str(e)}")
    
    drive_links = {
        "en": "https://drive.google.com/uc?id=1Q3WFKlNe12qXcwDnUmrrf6OkamwiXLG-"
    }
    
    model_path = f"{model_dir}/model.safetensors"
    if not os.path.exists(model_path):
        try:
            gdown.download(drive_links[lang_code], model_path, quiet=False)
        except Exception as e:
            st.error(f"Error downloading model.safetensors: {str(e)}")
            return False
    return True

@st.cache_resource
def load_model(language):
    """Load model from local directory"""
    if language == "Arabic":
        model_path = "./ar_model"  # Path to fine-tuned model
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            model.eval()
            return model, tokenizer
        except Exception as e:
            st.error(f"Error loading Arabic model: {str(e)}")
            return None, None
    else:
        model_path = "models/en"
        try:
            tokenizer = BertTokenizer.from_pretrained(model_path)
            model = BertForSequenceClassification.from_pretrained(model_path)
            model.eval()
            return model, tokenizer
        except Exception as e:
            st.error(f"Error loading English model: {str(e)}")
            return None, None

# Text processing functions
def convert_emojis(text):
    """Convert emojis to Arabic text"""
    text = emoji.demojize(text, language='en')
    emoji_translations = {
        "face_with_tears_of_joy": "ضحك",
        "red_heart": "حب",
        "angry_face": "غضب",
        "crying_face": "حزن",
        "smiling_face_with_smiling_eyes": "سعادة",
        "thumbs_up": "اعجاب",
        "clapping_hands": "تصفيق",
        "fire": "رائع",
        "😂": "ضحك", "❤": "حب", "😍": "حب",
        "😊": "سعادة", "👍": "موافقة", "😢": "حزن",
        "👏": "تصفيق", "🔥": "رائع", "😠": "غضب"
    }

    for emoji_code, arabic_word in emoji_translations.items():
        text = text.replace(f":{emoji_code}:", arabic_word)
    return text

def has_emoji(text):
    """Check if text contains emojis"""
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        u"\U0001F680-\U0001F6FF"  # Transport & Map
        u"\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        u"\U00002500-\U00002BEF"  # Chinese/Japanese/Korean
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        "]+", flags=re.UNICODE)
    return bool(emoji_pattern.search(text))

# Arabic stopwords handling
arabic_stopwords = set(stopwords.words("arabic"))
keep_words = {'لا', 'لم', 'لن', 'ما', 'مش', 'ليس', 'بدون', 'غير', 'إن', 'إذ', 'إذا'}
custom_stopwords = arabic_stopwords - keep_words

def remove_custom_stopwords(tokens):
    """Remove custom stopwords"""
    return [word for word in tokens if word not in custom_stopwords]

def normalize_arabic(text):
    """Advanced Arabic text normalization"""
    if has_emoji(text):
        text = convert_emojis(text)
    
    # Basic cleaning
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Remove non-Arabic
    
    # Character normalization
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'ء', text)
    text = re.sub(r'ئ', 'ء', text)
    text = re.sub(r'ة', 'ه', text)
    
    # Dialect handling
    text = re.sub(r'\bمش\b', 'ليس', text)
    text = re.sub(r'\bمو\b', 'ليس', text)
    text = re.sub(r'\bما (\w+)', r'ليس \1', text)
    text = re.sub(r'\b(\w+)ش\b', r'\1', text)
    
    # Remove diacritics and punctuation
    text = strip_tashkeel(text)
    text = strip_diacritics(text)
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove digits
    text = re.sub(r'[a-zA-Z]', '', text) # Remove English
    text = re.sub(r'[^\u0621-\u064A]', ' ', text) # Keep Arabic only
    text = re.sub(r'[\u061F\u060C\u061B]', '', text)
    
    # Tokenization and stopword removal
    tokens = word_tokenize(text)
    tokens = remove_custom_stopwords(tokens)
    
    return ' '.join(tokens)

def clean_english_text(text):
    """Clean English text"""
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def predict_sentiment(text, language):
    """Predict sentiment with threshold adjustment for neutral"""
    if language == "arabic":
        # Arabic text processing
        text = normalize_arabic(text)
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            
            # Adjust threshold for neutral class
            if probs[1] < 0.65:  # If neutral confidence < 65%
                final_pred = torch.argmax(probs * torch.tensor([1.2, 1.0, 1.2]))  # Reduce neutral weight
            else:
                final_pred = torch.argmax(probs)
            
            label_map = {0: "سلبي", 1: "محايد", 2: "إيجابي"}
            colors = {0: "🔴", 1: "🟡", 2: "🟢"}
    else:
        # English text processing
        text = clean_english_text(text)
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
            final_pred = torch.argmax(probs)
            label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            colors = {0: "🔴", 1: "🟡", 2: "🟢"}
    
    return label_map[final_pred.item()], probs[final_pred].item(), colors[final_pred.item()]

def extract_video_id(url):
    """Extract video ID from URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_comments_without_api(video_url, max_comments=100):
    """Get comments without YouTube API"""
    video_id = extract_video_id(video_url)
    if not video_id:
        return []
    
    downloader = YoutubeCommentDownloader()
    comments = []
    try:
        for comment in downloader.get_comments_from_url(f"https://www.youtube.com/watch?v={video_id}"):
            comments.append({
                'author': comment['author'],
                'text': comment['text'],
                'likes': int(comment['votes']),
                'published': ''
            })
            if len(comments) >= max_comments:
                break
    except Exception as e:
        st.error(f"Error during scraping: {str(e)}")
    return comments

def analyze_comments(comments, language="English"):
    """Analyze sentiment of comments"""
    language_code = "arabic" if language == "Arabic" else "english"
    
    results = []
    for comment in comments:
        sentiment, confidence, emoji_icon = predict_sentiment(comment['text'], language_code)
        results.append({
            'comment': comment['text'][:100] + "..." if len(comment['text']) > 100 else comment['text'],
            'author': comment['author'],
            'sentiment': sentiment,
            'confidence': confidence,
            'emoji': emoji_icon,
            'likes': comment['likes']
        })
    return results

def create_visualizations(results, language):
    """Create visualizations for results"""
    df = pd.DataFrame(results)
    
    if language == "Arabic":
        titles = {
            'pie': "توزيع المشاعر",
            'bar': "عدد التعليقات حسب المشاعر",
            'hist': "توزيع مستويات الثقة",
            'wordcloud': "الكلمات الأكثر تكراراً"
        }
        colors = {'إيجابي': '#2ecc71', 'سلبي': '#e74c3c', 'محايد': '#f39c12'}
    else:
        titles = {
            'pie': "Sentiment Distribution",
            'bar': "Number of Comments by Sentiment",
            'hist': "Confidence Level Distribution",
            'wordcloud': "Most Frequent Words"
        }
        colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#f39c12'}
    
    # Calculate sentiment counts
    sentiment_counts = df['sentiment'].value_counts()
    
    # Pie chart
    fig_pie = px.pie(
        names=sentiment_counts.index,
        values=sentiment_counts.values,
        title=titles['pie'],
        color=sentiment_counts.index,
        color_discrete_map=colors
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')

    # Bar chart
    fig_bar = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        title=titles['bar'],
        labels={'x': 'Sentiment', 'y': 'Number of Comments'},
        color=sentiment_counts.index,
        color_discrete_map=colors
    )

    # Confidence histogram
    fig_confidence = px.histogram(
        df,
        x='confidence',
        color='sentiment',
        title=titles['hist'],
        labels={'confidence': 'Confidence Level', 'count': 'Count'},
        color_discrete_map=colors,
        nbins=20
    )

    return fig_pie, fig_bar, fig_confidence, df

# Load models
model, tokenizer = load_model(language)

# Test samples
test_samples = [
    ("الفيلم رائع وممتع", "إيجابي"),
    ("سيء جداً ولا أنصح به", "سلبي"),
    ("شاهدت الفيلم البارحة", "محايد"),
    ("This movie is amazing!", "Positive"),
    ("Terrible movie, don't watch it", "Negative"),
    ("I watched the movie yesterday", "Neutral")
]

# Display test samples
st.subheader("🧪 Model Test Samples")
for text, expected in test_samples:
    lang = "Arabic" if any(char in text for char in "اأإآبتثجحخدذرزسشصضطظعغفقكلمنهوي") else "English"
    pred, conf, emoji_icon = predict_sentiment(text, lang.lower())
    st.write(f"{emoji_icon} Text: {text} | Expected: {expected} | Result: {pred} | Confidence: {conf:.2%}")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
video_url = st.sidebar.text_input(
    "YouTube Video URL:",
    placeholder="https://www.youtube.com/watch?v=..."
)

max_comments = st.sidebar.slider("Number of comments to analyze:", 10, 500, 100)
analyze_button = st.sidebar.button("🔍 Analyze Comments", type="primary")

# Single comment analysis
st.sidebar.markdown("---")
st.sidebar.header("📝 Single Comment Analysis")
single_comment = st.sidebar.text_area("Enter a comment to analyze:")

if st.sidebar.button("Analyze Single Comment"):
    if single_comment:
        lang = "arabic" if any(char in single_comment for char in "اأإآبتثجحخدذرزسشصضطظعغفقكلمنهوي") else "english"
        sentiment, confidence, emoji_icon = predict_sentiment(single_comment, lang)
        st.sidebar.markdown(f"**Result:** {emoji_icon} {sentiment}")
        st.sidebar.markdown(f"**Confidence Level:** {confidence:.2%}")
    else:
        st.sidebar.warning("Please enter a comment to analyze")

# Main content
if analyze_button:
    if not video_url:
        st.error("⚠️ Please enter the YouTube video URL")
    else:
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("⚠️ Invalid video URL")
        else:
            with st.spinner("🔄 Fetching and analyzing comments..."):
                comments = get_comments_without_api(video_url, max_comments)
                
                if not comments:
                    st.error("❌ No comments found or an error occurred")
                else:
                    results = analyze_comments(comments, language)
                    fig_pie, fig_bar, fig_hist, df = create_visualizations(results, language)
                    
                    st.success(f"✅ Successfully analyzed {len(results)} comments!")
                    
                    # Display quick statistics
                    col1, col2, col3, col4 = st.columns(4)
                    sentiment_counts = df['sentiment'].value_counts()
                    
                    with col1:
                        pos_label = "إيجابي" if language == "Arabic" else "Positive"
                        positive = sentiment_counts.get(pos_label, 0)
                        st.metric(pos_label, f"{positive} ({positive/len(results):.1%})")
                    
                    with col2:
                        neg_label = "سلبي" if language == "Arabic" else "Negative"
                        negative = sentiment_counts.get(neg_label, 0)
                        st.metric(neg_label, f"{negative} ({negative/len(results):.1%})")
                    
                    with col3:
                        neu_label = "محايد" if language == "Arabic" else "Neutral"
                        neutral = sentiment_counts.get(neu_label, 0)
                        st.metric(neu_label, f"{neutral} ({neutral/len(results):.1%})")
                    
                    with col4:
                        avg_conf = df['confidence'].mean()
                        conf_label = "متوسط الثقة" if language == "Arabic" else "Avg. Confidence"
                        st.metric(conf_label, f"{avg_conf:.2%}")
                    
                    st.markdown("---")
                    
                    # Display charts
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("📋 Comments Details")
                    
                    # Filter results
                    filter_label = "تصفية حسب المشاعر" if language == "Arabic" else "Filter by sentiment"
                    filter_sentiment = st.selectbox(
                        filter_label,
                        ["All"] + list(df['sentiment'].unique())
                    )
                    if filter_sentiment != "All":
                        filtered_df = df[df['sentiment'] == filter_sentiment]
                    else:
                        filtered_df = df
                    
                    # Display table
                    display_cols = ['author', 'comment', 'sentiment', 'confidence', 'likes']
                    display_df = filtered_df[display_cols].copy()
                    
                    if language == "Arabic":
                        display_df.columns = ['المؤلف', 'التعليق', 'المشاعر', 'الثقة', 'الإعجابات']
                    else:
                        display_df.columns = ['Author', 'Comment', 'Sentiment', 'Confidence', 'Likes']
                    
                    display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.2%}")
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True,
                        height=400
                    )
                    
                    # Download button
                    csv = df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 Download Results (CSV)" if language == "English" else "📥 تنزيل النتائج (CSV)",
                        data=csv,
                        file_name=f"youtube_sentiment_{video_id}.csv",
                        mime="text/csv"
                    )
else:
    st.markdown("""
    ## 📊 Features:
    - Automatic sentiment analysis of YouTube comments
    - Support for both Arabic and English
    - Interactive visualizations
    - Detailed statistics
    - Filtering and CSV download
    - Single comment analysis
    
    ## 🚀 How to use:
    1. Enter a YouTube video URL in the sidebar
    2. Select the number of comments to analyze
    3. Click "Analyze Comments" button
    4. View results and download as CSV
    """)

    if language == "Arabic":
        st.markdown("""
        ## 📊 الميزات:
        - تحليل تلقائي لمشاعر تعليقات اليوتيوب
        - دعم اللغتين العربية والإنجليزية
        - رسومات بيانية تفاعلية
        - إحصائيات مفصلة
        - تصفية النتائج وتنزيلها كملف CSV
        - تحليل تعليق مفرد
        
        ## 🚀 طريقة الاستخدام:
        1. أدخل رابط فيديو يوتيوب في الشريط الجانبي
        2. اختر عدد التعليقات المراد تحليلها
        3. اضغط على زر "تحليل التعليقات"
        4. عرض النتائج وتنزيلها كملف CSV
        """)