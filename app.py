import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from googleapiclient.discovery import build
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re
from collections import Counter
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

nltk.download('punkt_tab')
nltk.download('stopwords')
# def download_model_files():
#     github_files = [
#         "en/config.json",
#         "en/special_tokens_map.json",
#         "en/tokenizer_config.json",
#         "en/vocab.txt",
#         "ar/config.json",
#         "ar/special_tokens_map.json",
#         "ar/tokenizer_config.json",
#         "ar/vocab.txt",
#     ]
    
#     if not os.path.exists("model.safetensors"):
#         model_url_en = "https://drive.google.com/uc?id=1Q3WFKlNe12qXcwDnUmrrf6OkamwiXLG-"
#         model_url_ar = "https://drive.google.com/uc?id=1ig3la7xbgKI0Q9iz79b2_OD5cpf_Jx-X"

#         gdown.download(model_url_en, "model.safetensors", quiet=False)
st.set_page_config(page_title="YouTube Comments Sentiment Analysis", layout="wide")
st.title("🎥 YouTube Comments Sentiment Analysis")
st.markdown("---")
def download_model_files(language):
    """إعداد ملفات النموذج حسب اللغة"""
    # تحديد المسارات بناءً على اللغة
    lang_code = "ar" if language == "Arabic" else "en"
    model_dir = f"models/{lang_code}"
    os.makedirs(model_dir, exist_ok=True)
    
    # نسخ الملفات الخفيفة من المجلدات المحلية
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
# https://drive.google.com/file/d/1dceNrR-xO-UclWEAZBCNC3YgzykdNnnH/view?usp=drive_link
    # تحميل model.safetensors من Google Drive
    
    drive_links = {
        "ar": "https://drive.google.com/uc?id=1dceNrR-xO-UclWEAZBCNC3YgzykdNnnH",
        "en": "https://drive.google.com/uc?id=1Q3WFKlNe12qXcwDnUmrrf6OkamwiXLG-"
    }
    
    model_path = f"{model_dir}/model.safetensors"
    if not os.path.exists(model_path):
        try:
            gdown.download(drive_links[lang_code], model_path, quiet=False)
        except Exception as e:
            st.error(f"Error downloading model.safetensors: {str(e)}")

@st.cache_resource
def load_model(language):
    """تحميل النموذج من المجلد المحلي"""
    lang_code = "ar" if language == "Arabic" else "en"
    model_path = f"models/{lang_code}"
    
    download_model_files(language)
    
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# إعدادات اللغة في الشريط الجانبي
st.sidebar.header("🌍 Language Settings")
language = st.sidebar.radio(
    "Select Comment Language:",
    ("Arabic", "English"),
    index=0
)
arabert_prep = ArabertPreprocessor(model_name="models/ar")
def clean_arabic_text(text):
    """معالجة متقدمة للنص العربي"""
    # التنظيف الأساسي
    text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)  # إزالة الإيموجي
    text = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)  # إبقاء الحروف العربية فقط
    
    # المعالجة باستخدام AraBERT
    text = arabert_prep.preprocess(text)
    
    # إزالة الفراغات الزائدة
    text = ' '.join(text.split())
    return text.strip()

# تحميل النموذج المناسب
language_code = "arabic" if language == "Arabic" else "english"
model, tokenizer = load_model(language_code)

def predict_sentiment(text, language):
    """تحليل المشاعر مع تعديل العتبة للمحايد"""
    if language == "arabic":
        # معالجة النص العربي
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
            
            # تعديل العتبة للفئة المحايدة
            if probs[1] < 0.65:  # إذا كانت ثقة المحايد أقل من 65%
                final_pred = torch.argmax(probs * torch.tensor([1.2, 1.0, 1.2]))  # تقليل وزن المحايد
            else:
                final_pred = torch.argmax(probs)
            
            label_map = {0: "سلبي", 1: "محايد", 2: "إيجابي"}
            colors = {0: "🔴", 1: "🟡", 2: "🟢"}
    else:
        # النص الإنجليزي (يبقى كما هو)
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

# إضافة دوال معالجة النص من Notebook
def convert_emojis(text):
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
arabic_stopwords = set(stopwords.words("arabic"))
keep_words = {'لا', 'لم', 'لن', 'ما', 'مش', 'ليس', 'بدون', 'غير', 'إن', 'إذ', 'إذا'}
custom_stopwords = arabic_stopwords - keep_words
def remove_custom_stopwords(tokens):
    """إزالة الكلمات التوقفية المخصصة"""
    return [word for word in tokens if word not in custom_stopwords]
def normalize_arabic(text):
    if has_emoji(text):
        text = convert_emojis(text)
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Remove non-Arabic
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'ء', text)
    text = re.sub(r'ئ', 'ء', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'\bمش\b', 'ليس', text)
    text = re.sub(r'\bمو\b', 'ليس', text)
    text = re.sub(r'\bما (\w+)', r'ليس \1', text)
    text = re.sub(r'\b(\w+)ش\b', r'\1', text)
    text = strip_tashkeel(text)
    text = strip_diacritics(text)
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove digits
    text = re.sub(r'[a-zA-Z]', '', text) # Remove English
    text = re.sub(r'[^\u0621-\u064A]', ' ', text) # Keep Arabic only
    text = re.sub(r'[\u061F\u060C\u061B]', '', text)
    tokens = word_tokenize(text)
    tokens = remove_custom_stopwords(tokens)
    return ' '.join(tokens)

def extract_video_id(url):
    """استخراج معرف الفيديو من الرابط"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com\/watch\?.*v=([a-zA-Z0-9_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# def get_comments_without_api(video_url, max_comments=100):
#     """جلب التعليقات بدون استخدام API"""
#     video_id = extract_video_id(video_url)
#     downloader = YoutubeCommentDownloader()
#     comments = []
#     try:
#         for comment in downloader.get_comments_from_url(f"https://www.youtube.com/watch?v={video_id}"):
#             comments.append({
#                 'author': comment['author'],
#                 'text': comment['text'],
#                 'likes': int(comment['votes']),
#                 'published': ''
#             })
#             if len(comments) >= max_comments:
#                 break
#     except Exception as e:
#         st.error(f"Error during scraping: {str(e)}")
#     return comments

def get_comments_without_api(video_url, max_comments=100):
    video_id = extract_video_id(video_url)
    downloader = YoutubeCommentDownloader()
    comments = []
    try:
        for comment in downloader.get_comments_from_url(f"https://www.youtube.com/watch?v={video_id}"):
            comments.append({
                'author': comment['author'],
                'text': comment['text'],
                'likes': int(comment['votes']),
                'published': ''  # Not available
            })
            if len(comments) >= max_comments:
                break
    except Exception as e:
        st.error(f"Error during scraping: {str(e)}")
    return comments

def get_youtube_comments(video_id, api_key=None, max_comments=100):
    """Fetch video comments from YouTube"""
    return get_comments_without_api(f"https://www.youtube.com/watch?v={video_id}", max_comments)

    # youtube = build('youtube', 'v3', developerKey=api_key)

    # comments = []
    # try:
    #     request = youtube.commentThreads().list(
    #         part='snippet',
    #         videoId=video_id,
    #         maxResults=min(max_comments, 100),
    #         order='relevance'
    #     )

    #     while request and len(comments) < max_comments:
    #         response = request.execute()

    #         for item in response['items']:
    #             comment = item['snippet']['topLevelComment']['snippet']
    #             comments.append({
    #                 'author': comment['authorDisplayName'],
    #                 'text': comment['textDisplay'],
    #                 'likes': comment['likeCount'],
    #                 'published': comment['publishedAt']
    #             })

    #         # Fetch more comments if available
    #         if 'nextPageToken' in response and len(comments) < max_comments:
    #             request = youtube.commentThreads().list(
    #                 part='snippet',
    #                 videoId=video_id,
    #                 pageToken=response['nextPageToken'],
    #                 maxResults=min(max_comments - len(comments), 100),
    #                 order='relevance'
    #             )
    #         else:
    #             break

    # except Exception as e:
    #     st.error(f"Error fetching comments: {str(e)}")
    #     return []

    # return comments

def analyze_comments(comments, language_code="english"):
    """Analyze sentiment of comments with language support"""
    if language_code == "arabic":
        label_map = {0: "سلبي", 1: "إيجابي", 2: "محايد"}
    else:
        label_map = {0: "Negative", 1: "Positive", 2: "Neutral"}

    results = []
    for comment in comments:
        sentiment, confidence, emoji = predict_sentiment(comment['text'], language_code)
        results.append({
            'comment': comment['text'][:100] + "..." if len(comment['text']) > 100 else comment['text'],
            'author': comment['author'],
            'sentiment': sentiment,
            'confidence': confidence,
            'emoji': emoji,
            'likes': comment['likes']
        })
    return results

def create_visualizations(results, language):
    """Create visualizations"""
    df = pd.DataFrame(results)
    
    if language == "arabic":
        titles = {
            'pie': "توزيع المشاعر",
            'bar': "عدد التعليقات حسب المشاعر",
            'hist': "توزيع مستويات الثقة"
        }
        colors = {'إيجابي': '#2ecc71', 'سلبي': '#e74c3c', 'محايد': '#f39c12'}
    else:
        titles = {
            'pie': "Sentiment Distribution",
            'bar': "Number of Comments by Sentiment",
            'hist': "Confidence Level Distribution"
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
        color_discrete_map=colors
    )

    return fig_pie, fig_bar, fig_confidence, df
# App UI
st.set_page_config(page_title="YouTube Comments Sentiment Analysis", layout="wide")

test_samples = [
    ("الفيلم رائع وممتع", "إيجابي"),
    ("سيء جداً ولا أنصح به", "سلبي"),
    ("شاهدت الفيلم البارحة", "محايد")
]

for text, expected in test_samples:
    pred, conf, _ = predict_sentiment(text, "arabic")
    st.write(f"النص: {text} | المتوقع: {expected} | النتيجة: {pred} | الثقة: {conf:.2%}")
# Sidebar
st.sidebar.header("⚙️ Settings")

# API Key input
# api_key = st.sidebar.text_input(
#     "YouTube API Key:",
#     type="password",
#     help="You can obtain an API Key from Google Cloud Console"
# )

# Video URL input
video_url = st.sidebar.text_input(
    "YouTube Video URL:",
    placeholder="https://www.youtube.com/watch?v=..."
)

# Number of comments
max_comments = st.sidebar.slider("Number of comments to analyze:", 10, 500, 100)

# Analyze button
analyze_button = st.sidebar.button("🔍 Analyze Comments", type="primary")

# Single comment analysis
st.sidebar.markdown("---")
st.sidebar.header("📝 Single Comment Analysis")
single_comment = st.sidebar.text_area("Enter a comment to analyze:")

if st.sidebar.button("Analyze Comment"):
    if single_comment:
        sentiment_id, confidence = predict_sentiment(single_comment)
        label_map_ar = {0: "Negative", 1: "Positive", 2: "Neutral"}
        colors = {0: "🔴", 1: "🟢", 2: "🟡"}

        st.sidebar.markdown(f"**Result:** {colors[sentiment_id]} {label_map_ar[sentiment_id]}")
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
                    results = analyze_comments(comments, language_code)
                    fig_pie, fig_bar, fig_hist, df = create_visualizations(results, language_code)
                    
                    st.success(f"✅ Successfully analyzed {len(results)} comments!")
                    
                    # عرض الإحصائيات السريعة
                    col1, col2, col3, col4 = st.columns(4)
                    sentiment_counts = df['sentiment'].value_counts()
                    
                    with col1:
                        positive = sentiment_counts.get('إيجابي' if language_code == "arabic" else 'Positive', 0)
                        st.metric("Positive" if language == "English" else "إيجابي", 
                                 f"{positive} ({positive/len(results):.1%})")
                    
                    with col2:
                        negative = sentiment_counts.get('سلبي' if language_code == "arabic" else 'Negative', 0)
                        st.metric("Negative" if language == "English" else "سلبي", 
                                 f"{negative} ({negative/len(results):.1%})")
                    
                    with col3:
                        neutral = sentiment_counts.get('محايد' if language_code == "arabic" else 'Neutral', 0)
                        st.metric("Neutral" if language == "English" else "محايد", 
                                 f"{neutral} ({neutral/len(results):.1%})")
                    
                    with col4:
                        avg_conf = df['confidence'].mean()
                        st.metric("Avg. Confidence" if language == "English" else "متوسط الثقة", 
                                 f"{avg_conf:.2%}")
                    
                    st.markdown("---")
                    
                    # عرض الرسوم البيانية
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("📋 Comments Details")
                    
                    # تصفية النتائج
                    filter_sentiment = st.selectbox(
                        "Filter by sentiment:" if language == "English" else "تصفية حسب المشاعر",
                        ["All"] + list(df['sentiment'].unique())
                    )
                    if filter_sentiment != "All":
                        filtered_df = df[df['sentiment'] == filter_sentiment]
                    else:
                        filtered_df = df
                    
                    # عرض الجدول
                    display_cols = ['author', 'comment', 'sentiment', 'confidence', 'likes']
                    display_df = filtered_df[display_cols].copy()
                    display_df.columns = ['Author', 'Comment', 'Sentiment', 'Confidence', 'Likes']
                    display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.2%}")
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # زر التنزيل
                    csv = df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 Download Results (CSV)",
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
    """)