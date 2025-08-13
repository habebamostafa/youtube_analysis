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
import shutil
st.set_page_config(page_title="YouTube Comments Sentiment Analysis", layout="wide")
st.title("🎥 YouTube Comments Sentiment Analysis")
st.markdown("---")
def download_model_files(language):
    """إعداد ملفات النموذج حسب اللغة"""
    lang_code = "ar" if language == "Arabic" else "en"
    model_dir = f"models/{lang_code}"
    os.makedirs(model_dir, exist_ok=True)
    
    # نسخ ملفات التكوين
    config_files = ["config.json", "vocab.txt", "special_tokens_map.json", "tokenizer_config.json"]
    
    for filename in config_files:
        src_path = f"{lang_code}/{filename}"
        dst_path = f"{model_dir}/{filename}"
        
        if not os.path.exists(dst_path):
            try:
                shutil.copyfile(src_path, dst_path)
            except Exception as e:
                st.error(f"خطأ في نسخ {filename}: {str(e)}")

    # تحميل ملف النموذج
    model_files = {
        "ar": {
            "url": "https://drive.google.com/uc?id=1dceNrR-xO-UclWEAZBCNC3YgzykdNnnH",
            "dest": f"{model_dir}/model.safetensors"
        },
        "en": {
            "url": "https://drive.google.com/uc?id=1Q3WFKlNe12qXcwDnUmrrf6OkamwiXLG-",
            "dest": f"{model_dir}/model.safetensors"
        }
    }
    
    if not os.path.exists(model_files[lang_code]["dest"]):
        try:
            gdown.download(model_files[lang_code]["url"], model_files[lang_code]["dest"], quiet=False)
        except Exception as e:
            st.error(f"خطأ في تحميل النموذج: {str(e)}")
@st.cache_resource
def load_model(language):
    """تحميل النموذج من المجلد المحلي"""
    lang_code = "ar" if language == "Arabic" else "en"
    model_path = f"models/{lang_code}"
    
    # التأكد من وجود جميع الملفات المطلوبة
    required_files = [
        "config.json",
        "vocab.txt",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "model.safetensors"
    ]
    
    # تحميل الملفات الناقصة
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    if missing_files:
        st.warning(f"جاري تحميل الملفات الناقصة: {', '.join(missing_files)}")
        download_model_files(language)
    
    # التحقق مرة أخرى بعد التحميل
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    if missing_files:
        st.error(f"لا تزال الملفات الناقصة: {', '.join(missing_files)}")
        return None, None
    
    try:
        # استخدم AutoModel بدلاً من BertModel المحدد
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # التحقق من عدد الفئات
        if model.config.num_labels != 3:
            st.error(f"النموذج يحتوي على {model.config.num_labels} فئات بدلاً من 3")
            return None, None
            
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"خطأ في تحميل النموذج: {str(e)}")
        return None, None
# إعدادات اللغة في الشريط الجانبي
st.sidebar.header("🌍 Language Settings")
language = st.sidebar.radio(
    "Select Comment Language:",
    ("Arabic", "English"),
    index=0
)

# تحميل النموذج المناسب
# تحميل النموذج المناسب
language_code = "arabic" if language == "Arabic" else "english"
model, tokenizer = load_model(language)  # هنا يجب تمرير language وليس language_code
def predict_sentiment(text, language):
    """تحليل المشاعر للنص"""
    if not text.strip():
        return "غير محدد", 0.0, "⚪"
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Get probabilities using softmax
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            
            # Get the predicted class with highest probability
            predicted_class = torch.argmax(probabilities).item()
            
            # Verify the predicted class is within valid range
            num_labels = model.config.num_labels
            if predicted_class >= num_labels:
                st.error(f"Invalid class index {predicted_class} for model with {num_labels} labels")
                predicted_class = num_labels - 1  # Fallback to last class
            
            confidence = probabilities[predicted_class].item()
            
            # Define labels and colors based on language
            if language == "arabic":
                labels = ["سلبي", "إيجابي", "محايد"]
                colors = ["🔴", "🟢", "🟡"]
            else:
                labels = ["Negative", "Positive", "Neutral"]
                colors = ["🔴", "🟢", "🟡"]
            
            # Ensure we have enough labels
            if predicted_class >= len(labels):
                return "غير محدد", 0.0, "⚪"
                
            return labels[predicted_class], confidence, colors[predicted_class]
            
    except Exception as e:
        st.error(f"خطأ في تحليل المشاعر: {str(e)}")
        return "خطأ", 0.0, "⚪"
    
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
                    results = analyze_comments(comments, language.lower())
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