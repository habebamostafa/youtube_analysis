import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from collections import Counter
from youtube_comment_downloader import YoutubeCommentDownloader
import gdown
import os
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
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        
        # Debug output
        st.write("Model loaded successfully")
        st.write(f"Model architecture: {model.__class__.__name__}")
        st.write(f"Number of classes: {model.config.num_labels}")
        
        if hasattr(model.config, 'id2label'):
            st.write("Class labels:", model.config.id2label)
        else:
            st.warning("No class label mapping found in model config")
        
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

# إعدادات اللغة في الشريط الجانبي
st.sidebar.header("🌍 Language Settings")
language = st.sidebar.radio(
    "Select Comment Language:",
    ("Arabic", "English"),
    index=0
)

language_code = "arabic" if language == "Arabic" else "english"
model, tokenizer = load_model(language) 

if model is None or tokenizer is None:
    st.error("Failed to load model - please check the error messages above")
    st.stop()

def predict_sentiment(text, language):
    """تحليل المشاعر للنص - Fixed version"""
    if not text.strip():
        return "غير محدد" if language == "arabic" else "Unknown", 0.0, "⚪"
    
    try:
        # Tokenize input
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Verify model output dimensions
            if logits.shape[1] != model.config.num_labels:
                st.error(f"Model output dimension mismatch! Expected {model.config.num_labels} classes, got {logits.shape[1]}")
                return "خطأ" if language == "arabic" else "Error", 0.0, "⚪"
            
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            predicted_class = torch.argmax(logits, dim=1).item()
            
            # Ensure predicted class is valid
            if predicted_class >= model.config.num_labels:
                st.error(f"Invalid class prediction: {predicted_class} (max is {model.config.num_labels-1})")
                return "خطأ" if language == "arabic" else "Error", 0.0, "⚪"
            
            confidence = probabilities[predicted_class].item()
            
            # Use model's built-in labels if available
            if hasattr(model.config, 'id2label') and model.config.id2label:
                model_label = model.config.id2label[str(predicted_class)]
                
                # Map English model labels to desired language
                if language == "arabic":
                    label_mapping = {
                        "Negative": "سلبي",
                        "Positive": "إيجابي", 
                        "Neutral": "محايد"
                    }
                    sentiment_label = label_mapping.get(model_label, model_label)
                else:
                    sentiment_label = model_label
                
                # Color mapping
                color_mapping = {
                    "Negative": "🔴", "سلبي": "🔴",
                    "Positive": "🟢", "إيجابي": "🟢", 
                    "Neutral": "🟡", "محايد": "🟡"
                }
                color = color_mapping.get(sentiment_label, "⚪")
                
            else:
                # Fallback if no model labels available
                if language == "arabic":
                    labels = ["سلبي", "محايد", "إيجابي"]  # Reordered to match model: 0=Negative, 1=Neutral, 2=Positive
                    colors = ["🔴", "🟡", "🟢"]
                else:
                    labels = ["Negative", "Neutral", "Positive"]  # Reordered to match model
                    colors = ["🔴", "🟡", "🟢"]
                
                # Ensure we have enough labels
                if predicted_class >= len(labels):
                    return "غير محدد" if language == "arabic" else "Unknown", 0.0, "⚪"
                    
                sentiment_label = labels[predicted_class]
                color = colors[predicted_class]
            
            return sentiment_label, confidence, color
            
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return "خطأ" if language == "arabic" else "Error", 0.0, "⚪"

# Display model info for debugging
st.write(f"Model configuration: {model.config}")
st.write(f"Model class names: {model.config.id2label if hasattr(model.config, 'id2label') else 'Not available'}")

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

def get_comments_without_api(video_url, max_comments=100):
    """Fetch comments without API"""
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

def analyze_comments(comments, language_code="english"):
    """Analyze sentiment of comments with language support"""
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

# Sidebar
st.sidebar.header("⚙️ Settings")

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
        sentiment, confidence, emoji = predict_sentiment(single_comment, language_code)
        st.sidebar.markdown(f"**Result:** {emoji} {sentiment}")
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
                    
                    # Get labels based on language
                    if language_code == "arabic":
                        pos_label, neg_label, neu_label = "إيجابي", "سلبي", "محايد"
                        pos_text, neg_text, neu_text, conf_text = "إيجابي", "سلبي", "محايد", "متوسط الثقة"
                    else:
                        pos_label, neg_label, neu_label = "Positive", "Negative", "Neutral"
                        pos_text, neg_text, neu_text, conf_text = "Positive", "Negative", "Neutral", "Avg. Confidence"
                    
                    with col1:
                        positive = sentiment_counts.get(pos_label, 0)
                        st.metric(pos_text, f"{positive} ({positive/len(results):.1%})")
                    
                    with col2:
                        negative = sentiment_counts.get(neg_label, 0)
                        st.metric(neg_text, f"{negative} ({negative/len(results):.1%})")
                    
                    with col3:
                        neutral = sentiment_counts.get(neu_label, 0)
                        st.metric(neu_text, f"{neutral} ({neutral/len(results):.1%})")
                    
                    with col4:
                        avg_conf = df['confidence'].mean()
                        st.metric(conf_text, f"{avg_conf:.2%}")
                    
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
                        "Filter by sentiment:" if language == "English" else "تصفية حسب المشاعر:",
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
    
    ## 🚀 How to Use:
    1. Select your preferred language (Arabic/English)
    2. Enter a YouTube video URL in the sidebar
    3. Choose the number of comments to analyze
    4. Click "🔍 Analyze Comments"
    5. View results and download CSV if needed
    """)