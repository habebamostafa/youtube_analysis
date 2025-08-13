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
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt_tab')
ARABIC_STOPWORDS = set(stopwords.words("arabic"))
import emoji

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
    """تحقق إذا كان النص يحتوي على إيموجي"""
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
    text = re.sub(r'\b(\w+)ش\b', r'\1', text)  # مثل: "فهمت" بدل "فهمتش"
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove digits
    text = re.sub(r'[a-zA-Z]', '', text) # Remove English
    text = re.sub(r'[^\u0621-\u064A]', ' ', text) # Keep Arabic only
    text = re.sub(r'[\u061F\u060C\u061B]', '', text)

    tokens = word_tokenize(text)

    # إزالة ستوب وورد NLTK و الستوب وورد المخصصة مع بعض
    tokens = [word for word in tokens if word not in ARABIC_STOPWORDS]
    tokens = remove_custom_stopwords(tokens)

    return ' '.join(tokens)

st.set_page_config(page_title="YouTube Comments Sentiment Analysis", layout="wide")
st.title("🎥 YouTube Comments Sentiment Analysis")
st.markdown("---")

def download_model_files(language):
    lang_code = "ar" if language == "Arabic" else "en"
    model_dir = f"models/{lang_code}"
    os.makedirs(model_dir, exist_ok=True)
    
    config_files = ["config.json", "vocab.txt", "special_tokens_map.json", "tokenizer_config.json"]
    
    for filename in config_files:
        src_path = f"{lang_code}/{filename}"
        dst_path = f"{model_dir}/{filename}"
        
        if not os.path.exists(dst_path):
            try:
                shutil.copyfile(src_path, dst_path)
            except Exception as e:
                st.error(f"  error {filename}: {str(e)}")

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
            st.error(f"error: {str(e)}")

@st.cache_resource
def load_model(language):
    lang_code = "ar" if language == "Arabic" else "en"
    model_path = f"models/{lang_code}"
    
    try:
        # Load tokenizer and model with error handling
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Validate tokenizer and model compatibility
        # if hasattr(tokenizer, 'vocab_size'):
        #     tokenizer_vocab_size = tokenizer.vocab_size
        # else:
        #     tokenizer_vocab_size = len(tokenizer.get_vocab()) if hasattr(tokenizer, 'get_vocab') else 30000
        
        # model_vocab_size = model.config.vocab_size if hasattr(model.config, 'vocab_size') else 30000
        
        # Debug output
        # st.write("✅ Model loaded successfully")
        # st.write(f"Model architecture: {model.__class__.__name__}")
        # st.write(f"Number of classes: {model.config.num_labels}")
        # st.write(f"Tokenizer vocab size: {tokenizer_vocab_size}")
        # st.write(f"Model vocab size: {model_vocab_size}")
        model.resize_token_embeddings(len(tokenizer))

        # Check vocab size compatibility
        # if tokenizer_vocab_size != model_vocab_size:
        #     st.warning(f"⚠️ Vocab size mismatch! Tokenizer: {tokenizer_vocab_size}, Model: {model_vocab_size}")
        #     st.info("This may cause 'index out of range' errors. Using fallback method for problematic tokens.")
        
        # if hasattr(model.config, 'id2label'):
        #     st.write("Class labels:", model.config.id2label)
        # else:
        #     st.warning("No class label mapping found in model config")
        
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"❌ Model loading failed: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None, None
download_model_files("English")
download_model_files("Arabic")

st.sidebar.header("🌍 Language Settings")
language = st.sidebar.radio(
    "Select Comment Language:",
    ("Arabic", "English"),
    index=0
)

# Add vocab mismatch warning and solution
st.sidebar.markdown("---")
# st.sidebar.header("⚙️ Model Status")

language_code = "arabic" if language == "Arabic" else "english"
model, tokenizer = load_model(language) 

if model is None or tokenizer is None:
    st.error("Failed to load model - please check the error messages above")
    st.stop()

# Check vocab compatibility
model_vocab_size = model.config.vocab_size if hasattr(model.config, 'vocab_size') else 30000
tokenizer_vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer.get_vocab()) if hasattr(tokenizer, 'get_vocab') else 64000

# if tokenizer_vocab_size != model_vocab_size:
#     st.sidebar.warning(f"Vocab Mismatch Detected!")
#     st.sidebar.info(f"Tokenizer: {tokenizer_vocab_size:,} tokens")
#     st.sidebar.info(f"Model: {model_vocab_size:,} tokens")
#     st.sidebar.markdown("**Status:** Using token filtering + fallback")
    
#     with st.sidebar.expander("🔧 How to Fix This"):
#         st.markdown("""
#         **Option 1: Use a compatible tokenizer**
#         ```bash
#         # Download matching tokenizer for your model
#         from transformers import AutoTokenizer
#         tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
#         tokenizer.save_pretrained("models/ar")  # or models/en
#         ```
        
#         **Option 2: Retrain with matching vocab**
#         - Ensure tokenizer and model use same vocabulary file
        
#         **Current Solution:**
#         - Filtering out-of-vocabulary tokens
#         - Using fallback sentiment analysis when needed
#         """)
# else:
#     st.sidebar.success("✅ Tokenizer-Model Compatible")

# Add a toggle for debugging
# debug_mode = st.sidebar.checkbox("🐛 Debug Mode", help="Show detailed processing info")

def predict_sentiment(text, language):
    if not text.strip():
        return " not defined" if language == "arabic" else "Unknown", 0.0, "⚪"
    if language.lower() == "arabic":
        text = normalize_arabic(text)
    try:
        # Clean and preprocess text
        text = text.strip()
        if len(text) > 500:  # Limit text length
            text = text[:500]
        
        # Get vocabulary size from model config
        model_vocab_size = model.config.vocab_size if hasattr(model.config, 'vocab_size') else 30000
        
        # Tokenize with additional safety checks
        try:
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512,
                add_special_tokens=True,
                return_attention_mask=True
            )
            
            # CRITICAL FIX: Filter out-of-vocabulary tokens
            input_ids = inputs['input_ids'][0]
            attention_mask = inputs['attention_mask'][0]
            
            # Replace OOV tokens with [UNK] token ID (usually 1 or 100)
            unk_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 1
            
            # Create mask for valid tokens (within vocab range)
            valid_mask = input_ids < model_vocab_size
            
            # Replace invalid tokens with UNK token
            filtered_input_ids = torch.where(valid_mask, input_ids, torch.tensor(unk_token_id))
            
            # Update inputs with filtered token IDs
            inputs['input_ids'] = filtered_input_ids.unsqueeze(0)
            
            # Verify all tokens are now within range
            if torch.any(inputs['input_ids'] >= model_vocab_size):
                st.warning("Still found OOV tokens after filtering, using fallback")
                return get_fallback_sentiment(text, language)
                
        except Exception as tokenizer_error:
            st.error(f"Tokenization error: {str(tokenizer_error)}")
            return get_fallback_sentiment(text, language)
        
        # Model inference
        with torch.no_grad():
            try:
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Verify model output dimensions
                if logits.shape[1] != model.config.num_labels:
                    return get_fallback_sentiment(text, language)
                
                probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
                predicted_class = torch.argmax(logits, dim=1).item()
                
                # Ensure predicted class is valid
                if predicted_class >= model.config.num_labels or predicted_class < 0:
                    return get_fallback_sentiment(text, language)
                
                confidence = probabilities[predicted_class].item()
                
            except Exception as model_error:
                # Model still failed, use fallback
                return get_fallback_sentiment(text, language)
        
        # Label mapping with safer access
        try:
            # Use integer keys for id2label access
            if hasattr(model.config, 'id2label') and model.config.id2label:
                # Try integer key first, then string key
                model_label = None
                if predicted_class in model.config.id2label:
                    model_label = model.config.id2label[predicted_class]
                elif str(predicted_class) in model.config.id2label:
                    model_label = model.config.id2label[str(predicted_class)]
                label_normalization = {
                    "LABEL_0": "Negative",
                    "LABEL_1": "Positive",
                    "LABEL_2": "Neutral",
                    "0": "Negative",
                    "2": "Positive",
                    "1": "Neutral"
                }
                if model_label in label_normalization:
                    model_label = label_normalization[model_label]
                if model_label:
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
                    raise ValueError("Could not find model label")
                    
            else:
                raise ValueError("No id2label found")
                
        except Exception:
            # Fallback label mapping
            if language == "arabic":
                labels = ["سلبي", "محايد", "إيجابي"]  # 0=Negative, 1=Neutral, 2=Positive
                colors = ["🔴", "🟡", "🟢"]
            else:
                labels = ["Negative", "Neutral", "Positive"]  # 0=Negative, 1=Neutral, 2=Positive
                colors = ["🔴", "🟡", "🟢"]
            
            if predicted_class >= len(labels):
                return get_fallback_sentiment(text, language)
                
            sentiment_label = labels[predicted_class]
            color = colors[predicted_class]
        
        return sentiment_label, confidence, color
            
    except Exception:
        return get_fallback_sentiment(text, language)

def get_fallback_sentiment(text, language):
    """Fallback sentiment analysis using simple keyword matching"""
    text = text.lower()
    
    # Simple keyword-based sentiment
    positive_words = ['good', 'great', 'awesome', 'amazing', 'excellent', 'love', 'like', 'best', 'wonderful', 'fantastic',
                     'جميل', 'رائع', 'ممتاز', 'حب', 'أحب', 'جيد', 'عظيم', 'مذهل']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting', 'stupid', 'ugly',
                     'سيء', 'فظيع', 'أكره', 'قبيح', 'غبي', 'سخيف']
    
    pos_count = sum(1 for word in positive_words if word in text)
    neg_count = sum(1 for word in negative_words if word in text)
    
    if pos_count > neg_count:
        return ("إيجابي" if language == "arabic" else "Positive"), 0.7, "🟢"
    elif neg_count > pos_count:
        return ("سلبي" if language == "arabic" else "Negative"), 0.7, "🔴"
    else:
        return ("محايد" if language == "arabic" else "Neutral"), 0.5, "🟡"

# Display model info and run validation
# if model is not None and tokenizer is not None:
#     st.write(f"Model configuration: {model.config}")
#     st.write(f"Model class names: {model.config.id2label if hasattr(model.config, 'id2label') else 'Not available'}")
    
#     # Test the model with a simple sentence
#     st.subheader("🧪 Model Test")
#     test_text = "This is a test sentence" if language == "English" else "هذه جملة تجريبية"
    
#     with st.expander("Click to run model test"):
#         try:
#             test_result = predict_sentiment(test_text, language_code)
#             st.success(f"✅ Model test successful: {test_result}")
#         except Exception as e:
#             st.error(f"❌ Model test failed: {str(e)}")
# else:
#     st.error("❌ Model or tokenizer not loaded properly!")

def extract_video_id(url):
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
        st.sidebar.write("🔍 **Debug Info:**")
        # Show token analysis
        try:
            inputs = tokenizer(single_comment, return_tensors="pt", truncation=True, padding=True, max_length=512)
            input_ids = inputs['input_ids'][0]
            model_vocab_size = model.config.vocab_size
            
            oov_tokens = (input_ids >= model_vocab_size).sum().item()
            st.sidebar.write(f"Total tokens: {len(input_ids)}")
            st.sidebar.write(f"OOV tokens: {oov_tokens}")
            st.sidebar.write(f"Max token ID: {input_ids.max().item()}")
            st.sidebar.write(f"Model vocab limit: {model_vocab_size}")
        except Exception as e:
            st.sidebar.write(f"Debug error: {e}")
    
        sentiment, confidence, emoji = predict_sentiment(single_comment, language_code)
        st.sidebar.markdown(f"**Result:** {emoji} {sentiment}")
        st.sidebar.markdown(f"**Confidence:** {confidence:.2%}")
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
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig_bar, use_container_width=True)
                    
                    st.plotly_chart(fig_hist, use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("📋 Comments Details")
                    
                    filter_sentiment = st.selectbox(
                        "Filter by sentiment:" if language == "English" else "تصفية حسب المشاعر:",
                        ["All"] + list(df['sentiment'].unique())
                    )
                    if filter_sentiment != "All":
                        filtered_df = df[df['sentiment'] == filter_sentiment]
                    else:
                        filtered_df = df
                    
                    display_cols = ['author', 'comment', 'sentiment', 'confidence', 'likes']
                    display_df = filtered_df[display_cols].copy()
                    display_df.columns = ['Author', 'Comment', 'Sentiment', 'Confidence', 'Likes']
                    display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.2%}")
                    
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
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
