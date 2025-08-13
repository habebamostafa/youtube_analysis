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
import emoji

# Download NLTK data (only once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# ==============================================
# Helper Functions
# ==============================================

def has_emoji(text):
    """Check if text contains emoji"""
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

def improved_convert_emojis(text):
    """Convert emojis to Arabic words with enhanced sentiment weighting"""
    text = emoji.demojize(text, language='en')
    
    # Enhanced Arabic emoji translations
    emoji_translations = {
        # Strong positive
        "face_with_tears_of_joy": "ضحك سعادة فرح جميل",
        "red_heart": "حب عشق محبة رائع",
        "smiling_face_with_smiling_eyes": "سعادة فرح بهجة جميل",
        "thumbs_up": "إعجاب موافقة رائع ممتاز",
        "clapping_hands": "تصفيق إعجاب برافو رائع",
        "fire": "رائع ممتاز جميل يجنن",
        "party_popper": "احتفال فرح سعادة رائع",
        "grinning_face": "ضحك سعادة فرح جميل",
        "heart_eyes": "حب إعجاب جميل رائع ممتاز",
        "rolling_on_the_floor_laughing": "ضحك سعادة فرح قوي جميل",
        "face_blowing_a_kiss": "حب قبلة سعادة جميل",
        "smiling_face_with_heart-eyes": "حب سعادة جميل رائع",
        
        # Negative
        "crying_face": "حزن بكاء أسف سيء",
        "angry_face": "غضب زعل سيء",
        "broken_heart": "حزن ألم فراق سيء",
        "disappointed_face": "خيبة أمل حزن سيء",
        "face_with_steam_from_nose": "غضب زعل سيء",
        
        # Direct symbols
        "😂": "ضحك سعادة فرح جميل", "❤": "حب محبة رائع", "😍": "حب إعجاب جميل رائع",
        "😊": "سعادة فرح جميل", "👍": "إعجاب موافقة رائع", "😢": "حزن بكاء سيء",
        "👏": "تصفيق إعجاب رائع", "🔥": "رائع ممتاز جميل يجنن", "😠": "غضب زعل سيء",
        "🎉": "فرح احتفال سعادة رائع", "🥰": "حب سعادة جميل", "😘": "حب قبلة جميل",
        "🤣": "ضحك سعادة فرح قوي جميل", "💔": "حزن ألم فراق سيء", "😞": "حزن خيبة أمل سيء",
        "✨": "جميل رائع ممتاز", "💕": "حب محبة جميل", "🌟": "رائع ممتاز جميل",
        "😃": "سعادة فرح جميل", "😄": "سعادة فرح جميل", "😆": "ضحك سعادة جميل"
    }

    for emoji_code, arabic_words in emoji_translations.items():
        text = text.replace(f":{emoji_code}:", arabic_words)
    
    return text

def enhanced_normalize_arabic(text):
    """Enhanced Arabic text normalization preserving important words"""
    if has_emoji(text):
        text = improved_convert_emojis(text)
    
    # Important sentiment indicators to preserve
    positive_indicators = [
        'رائع', 'جميل', 'ممتاز', 'حلو', 'جامد', 'يجنن', 'عظيم', 'مذهل', 'تحفة',
        'أحب', 'بحب', 'حب', 'محبة', 'اعجاب', 'موافقة', 'برافو', 'تصفيق',
        'سعادة', 'فرح', 'بهجة', 'ضحك', 'مبسوط', 'منبسط', 'حبيب', 'حبيبتي',
        'يحبيبتي', 'روعة', 'بديع', 'لذيذ', 'استمري', 'شكرا', 'مشكور', 'الف شكر',
        'احسنت', 'مبروك', 'موفق', 'ناجح', 'جيد', 'حسن', 'منيح', 'كويس'
    ]
    
    negative_indicators = [
        'سيء', 'وحش', 'مش حلو', 'مخرب', 'فظيع', 'قبيح', 'غبي', 'سخيف',
        'أكره', 'كرهت', 'زهقت', 'ملل', 'حزن', 'زعل', 'غضب', 'خيبة',
        'تعبان', 'مملل', 'بطال', 'زفت', 'محبط', 'مستاء', 'متضايق'
    ]
    
    # Add sentiment indicators
    sentiment_boost = ""
    for word in positive_indicators:
        if word in text.lower():
            sentiment_boost += " إيجابي_قوي "
    
    for word in negative_indicators:
        if word in text.lower():
            sentiment_boost += " سلبي_قوي "
    
    # Basic text normalization
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Keep Arabic only
    text = re.sub(r'[إأآا]', 'ا', text)  # Unify alef
    text = re.sub(r'ى', 'ي', text)  # Unify yeh
    text = re.sub(r'ؤ', 'ء', text)  # Unify hamza
    text = re.sub(r'ئ', 'ء', text)  # Unify hamza
    text = re.sub(r'ة', 'ه', text)  # Unify ta marbuta
    
    # Handle colloquial negation
    text = re.sub(r'\bمش\b', 'ليس', text)
    text = re.sub(r'\bمو\b', 'ليس', text)
    text = re.sub(r'\bما\s+(\w+)', r'ليس \1', text)
    text = re.sub(r'\b(\w+)ش\b', r'\1', text)  # Remove negation "sh"
    
    # Remove numbers and English
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[a-zA-Z]', '', text)
    text = re.sub(r'[^\u0621-\u064A\s]', ' ', text)  # Arabic only
    
    # Combine processed text with sentiment indicators
    final_text = (text + " " + sentiment_boost).strip()
    
    return ' '.join(final_text.split())

def post_process_sentiment(text, sentiment, confidence):
    """Post-processing to improve sentiment accuracy"""
    original_text = text.lower()
    
    # Rules for religious content
    religious_keywords = ['الله', 'النبي', 'صلى', 'استغفر', 'دعاء', 'رب', 'يارب', 'اللهم', 'سبحان', 'الحمد']
    if any(word in original_text for word in religious_keywords):
        if sentiment == "سلبي":
            return "إيجابي", max(confidence, 0.80)
        elif sentiment == "محايد":
            return "إيجابي", max(confidence, 0.75)
    
    # Rules for strong positive emojis
    strong_positive_emojis = ['🎉', '❤', '😍', '🥰', '🔥', '👏', '😂', '🤣', '✨', '💕', '😘', '😊']
    emoji_count = sum([original_text.count(emoji) for emoji in strong_positive_emojis])
    if emoji_count >= 2:
        if sentiment == "محايد":
            return "إيجابي", max(confidence, 0.85)
        elif sentiment == "سلبي" and emoji_count >= 3:
            return "إيجابي", max(confidence, 0.80)
    
    # Rules for strong positive words
    strong_positive_words = ['رائع', 'جميل', 'ممتاز', 'روعة', 'يجنن', 'أحب', 'بحب', 'جامد', 'تحفة', 'حبيبتي']
    positive_count = sum([1 for word in strong_positive_words if word in original_text])
    if positive_count >= 1 and sentiment == "محايد":
        return "إيجابي", max(confidence, 0.75)
    
    # Rules for strong negative words
    strong_negative_words = ['مخرب', 'وحش', 'سيء', 'فظيع', 'أكره', 'كرهت', 'قبيح', 'غبي']
    if any(word in original_text for word in strong_negative_words):
        if sentiment == "محايد":
            return "سلبي", max(confidence, 0.80)
        elif sentiment == "إيجابي" and confidence < 0.7:
            return "سلبي", max(confidence, 0.75)
    
    # Rules for simple questions
    question_patterns = ['ايش اسم', 'مين', 'وين', 'كيف', 'متى', 'ممكن اعرف', 'كم حلقة', 'شو اسم']
    if any(pattern in original_text for pattern in question_patterns):
        return "محايد", max(confidence, 0.70)
    
    # Rules for thanks and appreciation
    thanks_patterns = ['شكرا', 'مشكور', 'الف شكر', 'يعطيك', 'بارك الله', 'جزاك الله']
    if any(pattern in original_text for pattern in thanks_patterns):
        return "إيجابي", max(confidence, 0.85)
    
    # Fallback for very low confidence
    if confidence < 0.4:
        return get_enhanced_fallback_sentiment(text, "arabic")[:2]
    
    return sentiment, confidence

def get_enhanced_fallback_sentiment(text, language):
    """Enhanced fallback sentiment analysis with expanded rules"""
    text_lower = text.lower()
    
    # Expanded word lists
    positive_words = {
        'ar': ['رائع', 'جميل', 'ممتاز', 'روعة', 'يجنن', 'أحب', 'بحب', 'جامد', 'تحفة', 'حبيبتي',
               'سعادة', 'فرح', 'بهجة', 'ضحك', 'مبسوط', 'منبسط', 'حبيب', 'روعة', 'بديع', 'لذيذ',
               'استمري', 'شكرا', 'مشكور', 'الف شكر', 'احسنت', 'مبروك', 'موفق', 'ناجح', 'جيد', 'حسن'],
        'en': ['good', 'great', 'awesome', 'amazing', 'excellent', 'love', 'like', 'best', 
               'wonderful', 'fantastic', 'perfect', 'nice', 'beautiful', 'happy']
    }
    
    negative_words = {
        'ar': ['سيء', 'وحش', 'مش حلو', 'مخرب', 'فظيع', 'قبيح', 'غبي', 'سخيف',
               'أكره', 'كرهت', 'زهقت', 'ملل', 'حزن', 'زعل', 'غضب', 'خيبة',
               'تعبان', 'مملل', 'بطال', 'زفت', 'محبط', 'مستاء', 'متضايق'],
        'en': ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disgusting',
               'stupid', 'ugly', 'boring', 'annoying', 'angry', 'sad']
    }
    
    lang_code = 'ar' if language == 'arabic' else 'en'
    
    pos_count = sum(1 for word in positive_words[lang_code] if word in text_lower)
    neg_count = sum(1 for word in negative_words[lang_code] if word in text_lower)
    
    # Emoji analysis
    emoji_sentiment = 0
    if has_emoji(text):
        converted = improved_convert_emojis(text)
        pos_emoji = sum(1 for word in positive_words[lang_code] if word in converted)
        neg_emoji = sum(1 for word in negative_words[lang_code] if word in converted)
        emoji_sentiment = pos_emoji - neg_emoji
    
    # Calculate total sentiment
    total_sentiment = (pos_count - neg_count) + emoji_sentiment
    
    if total_sentiment > 0:
        return ("إيجابي" if lang_code == 'ar' else "Positive"), 0.75, "🟢"
    elif total_sentiment < 0:
        return ("سلبي" if lang_code == 'ar' else "Negative"), 0.75, "🔴"
    else:
        return ("محايد" if lang_code == 'ar' else "Neutral"), 0.5, "🟡"

def improved_predict_sentiment(text, language, model, tokenizer):
    """Enhanced sentiment prediction with better Arabic handling"""
    if not text.strip():
        return "محايد" if language == "arabic" else "Neutral", 0.5, "🟡"
    
    original_text = text
    
    # Text preprocessing
    if language.lower() == "arabic":
        processed_text = enhanced_normalize_arabic(text)
    else:
        processed_text = text.strip()
    
    try:
        # Model prediction attempt
        if len(processed_text) > 500:
            processed_text = processed_text[:500]
        
        model_vocab_size = model.config.vocab_size if hasattr(model.config, 'vocab_size') else 30000
        
        inputs = tokenizer(
            processed_text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512,
            add_special_tokens=True,
            return_attention_mask=True
        )
        
        # Filter out-of-vocabulary tokens
        input_ids = inputs['input_ids'][0]
        unk_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 1
        valid_mask = input_ids < model_vocab_size
        filtered_input_ids = torch.where(valid_mask, input_ids, torch.tensor(unk_token_id))
        inputs['input_ids'] = filtered_input_ids.unsqueeze(0)
        
        if torch.any(inputs['input_ids'] >= model_vocab_size):
            return get_enhanced_fallback_sentiment(original_text, language)
        
        # Model prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probabilities[predicted_class].item()
        
        # Label mapping
        if hasattr(model.config, 'id2label') and model.config.id2label:
            model_label = None
            if predicted_class in model.config.id2label:
                model_label = model.config.id2label[predicted_class]
            elif str(predicted_class) in model.config.id2label:
                model_label = model.config.id2label[str(predicted_class)]
            
            label_normalization = {
                "LABEL_0": "Negative", "LABEL_1": "Positive", "LABEL_2": "Neutral",
                "0": "Negative", "1": "Neutral", "2": "Positive"
            }
            
            if model_label in label_normalization:
                model_label = label_normalization[model_label]
            
            if model_label:
                if language == "arabic":
                    label_mapping = {
                        "Negative": "سلبي", "Positive": "إيجابي", "Neutral": "محايد"
                    }
                    sentiment_label = label_mapping.get(model_label, model_label)
                else:
                    sentiment_label = model_label
                
                color_mapping = {
                    "Negative": "🔴", "سلبي": "🔴",
                    "Positive": "🟢", "إيجابي": "🟢", 
                    "Neutral": "🟡", "محايد": "🟡"
                }
                color = color_mapping.get(sentiment_label, "⚪")
            else:
                raise ValueError("Could not find model label")
        else:
            # Fallback labels
            if language == "arabic":
                labels = ["سلبي", "محايد", "إيجابي"]
                colors = ["🔴", "🟡", "🟢"]
            else:
                labels = ["Negative", "Neutral", "Positive"]
                colors = ["🔴", "🟡", "🟢"]
            
            if predicted_class >= len(labels):
                return get_enhanced_fallback_sentiment(original_text, language)
            
            sentiment_label = labels[predicted_class]
            color = colors[predicted_class]
        
        # Post-processing for better accuracy
        final_sentiment, final_confidence = post_process_sentiment(
            original_text, sentiment_label, confidence
        )
        
        return final_sentiment, final_confidence, color
            
    except Exception as e:
        return get_enhanced_fallback_sentiment(original_text, language)

# ==============================================
# Model Download and Loading
# ==============================================

def download_model_files(language):
    """Download model files from Google Drive"""
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
                st.error(f"Error copying {filename}: {str(e)}")

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
            st.error(f"Download error: {str(e)}")

@st.cache_resource
def load_model(language):
    """Load the sentiment analysis model"""
    lang_code = "ar" if language == "Arabic" else "en"
    model_path = f"models/{lang_code}"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        model.resize_token_embeddings(len(tokenizer))
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

# ==============================================
# YouTube Comment Functions
# ==============================================

def extract_video_id(url):
    """Extract YouTube video ID from URL"""
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
    """Fetch YouTube comments without API"""
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

def analyze_comments(comments, language_code, model, tokenizer):
    """Analyze sentiment of comments"""
    results = []
    for comment in comments:
        sentiment, confidence, emoji = improved_predict_sentiment(
            comment['text'], 
            language_code,
            model,
            tokenizer
        )
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
    """Create sentiment visualizations"""
    df = pd.DataFrame(results)
    
    if language == "Arabic":
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

# ==============================================
# Streamlit UI
# ==============================================

def main():
    st.set_page_config(page_title="YouTube Comments Sentiment Analysis", layout="wide")
    st.title("🎥 YouTube Comments Sentiment Analysis")
    st.markdown("---")
    
    # Download models
    download_model_files("English")
    download_model_files("Arabic")
    
    # Language selection
    st.sidebar.header("🌍 Language Settings")
    language = st.sidebar.radio(
        "Select Comment Language:",
        ("Arabic", "English"),
        index=0
    )
    language_code = "arabic" if language == "Arabic" else "english"
    
    # Load model
    model, tokenizer = load_model(language) 
    if model is None or tokenizer is None:
        st.error("Failed to load model - please check the error messages")
        st.stop()
    
    # Video URL input
    st.sidebar.header("⚙️ Settings")
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
            sentiment, confidence, emoji_icon = improved_predict_sentiment(
                single_comment, 
                language_code,
                model,
                tokenizer
            )
            st.sidebar.markdown(f"**Result:** {emoji_icon} {sentiment}")
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
                        results = analyze_comments(comments, language_code, model, tokenizer)
                        fig_pie, fig_bar, fig_hist, df = create_visualizations(results, language)
                        
                        st.success(f"✅ Successfully analyzed {len(results)} comments!")
                        
                        # Display quick stats
                        col1, col2, col3, col4 = st.columns(4)
                        sentiment_counts = df['sentiment'].value_counts()
                        
                        # Get labels based on language
                        if language == "Arabic":
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
                        
                        # Display charts
                        col1, col2 = st.columns(2)
                        with col1:
                            st.plotly_chart(fig_pie, use_container_width=True)
                        with col2:
                            st.plotly_chart(fig_bar, use_container_width=True)
                        
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                        st.markdown("---")
                        st.subheader("📋 Comments Details")
                        
                        # Filter options
                        if language == "Arabic":
                            filter_label = "تصفية حسب المشاعر:"
                            all_option = "الكل"
                        else:
                            filter_label = "Filter by sentiment:"
                            all_option = "All"
                            
                        filter_sentiment = st.selectbox(
                            filter_label,
                            [all_option] + list(df['sentiment'].unique())
                        )
                        
                        if filter_sentiment != all_option:
                            filtered_df = df[df['sentiment'] == filter_sentiment]
                        else:
                            filtered_df = df
                        
                        # Display results table
                        display_cols = ['author', 'comment', 'sentiment', 'confidence', 'likes']
                        display_df = filtered_df[display_cols].copy()
                        display_df.columns = ['Author', 'Comment', 'Sentiment', 'Confidence', 'Likes']
                        display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.2%}")
                        
                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True
                        )
                        
                        # Download button
                        csv = df.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=csv,
                            file_name=f"youtube_sentiment_{video_id}.csv",
                            mime="text/csv"
                        )
    else:
        # Display instructions
        if language == "Arabic":
            st.markdown("""
            ## 📊 المميزات:
            - تحليل تلقائي لمشاعر تعليقات اليوتيوب
            - دعم اللغتين العربية والإنجليزية
            - رسوم بيانية تفاعلية
            - إحصائيات مفصلة
            - تصفية النتائج وتنزيلها بصيغة CSV
            - تحليل التعليقات الفردية
            
            ## 🚀 طريقة الاستخدام:
            1. اختر اللغة المفضلة (عربي/إنجليزي)
            2. أدخل رابط فيديو اليوتيوب في الشريط الجانبي
            3. اختر عدد التعليقات المراد تحليلها
            4. اضغط على "تحليل التعليقات"
            5. استعرض النتائج وقم بتنزيلها إذا رغبت
            """)
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
            4. Click "Analyze Comments"
            5. View results and download CSV if needed
            """)

if __name__ == "__main__":
    main()
