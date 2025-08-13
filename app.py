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

def improved_convert_emojis(text):
    """تحسين معالجة الإيموجي مع إعطاء أوزان أكبر للمشاعر"""
    # تحويل الإيموجي إلى نص
    text = emoji.demojize(text, language='en')
    
    # قاموس محسن للإيموجي العربية
    emoji_translations = {
        # إيجابي قوي
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
        
        # سلبي
        "crying_face": "حزن بكاء أسف سيء",
        "angry_face": "غضب زعل سيء",
        "broken_heart": "حزن ألم فراق سيء",
        "disappointed_face": "خيبة أمل حزن سيء",
        "face_with_steam_from_nose": "غضب زعل سيء",
        
        # رموز مباشرة
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
    """تطبيع محسن للنص العربي مع الحفاظ على الكلمات المهمة"""
    original_text = text
    
    # تحويل الإيموجي أولاً
    if has_emoji(text):
        text = improved_convert_emojis(text)
    
    # كلمات إيجابية مهمة يجب الحفاظ عليها
    positive_indicators = [
        'رائع', 'جميل', 'ممتاز', 'حلو', 'جامد', 'يجنن', 'عظيم', 'مذهل', 'تحفة',
        'أحب', 'بحب', 'حب', 'محبة', 'اعجاب', 'موافقة', 'برافو', 'تصفيق',
        'سعادة', 'فرح', 'بهجة', 'ضحك', 'مبسوط', 'منبسط', 'حبيب', 'حبيبتي',
        'يحبيبتي', 'روعة', 'بديع', 'لذيذ', 'استمري', 'شكرا', 'مشكور', 'الف شكر',
        'احسنت', 'مبروك', 'موفق', 'ناجح', 'جيد', 'حسن', 'منيح', 'كويس'
    ]
    
    # كلمات سلبية مهمة
    negative_indicators = [
        'سيء', 'وحش', 'مش حلو', 'مخرب', 'فظيع', 'قبيح', 'غبي', 'سخيف',
        'أكره', 'كرهت', 'زهقت', 'ملل', 'حزن', 'زعل', 'غضب', 'خيبة',
        'تعبان', 'مملل', 'بطال', 'زفت', 'محبط', 'مستاء', 'متضايق'
    ]
    
    # إضافة مؤشرات المشاعر
    sentiment_boost = ""
    for word in positive_indicators:
        if word in text.lower():
            sentiment_boost += " إيجابي_قوي "
    
    for word in negative_indicators:
        if word in text.lower():
            sentiment_boost += " سلبي_قوي "
    
    # تطبيع النص الأساسي
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # الاحتفاظ بالعربي فقط
    text = re.sub(r'[إأآا]', 'ا', text)  # توحيد الألف
    text = re.sub(r'ى', 'ي', text)  # توحيد الياء
    text = re.sub(r'ؤ', 'ء', text)  # توحيد الهمزة
    text = re.sub(r'ئ', 'ء', text)  # توحيد الهمزة
    text = re.sub(r'ة', 'ه', text)  # توحيد التاء المربوطة
    
    # معالجة النفي العامي
    text = re.sub(r'\bمش\b', 'ليس', text)
    text = re.sub(r'\bمو\b', 'ليس', text)
    text = re.sub(r'\bما\s+(\w+)', r'ليس \1', text)
    text = re.sub(r'\b(\w+)ش\b', r'\1', text)  # إزالة "ش" النفي
    
    # إزالة الأرقام والإنجليزي
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[a-zA-Z]', '', text)
    text = re.sub(r'[^\u0621-\u064A\s]', ' ', text)  # العربي فقط
    
    # دمج النص المعالج مع مؤشرات المشاعر
    final_text = (text + " " + sentiment_boost).strip()
    
    return ' '.join(final_text.split())

def post_process_sentiment(text, sentiment, confidence):
    """معالجة لاحقة لتحسين دقة التصنيف"""
    original_text = text.lower()
    
    # قواعد خاصة للنصوص الدينية
    religious_keywords = ['الله', 'النبي', 'صلى', 'استغفر', 'دعاء', 'رب', 'يارب', 'اللهم', 'سبحان', 'الحمد']
    if any(word in original_text for word in religious_keywords):
        if sentiment == "سلبي":
            return "إيجابي", max(confidence, 0.80)
        elif sentiment == "محايد":
            return "إيجابي", max(confidence, 0.75)
    
    # قواعد للإيموجي الإيجابية القوية
    strong_positive_emojis = ['🎉', '❤', '😍', '🥰', '🔥', '👏', '😂', '🤣', '✨', '💕', '😘', '😊']
    emoji_count = sum([original_text.count(emoji) for emoji in strong_positive_emojis])
    if emoji_count >= 2:
        if sentiment == "محايد":
            return "إيجابي", max(confidence, 0.85)
        elif sentiment == "سلبي" and emoji_count >= 3:
            return "إيجابي", max(confidence, 0.80)
    
    # قواعد للكلمات الإيجابية القوية
    strong_positive_words = ['رائع', 'جميل', 'ممتاز', 'روعة', 'يجنن', 'أحب', 'بحب', 'جامد', 'تحفة', 'حبيبتي']
    positive_count = sum([1 for word in strong_positive_words if word in original_text])
    if positive_count >= 1 and sentiment == "محايد":
        return "إيجابي", max(confidence, 0.75)
    
    # قواعد للكلمات السلبية القوية
    strong_negative_words = ['مخرب', 'وحش', 'سيء', 'فظيع', 'أكره', 'كرهت', 'قبيح', 'غبي']
    if any(word in original_text for word in strong_negative_words):
        if sentiment == "محايد":
            return "سلبي", max(confidence, 0.80)
        elif sentiment == "إيجابي" and confidence < 0.7:
            return "سلبي", max(confidence, 0.75)
    
    # قواعد للأسئلة البسيطة
    question_patterns = ['ايش اسم', 'مين', 'وين', 'كيف', 'متى', 'ممكن اعرف', 'كم حلقة', 'شو اسم']
    if any(pattern in original_text for pattern in question_patterns):
        return "محايد", max(confidence, 0.70)
    
    # قواعد لعبارات الشكر والتقدير
    thanks_patterns = ['شكرا', 'مشكور', 'الف شكر', 'يعطيك', 'بارك الله', 'جزاك الله']
    if any(pattern in original_text for pattern in thanks_patterns):
        return "إيجابي", max(confidence, 0.85)
    
    # إذا كانت الثقة منخفضة جداً، استخدم التحليل الاحتياطي
    if confidence < 0.4:
        return get_enhanced_fallback_sentiment(text, "arabic")[:2]
    
    return sentiment, confidence

def improved_predict_sentiment(text, language, model, tokenizer):
    """دالة محسنة لتحليل المشاعر مع معالجة أفضل للعربية"""
    if not text.strip():
        return "محايد" if language == "arabic" else "Neutral", 0.5, "🟡"
    
    original_text = text
    
    # تحسين النص للمعالجة
    if language.lower() == "arabic":
        processed_text = enhanced_normalize_arabic(text)
    else:
        processed_text = text.strip()
    
    try:
        # محاولة التنبؤ بالنموذج
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
        
        # تصفية الرموز خارج المفردات
        input_ids = inputs['input_ids'][0]
        unk_token_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 1
        valid_mask = input_ids < model_vocab_size
        filtered_input_ids = torch.where(valid_mask, input_ids, torch.tensor(unk_token_id))
        inputs['input_ids'] = filtered_input_ids.unsqueeze(0)
        
        if torch.any(inputs['input_ids'] >= model_vocab_size):
            return get_enhanced_fallback_sentiment(original_text, language)
        
        # التنبؤ بالنموذج
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probabilities[predicted_class].item()
        
        # تحويل التسميات
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
            # تسميات احتياطية
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
        
        # معالجة لاحقة لتحسين الدقة
        final_sentiment, final_confidence = post_process_sentiment(
            original_text, sentiment_label, confidence
        )
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
