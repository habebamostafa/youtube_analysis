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

# def download_file_from_drive(file_id, filename):
#     if not os.path.exists(filename):
#         url = f"https://drive.google.com/uc?id={file_id}"
#         gdown.download(url, filename, quiet=False)
import shutil

def download_model():
    files = {
        "181NGDNj-jTUY9JH5AtMW9Ez7FAiJPtqR": "config.json",
        "1Q3WFKlNe12qXcwDnUmrrf6OkamwiXLG-": "model.safetensors", 
        "1DKsomb6RgIqombyJ3IsVemmJUu16yYDh": "special_tokens_map.json",
        "1ZM-u0_4zB21ZpL6507_ZiOm5Aa0n1x1T": "tokenizer_config.json",
        "1X-YW8e54-O63z_oFFzZnFK54bTHBvx0y": "training_args.bin",
        "1v5y-ffp9O6FW7T3G2tST26O1RmdugxXf": "vocab.txt"
    }

    for file_id, filename in files.items():
        filepath = os.path.join(".", filename)  # مباشرة في نفس المسار
        if not os.path.exists(filepath):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, filepath, quiet=False)

    return "."  # المسار الحالي

# Load model


@st.cache_resource
def load_model():
    # Ensure model_path is a valid Hugging Face model ID or a local path
    download_model()
    model = BertForSequenceClassification.from_pretrained(
        ".",
        local_files_only=True ,# Explicitly specify loading from local files
    )
    tokenizer = BertTokenizer.from_pretrained(
        ".",
        local_files_only=True # Explicitly specify loading from local files
    )

    model.eval()
    return model, tokenizer


model, tokenizer = load_model()

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        confidence = probabilities[predicted_class].item()
        return predicted_class, confidence

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
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

def analyze_comments(comments):
    """Analyze sentiment of comments"""
    label_map = {0: "Negative", 1: "Positive", 2: "Neutral"}
    label_map_ar = {0: "Negative", 1: "Positive", 2: "Neutral"}

    results = []
    for comment in comments:
        sentiment_id, confidence = predict_sentiment(comment['text'])
        results.append({
            'comment': comment['text'][:100] + "..." if len(comment['text']) > 100 else comment['text'],
            'author': comment['author'],
            'sentiment': label_map[sentiment_id],
            'sentiment_ar': label_map_ar[sentiment_id],
            'confidence': confidence,
            'likes': comment['likes']
        })

    return results

def create_visualizations(results):
    """Create visualizations"""
    df = pd.DataFrame(results)
    sentiment_counts = df['sentiment_ar'].value_counts()

    # Pie chart
    fig_pie = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title="Sentiment Distribution",
        color_discrete_map={
            'Positive': '#2ecc71',
            'Negative': '#e74c3c',
            'Neutral': '#f39c12'
        }
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')

    # Bar chart
    fig_bar = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        title="Number of Comments by Sentiment",
        labels={'x': 'Sentiment', 'y': 'Number of Comments'},
        color=sentiment_counts.index,
        color_discrete_map={
            'Positive': '#2ecc71',
            'Negative': '#e74c3c',
            'Neutral': '#f39c12'
        }
    )

    # Confidence histogram
    fig_confidence = px.histogram(
        df,
        x='confidence',
        color='sentiment_ar',
        title="Confidence Level Distribution",
        labels={'confidence': 'Confidence Level', 'count': 'Count'},
        color_discrete_map={
            'Positive': '#2ecc71',
            'Negative': '#e74c3c',
            'Neutral': '#f39c12'
        }
    )

    return fig_pie, fig_bar, fig_confidence, df

# App UI
st.set_page_config(page_title="YouTube Comments Sentiment Analysis", layout="wide")

st.title("🎥 YouTube Comments Sentiment Analysis")
st.markdown("---")

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
                comments = get_youtube_comments(video_id, max_comments)

                if not comments:
                    st.error("❌ No comments found or an error occurred")
                else:
                    results = analyze_comments(comments)
                    fig_pie, fig_bar, fig_confidence, df = create_visualizations(results)

                    st.success(f"✅ Successfully analyzed {len(results)} comments!")

                    # Quick stats
                    col1, col2, col3, col4 = st.columns(4)
                    sentiment_counts = df['sentiment_ar'].value_counts()

                    with col1:
                        positive_count = sentiment_counts.get('Positive', 0)
                        st.metric("Positive Comments", positive_count, f"{positive_count/len(results)*100:.1f}%")

                    with col2:
                        negative_count = sentiment_counts.get('Negative', 0)
                        st.metric("Negative Comments", negative_count, f"{negative_count/len(results)*100:.1f}%")

                    with col3:
                        neutral_count = sentiment_counts.get('Neutral', 0)
                        st.metric("Neutral Comments", neutral_count, f"{neutral_count/len(results)*100:.1f}%")

                    with col4:
                        avg_confidence = df['confidence'].mean()
                        st.metric("Average Confidence", f"{avg_confidence:.2%}")

                    st.markdown("---")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig_bar, use_container_width=True)

                    st.plotly_chart(fig_confidence, use_container_width=True)

                    st.markdown("---")
                    st.subheader("📋 Comments Details")

                    filter_sentiment = st.selectbox(
                        "Filter by sentiment:",
                        ["All"] + list(df['sentiment_ar'].unique())
                    )

                    if filter_sentiment != "All":
                        filtered_df = df[df['sentiment_ar'] == filter_sentiment]
                    else:
                        filtered_df = df

                    display_df = filtered_df[['author', 'comment', 'sentiment_ar', 'confidence', 'likes']].copy()
                    display_df.columns = ['Author', 'Comment', 'Sentiment', 'Confidence Level', 'Likes']
                    display_df['Confidence Level'] = display_df['Confidence Level'].apply(lambda x: f"{x:.2%}")

                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    csv = df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=csv,
                        file_name=f"youtube_sentiment_analysis_{video_id}.csv",
                        mime="text/csv"
                    )

else:
    st.markdown("""
    ## 📊 Features:
    - Automatic sentiment analysis of YouTube comments
    - Interactive visualizations
    - Detailed statistics
    - Filtering and CSV download
    - Single comment sentiment analysis
    """)
