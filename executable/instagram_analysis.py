import streamlit as st
import pandas as pd
import warnings
import os

import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ignore warnings
warnings.filterwarnings("ignore")


# Cache the model and tokenizer to avoid reloading on every run
@st.cache_resource
def load_model():
    MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model


tokenizer, model = load_model()


def format_instagram_comments(instagram: list) -> list:
    """
    Convert txt format of Instagram comments to list of comments
    """
    # Collect comments
    lines = [comment.strip() for comment in instagram]

    comments = []
    for i, line in enumerate(lines):
        if line.strip() == "":
            if i + 2 < len(lines):
                comments.append(lines[i + 2].strip())

    return comments


def get_sentiment(sentence: str, tokenizer, model) -> str:
    """
    Calculate sentiment of the sentence using the XLM-RoBERTa model
    """
    # Tokenize the sentence and convert to input format
    inputs = tokenizer(
        sentence, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    # Predict
    with torch.no_grad():
        logits = model(**inputs).logits

    # Apply softmax to get probabilities
    probabilities = softmax(logits, dim=1).flatten()

    # Map model output to sentiment
    sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment_score = torch.argmax(probabilities).item()

    sentiment = sentiment_mapping[sentiment_score]

    return sentiment


def print_stats_xlm(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate sentiment statistics from the data
    """
    num_pos = len(data[data["Sentiment_XLM"] == "Positive"])
    num_neu = len(data[data["Sentiment_XLM"] == "Neutral"])
    num_neg = len(data[data["Sentiment_XLM"] == "Negative"])
    num_total = len(data)

    percent_pos = num_pos / num_total * 100 if num_total > 0 else 0
    percent_neu = num_neu / num_total * 100 if num_total > 0 else 0
    percent_neg = num_neg / num_total * 100 if num_total > 0 else 0

    return pd.DataFrame(
        {
            "Sentiment": ["Positive", "Neutral", "Negative", "Total"],
            "Count": [num_pos, num_neu, num_neg, num_total],
            "Percentage": [percent_pos, percent_neu, percent_neg, 100.0],
        }
    )


def main():
    st.title("Instagram Sentiment Analysis")

    uploaded_file = st.file_uploader(
        "Choose a CSV, TXT, or XLSX file", type=["csv", "txt", "xlsx"]
    )

    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        if file_extension == ".csv":
            file_type = "csv"
        elif file_extension == ".txt":
            file_type = "txt"
        elif file_extension == ".xlsx":
            file_type = "xlsx"
        else:
            st.error("Invalid file type. Please upload a CSV, TXT, or XLSX file.")
            st.stop()

        if file_type == "csv":
            try:
                df_instagram_comments = pd.read_csv(uploaded_file).rename(
                    columns={"Message": "Comment"}
                )
            except Exception as e:
                st.error(f"Failed to read the CSV file: {e}")
                st.stop()
        elif file_type == "xlsx":
            try:
                df_instagram_comments = pd.read_excel(uploaded_file).rename(
                    columns={"Message": "Comment"}
                )
            except Exception as e:
                st.error(f"Failed to read the XLSX file: {e}")
                st.stop()
        elif file_type == "txt":
            try:
                instagram = uploaded_file.read().decode("utf-8").splitlines()
            except Exception as e:
                st.error(f"Failed to read the TXT file: {e}")
                st.stop()
            # Extract all comments
            insta_comments = set(format_instagram_comments(instagram))

            # Convert to DataFrame
            df_instagram_comments = pd.DataFrame(insta_comments, columns=["Comment"])

        if "Comment" not in df_instagram_comments.columns:
            st.error("The CSV file does not contain a 'Comment' column.")
            st.stop()

        # Perform sentiment analysis
        st.write("Performing sentiment analysis...")
        sentiments = []
        total_comments = len(df_instagram_comments)
        progress_bar = st.progress(0)
        for idx, comment in enumerate(df_instagram_comments["Comment"]):
            sentiment = get_sentiment(str(comment), tokenizer, model)
            sentiments.append(sentiment)
            progress_bar.progress((idx + 1) / total_comments)
        df_instagram_comments["Sentiment_XLM"] = sentiments

        # Display sentiment statistics
        df_stats = print_stats_xlm(df_instagram_comments)
        st.subheader("Sentiment Statistics")
        st.dataframe(df_stats)

        # Display comments with sentiments
        st.subheader("Comments with Sentiments")
        st.dataframe(df_instagram_comments)

        # Provide option to download the results
        csv = df_instagram_comments.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="sentiment_analysis_results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
