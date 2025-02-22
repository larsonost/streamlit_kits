import json
import random
import re
import time
import warnings
from http.cookiejar import CookieJar

import facebook_scraper as fs
import pandas as pd
import requests
import spacy
import torch
from deep_translator import GoogleTranslator
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ignore warnings
warnings.filterwarnings("ignore")

nlp = spacy.load("en_core_web_sm")

# Load Aspect-Based Sentiment Analysis model
absa_tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
absa_model = AutoModelForSequenceClassification.from_pretrained(
    "yangheng/deberta-v3-base-absa-v1.1"
)

# Load the tokenizer and model
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


def get_sentiment(sentence):
    # Tokenize the sentence and convert to input format
    inputs = tokenizer(
        sentence, return_tensors="pt", padding=True, truncation=True, max_length=512
    )

    # Predict
    with torch.no_grad():
        logits = model(**inputs).logits

    # Apply softmax to get probabilities
    probabilities = softmax(logits, dim=1).flatten()

    # Map model output (1 to 3 stars) to sentiment
    sentiment_mapping = {1: "Negative", 2: "Neutral", 3: "Positive"}
    sentiment_score = (
        torch.argmax(probabilities).item() + 1
    )  # Adding 1 because model outputs are 1-indexed (0 to 2)

    sentiment = sentiment_mapping[sentiment_score]
    return sentiment


def format_instagram_comments(instagram) -> list:
    # Collect comments
    lines = [comment.strip() for comment in instagram]

    comments = []
    for i, line in enumerate(lines):
        if line.strip() == "":
            if i + 2 < len(lines):
                comments.append(lines[i + 2].strip())

    return comments


def clean_text(text):
    """
    Clean the text of unwanted characters
    """
    # replace all @ symbols and the person they are @ing
    text = re.sub(r"@\w+", "", text)
    # Remove newlines
    text = re.sub(r"\n", " ", text)
    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)
    return text


def translate_text(text: str) -> str:
    """
    Translate text to English
    """
    try:
        translator = GoogleTranslator(source="auto", target="en")
        translated_text = translator.translate(text)
        if translated_text == None:
            return text
        else:
            return translated_text
    except:
        return text


def print_stats_XLM(data: pd.DataFrame) -> None:
    """
    Print the sentiment stats from Custom Model
    """
    print("\n\nXLM-RoBERTa Multilingual Sentiment Analysis Stats")
    num_pos = len(data[data["Sentiment_XLM"] == "Positive"])
    num_neu = len(data[data["Sentiment_XLM"] == "Neutral"])
    num_neg = len(data[data["Sentiment_XLM"] == "Negative"])
    num_total = len(data)

    print(f"Percent Positive: {num_pos / num_total * 100:.2f}%")
    print(f"Percent Neutral: {num_neu / num_total * 100:.2f}%")
    print(f"Percent Negative: {num_neg / num_total * 100:.2f}%")


def get_cookie_jar(cookies_path: str) -> CookieJar:
    """
    Get cookie jar from cookies.txt file
    :param cookies_path: path to cookies.txt file
    :return: cookie jar
    """

    # Make cookie jar
    cookie_jar = CookieJar()

    # Load cookies
    with open(cookies_path, "r") as file:
        cookies = json.load(file)

    # Add cookies to cookie jar
    for cookie in cookies:
        c = requests.cookies.create_cookie(
            domain=cookie["domain"],
            name=cookie["name"],
            value=cookie["value"],
            path=cookie["path"],
            secure=cookie["secure"],
            expires=cookie.get("expirationDate"),
            rest={
                "HttpOnly": cookie["httpOnly"],
                "SameSite": cookie["sameSite"],
            },
        )
        cookie_jar.set_cookie(c)

    return cookie_jar


def get_facebook_post_comments(
    POST_ID: str, MAX_COMMENTS: int, cookie_jar: CookieJar
) -> list:
    """
    Get the comments of a facebook post
    """
    gen = fs.get_posts(
        post_urls=[int(POST_ID)],
        options={"comments": MAX_COMMENTS, "progress": True},
        cookies=cookie_jar,
    )
    try:
        post = next(gen)
    except:
        return []

    # Extract the comments from the post
    comments = post["comments_full"]
    if comments is None:
        return []
        # Introduce a random delay to avoid rate limiting
    time.sleep(random.uniform(10, 20))
    return [com["comment_text"] for com in comments]


def get_facebook_post_ids(df: pd.DataFrame) -> list:
    """
    Get the post ids from the dataframe
    """
    df = df[df["MessageType"] == "Facebook Post"]
    df["PostID"] = df["ConversationId"].apply(lambda x: x.split("_")[1])
    return list(set(df["PostID"].tolist()))


def get_twitter_replies_query(
    df: pd.DataFrame, twitter_link_1: str, twitter_link_2: str
) -> str:
    """
    Get the query string for replies
    """
    # pick the top 100 tweets (-2 for hardcoded) with the highest number of followers
    df = df[df["MessageType"] == "X Mention"].sort_values(
        by="Sender Followers Count", ascending=False
    )[:98]
    # get the tweet ids
    df["tweet_id"] = df["Permalink"].str.split("/").str[-1]
    # convert to list & sprinklr form
    tweet_ids = df["tweet_id"].tolist()
    # Add the twitter links to the list
    tweet_ids.append(twitter_link_1.split("/")[-1])
    tweet_ids.append(twitter_link_2.split("/")[-1])
    string = ""
    for tweet_id in tweet_ids:
        string += f"(engagingWithGuid: {tweet_id}) OR "
    string = string[:-4]
    return string


def clean_twitter_replies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the twitter replies dataframe
    """
    # drop duplicates and reposts
    df = df[df["MessageType"] != "X Repost"]
    df = df[
        [
            "SenderScreenName",
            "SenderListedName",
            "Sender Followers Count",
            "OriginalAuthor",
            "Message",
            "MessageType",
            "Sentiment",
            "Language",
        ]
    ]
    df = df.drop_duplicates(subset=["Message"])
    return df


def extract_facebook_comments(comm_link: list) -> list:
    """
    Extract the comments from the facebook comments
    """
    comments = []
    with open(comm_link, "r") as file:
        lines = file.readlines()
    for i in range(len(lines) - 1):
        if "days ago" in lines[i + 1] or "weeks ago" in lines[i + 1]:
            comments.append(lines[i].strip())
    return comments


def extract_aspects(text: str) -> list[str]:
    """
    Extracts aspects from a given text.

    Args:
        text: The text to extract aspects from.

    Returns:
        A list of aspects.
    """
    doc = nlp(text)
    aspects = []

    for token in doc:
        if token.pos_ in ["NOUN", "ADJ", "VERB"]:
            aspects.append(token.text)
        elif token.dep_ == "amod" and token.head.pos_ in ["NOUN", "ADJ", "VERB"]:
            aspects.append(token.text)

    return aspects


def perform_absa(post_comment: list, announcement_keywords: list) -> bool:
    review = post_comment.strip().lower()
    aspects = extract_aspects(review)

    # if any of the aspects are in the announcement keywords, then it is an announcement
    if any(aspect in announcement_keywords for aspect in aspects):
        is_announcement = True
    else:
        is_announcement = False

    return is_announcement
