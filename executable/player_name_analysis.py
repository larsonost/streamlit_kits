import streamlit as st
import pandas as pd
import re
import unidecode
import spacy
from tqdm import tqdm
import io


# Function to remove profile mentions
def remove_profile_mentions(text):
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"@[\w_]+", "", text)


# Function to find conversations and calculate sentiments
def find_conversations(comments, sentiments):
    people = {}
    nlp = spacy.load("en_core_web_sm")
    for comment, sentiment in tqdm(zip(comments, sentiments), total=len(comments)):
        doc = nlp(comment)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                ent = ent.text.lower()
                ent = unidecode.unidecode(ent)
                if ent not in people:
                    people[ent] = {
                        "POSITIVE": 0,
                        "NEGATIVE": 0,
                        "NEUTRAL": 0,
                        "total": 0,
                    }
                # Update sentiment counts
                people[ent][sentiment] += 1
                people[ent]["total"] += 1
                break
    return people


# Streamlit App
def main():
    st.title("Player Name Analysis")

    # Upload file
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

    if uploaded_file:
        # Read the uploaded file
        df = pd.read_excel(uploaded_file)

        # Data cleaning
        st.write("Cleaning data...")
        df["Message"] = df["Message"].str.replace("RT ", "")
        df["Message"] = df["Message"].apply(remove_profile_mentions)

        # Process the data to find conversations and sentiments
        st.write("Finding names & sentiments...")
        people = find_conversations(df["Message"], df["Sentiment"])

        # Convert results into a dataframe
        people_df = pd.DataFrame.from_dict(people, orient="index").reset_index()
        people_df.columns = [
            "person",
            "positive_comments",
            "negative_comments",
            "neutral_comments",
            "total_comments",
        ]
        people_df = people_df.sort_values(by="total_comments", ascending=False)

        # Display the final dataframe
        st.write("People mentioned and their sentiment counts:")
        st.dataframe(people_df)
        if st.button("Download CSV"):
            csv = people_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="people_sentiment_analysis.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
