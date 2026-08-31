import os
import json
from openai import OpenAI
import snowflake.connector
from dotenv import load_dotenv

load_dotenv(".env")
MODEL="gpt-4o-mini"

SAMPLE_N =5

TOPICS = ["food quality", "delivery", "pricing", "service", "packaging", "other"]

CLIENT=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = f"""
You classify customer reviews for a food delivery app.

For the review you are given, return:
- sentiment_label: positive, negative, or neutral
- sentiment_score: a number between -1.0 and 1.0
- topic: one of {TOPICS}
- key_issue: a short phrase of 6 words or less that describes the main issue in the review, if any. If there is no issue, return null

Reply as JSON in this exact format:
{{
    "sentiment_label": "<sentiment_label>",
    "sentiment_score": <sentiment_score>,
    "topic": "<topic>",
    "key_issue": "<key_issue>"}}
"""
def get_connection():
    return snowflake.connector.connect(
        user=os.environ.get("SNOWFLAKE_USER"),
        password=os.environ.get("SNOWFLAKE_PASSWORD"),
        account=os.environ.get("SNOWFLAKE_ACCOUNT"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE"),
        database=os.environ.get("SNOWFLAKE_DATABASE"),
        schema=os.environ.get("SNOWFLAKE_SCHEMA")
    )

def create_output_table(cursor):
    cursor.execute("""CREATE SCHEMA IF NOT EXISTS ZOMATO.AI""")
    cursor.execute("""CREATE OR REPLACE TABLE ZOMATO.AI.review_enriched (
        review_id STRING,
        sentiment_label STRING,
        sentiment_score FLOAT,
        topic STRING,
        key_issue STRING,
        Model STRING,
        Enriched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
        )"""
    )
def get_reviews_to_enrich(cursor):
    cursor.execute(f"""
        SELECT review_id, COMMENT
        FROM ZOMATO.RAW.reviews
        WHERE review_id NOT IN (SELECT review_id FROM ZOMATO.AI.review_enriched)
        ORDER BY RANDOM()
        LIMIT {SAMPLE_N}
    """)
    return cursor.fetchall()

def classify_review(review_text):
    response = CLIENT.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": review_text}
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


    
def save_results(cursor, results):
    """Insert all the enriched rows into Snowflake in one go."""
    print(f"Saving {len(results)} enriched reviews to Snowflake...")
    cursor.executemany(
        """
        INSERT INTO ZOMATO.AI.REVIEW_ENRICHED
            (review_id, sentiment_label, sentiment_score, topic, key_issue, model)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        results,
    )

def main():
    conn = get_connection()
    cursor = conn.cursor()
    create_output_table(cursor)
    reviews = get_reviews_to_enrich(cursor)

    if len(reviews) == 0:
        print("No new reviews to enrich.")
        return

    print(f"Enriching {len(reviews)} reviews...")

    results = []
    for review_id, comment in reviews:
        print(f"Classifying review {review_id}: {comment}")
        try:
            labels = classify_review(comment)
            print(f"{labels}")
            results.append((
                review_id,
                labels["sentiment_label"],
                labels["sentiment_score"],
                labels["topic"],
                labels["key_issue"],
                MODEL
            ))
        except Exception as e:
            print(f"Error occurred while classifying review {review_id}: {e}")

    save_results(cursor, results)
    print(f"Saved {len(results)} enriched reviews to Snowflake.")
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()