import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# -----------------------------------
# Configuration
# -----------------------------------

DATA_FILE = "tickets.csv"
CONFIDENCE_THRESHOLD = 0.60


# -----------------------------------
# Text preprocessing
# -----------------------------------

def clean_text(text):
    """Clean ticket text before vectorization."""

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Keep alphabetic characters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------------
# Priority detection
# -----------------------------------

def detect_priority(text):
    urgent_keywords = [
        "urgent",
        "down",
        "not working",
        "cannot access",
        "immediately",
        "critical",
        "crash",
        "failed"
    ]

    text = text.lower()

    for keyword in urgent_keywords:
        if keyword in text:
            return "Urgent"

    return "Normal"


# -----------------------------------
# Load dataset
# -----------------------------------

try:
    df = pd.read_csv(DATA_FILE)

except FileNotFoundError:
    print(f"Error: {DATA_FILE} was not found.")
    raise SystemExit(1)


required_columns = {"subject", "body", "category"}

if not required_columns.issubset(df.columns):
    print("Error: Dataset must contain subject, body and category columns.")
    raise SystemExit(1)


# -----------------------------------
# Handle missing values
# -----------------------------------

df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")
df["category"] = df["category"].fillna("")


# Remove rows without a category
df = df[df["category"].str.strip() != ""].copy()


# Combine subject and body
df["text"] = df["subject"] + " " + df["body"]

df["clean_text"] = df["text"].apply(clean_text)


# Remove rows where text becomes empty
df = df[df["clean_text"].str.strip() != ""].copy()


# -----------------------------------
# Train/test split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["category"],
    test_size=0.25,
    random_state=42,
    stratify=df["category"]
)


# -----------------------------------
# TF-IDF Vectorization
# -----------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# -----------------------------------
# Logistic Regression model
# -----------------------------------

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)


# -----------------------------------
# Evaluation
# -----------------------------------

predictions = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, predictions)

print("\n========== MODEL EVALUATION ==========\n")

print(f"Accuracy: {accuracy:.2%}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, predictions))


# -----------------------------------
# Real-time prediction
# -----------------------------------

def predict_ticket(subject, body):

    if not str(subject).strip() and not str(body).strip():
        return {
            "category": "Needs Human Review",
            "confidence": 0.0,
            "priority": "Normal"
        }

    original_text = f"{subject} {body}"
    cleaned = clean_text(original_text)

    if not cleaned:
        return {
            "category": "Needs Human Review",
            "confidence": 0.0,
            "priority": detect_priority(original_text)
        }

    features = vectorizer.transform([cleaned])

    probabilities = model.predict_proba(features)[0]

    best_index = probabilities.argmax()

    predicted_category = model.classes_[best_index]
    confidence = probabilities[best_index]

    priority = detect_priority(original_text)

    # Low-confidence fallback
    if confidence < CONFIDENCE_THRESHOLD:
        final_category = "Needs Human Review"
    else:
        final_category = predicted_category

    return {
        "category": final_category,
        "predicted_category": predicted_category,
        "confidence": confidence,
        "priority": priority
    }


# -----------------------------------
# Five unseen sample tickets
# -----------------------------------

sample_tickets = [
    (
        "Payment problem",
        "I was charged twice for my subscription."
    ),
    (
        "System down",
        "The application is down and our team cannot access it."
    ),
    (
        "Leave request",
        "I need to apply for annual leave next week."
    ),
    (
        "Office information",
        "Can you tell me your office working hours?"
    ),
    (
        "Password problem",
        "Urgent, my password reset link is not working."
    )
]


print("\n========== SAMPLE PREDICTIONS ==========\n")

for subject, body in sample_tickets:

    result = predict_ticket(subject, body)

    print(f"Subject    : {subject}")
    print(f"Category   : {result['category']}")
    print(f"Confidence : {result['confidence']:.2%}")
    print(f"Priority   : {result['priority']}")
    print("-" * 50)


# -----------------------------------
# Mini CLI Demo
# -----------------------------------

print("\n========== LIVE TICKET CATEGORIZER ==========")

print("Type 'exit' as the subject to stop.\n")

while True:

    subject = input("Ticket subject: ").strip()

    if subject.lower() == "exit":
        print("Ticket categorizer stopped.")
        break

    body = input("Ticket body: ").strip()

    result = predict_ticket(subject, body)

    print("\n--- Prediction ---")

    print(f"Category   : {result['category']}")
    print(f"Confidence : {result['confidence']:.2%}")
    print(f"Priority   : {result['priority']}")

    if result["category"] == "Needs Human Review":
        print(
            f"Suggested category: "
            f"{result.get('predicted_category', 'Unknown')}"
        )

    print()
