# 🎫 Support Ticket Categorizer

A machine learning-based support ticket classification system that automatically categorizes incoming support tickets into **Billing, Technical, HR, or General**.

The project uses **TF-IDF** for text feature extraction and **Logistic Regression** for classification. It also provides confidence scores, human-review routing for uncertain predictions, and priority tagging for urgent tickets.

## 📌 Project Overview

Support teams receive many tickets every day, and manually assigning each ticket to the correct department can be time-consuming.

This project automates the initial ticket triage process by analyzing the ticket's **subject and body** and predicting the appropriate category.

### Supported Categories

* Billing
* Technical
* HR
* General

## ✨ Features

* Cleans and preprocesses ticket text
* Combines ticket subject and body for classification
* Converts text into numerical features using TF-IDF
* Trains a Logistic Regression classifier
* Evaluates model accuracy
* Displays precision, recall, and F1-score
* Generates a confusion matrix
* Predicts categories for unseen tickets
* Returns prediction confidence scores
* Routes predictions below 60% confidence to **Needs Human Review**
* Detects urgent tickets using keyword-based priority rules
* Provides an interactive command-line demo
* Handles missing, empty, and invalid input

## 🗂️ Project Structure

```text
ticket-categorizer/
│
├── ticket_categorizer.py
├── tickets.csv
├── requirements.txt
├── README.md
```

## 🧠 Machine Learning Approach

The project follows the following pipeline:

```text
Ticket Subject + Body
        ↓
Text Cleaning
        ↓
TF-IDF Vectorization
        ↓
Logistic Regression
        ↓
Category Prediction
        ↓
Confidence Score
        ↓
Priority Detection
        ↓
Final Ticket Routing
```

### 1. Text Preprocessing

The ticket subject and body are combined and cleaned before being passed to the model.

Preprocessing includes:

* Converting text to lowercase
* Removing URLs
* Removing unnecessary characters
* Removing extra whitespace
* Handling missing values

### 2. TF-IDF Feature Extraction

Machine learning models cannot directly understand raw text.

**TF-IDF (Term Frequency-Inverse Document Frequency)** converts ticket text into numerical feature vectors while giving greater importance to informative words.

The vectorizer also uses unigrams and bigrams to capture both individual words and short phrases.

### 3. Logistic Regression

Logistic Regression is used as the classification model because it works effectively with sparse, high-dimensional TF-IDF features and is computationally efficient for real-time text classification.

It also supports probability estimates through `predict_proba()`, allowing the system to calculate a confidence score for every prediction.

## 📊 Model Evaluation

The dataset is divided into training and testing sets.

The model is evaluated using:

* **Accuracy** — overall percentage of correct predictions
* **Precision** — how often predicted categories are correct
* **Recall** — how well the model identifies tickets belonging to each category
* **F1-score** — balance between precision and recall
* **Confusion Matrix** — shows which categories are correctly classified or confused with others

> Model performance depends heavily on the size, quality, and diversity of the training dataset. A small demonstration dataset should not be treated as representative of production performance.

## 🎯 Confidence-Based Human Review

The classifier returns a probability for each category.

The category with the highest probability is selected as the predicted category.

A confidence threshold of **60%** is used:

```text
Confidence >= 60%
        ↓
Automatically assign predicted category

Confidence < 60%
        ↓
Needs Human Review
```

This prevents uncertain predictions from being automatically assigned to the wrong department.

## 🚨 Priority Tagging

Along with category prediction, the system assigns a simple priority level.

Keywords such as:

```text
urgent
down
not working
cannot access
immediately
critical
crash
failed
```

cause the ticket to be tagged as:

```text
Urgent
```

Otherwise, the ticket is tagged as:

```text
Normal
```

This rule-based priority layer operates independently from the machine learning category prediction.

## 📄 Dataset Format

The input dataset should be stored in `tickets.csv`.

Required columns:

```csv
subject,body,category
Payment failed,"My payment was declined but money was deducted.",Billing
Login issue,"I cannot log in after resetting my password.",Technical
Leave request,"I would like to apply for annual leave.",HR
Office hours,"What are the office working hours?",General
```

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone YOUR_REPOSITORY_URL
```

### 2. Navigate to the Project

```bash
cd ticket-categorizer
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 📦 Requirements

The project requires:

```text
pandas
scikit-learn
```

Python 3.9 or newer is recommended.

## ▶️ Running the Project

Run:

```bash
python ticket_categorizer.py
```

The program will:

1. Load the ticket dataset
2. Clean and preprocess the text
3. Train the classification model
4. Evaluate its performance
5. Display predictions for sample unseen tickets
6. Start the interactive ticket categorizer

## 💻 Live CLI Demo

After training, the program allows a new ticket to be entered directly from the terminal.

Example:

```text
========== LIVE TICKET CATEGORIZER ==========

Ticket subject: Application not working
Ticket body: Urgent, the application is down and our team cannot access it.

--- Prediction ---
Category   : Technical
Confidence : 82.45%
Priority   : Urgent
```

If the confidence score is too low:

```text
Category   : Needs Human Review
Confidence : 48.32%
Priority   : Normal
Suggested category: General
```

Type:

```text
exit
```

as the ticket subject to close the program.

## 🧩 Edge-Case Handling

The project handles several common edge cases:

* Missing dataset file
* Missing required CSV columns
* Missing subject or body values
* Empty ticket text
* Invalid or unusable records
* Low-confidence predictions
* Tickets that do not clearly belong to one category

Uncertain predictions are routed to **Needs Human Review** rather than being automatically assigned.

## 📝 Approach Summary

Used TF-IDF to convert cleaned ticket text into numerical features and trained a Logistic Regression classifier to predict Billing, Technical, HR, or General categories. The model returns confidence scores and routes predictions below 60% confidence to human review. Missing or empty text is handled safely, and keyword-based priority tagging identifies urgent tickets.

## 🔮 Future Improvements

With more data, the model could be trained using a larger and more diverse collection of real support tickets and optimized using cross-validation and hyperparameter tuning.

Future improvements could include:

* Larger production-quality training datasets
* Class imbalance handling
* Hyperparameter optimization
* Better confidence calibration
* Automatic retraining using reviewed tickets
* REST API integration
* Streamlit web interface
* Database integration
* Real-time support platform integration
* Model performance monitoring

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF
* Logistic Regression
* Natural Language Processing
* Machine Learning
* Regular Expressions

## 📌 Conclusion

This project demonstrates an end-to-end text classification workflow for automated support ticket triage. It combines NLP preprocessing, TF-IDF feature extraction, machine learning classification, confidence-based routing, and rule-based priority detection to create a simple and practical ticket categorization system.
