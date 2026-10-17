
# ============================================================
# RETAILSENSE PRODUCT REVIEW SENTIMENT ANALYSIS
# Cynaris AI/ML Internship - RetailSense Pvt Ltd
# ============================================================

import json
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


# ------------------------------------------------------------
# 1. FILE LOCATIONS
# ------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent

DATA_FILE = APP_DIR / "aiml_product_reviews.csv"
MODEL_FILE = APP_DIR / "retailsense_model.pkl"
METRICS_FILE = APP_DIR / "retailsense_metrics.json"
PREDICTIONS_FILE = APP_DIR / "retailsense_predictions.csv"


# ------------------------------------------------------------
# 2. STREAMLIT PAGE
# ------------------------------------------------------------

st.set_page_config(
    page_title="RetailSense Sentiment Dashboard",
    page_icon="🛍️",
    layout="wide",
)

st.title("RetailSense Pvt Ltd")
st.subheader("Product Review Sentiment Analysis Dashboard")

st.write(
    "AI/ML Internship Project | Cynaris Solutions"
)

st.markdown(
    """
    This application analyzes product reviews and classifies
    them as **Positive, Negative, or Neutral**.

    It also displays sentiment trends, compares machine learning
    models, evaluates accuracy, and predicts sentiment for new reviews.
    """
)


# ------------------------------------------------------------
# 3. TEXT CLEANING
# ------------------------------------------------------------

def clean_text(text):
    """Clean review text before machine learning."""

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9\s!?.,']",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ------------------------------------------------------------
# 4. FIND REQUIRED DATASET COLUMNS
# ------------------------------------------------------------

def find_column(columns, possible_names):
    """Find a column without depending on capitalization."""

    lookup = {
        str(column).strip().lower(): column
        for column in columns
    }

    for name in possible_names:
        if name in lookup:
            return lookup[name]

    for column in columns:
        column_name = str(column).strip().lower()

        for name in possible_names:
            if name in column_name:
                return column

    return None


# ------------------------------------------------------------
# 5. STANDARDIZE SENTIMENT LABELS
# ------------------------------------------------------------

def normalize_sentiment(value):
    """Convert common sentiment labels to standard names."""

    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    labels = {
        "positive": "positive",
        "pos": "positive",
        "negative": "negative",
        "neg": "negative",
        "neutral": "neutral",
        "neu": "neutral",
        "1": "positive",
        "-1": "negative",
        "0": "neutral",
    }

    return labels.get(value)


# ------------------------------------------------------------
# 6. LOAD AND PREPARE DATASET
# ------------------------------------------------------------

def load_dataset(uploaded_file):

    if uploaded_file is not None:

        # Read uploaded CSV data
        uploaded_bytes = uploaded_file.getvalue()

        from io import BytesIO

        df = pd.read_csv(BytesIO(uploaded_bytes))

        # Save a local copy beside this Python file
        df.to_csv(DATA_FILE, index=False)

    elif DATA_FILE.exists():

        df = pd.read_csv(DATA_FILE)

    else:

        return None, None

    if df.empty:
        raise ValueError("The CSV file contains no records.")

    # Detect important columns
    text_column = find_column(
        df.columns,
        ["review_text", "review", "text", "comment", "feedback"]
    )

    sentiment_column = find_column(
        df.columns,
        ["sentiment", "sentiment_label", "label", "target", "polarity"]
    )

    rating_column = find_column(
        df.columns,
        ["rating", "stars", "score"]
    )

    category_column = find_column(
        df.columns,
        ["category", "product_category", "department"]
    )

    if text_column is None:
        raise ValueError(
            "Review text column not found. "
            "Expected a column such as review_text."
        )

    prepared = df.copy()

    # Clean review text
    prepared["_review_text"] = (
        prepared[text_column]
        .fillna("")
        .astype(str)
        .apply(clean_text)
    )

    # Normalize sentiment labels
    if sentiment_column is not None:

        prepared["_sentiment"] = (
            prepared[sentiment_column]
            .apply(normalize_sentiment)
        )

    else:

        prepared["_sentiment"] = None

    # Use ratings only when a sentiment label is unavailable
    missing_labels = prepared["_sentiment"].isna()

    if missing_labels.any() and rating_column is not None:

        ratings = pd.to_numeric(
            prepared[rating_column],
            errors="coerce"
        )

        rating_sentiments = pd.Series(
            index=prepared.index,
            dtype="object"
        )

        rating_sentiments.loc[ratings <= 2] = "negative"
        rating_sentiments.loc[ratings == 3] = "neutral"
        rating_sentiments.loc[ratings >= 4] = "positive"

        prepared.loc[
            missing_labels,
            "_sentiment"
        ] = rating_sentiments.loc[missing_labels]

    # Remove records without usable text or labels
    prepared = prepared[
        prepared["_review_text"].str.len().gt(0)
        & prepared["_sentiment"].isin(
            ["positive", "negative", "neutral"]
        )
    ].copy()

    if prepared.empty:
        raise ValueError(
            "No usable reviews found. Check your text, sentiment, "
            "and rating columns."
        )

    if prepared["_sentiment"].nunique() < 2:
        raise ValueError(
            "At least two sentiment classes are required for training."
        )

    # Prepare product categories
    if category_column is not None:

        prepared["_category"] = (
            prepared[category_column]
            .fillna("Unspecified")
            .astype(str)
        )

    else:

        prepared["_category"] = "Unspecified"

    return prepared.reset_index(drop=True), {
        "original_rows": len(df),
        "text_column": text_column,
        "sentiment_column": sentiment_column,
        "rating_column": rating_column,
        "category_column": category_column,
    }


# ------------------------------------------------------------
# 7. BUILD MACHINE LEARNING MODELS
# ------------------------------------------------------------

def create_model(model_name):

    if model_name == "Logistic Regression":

        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )

    elif model_name == "Linear SVM":

        classifier = LinearSVC(
            class_weight="balanced"
        )

    else:

        classifier = MultinomialNB()

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=50000,
                sublinear_tf=True
            )
        ),
        ("classifier", classifier)
    ])

    return model


# ------------------------------------------------------------
# 8. UPLOAD DATASET
# ------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload aiml_product_reviews.csv",
    type=["csv"]
)

if uploaded_file is None and not DATA_FILE.exists():

    st.info(
        "Please upload the official product reviews CSV "
        "from your Cynaris project portal."
    )

    st.stop()


# ------------------------------------------------------------
# 9. LOAD DATA SAFELY
# ------------------------------------------------------------

try:

    df, column_info = load_dataset(uploaded_file)

except Exception as error:

    st.error(f"Dataset error: {error}")
    st.stop()


if df is None or df.empty:

    st.error("No usable dataset is available.")
    st.stop()


# ------------------------------------------------------------
# 10. DATASET OVERVIEW
# ------------------------------------------------------------

st.header("1. Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Original records",
    f"{column_info['original_rows']:,}"
)

col2.metric(
    "Usable reviews",
    f"{len(df):,}"
)

col3.metric(
    "Product categories",
    f"{df['_category'].nunique():,}"
)

if len(df) < 1000:

    st.warning(
        f"The dataset currently contains {len(df)} usable reviews. "
        "The project brief mentions 1,000 reviews. "
        "Use the complete official dataset if available."
    )

with st.expander("View dataset details"):

    st.write("Detected columns")
    st.json(column_info)

    st.write("Sentiment counts")

    st.dataframe(
        df["_sentiment"]
        .value_counts()
        .rename_axis("Sentiment")
        .reset_index(name="Count"),
        use_container_width=True
    )

    st.write("First 10 records")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )


# ------------------------------------------------------------
# 11. TRAINING SETTINGS
# ------------------------------------------------------------

st.header("2. Train Machine Learning Models")

test_size = st.select_slider(
    "Test dataset percentage",
    options=[15, 20, 25, 30],
    value=20
)

st.write(
    f"{100 - test_size}% training data and "
    f"{test_size}% held-out testing data."
)

if st.button("Train Models and Evaluate Accuracy", type="primary"):

    class_counts = df["_sentiment"].value_counts()

    if len(df) < 10 or class_counts.min() < 2:

        st.error(
            "There are too few records in one or more sentiment classes. "
            "Please check your dataset."
        )

        st.stop()

    X = df["_review_text"]
    y = df["_sentiment"]

    # Split data into training and test sets
    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size / 100,
            random_state=42,
            stratify=y
        )

    except ValueError as error:

        st.error(f"Unable to split the dataset: {error}")
        st.stop()

    st.write(f"Training records: {len(X_train)}")
    st.write(f"Testing records: {len(X_test)}")

    # Compare three baseline models
    model_names = [
        "Logistic Regression",
        "Linear SVM",
        "Multinomial Naive Bayes"
    ]

    trained_models = {}
    results = []

    progress = st.progress(0)
    status = st.empty()

    for index, model_name in enumerate(model_names):

        status.write(f"Training {model_name}...")

        model = create_model(model_name)

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        trained_models[model_name] = model

        results.append({
            "Model": model_name,
            "Accuracy (%)": round(accuracy * 100, 2)
        })

        progress.progress(
            (index + 1) / len(model_names)
        )

    # Select the model with the best measured test accuracy
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "Accuracy (%)",
        ascending=False
    ).reset_index(drop=True)

    best_model_name = results_df.loc[0, "Model"]
    best_model = trained_models[best_model_name]

    best_predictions = best_model.predict(X_test)

    best_accuracy = accuracy_score(
        y_test,
        best_predictions
    )

    status.success("Model comparison completed.")

    # Save trained model
    joblib.dump(best_model, MODEL_FILE)

    # Save test predictions
    predictions_df = pd.DataFrame({
        "review_text": X_test.values,
        "actual_sentiment": y_test.values,
        "predicted_sentiment": best_predictions
    })

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    # Save evaluation metrics
    labels = sorted(y.unique())

    report = classification_report(
        y_test,
        best_predictions,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    metrics = {
        "original_dataset_rows": int(column_info["original_rows"]),
        "usable_reviews": int(len(df)),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "best_model": best_model_name,
        "test_accuracy_percent": round(
            best_accuracy * 100, 2
        ),
        "target_accuracy_percent": 82,
        "model_comparison": results,
        "classification_report": report
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # 12. MODEL COMPARISON
    # --------------------------------------------------------

    st.header("3. Model Comparison")

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots()

    ax.bar(
        results_df["Model"],
        results_df["Accuracy (%)"]
    )

    ax.set_title("Machine Learning Model Accuracy")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
    ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # --------------------------------------------------------
    # 13. BEST MODEL ACCURACY
    # --------------------------------------------------------

    st.header("4. Evaluation Results")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Best model",
        best_model_name
    )

    col2.metric(
        "Test accuracy",
        f"{best_accuracy * 100:.2f}%"
    )

    col3.metric(
        "Required target",
        "82%"
    )

    if best_accuracy >= 0.82:

        st.success(
            "The model reached the 82% accuracy target "
            "on this test split."
        )

    else:

        st.warning(
            "The model has not reached the 82% target on this test split. "
            "Do not report the target as achieved. Review the data and "
            "model performance before making improvements."
        )

    # --------------------------------------------------------
    # 14. CLASSIFICATION REPORT
    # --------------------------------------------------------

    st.subheader("Classification Report")

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True
    )

    # --------------------------------------------------------
    # 15. CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        best_predictions,
        labels=labels
    )

    fig, ax = plt.subplots()

    image = ax.imshow(cm)

    ax.set_xticks(
        range(len(labels)),
        labels=labels,
        rotation=25
    )

    ax.set_yticks(
        range(len(labels)),
        labels=labels
    )

    ax.set_xlabel("Predicted sentiment")
    ax.set_ylabel("Actual sentiment")
    ax.set_title("Confusion Matrix")

    for row in range(cm.shape[0]):

        for column in range(cm.shape[1]):

            ax.text(
                column,
                row,
                str(cm[row, column]),
                ha="center",
                va="center"
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # --------------------------------------------------------
    # 16. SENTIMENT DISTRIBUTION
    # --------------------------------------------------------

    st.header("5. Sentiment Distribution")

    sentiment_counts = (
        df["_sentiment"]
        .value_counts()
        .reindex(
            ["positive", "neutral", "negative"],
            fill_value=0
        )
    )

    fig, ax = plt.subplots()

    sentiment_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Product Review Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Number of reviews")
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        sentiment_counts.rename_axis("Sentiment")
        .reset_index(name="Review Count"),
        use_container_width=True
    )

    # --------------------------------------------------------
    # 17. CATEGORY-WISE SENTIMENT ANALYSIS
    # --------------------------------------------------------

    st.header("6. Sentiment Trends by Product Category")

    category_table = pd.crosstab(
        df["_category"],
        df["_sentiment"]
    )

    st.dataframe(
        category_table,
        use_container_width=True
    )

    fig, ax = plt.subplots()

    category_table.plot(
        kind="bar",
        stacked=True,
        ax=ax
    )

    ax.set_title("Sentiment by Product Category")
    ax.set_xlabel("Product category")
    ax.set_ylabel("Number of reviews")
    ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)

    # --------------------------------------------------------
    # 18. PREDICT NEW REVIEW SENTIMENT
    # --------------------------------------------------------

    st.header("7. Predict Sentiment of a New Review")

    new_review = st.text_area(
        "Enter a product review",
        placeholder=(
            "Example: The product quality is excellent "
            "and delivery was quick."
        )
    )

    if st.button("Predict Review Sentiment"):

        cleaned_review = clean_text(new_review)

        if not cleaned_review:

            st.warning(
                "Please enter a review before predicting."
            )

        else:

            predicted_sentiment = best_model.predict(
                [cleaned_review]
            )[0]

            st.success(
                f"Predicted Sentiment: "
                f"{predicted_sentiment.upper()}"
            )

            if hasattr(best_model, "predict_proba"):

                probabilities = best_model.predict_proba(
                    [cleaned_review]
                )[0]

                probability_df = pd.DataFrame({
                    "Sentiment": best_model.classes_,
                    "Probability (%)": (
                        probabilities * 100
                    ).round(2)
                })

                st.dataframe(
                    probability_df,
                    use_container_width=True,
                    hide_index=True
                )

    # --------------------------------------------------------
    # 19. DOWNLOAD TEST PREDICTIONS
    # --------------------------------------------------------

    st.header("8. Download Results")

    st.download_button(
        label="Download Test Predictions CSV",
        data=predictions_df.to_csv(index=False),
        file_name="retailsense_predictions.csv",
        mime="text/csv"
    )

    st.download_button(
        label="Download Evaluation Metrics JSON",
        data=json.dumps(metrics, indent=4),
        file_name="retailsense_metrics.json",
        mime="application/json"
    )

    st.info(
        f"Saved model: {MODEL_FILE.name}\n\n"
        f"Saved metrics: {METRICS_FILE.name}\n\n"
        f"Saved predictions: {PREDICTIONS_FILE.name}"
    )


# ------------------------------------------------------------
# 20. FOOTER
# ------------------------------------------------------------

st.divider()

st.caption(
    "RetailSense | Cynaris AI/ML Internship | "
    "TF-IDF + Logistic Regression + Linear SVM + Naive Bayes"
)
