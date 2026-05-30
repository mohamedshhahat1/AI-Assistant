"""
Intent Detection Model for AI Assistant
========================================

This module implements an NLP-based intent classification system built from scratch
using scikit-learn. It detects what a user is trying to accomplish (their "intent")
from natural language input.

Key Concepts for Beginners:
---------------------------

TF-IDF (Term Frequency - Inverse Document Frequency):
    A technique to convert text into numerical features that ML models can understand.
    - Term Frequency (TF): How often a word appears in a document.
    - Inverse Document Frequency (IDF): How rare/important a word is across all documents.
    - Words that appear frequently in one document but rarely in others get higher scores.
    - Example: "hello" in a greeting pattern gets a high TF-IDF score for that category.

Logistic Regression:
    Despite its name, this is a classification algorithm (not regression).
    - It draws decision boundaries between classes in the feature space.
    - For multi-class problems, it uses a one-vs-rest strategy.
    - It outputs probability scores, so we can measure confidence.
    - Fast to train, works well with TF-IDF text features, and is highly interpretable.

Pipeline:
    User text -> Preprocessing -> TF-IDF Vectorization -> Logistic Regression -> Intent + Confidence
"""

import json
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class IntentDetector:
    """
    ML-based intent classification system.

    This class handles the full lifecycle of intent detection:
    1. Loading training data from a JSON file
    2. Training a TF-IDF + Logistic Regression pipeline
    3. Saving/loading trained models for fast inference
    4. Predicting intents from user input with confidence scores

    Attributes:
        intents_path (str): Path to the intents training data JSON file.
        model_path (str): Path to save/load the trained model pickle file.
        vectorizer (TfidfVectorizer): Converts text to TF-IDF feature vectors.
        classifier (LogisticRegression): Classifies feature vectors into intents.
        intent_names (list): List of all known intent tag names.
    """

    def __init__(self, intents_path="backend/data/intents.json", model_path="models/intent_model.pkl"):
        """
        Initialize the IntentDetector.

        Attempts to load a pre-trained model from disk. If no model exists,
        automatically trains a new one from the intents JSON data.

        Args:
            intents_path (str): Path to the JSON file containing training patterns.
                                Defaults to "backend/data/intents.json".
            model_path (str): Path to save/load the serialized model.
                              Defaults to "models/intent_model.pkl".
        """
        self.intents_path = intents_path
        self.model_path = model_path

        # These will hold our trained components
        self.vectorizer = None
        self.classifier = None
        self.intent_names = []

        # Load the intents training data from JSON
        self._load_intents_data()

        # Try to load a pre-trained model from disk
        # If no model file exists, train a new one automatically
        if os.path.exists(self.model_path):
            self._load_model()
            print(f"[IntentDetector] Loaded pre-trained model from: {self.model_path}")
        else:
            print("[IntentDetector] No pre-trained model found. Training a new model...")
            self.train()

    def _load_intents_data(self):
        """
        Load and parse the intents JSON training data file.

        The JSON file contains intent tags and their associated training patterns.
        This method extracts all unique intent names for reference.
        """
        with open(self.intents_path, "r") as f:
            self.intents_data = json.load(f)

        # Extract all intent tag names
        self.intent_names = [intent["tag"] for intent in self.intents_data["intents"]]

    def _load_model(self):
        """
        Load a previously trained model from a pickle file using joblib.

        The pickle file contains a dictionary with:
        - 'vectorizer': The fitted TF-IDF vectorizer
        - 'classifier': The trained Logistic Regression classifier
        - 'intent_names': The list of intent names the model was trained on
        """
        # joblib is efficient for loading large numpy arrays (used inside sklearn models)
        model_data = joblib.load(self.model_path)
        self.vectorizer = model_data["vectorizer"]
        self.classifier = model_data["classifier"]
        self.intent_names = model_data["intent_names"]

    def train(self):
        """
        Train the intent classification model from scratch.

        Training Pipeline:
        1. Extract patterns and labels from the intents JSON data
        2. Create a TF-IDF vectorizer to convert text into numerical features
        3. Fit a Logistic Regression classifier on the TF-IDF features
        4. Evaluate accuracy on a train/test split
        5. Retrain on full data for production use
        6. Save the trained model to disk

        Returns:
            float: The training accuracy score (0.0 to 1.0).
        """
        # Step 1: Prepare training data
        # Extract all patterns (X) and their corresponding intent tags (y)
        patterns = []  # Input texts
        labels = []    # Target intent tags

        for intent in self.intents_data["intents"]:
            tag = intent["tag"]
            for pattern in intent["patterns"]:
                patterns.append(pattern.lower().strip())
                labels.append(tag)

        print(f"[IntentDetector] Training with {len(patterns)} patterns across {len(self.intent_names)} intents")

        # Step 2: Create the TF-IDF Vectorizer
        # This converts raw text into numerical feature vectors
        # - ngram_range=(1,2): Use both single words and word pairs as features
        #   Example: "good morning" creates features for "good", "morning", AND "good morning"
        # - max_features=1000: Limit to top 1000 most informative features
        # - stop_words='english': Remove common English words like "the", "is", "at"
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1000,
            stop_words="english"
        )

        # Step 3: Transform text patterns into TF-IDF feature vectors
        X = self.vectorizer.fit_transform(patterns)

        # Step 4: Evaluate using a train/test split
        # Split data: 80% for training, 20% for testing accuracy
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Step 5: Train the Logistic Regression Classifier
        # - max_iter=1000: Allow enough iterations for convergence
        # - C=10: Regularization parameter (higher = less regularization, fits training data more closely)
        #   Good for small datasets where we want to capture all patterns
        # - solver='lbfgs': Efficient solver that supports multinomial loss
        self.classifier = LogisticRegression(
            max_iter=1000,
            C=10,
            random_state=42
        )

        # Fit on training split to evaluate
        self.classifier.fit(X_train, y_train)

        # Calculate accuracy on the test set
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"[IntentDetector] Test accuracy: {accuracy:.2%}")

        # Step 6: Retrain on ALL data for production use
        # After evaluating, we use all available data for the final model
        self.classifier.fit(X, labels)
        print("[IntentDetector] Model retrained on full dataset for production use.")

        # Step 7: Save the trained model to disk
        self._save_model()

        return accuracy

    def _save_model(self):
        """
        Save the trained model components to a pickle file using joblib.

        Creates the models/ directory if it doesn't exist.
        Saves the vectorizer, classifier, and intent names together.
        """
        # Create the models directory if it doesn't exist
        model_dir = os.path.dirname(self.model_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)

        # Bundle all components needed for inference
        model_data = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "intent_names": self.intent_names
        }

        # joblib is preferred over pickle for sklearn models
        # because it efficiently handles large numpy arrays
        joblib.dump(model_data, self.model_path)
        print(f"[IntentDetector] Model saved to: {self.model_path}")

    def predict(self, text):
        """
        Predict the intent of user input text.

        Pipeline:
        1. Preprocess: lowercase and strip whitespace
        2. Vectorize: Convert text to TF-IDF features using the fitted vectorizer
        3. Predict: Use the classifier to determine the most likely intent
        4. Confidence: Extract the probability score for the predicted intent

        Args:
            text (str): The user's input text to classify.

        Returns:
            tuple: A tuple of (intent_name, confidence_score).
                   - intent_name (str): The predicted intent tag (e.g., "greeting").
                   - confidence_score (float): Probability between 0.0 and 1.0.

        Example:
            >>> detector = IntentDetector()
            >>> intent, confidence = detector.predict("hello there")
            >>> print(intent, confidence)
            greeting 0.95
        """
        # Step 1: Preprocess the input text
        # Lowercase ensures "Hello" and "hello" are treated the same
        # Strip removes leading/trailing whitespace
        processed_text = text.lower().strip()

        # Step 2: Convert text to TF-IDF feature vector
        # We use transform() (not fit_transform()) because the vectorizer
        # is already fitted on the training data vocabulary
        text_vector = self.vectorizer.transform([processed_text])

        # Step 3: Predict the intent class
        predicted_intent = self.classifier.predict(text_vector)[0]

        # Step 4: Get the confidence score (probability)
        # predict_proba() returns probabilities for ALL classes
        # We extract the probability for the predicted class
        probabilities = self.classifier.predict_proba(text_vector)[0]
        confidence = max(probabilities)

        return (predicted_intent, confidence)

    def detect(self, text):
        """
        Detect the intent of user input (dict-based interface).

        This is a convenience wrapper around predict() that returns a dictionary
        instead of a tuple — used by the FastAPI endpoints.

        Args:
            text (str): The user's input text to classify.

        Returns:
            dict: {"intent": "greeting", "confidence": 0.95}
        """
        intent, confidence = self.predict(text)
        return {"intent": intent, "confidence": float(confidence)}

    def get_all_intents(self):
        """
        Get a list of all known intent names.

        Returns:
            list: A list of intent tag strings (e.g., ["greeting", "goodbye", ...]).
        """
        return self.intent_names


# Allow running this module directly for testing
if __name__ == "__main__":
    print("=" * 50)
    print("Intent Detection Model - Training & Testing")
    print("=" * 50)

    # Initialize the detector (will auto-train if no model exists)
    detector = IntentDetector()

    # Test with sample inputs
    test_sentences = [
        "hello there!",
        "goodbye, see you later",
        "thank you so much",
        "what is artificial intelligence?",
        "how do I learn Python?",
        "can you help me with something?",
        "my name is John",
        "I'm feeling great today",
        "I'm having a terrible day",
        "what can you do?",
    ]

    print("\n" + "=" * 50)
    print("Testing Predictions:")
    print("=" * 50)

    for sentence in test_sentences:
        intent, confidence = detector.predict(sentence)
        print(f"  Input: '{sentence}'")
        print(f"  -> Intent: {intent} (confidence: {confidence:.2%})")
        print()

    print(f"All known intents: {detector.get_all_intents()}")
