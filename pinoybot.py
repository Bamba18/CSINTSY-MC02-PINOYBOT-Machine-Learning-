"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", "CS", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import os
import pickle
from typing import List

# Main tagging function
def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.
    Args:
        tokens: List of word tokens (strings).
    Returns:
        tags: List of predicted tags ("ENG", "FIL", "CS", or "OTH"), one per token.
    """
    if _MODEL is None or _VEC is None:
            raise FileNotFoundError(
                "model.pkl not found! Please run your training script to generate model.pkl first."
            )

    # 1. Extract features from the input tokens
    dicts = [extract_features(t) for t in tokens]

    # 2. Transform features using pre-fitted DictVectorizer
    X = _VEC.transform(dicts)

    # 3. Predict tags
    predictions = _MODEL.predict(X)

    # 4. Return predicted tags as a list of strings
    return [str(tag) for tag in predictions]



# 1. Load your trained model from disk (e.g., using pickle or joblib)
#    Example: with open('trained_model.pkl', 'rb') as f: model = pickle.load(f)
#    (Replace with your actual model loading code)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

if os.path.exists(_MODEL_PATH):
    with open(_MODEL_PATH, "rb") as f:
        _ARTIFACT = pickle.load(f)
        _VEC = _ARTIFACT["vectorizer"]
        _MODEL = _ARTIFACT["model"]
else:
    _VEC = None
    _MODEL = None

# 2. Extract features from the input tokens to create the feature matrix
#    Example: features = ... (your feature extraction logic here)

def extract_features(word: str) -> dict:
    word = str(word)
    lower = word.lower()
    vowels = "aeiou"

    num_vowels = sum(c in vowels for c in lower)
    num_consonants = sum(c.isalpha() and c not in vowels for c in lower)

    return {
        "length": len(word),
        "starts_upper": word[:1].isupper(),
        "all_upper": word.isupper(),
        "has_digit": any(c.isdigit() for c in word),
        "has_hyphen": "-" in word,
        "has_apostrophe": "'" in word,
        "num_vowels": num_vowels,
        "num_consonants": num_consonants,
        "vowel_ratio": num_vowels / len(word) if len(word) else 0,
        "prefix1": lower[:1],
        "prefix2": lower[:2],
        "prefix3": lower[:3],
        "suffix1": lower[-1:],
        "suffix2": lower[-2:],
        "suffix3": lower[-3:],
        "contains_ng": "ng" in lower,
        "contains_mag": lower.startswith("mag"),
        "contains_nag": lower.startswith("nag"),
        "contains_pag": lower.startswith("pag"),
        "contains_um": "um" in lower,
        "contains_in": "in" in lower,
        "contains_ing": lower.endswith("ing"),
        "contains_ed": lower.endswith("ed"),
        "contains_ly": lower.endswith("ly"),
        "contains_tion": lower.endswith("tion"),
        "number_of_a": lower.count("a"),
        "contains_f": "f" in lower,
        "contains_c": "c" in lower,
        "contains_j": "j" in lower,
        "contains_q": "q" in lower,
        "contains_v": "v" in lower,
        "contains_x": "x" in lower,
        "contains_z": "z" in lower,
    }

# 3. Use the model to predict the tags for each token
#    Example: predicted = model.predict(features)


    # 4. Convert the predictions to a list of strings ("ENG", "FIL", or "OTH")
    #    Example: tags = [str(tag) for tag in predicted]

    # 5. Return the list of tags
    #    return tags

    # You can define other functions, import new libraries, or add other Python files as needed, as long as
    # the tag_language function is retained and correctly accomplishes the expected task.

    # Currently, the bot just tags every token as FIL. Replace this with your more intelligent predictions.
    #return ['FIL' for i in tokens]

if __name__ == "__main__":
    # Example usage
    example_tokens = ["tangina", "mo", "i", "love", "u", "nagmarch", "ako", "sa", "EDSA", "."]
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)
    print("Predicted Tags:", tags)  # <-- Added print so output is visible!