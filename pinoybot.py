"""
pinoybot.py

PinoyBot: Filipino Code-Switched Language Identifier

This module provides the main tagging function for the PinoyBot project, which identifies the language of each word in a code-switched Filipino-English text. The function is designed to be called with a list of tokens and returns a list of tags ("ENG", "FIL", "CS", or "OTH").

Model training and feature extraction should be implemented in a separate script. The trained model should be saved and loaded here for prediction.
"""

import os
import pickle
from typing import List

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

    dicts = [extract_features(t) for t in tokens]

    X = _VEC.transform(dicts)

    predictions = _MODEL.predict(X)

    return [str(tag) for tag in predictions]


_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

if os.path.exists(_MODEL_PATH):
    with open(_MODEL_PATH, "rb") as f:
        _ARTIFACT = pickle.load(f)
        _VEC = _ARTIFACT["vectorizer"]
        _MODEL = _ARTIFACT["model"]
else:
    _VEC = None
    _MODEL = None



def extract_features(word: str) -> dict:
    word = str(word)
    lower = word.lower()
    vowels = "aeiou"

    num_vowels = sum(c in vowels for c in lower)
    num_consonants = sum(c.isalpha() and c not in vowels for c in lower)

    has_fil_prefix = lower.startswith(("nag", "mag", "pag", "ipa", "pina", "naka", "maka", "ina"))
    has_foreign_letters = any(c in lower for c in "fjcqvxz")
    ends_eng_cluster = lower.endswith(("ch", "sh", "ct", "th", "nk", "ck", "st", "pt", "xt", "ble", "tion", "ment", "ness", "ity"))
    starts_eng_cluster = lower.startswith(("str", "spr", "thr", "sch", "chr", "ph", "wh", "kn", "wr", "sh", "th", "ch"))
    
    has_eng_traits = has_foreign_letters or ends_eng_cluster or starts_eng_cluster or lower.endswith(("ed", "ing"))

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

        "is_cs_pattern": has_fil_prefix and has_eng_traits,
        "pure_eng_no_fil_prefix": (not has_fil_prefix) and has_eng_traits,
        "has_foreign_letters": has_foreign_letters,
        "ends_eng_cluster": ends_eng_cluster,
        "starts_eng_cluster": starts_eng_cluster,

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

if __name__ == "__main__":
    example_tokens = ["mahal", "kita", "i", "love", "u", "nagdate", "ako", "sa", "EDSA", "."]
    print("Tokens:", example_tokens)
    tags = tag_language(example_tokens)
    print("Predicted Tags:", tags)