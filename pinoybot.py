import os
import cloudpickle
from typing import List


# Same feature extraction sa training
def extract_features(word: str) -> dict:
    word = str(word)
    lower = word.lower()
    vowels = "aeiou"

    num_vowels = sum(c in vowels for c in lower)
    num_consonants = sum(c.isalpha() and c not in vowels for c in lower)

    has_fil_prefix = lower.startswith(
        ("nag", "mag", "pag", "ipa", "pina", "naka", "maka", "ina")
    )

    has_foreign_letters = any(c in lower for c in "fjcqvxz")

    ends_eng_cluster = lower.endswith(
        (
            "ch", "sh", "ct", "th", "nk", "ck", "st", "pt",
            "xt", "ble", "tion", "ment", "ness", "ity"
        )
    )

    starts_eng_cluster = lower.startswith(
        (
            "str", "spr", "thr", "sch", "chr", "ph",
            "wh", "kn", "wr", "sh", "th", "ch"
        )
    )

    has_eng_traits = (
        has_foreign_letters
        or ends_eng_cluster
        or starts_eng_cluster
        or lower.endswith(("ed", "ing"))
    )

    return {
        "length": len(word),
        "starts_upper": word[:1].isupper(),
        "all_upper": word.isupper(),
        "has_digit": any(c.isdigit() for c in word),
        "has_hyphen": "-" in word,
        "has_apostrophe": "'" in word,

        "num_vowels": num_vowels,
        "num_consonants": num_consonants,
        "vowel_ratio": num_vowels / len(word) if word else 0,

        "prefix1": lower[:1],
        "prefix2": lower[:2],
        "prefix3": lower[:3],

        "suffix1": lower[-1:],
        "suffix2": lower[-2:],
        "suffix3": lower[-3:],

        "is_cs_pattern": has_fil_prefix and has_eng_traits,
        "pure_eng_no_fil_prefix": not has_fil_prefix and has_eng_traits,
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
        "contains_z": "z" in lower
    }


# Main tagging function
def tag_language(tokens: List[str]) -> List[str]:
    """
    Tags each token in the input list with its predicted language.

    Args:
        tokens: List of word tokens (strings).

    Returns:
        tags: List of predicted tags ("ENG", "FIL", "CS", or "OTH"), one per token.
    """

    # 1. Load the trained model from disk
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "model.pkl not found. Put model.pkl beside pinoybot.py."
        )

    with open(model_path, "rb") as file:
        artifact = cloudpickle.load(file)

    model = artifact["model"]
    vectorizer = artifact["vectorizer"]

    # Use saved feature function if available, otherwise use local function
    feature_function = artifact.get("extract_features", extract_features)

    # 2. Extract features from the input tokens
    features = [feature_function(token) for token in tokens]
    feature_matrix = vectorizer.transform(features)

    # 3. Use the model to predict the tags for each token
    predicted = model.predict(feature_matrix)

    # 4. Convert the predictions to a list of strings ("ENG", "FIL", or "OTH")
    tags = [str(tag) for tag in predicted]

    # 5. Return the predicted tags
    return tags


if __name__ == "__main__":
    # Example usage
    example_tokens = ["Love", "kita", "."]
    print("Tokens:", example_tokens)

    tags = tag_language(example_tokens)
    print("Predicted Tags:", tags)