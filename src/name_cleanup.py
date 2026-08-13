import re
import unicodedata


def value_to_cleaned_string(value):
    text = "" if value is None else str(value)
    text = text.split("\\K", 1)[0]
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_name_cols_db(event, cols_to_clean=["profession", "first_name", "patronym", "last_name"]):
    for col in cols_to_clean:
        event[col] = _clean(event.get(col)).split("föd", 1)[0].strip()

    return event


def get_cleaned_full_name_db(event):
    cleaned_event = clean_name_cols_db(event)
    full_name = ""
    for part in (
        cleaned_event.get("profession", ""),
        cleaned_event.get("first_name", ""),
        cleaned_event.get("patronym", ""),
        cleaned_event.get("last_name", ""),
    ):
        if part != "" and full_name != "":
            full_name += " "
        full_name += part.strip()

    return full_name
