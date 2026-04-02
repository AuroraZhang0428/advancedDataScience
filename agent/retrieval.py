import pandas as pd


def load_listings(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Clean price if needed
    if "price" in df.columns:
        df["price"] = df["price"].replace(r"[\$,]", "", regex=True).astype(float)

    # Fill missing text fields
    for col in ["name", "description", "amenities", "room_type", "neighbourhood_group_cleansed"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # Rename columns for easier downstream use
    rename_map = {
        "neighbourhood_group_cleansed": "location",
        "accommodates": "guests",
        "review_scores_rating": "rating",
    }
    df = df.rename(columns=rename_map)

    return df


def filter_listings(df: pd.DataFrame, prefs: dict) -> pd.DataFrame:
    filtered = df.copy()
    hard = prefs["hard_constraints"]

    if hard.get("max_price") is not None and "price" in filtered.columns:
        filtered = filtered[filtered["price"] <= hard["max_price"]]

    if hard.get("guests") is not None and "guests" in filtered.columns:
        filtered = filtered[filtered["guests"] >= hard["guests"]]

    if hard.get("location") and "location" in filtered.columns:
        filtered = filtered[
            filtered["location"].str.lower().str.contains(str(hard["location"]).lower(), na=False)
        ]

    if hard.get("room_type") and "room_type" in filtered.columns:
        filtered = filtered[
            filtered["room_type"].str.lower() == str(hard["room_type"]).lower()
        ]

    return filtered