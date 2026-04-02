import pandas as pd


def score_listing(row: pd.Series, prefs: dict) -> float:
    score = 0.0
    soft = prefs["soft_preferences"]

    # Higher ratings are better
    rating = row.get("rating", 0)
    if pd.notna(rating):
        score += float(rating)

    # Amenities match
    amenities_text = str(row.get("amenities", "")).lower()
    for amenity in soft.get("amenities", []):
        if amenity.lower() in amenities_text:
            score += 2.0

    # Purpose / vibe match through description
    description = str(row.get("description", "")).lower()

    if soft.get("vibe") and soft["vibe"].lower() in description:
        score += 1.5

    if soft.get("purpose") == "remote work":
        if "workspace" in amenities_text or "wifi" in amenities_text:
            score += 2.0

    if soft.get("preferred_area"):
        location = str(row.get("location", "")).lower()
        if soft["preferred_area"].lower() in location:
            score += 1.0

    return score


def rank_listings(df: pd.DataFrame, prefs: dict, top_k: int = 5) -> pd.DataFrame:
    ranked = df.copy()
    ranked["score"] = ranked.apply(lambda row: score_listing(row, prefs), axis=1)
    ranked = ranked.sort_values(by="score", ascending=False)
    return ranked.head(top_k)