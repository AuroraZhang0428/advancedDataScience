def explain_listing(row, prefs):
    reasons = []

    if "price" in row:
        reasons.append(f"price is ${row['price']:.0f} per night")

    if row.get("rating") is not None:
        reasons.append(f"rating is {row.get('rating')}")

    amenities_text = str(row.get("amenities", "")).lower()
    desired_amenities = prefs["soft_preferences"].get("amenities", [])
    matched = [a for a in desired_amenities if a.lower() in amenities_text]

    if matched:
        reasons.append(f"matches desired amenities: {', '.join(matched)}")

    if prefs["soft_preferences"].get("purpose") == "remote work":
        if "wifi" in amenities_text or "workspace" in amenities_text:
            reasons.append("supports remote work needs")

    return "Recommended because " + "; ".join(reasons) + "."