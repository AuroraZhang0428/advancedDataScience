import pandas as pd
import random

print("Loading matched_subset_dataset.csv...")
df_main = pd.read_csv('matched_subset_dataset.csv')

print("Loading reviews-3.csv (this might take a minute)...")
try:
    df_reviews = pd.read_csv('reviews-3.csv', usecols=['listing_id', 'comments'])
except Exception as e:
    print(f"Error: {e}")
    exit(1)

print("Filtering and sampling reviews...")
# Drop empty comments
df_reviews = df_reviews.dropna(subset=['comments'])
# Ensure string
df_reviews['comments'] = df_reviews['comments'].astype(str)

# Filter for reasonable length comments (e.g., 30 to 250 characters)
# This removes "ok" and massive walls of text.
mask = (df_reviews['comments'].str.len() > 30) & (df_reviews['comments'].str.len() < 250)
df_valid_reviews = df_reviews[mask]

# Group by listing_id and take up to 2 random reviews
# A fast way to do this in pandas without massive groupby overhead on 1M rows:
# Since we filtered down, let's just shuffle and drop duplicates to get 1-2 per listing.

df_shuffled = df_valid_reviews.sample(frac=1, random_state=42)

# Get the first review for each listing
df_rev1 = df_shuffled.drop_duplicates(subset=['listing_id'], keep='first')

# Get the second review for each listing
df_remaining = df_shuffled[~df_shuffled.index.isin(df_rev1.index)]
df_rev2 = df_remaining.drop_duplicates(subset=['listing_id'], keep='first')

# Combine them into a dictionary: listing_id -> "Review 1 | Review 2"
# We'll use a dictionary for fast mapping
review_dict = {}

for _, row in df_rev1.iterrows():
    lid = row['listing_id']
    comment = row['comments'].replace('\n', ' ').replace('\r', '').strip()
    review_dict[lid] = [comment]

for _, row in df_rev2.iterrows():
    lid = row['listing_id']
    comment = row['comments'].replace('\n', ' ').replace('\r', '').strip()
    if lid in review_dict:
        review_dict[lid].append(comment)

def get_sample_reviews(lid):
    if lid in review_dict:
        # Join the 1 or 2 reviews we found
        return " | ".join(review_dict[lid])
    return "No text reviews available."

print("Merging into main dataset...")
df_main['sample_reviews'] = df_main['id'].apply(get_sample_reviews)

df_main.to_csv('matched_subset_dataset.csv', index=False)
print("Successfully added a 'sample_reviews' column with real user comments!")
