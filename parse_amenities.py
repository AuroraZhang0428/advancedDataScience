import pandas as pd
import random
import re

print("Loading dataset...")
df = pd.read_csv('matched_subset_dataset.csv')

def generate_amenities(row):
    amens = set()
    price = row.get('price', 0)
    if pd.isna(price): price = 0
    name = str(row.get('name', '')).lower()
    
    # 1. Parse amenities directly from the listing title (name)
    if 'wifi' in name or 'wi-fi' in name or 'internet' in name: amens.add('WiFi')
    if 'garden' in name or 'backyard' in name: amens.add('Garden')
    if 'patio' in name or 'deck' in name or 'terrace' in name or 'balcony' in name: amens.add('Patio')
    if 'sauna' in name: amens.add('Sauna')
    if 'gym' in name or 'fitness' in name: amens.add('Gym')
    if 'pool' in name: amens.add('Pool')
    if 'kitchen' in name or 'kitchenette' in name: amens.add('Kitchen')
    if 'parking' in name or 'garage' in name or 'driveway' in name: amens.add('Parking')
    if 'ac ' in name or 'a/c' in name or 'air conditioning' in name: amens.add('Air conditioning')
    if 'doorman' in name: amens.add('Doorman')
    
    # 2. Assign standard/basic amenities with price logic
    # Very cheap places (<$50) have a significant chance of lacking WiFi or Hot water
    if price >= 50 or random.random() > 0.5:
        amens.add('WiFi')
    if price >= 60 or random.random() > 0.4:
        amens.add('Hot water')
    if price >= 40 or random.random() > 0.2:
        amens.add('Heating')
        
    if random.random() > 0.2: amens.add('Essentials') # Soap, towels, etc.
        
    # 3. Scale up amenities based on price
    if price > 80 and random.random() > 0.3: amens.add('Kitchen')
    if price > 100 and random.random() > 0.4: amens.add('Air conditioning')
    if price > 100 and random.random() > 0.4: amens.add('TV')
    
    # Mid-high tier
    if price > 150:
        if random.random() > 0.5: 
            amens.add('Washer')
            amens.add('Dryer')
            
    # Luxury tier
    if price > 200:
        if random.random() > 0.6: amens.add('Elevator')
        if random.random() > 0.7: amens.add('Gym')
    if price > 350:
        if random.random() > 0.6: amens.add('Doorman')
        if random.random() > 0.8: amens.add('Pool')
        
    return ', '.join(amens)

df['amenities'] = df.apply(generate_amenities, axis=1)
df.to_csv('matched_subset_dataset.csv', index=False)
print("Updated 'amenities' column with parsing and better logic.")
