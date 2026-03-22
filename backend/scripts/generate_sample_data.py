"""
Generate Sample Grievance Data for testing and development
Creates synthetic grievance records with realistic features
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Sample data
CATEGORIES = ['Pothole', 'Street Light', 'Sanitation', 'Water Supply', 
              'Electricity', 'Noise Pollution', 'Parking', 'Road Damage',
              'Community Safety', 'Traffic Light']

LOCATIONS = ['North District', 'South District', 'East Zone', 'West Zone',
             'Central City', 'Suburb Area', 'Industrial Zone', 'Residential Area',
             'Commercial Area', 'Downtown']

COMPLAINTS = {
    'Pothole': [
        "Deep pothole on Main Road causing accidents",
        "Multiple potholes near the market making driving dangerous",
        "Severe pothole damage on highway affecting commute",
        "New potholes appeared after rain",
    ],
    'Street Light': [
        "Street light not working for 3 weeks",
        "Multiple broken street lights on the avenue",
        "Dark road at night due to non-functional lights",
        "Street lighting system failure in the area",
    ],
    'Sanitation': [
        "Garbage not collected for days",
        "Waste accumulation causing health hazard",
        "Dirty streets need immediate cleaning",
        "Sanitation workers absent for a week",
    ],
    'Water Supply': [
        "No water supply for 5 days",
        "Low water pressure affecting households",
        "Contaminated water from municipal supply",
        "Water pipeline burst causing shortage",
    ],
    'Electricity': [
        "Frequent power outages affecting business",
        "Electricity bill overcharged without reason",
        "No electricity for one week",
        "Faulty electrical connection creating hazard",
    ],
    'Noise Pollution': [
        "Construction noise disturbing residents",
        "Loud music from nearby establishment",
        "Traffic noise intolerable at night",
        "Industrial noise exceeding permissible levels",
    ],
    'Parking': [
        "Insufficient parking spaces in market",
        "Vehicles blocking entire street",
        "No designated parking area created",
        "Parking violation enforcement needed",
    ],
    'Road Damage': [
        "Entire road surface needs repair",
        "Cracked asphalt creating hazard",
        "Road subsidence near municipal area",
        "Uneven road surface causing accidents",
    ],
    'Community Safety': [
        "Frequent theft in the area",
        "Stray dogs roaming streets dangerously",
        "Inadequate police patrolling",
        "Street crime increasing daily",
    ],
    'Traffic Light': [
        "Traffic light not functioning properly",
        "Signal timing causes congestion",
        "Traffic light stuck on red",
        "No coordination between signals",
    ]
}

PRIORITY_KEYWORDS = {
    'High': ['urgent', 'emergency', 'danger', 'severe', 'critical', 'hazard', 'safety', 'accident'],
    'Medium': ['important', 'repair', 'needed', 'issue', 'problem', 'required'],
    'Low': ['minor', 'small', 'slight', 'little', 'concern', 'might']
}


def generate_sample_data(n_samples=1000):
    """Generate sample grievance dataset"""
    
    print(f"Generating {n_samples} sample grievance records...")
    
    data = []
    base_date = datetime(2023, 1, 1)
    
    for i in range(n_samples):
        category = random.choice(CATEGORIES)
        location = random.choice(LOCATIONS)
        complaint_text = random.choice(COMPLAINTS[category])
        
        # Add significant variation to complaint text
        variations = [
            f"Please check the {category.lower()} issue at {location}.",
            f"Large {category.lower()} problem reported by citizens in {location}.",
            f"The {category.lower()} status in {location} is critical.",
            f"Immediate attention needed for {category.lower()} near {location}.",
            f"Concern regarding {category.lower()} on the main street of {location}."
        ]
        complaint_text = f"{complaint_text} {random.choice(variations)}. Reference ID: {random.randint(1000, 9999)}."
        
        # Generate timestamp
        days_offset = random.randint(0, 365)
        hours_offset = random.randint(0, 23)
        timestamp = base_date + timedelta(days=days_offset, hours=hours_offset)
        
        # Assign priority based on keywords
        priority = 'Low'
        for keyword_category, keywords in PRIORITY_KEYWORDS.items():
            if any(kw in complaint_text.lower() for kw in keywords):
                priority = keyword_category
                break
        else:
            # Random assignment for unmatched
            priority = random.choices(['High', 'Medium', 'Low'], weights=[0.2, 0.4, 0.4])[0]
        
        data.append({
            'complaint_id': f'GR{i+1:06d}',
            'complaint_text': complaint_text,
            'category': category,
            'location': location,
            'timestamp': timestamp,
            'priority': priority,
            'department': random.choice(['Public Works', 'Municipal Services', 'Police', 'Utilities']),
            'status': random.choice(['Open', 'In Progress', 'Resolved']),
            'resolution_days': random.randint(1, 60) if random.random() > 0.3 else None
        })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    os.makedirs('data', exist_ok=True)
    output_path = 'data/grievances_dataset.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✓ Generated {len(df)} records")
    print(f"✓ Saved to {output_path}")
    print(f"\nDataset Info:")
    print(f"  - Categories: {df['category'].nunique()} types")
    print(f"  - Locations: {df['location'].nunique()} areas")
    print(f"  - Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  - Priority Distribution:\n{df['priority'].value_counts().to_string()}")
    print(f"  - Category Distribution:\n{df['category'].value_counts().to_string()}")
    
    return df


if __name__ == "__main__":
    df = generate_sample_data(n_samples=3000)
    print("\n" + "="*50)
    print("Dataset Preview:")
    print("="*50)
    print(df.head(10).to_string())
