import pandas as pd
import numpy as np

# Generate datetime range (30 days hourly)
date_range = pd.date_range(start='2023-01-01', periods=24*30, freq='H')

# Create realistic energy pattern
np.random.seed(42)

energy = []
for dt in date_range:
    hour = dt.hour
    
    base = 100
    
    if 6 <= hour <= 10:
        base += 100
    elif 17 <= hour <= 21:
        base += 150
    
    noise = np.random.normal(0, 10)
    
    energy.append(base + noise)

df = pd.DataFrame({
    'Datetime': date_range,
    'Energy': energy
})

df.to_csv('data/energy.csv', index=False)

print("✅ Dataset generated successfully!")