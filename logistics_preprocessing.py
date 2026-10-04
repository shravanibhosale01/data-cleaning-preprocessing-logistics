"""Week 2: Data collection (simulated), cleaning and preprocessing for logistics analysis.

The dataset is simulated to resemble public e-commerce / supply chain delivery data.
Data-quality problems (duplicates, missing values, outliers, inconsistent text,
wrong data types) are injected on purpose so each cleaning step can be demonstrated.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 10_000

# ---------------------------------------------------------------- 1. Collect
df = pd.DataFrame({
    'order_id': np.arange(1, N + 1),
    'order_date': pd.to_datetime('2025-01-01') + pd.to_timedelta(rng.integers(0, 365, N), unit='D'),
    'warehouse': rng.choice(['Warehouse A', 'Warehouse B', 'Warehouse C'], N),
    'vehicle_type': rng.choice(['Bike', 'Van', 'Truck'], N, p=[0.5, 0.35, 0.15]),
    'distance_km': rng.gamma(4, 3, N).round(1),
    'parcel_weight_kg': rng.gamma(2, 2, N).round(2),
    'traffic_index': rng.integers(1, 11, N),
})
df['delivery_time_min'] = (10 + 3.2 * df['distance_km'] + 2 * df['traffic_index']
                           + rng.normal(0, 8, N)).round(0)
df['delivery_cost'] = (20 + 6 * df['distance_km'] + 4 * df['parcel_weight_kg']
                       + rng.normal(0, 10, N)).round(2)

# ------------------------------------------------ inject data-quality issues
df.loc[rng.choice(N, 600, replace=False), 'distance_km'] = np.nan
df.loc[rng.choice(N, 450, replace=False), 'traffic_index'] = np.nan
df.loc[rng.choice(N, 300, replace=False), 'vehicle_type'] = np.nan
df.loc[rng.choice(N, 40, replace=False), 'delivery_time_min'] = rng.integers(900, 3000, 40)   # outliers
df.loc[rng.choice(N, 25, replace=False), 'parcel_weight_kg'] = -1                              # invalid
df.loc[rng.choice(N, 500, replace=False), 'warehouse'] = ' warehouse a '                       # inconsistent text
df['order_date'] = df['order_date'].dt.strftime('%Y-%m-%d')                                    # stored as text
df = pd.concat([df, df.sample(120, random_state=1)], ignore_index=True)                        # duplicates

raw_rows = len(df)
print('RAW SHAPE:', df.shape)
print('\nMissing values per column:\n', df.isna().sum())
print('Duplicate rows:', df.duplicated().sum())
print('Negative weights:', (df['parcel_weight_kg'] < 0).sum())

# ---------------------------------------------------------------- 2. Clean
# 2.1 duplicates
df = df.drop_duplicates()

# 2.2 data types
df['order_date'] = pd.to_datetime(df['order_date'])

# 2.3 inconsistent text
df['warehouse'] = df['warehouse'].str.strip().str.title()

# 2.4 invalid values -> missing
df.loc[df['parcel_weight_kg'] < 0, 'parcel_weight_kg'] = np.nan

# 2.5 missing values
df['vehicle_type'] = df['vehicle_type'].fillna(df['vehicle_type'].mode()[0])          # categorical: mode
for col in ['distance_km', 'traffic_index', 'parcel_weight_kg']:
    df[col] = df[col].fillna(df.groupby('warehouse')[col].transform('median'))        # numeric: group median

# 2.6 outliers (IQR rule, capped rather than dropped)
q1, q3 = df['delivery_time_min'].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
n_out = ((df['delivery_time_min'] < low) | (df['delivery_time_min'] > high)).sum()
df['delivery_time_min'] = df['delivery_time_min'].clip(low, high)
print(f'\nOutliers capped (IQR): {n_out}  bounds=({low:.0f}, {high:.0f})')

# ------------------------------------------------------------ 3. Preprocess
num_cols = ['distance_km', 'parcel_weight_kg', 'traffic_index', 'delivery_time_min']
# 3.1 Min-Max normalisation (0-1)
for c in num_cols:
    df[c + '_minmax'] = (df[c] - df[c].min()) / (df[c].max() - df[c].min())
# 3.2 Z-score standardisation
for c in num_cols:
    df[c + '_z'] = (df[c] - df[c].mean()) / df[c].std()
# 3.3 Encoding and date features
df = pd.get_dummies(df, columns=['vehicle_type', 'warehouse'], drop_first=True)
df['month'] = df['order_date'].dt.month
df['day_of_week'] = df['order_date'].dt.dayofweek

print('\nCLEAN SHAPE:', df.shape, '| rows removed:', raw_rows - len(df))
print('Missing values left:', int(df.isna().sum().sum()))
print(df[['distance_km_minmax', 'delivery_time_min_z']].describe().loc[['min', 'max', 'mean', 'std']].round(3))
df.to_csv('logistics_clean.csv', index=False)
