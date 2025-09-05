import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Load data ---
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

y = train_df['SalePrice']
train_df = train_df.drop(['SalePrice'], axis=1)

# --- Combine train/test for consistent preprocessing ---
df_all = pd.concat([train_df, test_df], axis=0, ignore_index=True)

# --- Encode categorical variables ---
cat_cols = df_all.select_dtypes(include=['object']).columns.tolist()
for col in cat_cols:
    df_all[col] = df_all[col].fillna('NA')  # fill missing
    le = LabelEncoder()
    df_all[col] = le.fit_transform(df_all[col])

# --- Numeric columns ---
num_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
df_all[num_cols] = df_all[num_cols].fillna(df_all[num_cols].median())

# --- Feature engineering for years ---
df_all['Age'] = 2025 - df_all['YearBuilt']
df_all['RemodAge'] = 2025 - df_all['YearRemodAdd']
df_all['GarageAge'] = 2025 - df_all['GarageYrBlt']
df_all['SinceLastSold'] = 2025 - df_all['YrSold']
df_all.drop(['YearBuilt', 'YearRemodAdd', 'GarageYrBlt'], axis=1, inplace=True)

# --- Drop unnecessary columns ---
df_all.drop(['Id', 'MoSold'], axis=1, inplace=True)

# --- Separate train/test ---
X_train = df_all.iloc[:len(train_df), :]
X_test = df_all.iloc[len(train_df):, :]

# --- Split train for validation ---
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, test_size=0.15, random_state=42)

# --- LightGBM Dataset ---
lgb_train = lgb.Dataset(X_tr, label=y_tr)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

# --- LightGBM parameters ---
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42
}

# --- Train model ---
gbm = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_val]
)

# --- Predict on test ---
predictions = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# --- Save to CSV ---
output_df = pd.DataFrame({
    'Id': range(1461, 1461 + len(predictions)),
    'SalePrice': predictions
    })
output_df.to_csv('predictions_lgb.csv', index=False)
