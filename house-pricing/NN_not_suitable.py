import pandas as pd
from tensorflow.keras import models, layers
# from data_processing import preprocessing
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, MinMaxScaler


def preprocessing(df):
    # ONE SHOT
    onehot_cols = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
                   'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'Electrical', 'Functional', 'GarageType', 'MiscFeature', 'SaleType', 'SaleCondition']
    onehot = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = onehot.fit_transform(df[onehot_cols])

    encoded_df = pd.DataFrame(
        encoded, 
        columns=onehot.get_feature_names_out(onehot_cols),
        index=df.index
    )
    df = pd.concat([df.drop(onehot_cols, axis=1), encoded_df], axis=1)

    # ORDINAL
    ordinal_cols = ['OverallQual', 'OverallCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC',
                    'KitchenQual', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence'] 
    ordinal = OrdinalEncoder()
    encoded = ordinal.fit_transform(df[ordinal_cols])
    
    # encoded_df = pd.DataFrame(
    #     encoded, 
    #     columns=ordinal.get_feature_names_out(ordinal_cols)
    # )
    encoded_df = pd.DataFrame(encoded, columns=ordinal_cols, index=df.index)
    df = pd.concat([df.drop(ordinal_cols, axis=1), encoded_df], axis=1)

    # YEARS
    df['Age'] = 2025 - df['YearBuilt']
    df['RemodAge'] = 2025 - df['YearRemodAdd']
    df['GarageAge'] = 2025 - df['GarageYrBlt']
    df['SinceLastSold'] = 2025 - df['YrSold']
    df.drop(['YearBuilt', 'YearRemodAdd', 'GarageYrBlt'], axis=1, inplace=True)

    # STANDARD SCALER
    numeric_cols = ['LotFrontage', 'LotArea', 'Age', 'RemodAge', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                    '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
                    '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'GarageAge', 'SinceLastSold']
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # MIN_MAX_SCALER
    min_max_cols = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces']
    scaler = MinMaxScaler()
    df[min_max_cols] = scaler.fit_transform(df[min_max_cols])

    # BINARY
    df["CentralAir"] = df["CentralAir"].map({"Y": 1, "N": 0})

    # I THINK NOT IMPORTANT
    df.drop(['Id', 'MoSold'], axis=1, inplace=True)

    return df

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

n_train = len(train_df)
n_test = len(test_df)

y_scaler = StandardScaler()
y_train_full = train_df['SalePrice']
y_train_scaled = y_scaler.fit_transform(y_train_full.values.reshape(-1, 1))
print(y_train_scaled)
train_df = train_df.drop(['SalePrice'], axis=1)

df_all = pd.concat([train_df, test_df], axis=0, ignore_index=True)
df_all = preprocessing(df_all)

train_processed = df_all.iloc[:n_train, :]
test_processed = df_all.iloc[n_train:, :]


sparse_cols = (train_processed == 0).sum() / len(train_processed) > 0.99
train_processed = train_processed.loc[:, ~sparse_cols]
test_processed = test_processed.loc[:, ~sparse_cols]
print(list((train_processed == 0).sum(axis=0) > .99))
print(train_processed.shape)

model = models.Sequential([
    layers.Dense(512, input_dim=182),
    layers.LeakyReLU(alpha=0.1),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    layers.Dense(256, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='linear')
])

model.compile(
    optimizer=Adam(),
    loss=MeanSquaredError(),
    metrics=['mae']
)

history = model.fit(train_processed, y_train_scaled, validation_split=0.15, epochs=200, batch_size=32)


# PREDICTIONS
predictions_scaled = model.predict(test_processed)
print(predictions_scaled)
predictions = y_scaler.inverse_transform(predictions_scaled).flatten()
print(predictions)
output_df = pd.DataFrame({
    # 'Id': range(1, len(predictions) + 1),
    'SalePrice': predictions
})
output_df.to_csv('predictions.csv', index=False)