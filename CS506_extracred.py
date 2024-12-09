import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from geopy.distance import great_circle
import optuna

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

target = 'is_fraud'
id_col = 'id'


def process_datetime_features(df):
    df['datetime'] = pd.to_datetime(df['trans_date'] + ' ' + df['trans_time'], 
                                    format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_month'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    return df

def process_dob(df):
    df['dob'] = pd.to_datetime(df['dob'], format='%Y-%m-%d', errors='coerce')
    df['age'] = df.apply(lambda x: x['year'] - x['dob'].year if pd.notnull(x['dob']) else np.nan, axis=1)
    return df

def process_distance(df):
    def calc_dist(row):
        if (pd.notnull(row['lat']) and pd.notnull(row['long']) 
            and pd.notnull(row['merch_lat']) and pd.notnull(row['merch_long'])):
            return great_circle((row['lat'], row['long']), (row['merch_lat'], row['merch_long'])).km
        else:
            return np.nan
    df['distance'] = df.apply(calc_dist, axis=1)
    df['distance_amt_ratio'] = df['distance'] / (df['amt'] + 1)
    return df

def process_amount(df):
    df['amt_log'] = np.log1p(df['amt'])
    return df

def process_categorical(df):
    cat_cols = ['category', 'gender', 'state', 'job', 'city', 'merchant']
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

def encode_categoricals(train, test, cols):
    for c in cols:
        le = LabelEncoder()
        combined = pd.concat([train[c], test[c]], axis=0).astype(str)
        le.fit(combined)
        train[c] = le.transform(train[c].astype(str))
        test[c] = le.transform(test[c].astype(str))
    return train, test

def feature_engineering(train, test):
    train = process_datetime_features(train)
    test = process_datetime_features(test)
    
    train = process_dob(train)
    test = process_dob(test)
    
    train = process_distance(train)
    test = process_distance(test)
    
    train = process_amount(train)
    test = process_amount(test)
    
    train = process_categorical(train)
    test = process_categorical(test)
    
    cat_features = ['category', 'gender', 'state', 'job', 'city', 'merchant']
    train, test = encode_categoricals(train, test, cat_features)
    
    train['rolling_amt_mean'] = train.groupby('cc_num')['amt'].transform('mean')
    test['rolling_amt_mean'] = test.groupby('cc_num')['amt'].transform('mean')
    
    return train, test

train, test = feature_engineering(train, test)

drop_cols = [id_col, 'trans_date', 'trans_time', 'unix_time', 'first', 'last', 'street', 'zip', 'lat', 'long', 
             'merch_lat', 'merch_long', 'dob', 'datetime','trans_num']

train.drop(columns=drop_cols, inplace=True, errors='ignore')
test.drop(columns=drop_cols, inplace=True, errors='ignore')

X = train.drop([target], axis=1)
y = train[target]

y = y.values.ravel() if isinstance(y, pd.DataFrame) else y

X_test = test.copy()

def objective(trial):
    print(f"Starting trial {trial.number}")
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'auc',
        'learning_rate': 0.01,
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', -1, 15),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'random_state': 42
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))

    for train_idx, val_idx in skf.split(X, y):  # Corrected split
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_val],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=50)
            ],
            num_boost_round=500
        )
        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)

    score = roc_auc_score(y, oof_preds)
    print(f"Trial {trial.number} completed with score: {score}")
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10, timeout=1800, n_jobs=-1)  # Reduced trials, added timeout

best_params = study.best_params

final_model = lgb.train(
    best_params,
    lgb.Dataset(X, y),
    num_boost_round=1000,
    callbacks=[
        lgb.log_evaluation(period=50)
    ]
)

test_preds = final_model.predict(X_test)
optimal_threshold = 0.5
test_preds_binary = (test_preds > optimal_threshold).astype(int)

submission = sample_submission.copy()
submission['is_fraud'] = test_preds_binary
submission.to_csv('submissionfinal.csv', index=False)
print("Submission file saved as submission.csv")
