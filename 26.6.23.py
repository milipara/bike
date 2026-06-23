import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import mean_squared_error , r2_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# 1. 데이터 불러오기
df = pd.read_csv('dataset/bike.csv')
print("데이터 크기:", df.shape)

# 2. 간단한 전처리 및 새로운 피처 추가
df['datetime'] = pd.to_datetime(df['datetime'])
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['weekday'] = df['datetime'].dt.weekday

# 필요 없는 컬럼 제거
df = df.drop(['datetime', 'casual' , 'registered'], axis=1)

df['count'] = np.log1p(df['count'])

X = df.drop('count',axis=1)
y = df['count']

X_train , x_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test_scaled)

xgb = XGBRegressor(objective = 'reg:squarederror' , random_sta=42)

param_grid = {
    'n_estimators' : [100, 200],
    'max_depth' : [3,5],
    'learning_rage' : [0.05,0.1],
    'subsample' : [0.8 , 1],
    'colsample_bytree' : [0.8 , 1],
    'alpha' : [0 , 1],
    'lambda' : [1, 10]
}

grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring = 'neg_root_mean_squared_error'
    
    
)