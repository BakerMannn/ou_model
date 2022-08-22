######################################################
"""Import Packages"""
#Data Manipulation
import pandas as pd
import numpy as np

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns 

#Weather Data
from meteostat import Point, Daily

#Modeling
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

######################################################
"""Global Variables"""
#Misc
trade_prob_quantile = 0.85

#Model Training
cv = 5
scoring = 'accuracy'
n_jobs = -1
pca_n = 0.95

######################################################
"""Data Import/Clean"""
#Stadium Coordinates
df_coord = pd.read_excel('nfl_stadiums_coordinates.xlsx')

#Historical Odds
df_odds = pd.read_excel('historical_nfl_odds.xlsx')
df_odds_condensed = df_odds[['Date','Home Team', 'Away Team', 'Home Score', 'Away Score', 'Playoff Game?', 'Neutral Venue?', 'Total Score Close']]

df_odds_condensed[['Playoff Game?', 'Neutral Venue?']] = df_odds_condensed[['Playoff Game?', 'Neutral Venue?']].fillna(0).replace('Y', 1)
df_odds_condensed.dropna(inplace=True)

#Joined Dataframe
df_joined = df_odds_condensed.merge(df_coord, left_on='Home Team', right_on='Team')

#Weather
"""
def weather(row):
    location = Point(row['latitude'], row['longitude'])
    start = row['Date']
    end = row['Date']
    data = Daily(location, start, end)
    return data.fetch()

weather = df_joined.apply(weather, axis=1)

df_weather = pd.concat([x for x in weather])
df_weather.reset_index(inplace=True)

df_weather.to_excel('weather.xlsx')
"""

df_weather = pd.read_excel('weather.xlsx')

#Final Merge and Clean
df = pd.concat([df_joined, df_weather], axis=1)
df.drop(['wpgt', 'tsun'], axis=1, inplace=True)

df['snow'].fillna(0, inplace=True)

######################################################
"""Preprocessing"""
#Feature Engineering
df['result'] = (df['Home Score'] + df['Away Score']) > df['Total Score Close']
#df['result'].replace([True, False], ['Over', 'Under'], inplace=True)
df.columns
#Features & Labels
X = df[['Home Team', 
     'Away Team', 
     'Playoff Game?', 
     'Neutral Venue?', 
     'Conference', 
     'tavg', 
     'tmin', 
     'tmax', 
     'prcp', 
     'snow', 
     'wdir', 
     'wspd', 
     'pres']]
y = df['result']

#Data Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=555)

######################################################
"""Combined Pipeline"""
#Categorical and Numerical Features
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

#Numeric Preprocessing Pipeline
num_pipe = Pipeline([
                    ('imputer', SimpleImputer()),
                    ('scaler', StandardScaler())
                    ])

#Categorical Preprocessing Pipeline
cat_pipe = Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value='N/A')),
                    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
                    ])

#Combined Preprocessing Pipeline
combined_pipe = ColumnTransformer([('num', num_pipe, num_cols),
                                   ('cat', cat_pipe, cat_cols)]
                                  )
######################################################
"""Base Models Training"""
#Instantiate Models
xgb_clf = xgb.XGBClassifier(seed=555)
rf_clf = RandomForestClassifier(random_state=555)
lr_clf = LogisticRegression(random_state=555)

#XGB Grid/Train
xgb_pipeline = Pipeline([
                        ('combined_pipe', combined_pipe),
                        ('pca', PCA(n_components=pca_n, random_state=555)),
                        ('xgb_clf', xgb_clf)
                        ])

xgb_param_grid = {'xgb_clf__max_depth':[2,3,5,7,10],
                  'xgb_clf__n_estimators':[10,100,500],}

xgb_grid = GridSearchCV(xgb_pipeline, 
                    xgb_param_grid, 
                    cv=cv, 
                    scoring=scoring,
                    n_jobs = n_jobs)

xgb_grid.fit(X_train, y_train)
xgb_model = xgb_grid.best_estimator_
print(f'XGB Train Accuracy Score: {xgb_grid.best_score_}')

#Random Forest Grid/Train
rf_pipeline = Pipeline([
                        ('combined_pipe', combined_pipe),
                        ('pca', PCA(n_components=pca_n, random_state=555)),
                        ('rf_clf', rf_clf)
                        ])

rf_param_grid = {'rf_clf__max_depth':[2,3,5,7,10],
                  'rf_clf__n_estimators':[10,100,500],}

rf_grid = GridSearchCV(rf_pipeline, 
                    rf_param_grid, 
                    cv=cv, 
                    scoring=scoring,
                    n_jobs =n_jobs)

rf_grid.fit(X_train, y_train)

rf_model = rf_grid.best_estimator_
print(f'RF Train Accuracy Score: {rf_grid.best_score_}')

#Logistic Regression Grid/Train
lr_pipeline = Pipeline([
                        ('combined_pipe', combined_pipe),
                        ('pca', PCA(n_components=pca_n, random_state=555)),
                        ('lr_clf', lr_clf)
                        ])

lr_param_grid = {'lr_clf__C': [0.1, 1, 10, 100],
                  'lr_clf__solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                  'lr_clf__penalty': ['none', 'l1', 'l2', 'elasticnet']}

lr_grid = GridSearchCV(lr_pipeline, 
                    lr_param_grid, 
                    cv=cv, 
                    scoring=scoring,
                    n_jobs = n_jobs)

lr_grid.fit(X_train, y_train)

lr_model = lr_grid.best_estimator_
print(f'Log Reg Train Accuracy Score: {lr_grid.best_score_}')

############################################################################
"""Ensemble Training"""
# #Ensemble
# voting_clf = VotingClassifier(estimators=[('xgb', xgb_model),
#                                           ('rf', rf_model),
#                                           ('lr', lr_model)])

# #Ensmble Pipeline
# ensemble_pipeline = Pipeline([
#                         ('combined_pipe', combined_pipe),
#                         ('pca', PCA(n_components=pca_n, random_state=555)),
#                         ('clf', voting_clf)
#                         ])

# #Parameter Grid
# param_grid = {'clf__voting':['hard', 'soft']}

# #Grid Search
# ensemble_grid = GridSearchCV(ensemble_pipeline, 
#                     param_grid, 
#                     cv=cv, 
#                     scoring=scoring,
#                     n_jobs = n_jobs)

# #Model Train
# ensemble_grid.fit(X_train, y_train)
# ensemble_model = ensemble_grid.best_estimator_
# print(f'Ensemble Train Accuracy Score: {ensemble_grid.best_score_}')

############################################################################
"""Model Evaluation"""
#Dummy Estimator
dummy_clf = DummyClassifier(strategy='most_frequent', random_state=555)
dummy_clf.fit(X_train, y_train)

dummy_y_pred = dummy_clf.predict(X_test)
dummy_score = accuracy_score(y_test, dummy_y_pred)
dummy_report = classification_report(y_test, dummy_y_pred)

#XGB Score
xgb_y_pred = xgb_model.predict(X_test)
xgb_score = accuracy_score(y_test, xgb_y_pred)
xgb_report = classification_report(y_test, xgb_y_pred)

#Random Forest Score
rf_y_pred = rf_model.predict(X_test)
rf_score = accuracy_score(y_test, rf_y_pred)
rf_report = classification_report(y_test, rf_y_pred)

print(confusion_matrix(y_test, xgb_y_pred))

#Logistic Regression Score
lr_y_pred = lr_model.predict(X_test)
lr_score = accuracy_score(y_test, lr_y_pred)
lr_report = classification_report(y_test, lr_y_pred)

# #Ensemble Score
# ensemble_y_pred = ensemble_model.predict(X_test)
# ensemble_score = accuracy_score(y_test, ensemble_y_pred)
# ensemble_report = classification_report(y_test, ensemble_y_pred)

#Test Scores
model_scores = {xgb_model:xgb_score, 
                rf_model:rf_score, 
                lr_model:lr_score, 
                #ensemble_model:ensemble_score
                }
                

model_reports = [('Dummy', dummy_report), 
                ('XGB', xgb_report), 
                ('Random Forest', rf_report), 
                ('Log Reg', lr_report), 
                #('Ensemble', ensemble_report)
                ]

for name, report in model_reports:
    print(f'{name} Classification Report:\n'
          f'{report}')

############################################################################
"""Decision Function"""
# best_clf = max(model_scores, key=model_scores.get)

# #Probablity Function
# prob_df = X_test.copy()
# prob_df[['prob_under', 'prob_over']] = best_clf.predict_proba(X_test)
# upper_limit = prob_df['prob_over'].quantile(trade_prob_quantile)
# prob_df['decision'] = prob_df['prob_over'].apply(lambda x: 'Over' if x> upper_limit else ('Under' if x < (1-upper_limit) else 'no_bet'))
# trades = prob_df['decision'].value_counts()

# #Cumulative Returns
# cumulative_returns_df = prob_df.copy()
# cumulative_returns_df['decision'].replace(['Over', 'Under', 'no_bet'], [1,-1,0], inplace=True)
# cumulative_returns_df['close_offset'] = cumulative_returns_df['Close'].shift(-1)
# cumulative_returns_df.dropna(inplace=True)

# cumulative_returns_df = cumulative_returns_df.assign(PL = lambda x: (x['close_offset'] - x['Close']) * x['decision'])
# total_pl = cumulative_returns_df['PL'].sum()
# normalized_pl = round((total_pl/prob_df['Close'].mean()*100),0)

# #Print Statments
# print(f'Cumulative Returns: {round(total_pl,2)}')
# print(f'Normalized Return %: {normalized_pl}%')
# print(f'Over Threshold: {round(upper_limit,3)}')
# print(f'Under Threshold: {round((1-upper_limit),3)}')
# print(trades)

######################################################