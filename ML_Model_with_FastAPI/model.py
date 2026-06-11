import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report , accuracy_score
import pickle
import numpy as np


data = pd.read_csv("insurance.csv")

df_feat= data.copy()

df_feat['bmi'] = df_feat['weight'] / (df_feat['height']**2)

def age_group(age):
    if age < 25:
        return 'young'
    elif age < 60 :
        return 'middle_aged'
    else :
        return 'senior'
    
df_feat['age_group'] = df_feat['age'].apply(age_group)

#feature : Life style risk
def lifestyle_risk(row):
    if row['smoker'] == 'True' and row['bmi'] > 30:
        return 'high'
    elif row['smoker'] == 'True' and row['bmi'] > 27:
        return 'mdeium'
    else :
        return 'low'
    
df_feat['lifestyle_risk'] = df_feat.apply(lifestyle_risk , axis = 1)
# df_feat['lifestyle_risk'] = df_feat.apply(lifestyle_risk, axis=1)


tier_1_cities = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]

tier_2_cities = ["Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi",
    "Visakhapatnam", "Coimbatore", "Bhopal", "Nagpur", "Vadodara", "Surat",
    "Rajkot", "Jodhpur", "Raipur", "Amritsar", "Varanasi", "Agra",
    "Dehradun", "Mysore", "Jabalpur", "Guwahati", "Thiruvananthapuram",
    "Ludhiana", "Nashik", "Allahabad", "Udaipur", "Aurangabad", "Hubli",
    "Belgaum", "Salem", "Vijayawada", "Tiruchirappalli", "Bhavnagar",
    "Gwalior", "Dhanbad", "Bareilly", "Aligarh", "Gaya", "Kozhikode",
    "Warangal", "Kolhapur", "Bilaspur", "Jalandhar", "Noida", "Guntur",
    "Asansol", "Siliguri"]

#feature 4 : city tier
def city_tier(city):
    if city in tier_1_cities:
        return 1
    elif city in tier_2_cities:
        return 2
    else:
        return 3
    
df_feat['city_tier'] = df_feat['city'].apply(city_tier)

df_feat.drop(columns=['age' , 'weight' , 'height' , 'smoker' , 'city'] , inplace=True)


#define categorical and numeric features
categorical_features = ['age_group' , 'lifestyle_risk' , 'occupation' , 'city_tier']
numeric_features = ['bmi' , 'income_lpa']

preprocessor = ColumnTransformer(
    transformers=[
        ("cat" , OneHotEncoder() , categorical_features) , 
        ("num" , "passthrough" , numeric_features)
    ]
)

#create pipeline with preprocessing and random forest classifier
pipeline = Pipeline(
    steps=[
        ("preprocessor" , preprocessor) , 
        ("classifier" , RandomForestClassifier(random_state=42))
    ]
)

features = df_feat.drop('insurance_premium_category' , axis = 1)
target = df_feat['insurance_premium_category']

x_train , x_test , y_train , y_test = train_test_split(features , target , test_size=0.2 , random_state=42)
pipeline.fit(x_train , y_train)

y_pred = pipeline.predict(x_test)

print(accuracy_score(y_test , y_pred))
pickle_model_path = 'model.pkl'
with open(pickle_model_path , 'wb') as f:
    pickle.dump(pipeline , f)
