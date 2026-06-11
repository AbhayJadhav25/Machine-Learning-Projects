from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel , Field , computed_field
from typing import Literal , Annotated
import pickle
import pandas as pd

#import the model 
with open("model_path.pkl" , "rb") as f:
    model = pickle.load(f)

app = FastAPI()

#pydantic model to validate incoming data
class UserInput(BaseModel):
    age : Annotated[int , Field(... , gt = 0 , lt =120 , description='age of the user')]
    weight :Annotated[float , Field(... , gt = 0  , description='weight of the user')]
    height : Annotated[float , Field(... , gt = 0 , lt =2.5 , description='Height of the user')]
    income_lpa : Annotated[float , Field(... , gt = 0 , description='Annual income of the user')]
    smoker : Annotated[bool , Field(... , description='is user a smoker?')]
    city : Annotated[str ,Field(description='The city user belongs to') ]
    occuption :Annotated[Literal[ 'retired',     'freelancer',  'student', 'government_job','business_owner',     'unemployed',    'private_job'] ,Field(description='Occuption of the user') ]

    @computed_field
    @property
    def bmi(self)->float:
        return round(self.weight / (self.height**2))
    
    @computed_field
    @property
    def lifestyle_risk(self)->str:
        if self.smoker == 'True' and self.bmi > 30:
            return 'high'
        elif self.smoker == 'True' and self.bmi > 27:
            return 'mdeium'
        else :
         return 'low'


    @computed_field
    @property
    def age_group(self)->str:
        if self.age < 25:
            return 'young'
        elif self.age < 60 :
            return 'middle_aged'
        else :
            return 'senior'
        
    @computed_field
    @property
    def city_tier(self) -> int:
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
        if self.city in tier_1_cities:
            return 1
        elif self.city in tier_2_cities:
            return 2
        else:
            return 3
    
@app.post('/predict')
def predict_premium(data : UserInput):

   data =  pd.DataFrame([{
        'bmi' : data.bmi ,
        'age_group' : data.age_group,
        'lifestyle_risk' : data.lifestyle_risk , 
        'city_tier' : data.city_tier , 
        'income_lpa' : data.income_lpa ,
        'occupation' : data.occuption

    }])

   prediction = model.predict(data)[0]

   return JSONResponse(
       status_code=200 , 
       content={
           'predicted_category' : prediction
       }
   )