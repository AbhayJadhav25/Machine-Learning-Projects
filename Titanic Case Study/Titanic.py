import pandas as pd
import numpy as np
import joblib #using these we can save the model on harddisk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , confusion_matrix
#---------------------------------------------------
# Function name : LoadPreservedModel
# Description :   It is use to load preserved model.
# Parameters : model , filename
# return : None
# Data : 14/03/2026
# Author : Abhay Santosh Jadhav
#-----------------------------------------------------
def LoadPreservedModel(filename):

  loaded_model = joblib.load(filename)

  print("Model successfully loaded")
  return loaded_model
#---------------------------------------------------
# Function name : PreserveModel
# Description :   It is use to preserve model on secondary.
# Parameters : model , filename
# return : None
# Data : 14/03/2026
# Author : Abhay Santosh Jadhav
#-----------------------------------------------------
def PreserveModel(model , filename):
  joblib.dump(model , filename)

  print("Model preserved Sucesfully with name : ",filename)
  
#---------------------------------------------------
# Function name : TrainTitanicModel
# Description :   It does split X , Y , Training data , testing data .
# Parameters : df
# return : None
# Data : 14/03/2026
# Author : Abhay Santosh Jadhav
#-----------------------------------------------------
def TrainTitanicModel(df):
  #splite features and labels
  X = df.drop(["Survived"] , axis = 1)
  Y = df["Survived"]

  print("\n Features : ")
  print(X.head())

  print("\n Labels : ")
  print(Y.head())

  print("Shape of X : ",X.shape)
  print("Shape of Y : ",Y.shape)

  X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
  print("X_train shape = ",X_train.shape) 
  print("X_test shape = ",X_test.shape) 
  print("Y_train shape = ",Y_train.shape) 
  print("Y_test shape = ",Y_test.shape) 

  model = LogisticRegression(max_iter=1000)

  model.fit(X_train , Y_train)
  print("Model Trained Successfully")

  print("\n Intercept of model : ")
  print(model.intercept_)

  print("\nCoeeficiant of model")
  for feature , coeficent in zip(X.columns , model.coef_[0]):
    print(feature , " : " , coeficent)

  PreserveModel(model , "marvelloustitanic.pkl")

  loaded_model = LoadPreservedModel("marvelloustitanic.pkl")

  Y_pred = loaded_model.predict(X_test)

  accuracy = accuracy_score(Y_pred , Y_test)

  print("Accuracy is : ",accuracy)

  cm = confusion_matrix(Y_pred , Y_test)
  print("Confusion Matrix is : ")
  print(cm)
#---------------------------------------------------
# Function name : DisplayInfo
# Description :   It displays the formatted title.
# Parameters : title(str)
# return : None
# Data : 14/03/2026
# Author : Abhay Santosh Jadhav
#-----------------------------------------------------
def DisplayInfo(title):
  print("\n"+ "="*70)
  print(title)
  print("="*70)

#---------------------------------------------------
# Function name : showData
# Description :   It shows basic information about dataset.          
# Parameters : 1)Dataset(df)
#               df --> Pandas dataframe object
#              2)message
#                message --> Heading text to display.
# return : None
# Data : 14/03/2026
# Author : Abhay Santosh Jadhav
#-----------------------------------------------------
def showData(df , message):
  DisplayInfo(message)
  
  print("\n First 5 rows of Dataset")
  print(df.head())

  print("Shape of dataset")
  print(df.shape)

  print("\nColumn names : ")
  print(df.columns.tolist())

  print("\n Missing values in each column")
  print(df.isnull().sum())
#---------------------------------------------------
# Function name : CleanTitanicData
# Description : It does preprocessing
#               It removes unnecessary columns
#               It handales missing values
#               It converts text data to numeric format
#               It does encoding to categorical columns
# Parameters : df
#              df -> Pandas dataframe
# return : df  -> Cleaned Pandas dataframe.
# Data : 14/03/2026
# Author : Abhay Santosh Jadhav
#-----------------------------------------------------
def CleanTitanicData(df):
  DisplayInfo("Step 2 : Original Data")
  print(df.head())

  #remove unnecessary columns
  drop_columns = ["Passengerid" , "zero" , "Name" , "Cabin"]
  existing_columns = [col for col in drop_columns if col in df.columns]
  
  print("\n Columns to be dropped : ")
  print(existing_columns)

  #drop the unwanted column
  df = df.drop(columns = existing_columns)
  DisplayInfo("Step 2 : Data after column removal")
  print(df.head())

  #Handle age column
  if "Age" in df.columns:
    print("Age column before filing missing values")
    print(df["Age"].head(10))

    #coerce -> Invalid value gets converted as NaN
    df["Age"] = pd.to_numeric(df["Age"] , errors = "coerce")
    age_median = df["Age"].median()

    #Replace missing value with median.
    df["Age"] = df["Age"].fillna(age_median)

    print("Age column after preprocessing : ")
    print(df["Age"].head(10))
  
  if "Fare" in df.columns:
    print("\n Fare column before preprocessing")
    print(df["Fare"].head(10))

    df["Fare"] = pd.to_numeric(df["Fare"] , errors = "coerce")

    fare_mdeian = df["Fare"].median()

    print("Median of fare column is : ",fare_mdeian)
    #Replace missing value with median.
    df["Fare"] = df["Fare"].fillna(fare_mdeian)

    print("Fare column after preprocessing : ")
    print(df["Fare"].head(10))

  #Handle Embarked column
  if "Embarked" in df.columns:
    print("\n Embarked column before preprocessing")
    print(df["Embarked"].head(10))

    #Convert the data into string
    df["Embarked"] = df["Embarked"].astype(str).str.strip()

    #Remove the missing values
    df["Embarked"] = df["Embarked"].replace(['nan' , 'None' , ''] , np.nan)

    #Get most frequent value
    embarked_mode = df["Embarked"].mode()[0]
    print("\n Mode of embarked column : ",embarked_mode)

    #Replace missing value with mode.
    df["Embarked"] = df["Embarked"].fillna(embarked_mode)

    print("Embarked column after preprocessing : ")
    print(df["Embarked"].head(10))

    #Encode Embraked column
    df = pd.get_dummies(df , columns = ["Embarked"] , drop_first=True)  #used to one hot coding.
    print("\n Embarked Data before encoding")

    print(df.head())

    print("Shape of dataset = ",df.shape)

    #convert boolean columns into integer
    for col in df.columns:
      if df[col].dtype == bool:
        df[col] = df[col].astype(int)

    print("\n Embarked Data after encoding")
    print(df.head())

  #handle sex column
  if "Sex" in df.columns:
    print("\n Sex column before preprocessing")
    print(df["Sex"].head(10))

    df["Sex"] = pd.to_numeric(df["Sex"] , errors = "coerce")

    print("\n Sex column after preprocessing")
    print(df["Sex"].head(10))

    DisplayInfo("Data After preprocessing")
    print(df.head())

    print("\n Missing values after preprocessing")
    print(df.isnull().sum())

  return df

# ---------------------------------------------------
# Function name : MarvellousTitanicLogistic
# Description : This is main pipeline controller
#               It loads the dataset , shows raw data
#               It process the dataset & train the model                
# Parameters : Data path of dataset file
# return : None
# Data : 14/03/2026
# Author : Abhay Santosh Jadhav
#-----------------------------------------------------

def MarvellousTitanicLogistic(DataPath):
  DisplayInfo("Step 1 : Load the Dataset")
  df = pd.read_csv(DataPath)
  
  showData(df , "Initial Dataset")

  df = CleanTitanicData(df)
  TrainTitanicModel(df)
#---------------------------------------------------
# Function name : main
# Description : Starting point of the application
# Parameters : none
# return : None
# Data : 14/03/2026
# Author : Abhay Santosh Jadhav
#-----------------------------------------------------

def main():
  MarvellousTitanicLogistic("MarvellousTitanicDataset.csv")

if __name__ == "__main__":
  main()
