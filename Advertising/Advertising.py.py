import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score

def Advertise(path):
  Border = "_"*40
  #-----------------------------------------------------------------#
  # step 1: Load Dataset
  #-----------------------------------------------------------------#
  print(Border)
  print("Step 1 : Load Dataset")
  df = pd.read_csv(path)
  print(Border)

  print("Few records from the dataset.")
  print(df.head()
        )
  #-----------------------------------------------------------------#
  # step 2: Remove Unwanted Columns 
  #-----------------------------------------------------------------#

  print(Border)
  print("Step 2 : Remove Unwanted Columns . ")
  print(Border)

  print("Shape of Dataset Before Removal : ",df.shape)
  if 'Unnamed: 0' in df.columns :
    df.drop(columns = ['Unnamed: 0']  , inplace=True)
  
  print("Shape of Dataset after Removal : ",df.shape)

  print(Border)
  print("Clean Dataset is : ")
  print(Border)
  print(df.head())

  #-----------------------------------------------------------------#
  # step 3: Check Missing Values
  #-----------------------------------------------------------------#
  print(Border)
  print("Step 3 : Check Missing Values . ")
  print(Border)

  print("Missing Values count : \n",df.isnull().sum())

  #-----------------------------------------------------------------#
  # step Correlation Between Columns
  #-----------------------------------------------------------------#
  print(Border)
  print("Step 4 : Display Statistical Summary. ")
  print(Border)

  print(df.describe())

  #-----------------------------------------------------------------#
  # step 5 : Correlation Between Columns
  #-----------------------------------------------------------------#
  print(Border)
  print("step 5 : Correlation Between Columns ")
  print(Border)

  print("Correlation Matrix : ")
  print(df.corr())

  #-----------------------------------------------------------------#
  # step 6 : Split Dataset into dependent and Independent Variable.  #-----------------------------------------------------------------#

  print(Border)
  print("step  : Split Dataset into dependent and Independent Variable.")
  print(Border)

  X = df[['TV' , 'radio' , 'newspaper']]
  Y = df['sales']

  print("Shape of Independent Variable : ",X.shape)
  print("Shape of Independent Variable : ",Y.shape)

  #-----------------------------------------------------------------#
  # step 7 : Split Dataset training and testing.  
  #-----------------------------------------------------------------#

  print(Border)
  print("step  7 : Split Dataset training and testing.")
  print(Border)

  X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size = 0.2 , random_state=42)

  print("X_train shape : ",X_train.shape)
  print("X_test shape : ",X_test.shape)
  print("Y_train shape : ",Y_train.shape)
  print("Y_test shape : ",Y_test.shape)


  #-----------------------------------------------------------------#
  # step 8 : Create and train the model.  
  #-----------------------------------------------------------------#

  print(Border)
  print("step  8 : Create and train the model.")
  print(Border)

  model = LinearRegression()
  model.fit(X_train , Y_train)
  

  #-----------------------------------------------------------------#
  # step 9 : Test the model.  
  #-----------------------------------------------------------------#

  print(Border)
  print("step  9 : Test the model.")
  print(Border)

  Y_pred = model.predict(X_test)

  #-----------------------------------------------------------------#
  # step 10 : Evaluate the model.  
  #-----------------------------------------------------------------#

  print(Border)
  print("step  10 : Evaluate the model.")
  print(Border)

  MSE = mean_squared_error(Y_pred , Y_test)
  RMSE = np.sqrt(MSE)
  R2 = r2_score(Y_pred , Y_pred)


  print("Mean Square Error : ",MSE)
  print(" Root Mean Square Error : ",RMSE)
  print("R square value : ",R2)

  #-----------------------------------------------------------------#
  # step 11 : Calculate Model Coefficient.  
  #-----------------------------------------------------------------#

  print(Border)
  print("step  11 : Calculate Model Coefficient.")
  print(Border)

  for column , value in zip(X.columns , model.coef_):
    print(f"{column} : {value}")
  
  print("Intercept : ",model.intercept_)

  #-----------------------------------------------------------------#
  # step 12 : Compare the actual and predicted Values.  
  #-----------------------------------------------------------------#

  print(Border)
  print("step  12 : Compare the actual and predicted Values.")
  print(Border)

  Result = pd.DataFrame({
    'Actual sale' : Y_test.values , 
    'Predicted Sale'  : Y_pred
    })
  
  print(Result.head())

  #-----------------------------------------------------------------#
  # step 13 : Plot actual vs Predicted.  
  #-----------------------------------------------------------------#

  print(Border)
  print("step  13 : Plot actual vs Predicted.")
  print(Border)

  plt.figure(figsize=(8,5))
  plt.scatter(Y_test , Y_pred)

  plt.xlabel("Actual Sales")
  plt.ylabel("Predicted Sales")
  plt.title("Actual Sales vs Predicted Sales.")
  plt.grid(True)
  plt.legend()
  plt.show()
  
def main():
  Advertise("Advertising.csv")
if __name__ == "__main__":
  main()