import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import mean_squared_error , r2_score 
def main():
  border = "_"*40
  #Step 1 : Load the Data
  print(border)
  print("Step 1 : Load The dataset")
  print(border)
  
  df = pd.read_csv("MarvellousHeadBrain.csv")
  print("Data loaded")
  print("Initial 5 value of CSV")
  print(df.head())

  #Step 2 : Check Missing Values
  print(border)
  print("Step 2 : Check Missing Values")

  print("Misiing Values \n",df.isnull().sum())
  print(border)

  #Step 3 : Describe Statistical Summary

  print(border)
  print("Step 3 : Describe Statistical Summary\n")
  print(border)
  print(df.describe())
  print(border)

  #Step 4 : Correlation Between Columns

  print(border)
  print("Step 4 : Correlation Between Columns\n")
  print(border)
  print("Coorelation Matrix")
  print(df.corr())
  print(border)

  #Step 5 : split Dataset into dependent and Independent Variable

  print(border)
  print("Step 5 : split Dataset into dependent and Independent Variable\n")
  print(border)
  X = df[['Age_Range','Head_Size(cm^3)','Brain_Weight(grams)']]
  y = df['Gender']

  print("Shape of Independent Variable : ",X.shape)
  print("Shape of dependent Variable : ",y.shape)
  print(border)

  #Step 6 : Split Dataset training and testing. 

  print(border)
  print("Step 6 : Split Dataset training and testing. \n")
  print(border)
  X_train , X_test , Y_train , Y_test = train_test_split(X , y , test_size = 0.2 , random_state = 42)

  print("X_train shape : ",X_train.shape)
  print("X_test shape : ",X_test.shape)
  print("Y_train shape : ",Y_train.shape)
  print("Y_test shape : ",Y_test.shape)

  print(border)

  #Step 7 : Create and train the model 

  print(border)
  print("Step 7 : Create and train the model \n")
  print(border)
  model = LinearRegression()
  model.fit(X_train , Y_train)
  print(border)

  #Step 8 : Test the model 

  print(border)
  print("Step 8 : Test the model \n")
  print(border)

  Y_pred = model.predict(X_test)

  #-----------------------------------------------------------------#
  # step 9: Evaluate the model.  
  #-----------------------------------------------------------------#

  print(border)
  print("step  9: Evaluate the model.")
  print(border)

  MSE = mean_squared_error(Y_pred , Y_test)
  RMSE = np.sqrt(MSE)
  R2 = r2_score(Y_pred , Y_pred)


  print("Mean Square Error : ",MSE)
  print(" Root Mean Square Error : ",RMSE)
  print("R square value : ",R2)
  print(border)
if __name__ == "__main__":
  main()