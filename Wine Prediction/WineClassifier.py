import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report
from sklearn.preprocessing import StandardScaler

def MarvellousClassifier(DataPath):
  border = "_"*40
  #--------------------------------------------------------------#
  # Step 1 : Load The Dataset
  #--------------------------------------------------------------#

  print(border)
  print("Step 1 : Load the dataset")
  print(border)
  df = pd.read_csv(DataPath)
  print(border)
  print("Some entries from Dataset ")
  print(df.head())
  print(border)

  #--------------------------------------------------------------#
  # Step 2 : Clean the dataset by removing empty rows 
  #--------------------------------------------------------------#

  print(border)
  print("Step 2 : Clean the dataset by removing empty rows")
  print(border)

  df.dropna(inplace = True)
  print("Total Records : ",df.shape[0])
  print("Total Columns : ",df.shape[1])
  print(border)

  #--------------------------------------------------------------#
  # Step 3 : Seprate independent & Dependent variables 
  #--------------------------------------------------------------#

  print(border)
  print("Step 3 : Seprate independent & Dependent variables")
  print(border)

  X = df.drop(columns=['Class'])
  Y = df['Class']

  print("Shape of X : " , X.shape)
  print("Shape of Y : ",Y.shape)
  
  print(border)

  print("Input Columns : ",X.columns.to_list())
  print("Output column : Class")
  #--------------------------------------------------------------#
  # Step 4 : Split the Dataset for training and testing 
  #--------------------------------------------------------------#

  print(border)
  print("Step 4 : Split the Dataset for training and testing")
  print(border)
  
  X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.2 , random_state=42 , stratify=Y) #startify -->if it yes(Y)  

  print("Information of Training and testing Data")
  print(border)
  print("X_train shape : ",X_train.shape)
  print("X_test shape : ",X_test.shape)
  print("Y_train shape : ",Y_train.shape)
  print("Y_test shape : ",Y_test.shape)
  print(border)

  #--------------------------------------------------------------#
  # Step 5 :Feature Scaling 
  #--------------------------------------------------------------#

  print(border)
  print("Step 5 :Feature Scaling")
  print(border)

  scalar = StandardScaler()
  # Independent Variable Scaling.
  X_train_scaled = scalar.fit_transform(X_train)
  X_test_scaled  = scalar.fit_transform(X_test)

  print("Featur Scaling is Done.")

  #--------------------------------------------------------------#
  # Step 6 : Explore the multiple values of K
  # Hyperparameter tuning (K) 
  #--------------------------------------------------------------#

  print(border)
  print("Step 6 : Explore the multiple values of K")
  print(border)

  accuracy_scores = []
  K_values = range(1 , 21)

  for k in K_values : 
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled ,Y_train)
    Y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(Y_pred , Y_test)
    accuracy_scores.append(accuracy)
  
  print(border)
  print("Accuracy report of all K values from 1 to 20")
  for i in range(len(accuracy_scores)):
    print(i+1,"->",accuracy_scores[i])

  print(border)
  
  #--------------------------------------------------------------#
  # Step 7 : Plot Graph of K vs Accuracy.
  #--------------------------------------------------------------#

  print(border)
  print("Step 7 : Plot Graph of K vs Accuracy.")
  print(border)

  plt.figure(figsize = (8,5))
  plt.plot(K_values , accuracy_scores , marker = 'o')
  plt.title("K values vs Accuracy")
  plt.xlabel("Value of K")
  plt.ylabel("Accuracy")
  plt.grid(True)
  plt.xticks(list(K_values))
  plt.show()

  #--------------------------------------------------------------#
  # Step 8 : find best value of K.
  #--------------------------------------------------------------#

  print(border)
  print("8 : find best value of K.")
  print(border)

  best_k = list(K_values)[accuracy_scores.index(max(accuracy_scores))]
  print("Best value of K is : ",best_k)

  #--------------------------------------------------------------#
  # Step 9 : Build final model using best value of K.
  #--------------------------------------------------------------#

  print(border)
  print("9 : Build final model using best value of K.")
  print(border)

  final_model = KNeighborsClassifier(n_neighbors=best_k)
  final_model.fit(X_train_scaled , Y_train)
  Y_pred = final_model.predict(X_test_scaled)

  #--------------------------------------------------------------#
  # Step 10 : Cakculate final accuracy.
  #--------------------------------------------------------------#

  print(border)
  print("10 : Cakculate final accuracy.")
  print(border)

  accuracy = accuracy_score(Y_test , Y_pred)
  print("Accuracy of Model is : ",accuracy)
  print(border)

  #--------------------------------------------------------------#
  # Step 11 : Display Confusion Matrix
  #--------------------------------------------------------------#

  print(border)
  print("11 : Display Confusion Matrix")
  print(border)

  cm = confusion_matrix(Y_test , Y_pred)
  print(cm)

  #--------------------------------------------------------------#
  # Step 12 : Display Classification Report
  #--------------------------------------------------------------#

  print(border)
  print("12 : Display Classification Report")
  print(border)

  print(classification_report(Y_test , Y_pred))
def main():
  border = "_"*40
  print(border)
  print("Wine Classifier Using KNN")
  print(border)
  
  MarvellousClassifier("WinePredictor.csv")

if __name__ == "__main__":
  main()