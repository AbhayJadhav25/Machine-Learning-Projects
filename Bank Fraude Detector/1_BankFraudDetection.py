import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split ,KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report , ConfusionMatrixDisplay
import matplotlib.pyplot as plt 


def header(step , message):
    border = "="*70

    print(border)
    print(f"step : {step} |  {message}".center(50))
    print(border)


'''
Step 1 : Load The Dataset
'''
def LoadDataset(dataset):
    header(1,"Load the Datset")
    data = pd.read_csv(dataset)
    print("Dataset Loaded")
    return data
    

'''
Step 2 : EDA
'''
def EDA(data):
    header(2,"Exploratory Data Analysis")
    print("\nFirst 5 rows of Datset")
    print(data.head())

    print("\nShape Of Dataset")
    print(data.shape)

    print("\nColumn Name")
    print(data.columns.tolist())

    print("\nDataset Info")
    print(data.info())

    print("\nDupliacte Rows")
    print(data.duplicated().sum())

    print("Missing Values")
    print(data.isnull().sum())

    print("\nStatistical Summary")
    print(data.describe())
    

'''
Step 3 : label Encoding of features whose values are strings
'''

def K_FOld_Target_Encoding(data , column , target , n_splits = 5):
      
    encoded_col = np.zeros(len(data))
    kf = KFold(n_splits=n_splits , shuffle=True , random_state=42)

    for train_idx , val_idx in kf.split(data):

        train_fold = data.iloc[train_idx]
        val_fold = data.iloc[val_idx]

        means = train_fold.groupby(column)[target].mean()

        encoded_col[val_idx] = val_fold[column].map(means)

    return encoded_col

def labelEncoding(data):
    header(3,"Label Encoding using K-Fold target Encoding")  
    categorical_cols = ['payment_channel' , 'authentication_type']

    for col in categorical_cols:

        data[col+'_encoded'] = K_FOld_Target_Encoding(
            data , 
            column = col ,
            target = 'fraud_flag'
        )

    print("\nLabel Encoding Complete")
    return data

'''
Step - 4 : Split Datset into Train and test 
'''
def splitData(data):
    header(4,"Split Dataset into train and test Dataset")
    X = data.drop(['payment_channel' , 'authentication_type' , 'fraud_flag'] , axis = 1)
    Y = data['fraud_flag']

    x_train , x_test , y_train , y_test = train_test_split(
        X ,
        Y ,
        test_size=0.2,
        random_state=42
    )

    print("Splitting Complete")
    return x_train , x_test , y_train , y_test

'''
Step -5 : Train the Datset
'''
def trainData(x_train , y_train):
    header(5,"Train the Model")
    #create Base Model
    base_model = DecisionTreeClassifier(random_state=42)

    model = BaggingClassifier(
        estimator=base_model,
        n_estimators=55,
        random_state=42
    )

    model.fit(x_train , y_train)
    print("Training Completed")
    return model

'''
step 6 : Test the model and calculate Accuracy
'''
def testModel(model,x_test , y_test):
    header(6,"\Test the Model Performance")
    y_pred = model.predict(x_test)

    print("Testing Complete")
    return y_pred

'''
Step 7 : Calculate Accuracy Score , Display Classification Report , confusion Matrix
'''
def EvaluateModel(y_test , y_pred):
    header(7,"\Model Evaluation")
    accuracy = accuracy_score(y_test , y_pred)
    print(f"\nModel Accuracy = {accuracy*100:.2f}")

    print("\nClassification Report ")
    print(classification_report(y_test , y_pred))

    print("\nConfusion Matrix")
    cm= confusion_matrix(y_test , y_pred)
    print(cm)

'''
step 8 : Graph
'''

def Graph(data , y_test , y_pred):
    header(8,"Graphs")
    data['fraud_flag'].value_counts().plot(kind='bar')

    plt.show()

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()

def main():
    dataset = "banking_transactions.csv"
    #step 1
    data = LoadDataset(dataset)

    #step -2
    EDA(data)

    #step - 3
    data = labelEncoding(data)
    print("Encoded Dataset\n")
    print(data.head())

    #step-4 : Split Dataset into Train and test
    x_train , x_test , y_train , y_test = splitData(data)

    #step-5 : train the Data

    model = trainData(x_train , y_train)

    #step 6 : Test the model and calculate Accuracy
    y_pred = testModel(model , x_test , y_test)

    #step 7 : Calaculate Accuracy Score , Showing Classification Report , confusion matrix
    EvaluateModel(y_test , y_pred)

    #step 8 : Graph

    Graph(data , y_test , y_pred)

if __name__ == "__main__":
    main()