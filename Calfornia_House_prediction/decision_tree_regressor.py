import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error , r2_score
'''
Step No. : 1
function : load_data()
Description : Load and return the data
return : loaded Data
Date : 20 June , 2026
'''
def load_data():
    data = pd.read_csv('california_housing.csv')
    return data

'''
Step No. : 2
Functon : EDA
Description : Analyze the data. Show information of data , total missing values , shape of dataset and other things.
return : Nothing
Date : 20 June , 2026
'''
def EDA(data):
    print('Shape of Dataset')
    print(data.shape)

    print(f'Initial Values of Dataset \n {data.head()}')

    print('Columns of Dataset')
    print(data.columns.tolist())

    print('Dataset Info')
    print(data.info())

    print('Duplicate Rows')
    print(data.duplicated().sum())

    print('Missing values per column')
    print(data.isnull().sum())

    print('Statistical Summary')
    print(data.describe())

'''
Step No. : 3
Function :SplitData()
Parameters : data
Description : Dataset into Features and target.
              Seprate the Features data and target data into train and test dataset
return : x_train , x_test , y_train , y_test
Date : 20 june , 2026
'''
def SplitData(data):
    X = data.drop('target',axis = 1)
    y = data['target']

    print('Shape of X (Features)')
    print(X.shape)

    print('Shape of Y (target)')
    print(y.shape)

    x_train , x_test , y_train , y_test = train_test_split(
        X , 
        y ,
        test_size=0.10 ,
        random_state=42
    )

    return x_train , x_test , y_train , y_test

'''
Step No. : 4
Function : create_model()
parameters : Nothing
Description : Create a model using ensemable learning Bagging classifier and DecisionTreeRegressor 
return : created model
Date : 20 june , 2026
'''
def create_model():
    #Base Model
    base_model = DecisionTreeRegressor(random_state=42 ,min_samples_split=3 )

    #create Bagging Model
    model = BaggingRegressor(
        estimator=base_model , 
        n_estimators=50 , 
        random_state=42
    )
    print('Model Created successfully')
    return model

'''
Step No. : 5
function : train_model()
parameters : created_model,x_train , y_train
Description : train the created model using x_train and y_train dataset 
return : trained model
Date : 20 June , 2026
'''
def train_model(model , x_train , y_train):
    model.fit(x_train,y_train)
    print('Training Complete')
    return model

'''
Step No. : 6
function : predict_model()
parameters : trained_model , x_test 
Description : predict the target values using x_test dataset.
return : predicted_values i.e. y_pred
date : 20 june , 2026
'''
def predict_target(model , x_test):
    y_pred = model.predict(x_test)
    print('Prediction Complete')
    return y_pred

'''
Step NO. : 7
function : test_model()
parameters : predicted dataset i.e y_pred , y_test
Description :Evaluate the model. Test the trained model by comparing prediction values and y_test dataset. calculate mean_squared_error , r2_score.
return : nothing
Date : 20 June , 2026
'''
def test_model(y_test , y_pred):
    print(f'Mean Squared Error : {mean_squared_error(y_test , y_pred)}')
    print(f'R Square : {r2_score(y_test , y_pred)}')
def main():
    data = load_data()
    print('Data loaded Successfully')

    EDA(data)

    x_train , x_test , y_train , y_test= SplitData(data)

    model = create_model()

    trained_model = train_model(model,x_train , y_train)

    y_pred = predict_target(trained_model , x_test)

    test_model(y_pred , y_test)


if __name__ == "__main__":
    main()