import numpy as np
import pandas as pd

# visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

def predict(income, age, room, bedroom, population):
    df = pd.read_csv("MCSReal_Estate.csv")
    
    def remove(row):
        if "$" in row:
            return float(row.replace("$", " "))
        else:
            return (float(row.replace("Rs", " ")))/75
    
    df["Price"] = df["Price"].apply(remove)
    
    def State(add):
        add_list = add.split(",")[-1]
        state = add_list.split()
        return state[-2]
    
    df["State"] = df["Address"].apply(State)
    
    # Replacing the "?" and "missing" from the data
    df["Avg. Area House Age"].replace("missing", np.nan, inplace=True)
    df["Avg. Area Number of Rooms"].replace("?", np.nan, inplace=True)
    
    # Changing the datatype 
    df["Avg. Area House Age"] = df["Avg. Area House Age"].astype("float64")
    df["Avg. Area Number of Rooms"] = df["Avg. Area Number of Rooms"].astype("float64")
    
    df.drop(["Avg Area Comfort", "ids"], axis=1, inplace=True)
    
    from sklearn.impute import SimpleImputer
    si = SimpleImputer(missing_values=np.nan, strategy='mean')
    df[["Avg. Area Number of Rooms", "Avg. Area Number of Bedrooms", "Avg. Area House Age"]] = si.fit_transform(df[["Avg. Area Number of Rooms", "Avg. Area Number of Bedrooms", "Avg. Area House Age"]])
    
    x = df.iloc[:,:5]
    y = df.iloc[:,-3]
    
    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30, random_state=1)
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    xtrain = sc.fit_transform(xtrain)
    xtest = sc.fit_transform(xtest)

    from sklearn.svm import SVR
    svm = SVR(kernel="linear", C=450, gamma='auto')
    svm.fit(xtrain, ytrain)
    ypred = svm.predict(xtest)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(ytest, ypred)
    mse = mean_squared_error(ytest, ypred)
    rmse = np.sqrt(mse)
    r2 = r2_score(ytest, ypred)
    
    new_obj = [income, age, room, bedroom, population]
    
    ypred = svm.predict([new_obj])

    return float(ypred)