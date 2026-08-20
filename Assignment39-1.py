import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# "load the dataset")
Data="student_performance_ml.csv"
df=pd.read_csv(Data)

X=df[["StudyHours","Attendance","PreviousScore","AssignmentsCompleted","SleepHours"]]
Y=df["FinalResult"]
print(X.shape)
print(Y.shape)

# split the dataset
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
print("x_train ",X_train.shape)  #24,5
print("x_test ",X_test.shape)   #6,5

print("y_train ",Y_train.shape)  #24,
print("y_test ",Y_test.shape)    #6,

#build the model
model=DecisionTreeClassifier(max_depth=5)

#train the model
model.fit(X_train,Y_train)
print("model trained successfully ")