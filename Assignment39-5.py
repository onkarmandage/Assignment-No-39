import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

Border="-"*30

# 1. Dataset Loading

Data="student_performance_ml.csv"
df=pd.read_csv(Data)
print("Dataset loaded successfully")
print("\n first  5 record :")
print(df.head())
print(Border)

#2 Data Analysis

print("Dataset shape :")
print(df.shape)

print("Column Names :")
print(df.columns.tolist())

print("Final result Destribution :")
print(df["FinalResult"].value_counts())

print("Average study Hours :")
print(df["StudyHours"].mean())

print("Average Attendance :")
print(df["Attendance"].mean())
print(Border)

# 3. Visualization
#histogram of student
 
plt.hist(
    df["StudyHours"],
    bins=10,
    edgecolor="black"
)
plt.xlabel("StudyHours")
plt.ylabel("Number of student")
plt.title("Study Hours Distribution")
plt.show()
print(Border)

# 4. independent and dependent

X=df[["StudyHours","Attendance","PreviousScore","AssignmentsCompleted","SleepHours"]]
Y=df["FinalResult"]
print(Border)

# 5. train and test 

X_train,X_text,Y_train,Y_text=train_test_split(X,Y,test_size=0.2,random_state=42)
print(Border)

# 6. Model Training

model=DecisionTreeClassifier(max_depth=3,random_state=42)
model.fit(X_train,Y_train)
print("Model Training completed")
print(Border)

# 7. Model prediction

Y_pred=model.predict(X_text)

print("Actual value")
print(Y_text.values)

print("predicted values")
print(Y_pred)
print(Border)

# 8. Accuracy Calculation

Accuracy=accuracy_score(Y_pred,Y_text)
print("model Accuracy ",Accuracy*100)