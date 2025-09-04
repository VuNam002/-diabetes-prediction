import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier # Mô hình RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load the dataset
data = pd.read_csv("diabetes.csv")
target = "Outcome"
# profile = ProfileReport(data, title="Diabetic Report", explorative=True)
# profile.to_file("diabetes_report.html")
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.25, random_state=42
)

#Tiền xử lý dữ liệu
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train) #Chú ý
x_test = scaler.transform(x_test)

#Lựa chọn mô hình
model = SVC()


#Huấn luyện mô hình
model.fit(x_train, y_train)

#Dự đoán
y_predict = model.predict(x_test)
# print(y_predict)


print("Accuracy:", accuracy_score(y_test, y_predict))
print("Precision:", precision_score(y_test, y_predict))
print("Recall:", recall_score(y_test, y_predict))
print("F1 Score:", f1_score(y_test, y_predict))