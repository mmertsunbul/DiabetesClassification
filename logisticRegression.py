import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Veri seti yüklenmesi
df = pd.read_csv("diabetes.csv")

#0 olan değerleri NaN ile değiştirme
cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

#NaN değerleri medyan ile doldurma
imputer = SimpleImputer(strategy='median')
df[cols_with_zeros] = imputer.fit_transform(df[cols_with_zeros])

#Bağımlı bağımsız değişken ayırma 
X = df.drop("Outcome", axis=1) #bağımsız değişken 
y = df["Outcome"] #bağımlı değişken

#Özellik ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Eğitim ve test verisini ayırma
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42, stratify=y)

#SMOTE tekniğinin uygulanması 
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
print("Önce:", X_train.shape, y_train.sum(), "pozitif örnek")
print("Sonra:", X_train_res.shape, y_train_res.sum(), "pozitif örnek")

#Logistic regression hiperparametre arama
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}


logreg = LogisticRegression(random_state=42)
grid_lr = GridSearchCV(logreg, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_lr.fit(X_train_res, y_train_res)


best_lr = grid_lr.best_estimator_
print("En iyi parametreler:", grid_lr.best_params_)

#Tahmin yapılması
y_pred_lr = best_lr.predict(X_test)

#Model başarı metrikleri
print("Logistic Regression")
print("Accuracy :", accuracy_score(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall   :", recall_score(y_test, y_pred_lr))
print("F1 Score :", f1_score(y_test, y_pred_lr))


#Karmaşıklık matrisi hesaplama
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

cm = confusion_matrix(y_test, y_pred_lr)
# cm = [[TN, FP], [FN, TP]]

#Karmaşıklık matrisi hücre etiketlerinin eklenmesi
labels = np.array([["TN", "FP"], ["FN", "TP"]])
annot = np.array([[f"{labels[i, j]}\n{cm[i, j]}" for j in range(2)] for i in range(2)])

#görselleştirme
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=annot, fmt='', cmap='Purples',
            xticklabels=['Negatif', 'Pozitif'],
            yticklabels=['Negatif', 'Pozitif'], cbar=False)
plt.xlabel('Tahmin')
plt.ylabel('Gerçek')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

