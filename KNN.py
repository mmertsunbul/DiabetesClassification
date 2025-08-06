import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Sınıflandırma Modeli
from sklearn.neighbors import KNeighborsClassifier

#Veri setini yüklenmesi ve incelemesi
df = pd.read_csv('diabetes.csv')
print("Veri Setinin Boyutu:", df.shape)
df.head()

#Veri seti temizleme işlemi 0 değerleri Nan ile değiştirme ve medyan ile doldurma
cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
imputer = SimpleImputer(strategy='median')
df[cols_with_zeros] = imputer.fit_transform(df[cols_with_zeros])

#Bağımlı bağımsız değişken ayırma 
X = df.drop(["Outcome"], axis = 1) #bağımsız değişken 
y = df["Outcome"] #bağımlı değişken

#Özellik ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Eğitim ve test setini ayırma
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42, stratify=y)

#SMOTE tekniğinin uygulanması
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

#Dengelenmiş eğitim verisinin boyutları
print("Önce:", X_train.shape, y_train.sum(), "pozitif örnek")
print("Sonra:", X_train_res.shape, y_train_res.sum(), "pozitif örnek")

knn_degerleri = {
    'n_neighbors': range(30, 51),
    'weights':    ['uniform', 'distance'],
    'p':          [1, 2]}
knn_cv_model = GridSearchCV(KNeighborsClassifier(), knn_degerleri, cv=5, scoring='recall')
knn_cv_model.fit(X_train_res, y_train_res)

eniyi_k_degeri=knn_cv_model.best_params_
print(eniyi_k_degeri)

model = KNeighborsClassifier(n_neighbors=30, p=2, weights='distance')
model_knn = model.fit(X_train_res,y_train_res)
model_knn.score(X_test,y_test)

#Modelin test setiyle karşılaştırılması
#gerçek sonuçlar(X_test) ile modelin tahmin ettiği sonuçlar (y_predict) karşılaştırılır 
#bu sayede modelin nerelerde doğru, nerelerde yanlış tahmin yaptığını görülür
y_pred = model_knn.predict(X_test)
tahmin_degeri = pd.DataFrame({"y_test" : y_test,
              "tahmin edilen sonuc" : y_pred})


pd.set_option('display.max_rows', None)
tahmin_degeri


from sklearn.model_selection import cross_val_score
#(SMOTE’u sadece eğitim seti için kullanıp, burada tüm veriyle gerçekçi bir değerlendirme yapılır)
scores = cross_val_score(
    model_knn,           # GridSearchCV’den gelen en iyi model
    X_scaled,           # Ölçeklenmiş tüm veri seti (X_imputed ve scaler sonrasında)
    y,                  # Orijinal etiketler
    cv=10, 
    scoring='f1'
)
print("10-Fold CV F1 ortalaması: %0.4f" % scores.mean())


#Model başarı metrikleri import etme
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#Accurary Score
print("Accurary:",accuracy_score(y_test,y_pred))
#Precision Score
print("Precision:",precision_score(y_test,y_pred))
#Recall Score
print("Recall:",recall_score(y_test,y_pred))
#F1 Score
print("F1:",f1_score(y_test,y_pred))


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Confusion matrix'in hesaplanması
cm = confusion_matrix(y_test, y_pred)
# cm = [[TN, FP], [FN, TP]]

#Karmaşıklık matrisi hücre etiketlerinin eklenmesi
labels = np.array([["TN", "FP"], ["FN", "TP"]])
annot = np.array([[f"{labels[i, j]}\n{cm[i, j]}" for j in range(2)] for i in range(2)])

#görselleştirme
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
            xticklabels=['Negatif', 'Pozitif'],
            yticklabels=['Negatif', 'Pozitif'], cbar=False)
plt.xlabel('Tahmin')
plt.ylabel('Gerçek')
plt.title('KNN Confusion Matrix')
plt.show()
