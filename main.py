import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("detection_dataset.csv")

print("VERİ SETİNİN İLK 5 SATIRI")
print(df.head())

print("\nVERİ SETİNİN BOYUTU")
print(df.shape)

print("\nSÜTUN İSİMLERİ")
print(df.columns)

print("\nVERİ SETİ BİLGİLERİ")
df.info()

print("\nMAKİNE ARIZASI DAĞILIMI")
failure_counts = df["Machine failure"].value_counts()
print(failure_counts)


failure_percentages = df["Machine failure"].value_counts(normalize=True) * 100
print("\nMAKİNE ARIZASI YÜZDELERİ")
print(failure_percentages)

print("\nMAKİNE TİPLERİNİN DAĞILIMI")
type_counts = df["Type"].value_counts()
print(type_counts)

print("\nMAKİNE TİPİNE GÖRE ARIZA SAYILARI")
type_failure_counts = pd.crosstab( df["Type"], df["Machine failure"])
print(type_failure_counts)

print("\nMAKİNE TİPİNE GÖRE ARIZA YÜZDELERİ")
type_failure_percentages = pd.crosstab(df["Type"], df["Machine failure"], normalize="index") * 100
print(type_failure_percentages.round(2))

print("\nSAYISAL DEĞİŞKENLERİN İSTATİSTİKLERİ")
print(df.describe())

print("\nARIZA DURUMUNA GÖRE DEĞİŞKEN ORTALAMALARI")
columns_to_compare = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]","Torque [Nm]","Tool wear [min]"]

failure_comparison = df.groupby("Machine failure")[columns_to_compare].mean()
print(failure_comparison.round(2))


failure_comparison.plot(kind="bar", subplots=True, layout=(3, 2), figsize=(12, 10), legend=False, sharex=True)

plt.suptitle("Arıza Durumuna Göre Değişken Ortalamaları")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


print("\nARIZA TÜRLERİNİN DAĞILIMI")
failure_types = ["TWF", "HDF", "PWF", "OSF", "RNF"]
for column in failure_types:
    print(column, ":", df[column].sum())
    

print("\nEKSİK DEĞER SAYILARI")
print(df.isnull().sum())

print("\nYİNELENEN SATIR SAYISI")
print(df.duplicated().sum())


features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

X = df[features]
y = df["Machine failure"]

print("\nÖZELLİKLERİN İLK 5 SATIRI")
print(X.head())

print("\nHEDEF DEĞİŞKENİN İLK 5 SATIRI")
print(y.head())

print("\nÖZELLİK BOYUTU:", X.shape)
print("HEDEF BOYUTU:", y.shape)

X_encoded = pd.get_dummies(
    X,
    columns=["Type"],
    dtype=int
)

print("\nDÖNÜŞTÜRÜLMÜŞ ÖZELLİKLER")
print(X_encoded.head())

print("\nDÖNÜŞTÜRÜLMÜŞ VERİ BOYUTU:", X_encoded.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nEĞİTİM VE TEST VERİLERİNİN BOYUTLARI")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\nEĞİTİM VERİSİNDEKİ ARIZALI KAYIT SAYISI:", y_train.sum())
print("TEST VERİSİNDEKİ ARIZALI KAYIT SAYISI:", y_test.sum())

print("\nLOJİSTİK REGRESYON MODELİ")

logistic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

logistic_model.fit(X_train, y_train)


print("\nMODEL TAHMİNLERİ")

y_pred = logistic_model.predict(X_test)

print(y_pred[:20])

print("\nMODEL DEĞERLENDİRME METRİKLERİ")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

print("\nCONFUSION MATRIX")

cm = confusion_matrix(y_test, y_pred)

print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Arıza Yok", "Arıza Var"]
)

display.plot()
plt.title("Lojistik Regresyon - Confusion Matrix")
plt.show()

print("\nDECISION TREE MODELİ")

decision_tree = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42
)

decision_tree.fit(X_train, y_train)

y_pred_tree = decision_tree.predict(X_test)

print("\nDECISION TREE - MODEL DEĞERLENDİRME")

accuracy_tree = accuracy_score(y_test, y_pred_tree)
precision_tree = precision_score(y_test, y_pred_tree, zero_division=0)
recall_tree = recall_score(y_test, y_pred_tree, zero_division=0)
f1_tree = f1_score(y_test, y_pred_tree, zero_division=0)

print("Accuracy:", accuracy_tree)
print("Precision:", precision_tree)
print("Recall:", recall_tree)
print("F1-score:", f1_tree)

y_train_pred_tree = decision_tree.predict(X_train)

print("\nDECISION TREE - EĞİTİM VE TEST KARŞILAŞTIRMASI")

print("Eğitim F1:",f1_score(y_train, y_train_pred_tree, zero_division=0))
print("Test F1:",f1_score(y_test, y_pred_tree, zero_division=0))

print("Eğitim Recall:",recall_score(y_train, y_train_pred_tree, zero_division=0))
print("Test Recall:",recall_score(y_test, y_pred_tree, zero_division=0))

print("\nDECISION TREE - CONFUSION MATRIX")

cm_tree = confusion_matrix(y_test, y_pred_tree)

print(cm_tree)

display_tree = ConfusionMatrixDisplay(
    confusion_matrix=cm_tree,
    display_labels=["Arıza Yok", "Arıza Var"]
)

display_tree.plot()
plt.title("Decision Tree - Confusion Matrix")
plt.show()

print("\nRANDOM FOREST MODELİ")

random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

random_forest.fit(X_train, y_train)

y_pred_forest = random_forest.predict(X_test)

print("\nRANDOM FOREST - MODEL DEĞERLENDİRME")

accuracy_forest = accuracy_score(y_test, y_pred_forest)
precision_forest = precision_score(y_test, y_pred_forest, zero_division=0)
recall_forest = recall_score( y_test, y_pred_forest, zero_division=0)
f1_forest = f1_score( y_test, y_pred_forest, zero_division=0)

print("Accuracy:", accuracy_forest)
print("Precision:", precision_forest)
print("Recall:", recall_forest)
print("F1-score:", f1_forest)

print("\nRANDOM FOREST - CONFUSION MATRIX")

cm_forest = confusion_matrix(y_test, y_pred_forest)

print(cm_forest)

display_forest = ConfusionMatrixDisplay(
    confusion_matrix=cm_forest,
    display_labels=["Arıza Yok", "Arıza Var"]
)

display_forest.plot()
plt.title("Random Forest - Confusion Matrix")
plt.show()

y_train_pred_forest = random_forest.predict(X_train)

print("\nRANDOM FOREST - EĞİTİM VE TEST KARŞILAŞTIRMASI")

print("Eğitim F1:",f1_score(y_train, y_train_pred_forest, zero_division=0))
print("Test F1:",f1_score(y_test, y_pred_forest, zero_division=0))
print("Eğitim Recall:",recall_score(y_train, y_train_pred_forest, zero_division=0))
print("Test Recall:",recall_score(y_test, y_pred_forest, zero_division=0))

print("\nXGBOOST MODELİ")

X_train_xgb = X_train.copy()
X_test_xgb = X_test.copy()

X_train_xgb.columns = [
    col.replace("[", "").replace("]", "").replace("<", "")
    for col in X_train_xgb.columns
]

X_test_xgb.columns = X_train_xgb.columns


xgboost_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=9661 / 339,
    eval_metric="logloss",
    random_state=42
)

xgboost_model.fit(X_train_xgb, y_train)
y_pred_xgb = xgboost_model.predict(X_test_xgb)


accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb, zero_division=0)
recall_xgb = recall_score(y_test, y_pred_xgb, zero_division=0)
f1_xgb = f1_score(y_test, y_pred_xgb, zero_division=0)

print("\nXGBOOST - MODEL DEĞERLENDİRME")
print("Accuracy:", accuracy_xgb)
print("Precision:", precision_xgb)
print("Recall:", recall_xgb)
print("F1-score:", f1_xgb)


print("\nXGBOOST - CONFUSION MATRIX")
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
print(cm_xgb)

display_xgb = ConfusionMatrixDisplay(
    confusion_matrix=cm_xgb,
    display_labels=["Arıza Yok", "Arıza Var"]
)

display_xgb.plot()
plt.title("XGBoost - Confusion Matrix")
plt.tight_layout()
plt.show()

results = pd.DataFrame([
    {
        "Model": "Logistic Regression",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    },
    {
        "Model": "Decision Tree",
        "Accuracy": accuracy_tree,
        "Precision": precision_tree,
        "Recall": recall_tree,
        "F1-score": f1_tree
    },
    {
        "Model": "Random Forest",
        "Accuracy": accuracy_forest,
        "Precision": precision_forest,
        "Recall": recall_forest,
        "F1-score": f1_forest
    },
    {
        "Model": "XGBoost",
        "Accuracy": accuracy_xgb,
        "Precision": precision_xgb,
        "Recall": recall_xgb,
        "F1-score": f1_xgb
    }
])

metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

model_names = results["Model"].tolist()

plt.figure(figsize=(11, 6))

x = np.arange(len(metrics))
width = 0.18

for i, row in results.iterrows():

    values = [
        row["Accuracy"],
        row["Precision"],
        row["Recall"],
        row["F1-score"]
    ]

    position = x + (i - (len(results) - 1) / 2) * width

    plt.bar(
        position,
        values,
        width,
        label=row["Model"]
    )

plt.xlabel("Değerlendirme Metrikleri")
plt.ylabel("Skor")
plt.title("Makine Öğrenmesi Modellerinin Performans Karşılaştırması")

plt.xticks(x, metrics)
plt.ylim(0, 1)

plt.legend()
plt.tight_layout()
plt.show()


feature_importance = pd.DataFrame({
    "Özellik": X_train.columns,
    "Önem": random_forest.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Önem",
    ascending=False
)

print("\nRANDOM FOREST - ÖZELLİK ÖNEMLERİ")
print(feature_importance.to_string(index=False))


plt.figure(figsize=(10, 6))
plt.barh(
    feature_importance["Özellik"],
    feature_importance["Önem"]
)
plt.xlabel("Önem Skoru")
plt.ylabel("Özellikler")
plt.title("Random Forest - Özellik Önemleri")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

xgb_importance = pd.DataFrame({
    "Özellik": X_train_xgb.columns,
    "Önem": xgboost_model.feature_importances_
})

xgb_importance = xgb_importance.sort_values(
    by="Önem",
    ascending=False
)

xgb_importance["Özellik"] = xgb_importance["Özellik"].replace({
    "Air temperature K": "Air temperature [K]",
    "Process temperature K": "Process temperature [K]",
    "Rotational speed rpm": "Rotational speed [rpm]",
    "Torque Nm": "Torque [Nm]",
    "Tool wear min": "Tool wear [min]"
})

print("\nXGBOOST - ÖZELLİK ÖNEMLERİ")
print(xgb_importance.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(xgb_importance["Özellik"], xgb_importance["Önem"])
plt.xlabel("Önem Skoru")
plt.ylabel("Özellikler")
plt.title("XGBoost - Özellik Önemleri")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


yeni_makine = pd.DataFrame([{
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1500,
    "Torque [Nm]": 45.0,
    "Tool wear [min]": 120,
    "Type_H": 0,
    "Type_L": 1,
    "Type_M": 0
}])
tahmin = random_forest.predict(yeni_makine)[0]
olasilik = random_forest.predict_proba(yeni_makine)[0][1]


yeni_makine_xgb = yeni_makine.copy()
yeni_makine_xgb.columns = [
    col.replace("[", "").replace("]", "").replace("<", "")
    for col in yeni_makine_xgb.columns
]
tahmin_xgb = xgboost_model.predict(yeni_makine_xgb)[0]
olasilik_xgb = xgboost_model.predict_proba(yeni_makine_xgb)[0][1]

print("\nBİRİNCİ MAKİNE - MODEL KARŞILAŞTIRMASI")

print("\nRandom Forest:")
print("Tahmin:", "Arıza Var" if tahmin == 1 else "Arıza Yok")
print(f"Arıza olasılığı: %{olasilik * 100:.2f}")

print("\nXGBoost:")
print("Tahmin:", "Arıza Var" if tahmin_xgb == 1 else "Arıza Yok")
print(f"Arıza olasılığı: %{olasilik_xgb * 100:.2f}")


arizali_makine = pd.DataFrame([{
    "Air temperature [K]": 300.5,
    "Process temperature [K]": 310.5,
    "Rotational speed [rpm]": 1350,
    "Torque [Nm]": 60.0,
    "Tool wear [min]": 200,
    "Type_H": 0,
    "Type_L": 1,
    "Type_M": 0
}])
ariza_tahmini = random_forest.predict(arizali_makine)[0]
ariza_olasiligi = random_forest.predict_proba(arizali_makine)[0][1]


arizali_makine_xgb = arizali_makine.copy()
arizali_makine_xgb.columns = [
    col.replace("[", "").replace("]", "").replace("<", "")
    for col in arizali_makine_xgb.columns
]
ariza_tahmini_xgb = xgboost_model.predict(arizali_makine_xgb)[0]
ariza_olasiligi_xgb = xgboost_model.predict_proba(arizali_makine_xgb)[0][1]

print("\nİKİNCİ MAKİNE - MODEL KARŞILAŞTIRMASI")

print("\nRandom Forest:")
print("Tahmin:", "Arıza Var" if ariza_tahmini == 1 else "Arıza Yok")
print(f"Arıza olasılığı: %{ariza_olasiligi * 100:.2f}")

print("\nXGBoost:")
print("Tahmin:", "Arıza Var" if ariza_tahmini_xgb == 1 else "Arıza Yok")
print(f"Arıza olasılığı: %{ariza_olasiligi_xgb * 100:.2f}")
