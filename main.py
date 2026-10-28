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
