import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = pd.read_csv(file_path, sep='\s+', header=None)
    return data

train = load_data('ECG5000_TRAIN.txt')
test = load_data('ECG5000_TEST.txt')

data = pd.concat([train, test], ignore_index=True)

data.rename(columns={0: 'label'}, inplace=True)

print("=== Analyse Exploratoire des Données ===\n")

print("Aperçu des données :")
print(data.head(), "\n")

print("Informations sur les données :")
print(data.info(), "\n")

print("Statistiques descriptives :")
print(data.describe(), "\n")

print("Valeurs manquantes par colonne :")
print(data.isnull().sum(), "\n")

plt.figure(figsize=(6,4))
sns.countplot(x='label', data=data)  
plt.title('Distribution des Classes')
plt.xlabel('Classe')
plt.ylabel('Nombre d\'instances')
plt.show()

# Matrice de corrélation
plt.figure(figsize=(10,8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=0.5)
plt.title('Matrice de Corrélation')
plt.show()

# Distribution des caractéristiques principales
plt.figure(figsize=(12,6))
for i in range(1, 6):  # Afficher les premières 5 caractéristiques
    plt.subplot(2, 3, i)
    sns.histplot(data[i], kde=True)
    plt.title(f'Distribution de la Caractéristique {i}')
plt.tight_layout()
plt.show()

# ---- Prétraitement des Données ----
print("=== Prétraitement des Données ===\n")

X = data.iloc[:, 1:].values  
y = data['label'].values    

# Distribution des classes
unique, counts = np.unique(y, return_counts=True)
print("Distribution des classes :")
for cls, cnt in zip(unique, counts):
    print(f"Classe {cls} : {cnt} instances")
print()

y = np.where(y == 1, 1, 0)

unique, counts = np.unique(y, return_counts=True)
print("Distribution des classes après binarisation :")
for cls, cnt in zip(unique, counts):
    print(f"Classe {cls} : {cnt} instances")
print()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Caractéristiques normalisées avec StandardScaler.\n")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Dimensions des ensembles de données après prétraitement :")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}\n")

# Visualisation d'exemples de signaux ECG
plt.figure(figsize=(12, 6))

normal_indices = np.where(y_train == 0)[0]
if normal_indices.size > 0:
    normal_idx = normal_indices[0]
    plt.subplot(2, 1, 1)
    plt.plot(X_train[normal_idx])
    plt.title('Exemple de Signal ECG Normal')
else:
    print("Aucun signal ECG normal trouvé dans l'ensemble d'entraînement.")

# Exemple de signal ECG anormal
anomaly_indices = np.where(y_train == 1)[0]
if anomaly_indices.size > 0:
    anomaly_idx = anomaly_indices[0]
    plt.subplot(2, 1, 2)
    plt.plot(X_train[anomaly_idx])
    plt.title('Exemple de Signal ECG Anormal')
else:
    print("Aucun signal ECG anormal trouvé dans l'ensemble d'entraînement.")

plt.tight_layout()
plt.show()

# Sauvegarde des données prétraitées
np.savez('preprocessed_ECG5000.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

print("Prétraitement terminé et données sauvegardées dans 'preprocessed_ECG5000.npz'.")
