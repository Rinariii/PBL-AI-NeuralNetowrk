import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

"""##Membuat fungsi neural network"""

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # He initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)

        # output
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2

    def backward(self, X, t, y_pred, lr):
        delta2 = (y_pred - t)

        dW2 = np.dot(self.a1.T, delta2)
        db2 = delta2.sum(axis=0, keepdims=True)

        delta1 = np.dot(delta2, self.W2.T) * self.relu_derivative(self.z1)

        dW1 = np.dot(X.T, delta1)
        db1 = delta1.sum(axis=0, keepdims=True)

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def train(self, X, y, epochs, lr):
        self.loss_history = []
        self.mae_history = []
        self.r2_history = []


        for i in range(epochs):
            # Forward pass
            y_pred = self.forward(X)

            # Hitung metrik
            loss = np.mean((y - y_pred)**2)
            mae  = mean_absolute_error(y, y_pred)
            r2   = r2_score(y, y_pred)

            # Simpan history
            self.loss_history.append(loss)
            self.mae_history.append(mae)
            self.r2_history.append(r2)

            # Backprop
            self.backward(X, y, y_pred, lr)


            if i % 500 == 0 or i == epochs - 1:
                print(f"Epoch {i}, Loss: {loss:.5f}, MAE: {mae:.5f}, R2: {r2:.5f}")

"""##Load dataset"""

url = "https://drive.google.com/uc?id=1rimtD4J7RFrTZ6G1nOmF1Pm5TqaCmTNP"
df = pd.read_csv(url)
df.head()

# Cek info dataset
df.info()

# Cek statistik deskriptif untuk dataset
df.describe()

"""##Exploratory Data Analysis"""

# Korelasi Matriks
num_cols = df.select_dtypes(include=['number']).columns
corr = df[num_cols].corr()

plt.figure(figsize=(8, 6))
plt.imshow(corr, cmap='Blues')
plt.colorbar(label='Correlation')
plt.xticks(np.arange(len(num_cols)), num_cols, rotation=30, ha='right')
plt.yticks(np.arange(len(num_cols)), num_cols)
plt.title("Correlation Matrix", fontsize=14)

for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', fontsize=8)
plt.tight_layout()
plt.show()

# Membuat Boxplot
plt.figure(figsize=(12,6))
df.select_dtypes(include=['number']).boxplot()
plt.xticks(rotation=45)
plt.title("Boxplot Fitur")
plt.show()

# Membuat scatter plot
num_df = df.select_dtypes(include=['int64', 'float64'])
num_df.head()
sns.pairplot(num_df)
plt.show()

"""##Data Cleaning & Preprocessing"""

# Cek apakah ada data null atau tidak
df.isnull().sum()

# Cek apakah ada data yang double
df.duplicated().sum()

"""##Training Model"""

# Membagi dataset dengan train test split
X = df.drop(columns=['SoilMoisture'])
y = df['SoilMoisture']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#Scalling model
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Normalisasi X
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)

# Normalisasi Y
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1,1))
y_test_scaled  = scaler_y.transform(y_test.values.reshape(-1,1))

# Membuat model
model = SimpleNeuralNetwork(
    input_size=X_train.shape[1],
    hidden_size=8,
    output_size=1
)

# Training Model
model.train(X_train_scaled, y_train_scaled, epochs=5000, lr = 0.0001)

y_pred = model.forward(X_test_scaled)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test_scaled, y_pred)
mae = mean_absolute_error(y_test_scaled, y_pred)
r2  = r2_score(y_test_scaled, y_pred)

print("MSE:", mse)
print("MAE:", mae)
print("R2 Score:", r2)

# Ambil 1 data uji
x_sample = X_test_scaled[0].reshape(1, -1)

# Ambil nilai aktual dari y_test
y_test_np = y_test.values.reshape(-1,1)
y_actual = y_test_np[0][0]

# Prediksi
y_pred_scaled = model.forward(x_sample)
y_pred = scaler_y.inverse_transform(y_pred_scaled)[0][0]

print(f"Prediksi Soil Mosture (data uji): {y_pred:.2f} (aktual: {y_actual:.2f})")

mse_history = np.array(model.loss_history)
mae_history = []
r2_history = []


for epoch in range(len(model.loss_history)):
    y_pred_epoch = model.forward(X_train_scaled)
    mae_history.append(mean_absolute_error(y_train_scaled, y_pred_epoch))
    r2_history.append(r2_score(y_train_scaled, y_pred_epoch))

mae_history = np.array(mae_history)
r2_history = np.array(r2_history)

# Visualisasi loss
plt.figure(figsize=(15,4))

# MSE
plt.subplot(1,3,1)
plt.plot(mse_history, color='orange')
plt.title("MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)

# MAE
plt.subplot(1,3,2)
plt.plot(mae_history, color='green')
plt.title("MAE")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.grid(True)

# R²
plt.subplot(1,3,3)
plt.plot(r2_history, color='blue')
plt.title("R²")
plt.xlabel("Epoch")
plt.ylabel("R² Score")
plt.grid(True)

plt.tight_layout()
plt.show()

