# ===================================================================
# UJIAN AKHIR SEMESTER - INTELIGENSI BUATAN
# Dataset: Incomes of 30K USA Citizens
# Model: Artificial Neural Network (ANN)
# ===================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import warnings
import os
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Make matplotlib non-interactive
plt.ioff()

import os
if not os.path.exists('income.csv'):
    print("Dataset 'income.csv' not found. Downloading Adult dataset from UCI...")
    import requests
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    response = requests.get(url)
    # Add headers
    headers = "age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income\n"
    data = response.text
    with open('income.csv', 'w') as f:
        f.write(headers + data)
    print("Dataset downloaded and saved as 'income.csv'.")

# Set random seed untuk reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ===================================================================
# 1. LOAD DAN EKSPLORASI DATASET
# ===================================================================

# Load dataset
# Catatan: Ganti path sesuai lokasi file Anda
df = pd.read_csv('income.csv')

print("=" * 70)
print("INFORMASI DATASET")
print("=" * 70)
print(f"Jumlah Data: {df.shape[0]} baris, {df.shape[1]} kolom")
print(f"\nKolom Dataset:\n{df.columns.tolist()}")
print(f"\nInfo Dataset:")
print(df.info())
print(f"\nStatistik Deskriptif:")
print(df.describe())
print(f"\nMissing Values:")
print(df.isnull().sum())
print(f"\nDistribusi Target Variable (income):")
print(df['income'].value_counts())

# ===================================================================
# 2. DATA PREPROCESSING
# ===================================================================

print("\n" + "=" * 70)
print("DATA PREPROCESSING")
print("=" * 70)

# Handle missing values
print("\nMenangani missing values...")
df = df.dropna()

# Identifikasi kolom kategorikal dan numerikal
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('income')  # Hapus target variable
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Kolom Kategorikal: {categorical_cols}")
print(f"Kolom Numerikal: {numerical_cols}")

# Label Encoding untuk fitur kategorikal
label_encoders = {}
df_encoded = df.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded {col}: {len(le.classes_)} unique values")

# Encoding target variable
target_encoder = LabelEncoder()
df_encoded['income'] = target_encoder.fit_transform(df['income'])
print(f"\nTarget Classes: {target_encoder.classes_}")

# Pisahkan features dan target
X = df_encoded.drop('income', axis=1)
y = df_encoded['income']

print(f"\nShape Features (X): {X.shape}")
print(f"Shape Target (y): {y.shape}")

# Split data: 70% training, 15% validation, 15% testing
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
)

print(f"\nData Training: {X_train.shape[0]} samples")
print(f"Data Validation: {X_val.shape[0]} samples")
print(f"Data Testing: {X_test.shape[0]} samples")

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed!")

# ===================================================================
# 3. MEMBANGUN MODEL ANN
# ===================================================================

print("\n" + "=" * 70)
print("MEMBANGUN MODEL ANN")
print("=" * 70)

def create_ann_model(input_dim, num_classes=2):
    """
    Membuat model ANN dengan arsitektur yang optimal
    """
    model = Sequential([
        # Input Layer + Hidden Layer 1
        Dense(128, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden Layer 2
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Hidden Layer 3
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Hidden Layer 4
        Dense(16, activation='relu'),
        
        # Output Layer
        Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
    ])
    
    return model

# Buat model
input_dimension = X_train_scaled.shape[1]
num_classes = len(np.unique(y))

model = create_ann_model(input_dimension, num_classes)

# Compile model
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Display model architecture
print("\nArsitektur Model ANN:")
model.summary()

# ===================================================================
# 4. TRAINING MODEL
# ===================================================================

print("\n" + "=" * 70)
print("TRAINING MODEL")
print("=" * 70)

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-7,
    verbose=1
)

# Training
print("\nMemulai training...")
history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ===================================================================
# 5. EVALUASI MODEL
# ===================================================================

print("\n" + "=" * 70)
print("EVALUASI MODEL")
print("=" * 70)

# Evaluasi pada data training
train_loss, train_accuracy = model.evaluate(X_train_scaled, y_train, verbose=0)
print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Training Loss: {train_loss:.4f}")

# Evaluasi pada data validation
val_loss, val_accuracy = model.evaluate(X_val_scaled, y_val, verbose=0)
print(f"\nValidation Accuracy: {val_accuracy:.4f}")
print(f"Validation Loss: {val_loss:.4f}")

# Evaluasi pada data testing
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Prediksi pada data testing
y_pred_proba = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)

# Classification Report
print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print("\n", classification_report(y_test, y_pred, 
                                  target_names=target_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ===================================================================
# 6. VISUALISASI HASIL
# ===================================================================

print("\n" + "=" * 70)
print("MEMBUAT VISUALISASI")
print("=" * 70)

# Plot 1: Training History
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=target_encoder.classes_,
            yticklabels=target_encoder.classes_,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Performance Metrics Comparison
metrics_data = {
    'Dataset': ['Training', 'Validation', 'Testing'],
    'Accuracy': [train_accuracy, val_accuracy, test_accuracy],
    'Loss': [train_loss, val_loss, test_loss]
}
metrics_df = pd.DataFrame(metrics_data)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
axes[0].bar(metrics_df['Dataset'], metrics_df['Accuracy'], 
            color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.8)
axes[0].set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_ylim(0, 1)
axes[0].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(metrics_df['Accuracy']):
    axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

# Loss comparison
axes[1].bar(metrics_df['Dataset'], metrics_df['Loss'], 
            color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.8)
axes[1].set_title('Loss Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].grid(True, alpha=0.3, axis='y')
for i, v in enumerate(metrics_df['Loss']):
    axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ===================================================================
# 7. SAVE MODEL
# ===================================================================

print("\n" + "=" * 70)
print("MENYIMPAN MODEL")
print("=" * 70)

model.save('income_prediction_model.keras')
print("Model berhasil disimpan sebagai 'income_prediction_model.keras'")

# ===================================================================
# 8. RINGKASAN HASIL
# ===================================================================

print("\n" + "=" * 70)
print("RINGKASAN HASIL EKSPERIMEN")
print("=" * 70)
print(f"\nArsitektur Model:")
print(f"  - Input Layer: {input_dimension} neurons")
print(f"  - Hidden Layer 1: 128 neurons (ReLU + BatchNorm + Dropout 0.3)")
print(f"  - Hidden Layer 2: 64 neurons (ReLU + BatchNorm + Dropout 0.3)")
print(f"  - Hidden Layer 3: 32 neurons (ReLU + BatchNorm + Dropout 0.2)")
print(f"  - Hidden Layer 4: 16 neurons (ReLU)")
print(f"  - Output Layer: {num_classes} neurons (Softmax)")
print(f"\nTotal Parameters: {model.count_params():,}")
print(f"\nOptimizer: Adam (lr=0.001)")
print(f"Loss Function: Sparse Categorical Crossentropy")
print(f"Batch Size: 32")
print(f"Epochs: {len(history.history['loss'])}")
print(f"\nPerforma Model:")
print(f"  - Training Accuracy: {train_accuracy:.4f}")
print(f"  - Validation Accuracy: {val_accuracy:.4f}")
print(f"  - Testing Accuracy: {test_accuracy:.4f}")
print("=" * 70)

print("\n✅ Eksperimen selesai! Semua file visualisasi telah disimpan.")
print("File yang dihasilkan:")
print("  1. training_history.png")
print("  2. confusion_matrix.png")
print("  3. performance_comparison.png")
print("  4. income_prediction_model.keras")