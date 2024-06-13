import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# استخراج ویژگی‌ها
def extract_features(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfccs.T

# بارگذاری داده‌ها
def load_data(data_dir, classes):
    X = []
    y = []
    for i, label in enumerate(classes):
        label_dir = os.path.join(data_dir, label)
        for file in os.listdir(label_dir):
            if file.endswith(".wav"):
                file_path = os.path.join(label_dir, file)
                features = extract_features(file_path)
                X.append(features)
                y.append(i)
    X = np.array(X)
    y = np.array(y)
    return X, y

# مسیر داده‌ها
data_dir = "data"

# لیست کلمات
classes = ["balee", "nah", "salam", "khodahafez", "lotfan", "tashakor", "bebakhshid", "komak", "tavaqof", "boro",
           "chap", "rast", "bal", "paein", "shoroe", "payan", "baz", "baste", "roshan", "khamosh"]

# بارگذاری داده‌ها
X, y = load_data(data_dir, classes)

# تبدیل به دسته‌های یک‌سردرجه‌ای
y = to_categorical(y, num_classes=len(classes))

# تقسیم داده‌ها به آموزشی و آزمایشی
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تعریف مدل
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# شکل‌دهی داده‌ها برای مدل
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# آموزش مدل
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# ارزیابی مدل
score = model.evaluate(X_test, y_test, verbose=0)
print(f"دقت مدل: {score[1] * 100:.2f}%")
