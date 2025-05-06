import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Generate synthetic dataset
def generate_data(num_patients=500, num_days=10):
    data = []
    for patient in range(num_patients):
        days_to_discharge = np.random.randint(1, num_days + 1)
        for day in range(days_to_discharge):
            for time_of_day in range(3):  # Morning, Afternoon, Evening
                heart_rate = np.random.randint(60, 100)
                blood_pressure = np.random.randint(110, 140)
                oxygen_saturation = np.random.uniform(95, 100)
                respiratory_rate = np.random.randint(12, 22)
                temperature = np.random.uniform(36.0, 37.5)
                data.append([patient, day, time_of_day, heart_rate, blood_pressure, 
                             oxygen_saturation, respiratory_rate, temperature, days_to_discharge])
    columns = ['patient_id', 'day_index', 'time_of_day', 'heart_rate', 'blood_pressure', 
               'oxygen_saturation', 'respiratory_rate', 'temperature', 'days_to_discharge']
    return pd.DataFrame(data, columns=columns)

# Generate and shuffle dataset
df = generate_data()
print("Generated Data Sample:\n", df.head(100))

df = df.sample(frac=1).reset_index(drop=True)

#print("Shuffled Data Sample:\n", df.head(100))

# Normalize numerical features
scaler = MinMaxScaler()
df[['heart_rate', 'blood_pressure', 'oxygen_saturation', 'respiratory_rate', 'temperature']] = \
    scaler.fit_transform(df[['heart_rate', 'blood_pressure', 'oxygen_saturation', 'respiratory_rate', 'temperature']])

# Prepare time-series data
def prepare_sequences(df, sequence_length=9):
    X, y = [], []
    grouped = df.groupby('patient_id')
    print("Grouped Data Sample:\n", grouped.head(10))
    for _, group in grouped:
        group = group.sort_values(['day_index', 'time_of_day']).reset_index(drop=True)
        for i in range(len(group) - sequence_length):
            X.append(group.iloc[i:i + sequence_length, 3:-1].values)  # Vital sign features
            y.append(group.iloc[i + sequence_length]['days_to_discharge'])
    return np.array(X), np.array(y)

X, y = prepare_sequences(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM Model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Make predictions
predictions = model.predict(X_test)

# Display sample predictions
for i in range(5):
    print(f"Actual Days to Discharge: {y_test[i]}, Predicted: {predictions[i][0]:.2f}")
