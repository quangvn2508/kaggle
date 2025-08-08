import pandas as pd
from tensorflow.keras import models, layers
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

df = pd.read_csv('data/train.csv')

y_train_full = df.iloc[:, 0]
X_train_full = df.iloc[:, 1:]

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42
)

print(X_train)

model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=[28 * 28]),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=Adam(),
    loss=SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=20,
)

# Plot and save loss
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Val Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss over Epochs')
# plt.legend()
# plt.savefig('loss_plot.png')

# # Plot and save accuracy
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Val Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.title('Accuracy over Epochs')
# plt.legend()
# plt.savefig('accuracy_plot.png')

new_df = pd.read_csv('data/test.csv')

predictions = model.predict(new_df)
predicted_classes = predictions.argmax(axis=1)
print(predicted_classes)

output_df = pd.DataFrame({
    'ImageId': range(1, len(predicted_classes) + 1),
    'Label': predicted_classes
})
output_df.to_csv('predictions.csv', index=False)