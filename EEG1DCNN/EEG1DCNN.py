from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, SpatialDropout2D,
    BatchNormalization, Flatten, Dense, LeakyReLU, Input
)
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder

def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        # Temporal Convolution
        Conv2D(25, (11, 1), padding='valid'),
        LeakyReLU(alpha=0.01),
        SpatialDropout2D(0.5),

        # Spatial Convolution
        Conv2D(25, (1, 2), padding='valid'),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        MaxPooling2D((3, 1), strides=(3, 1)),

        # Temporal Convolution
        Conv2D(50, (11, 1), padding='valid'),
        LeakyReLU(alpha=0.01),
        SpatialDropout2D(0.5),
        MaxPooling2D((3, 1), strides=(3, 1)),

        # Temporal Convolution
        Conv2D(100, (11, 1), padding='valid'),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        SpatialDropout2D(0.5),
        MaxPooling2D((3, 1), strides=(3, 1)),

        # Temporal Convolution
        Conv2D(200, (11, 1), padding='valid'),
        BatchNormalization(),
        LeakyReLU(alpha=0.01),
        MaxPooling2D((2, 1), strides=(2, 1)),

        Flatten(),
        Dense(4, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model(X_train, y_train, X_val, y_val, epochs=2000, batch_size=32):
    model = build_model(input_shape = X_train.shape[1:])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return model, history
