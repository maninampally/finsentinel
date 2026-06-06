import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPooling1D,
    LSTM,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def train_bilstm(X_train, X_val, y_train, y_val):
    tok = Tokenizer(num_words=10000, oov_token='<OOV>')
    tok.fit_on_texts(X_train)

    X_tr = pad_sequences(tok.texts_to_sequences(X_train), maxlen=128, padding='post')
    X_v  = pad_sequences(tok.texts_to_sequences(X_val),   maxlen=128, padding='post')

    with mlflow.start_run(run_name="tensorflow_bilstm"):
        mlflow.log_params({
            "model":         "Bidirectional LSTM",
            "vocab_size":    10000,
            "embedding_dim": 128,
            "lstm_units":    64,
            "epochs":        10
        })

        model = tf.keras.Sequential([
            Embedding(10000, 128, input_length=128),
            Bidirectional(LSTM(64, return_sequences=True)),
            GlobalMaxPooling1D(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(3,  activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            X_tr, y_train,
            epochs=10, batch_size=64,
            validation_data=(X_v, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(
                patience=3, restore_best_weights=True
            )]
        )

        for i, (loss, acc) in enumerate(zip(
            history.history['val_loss'],
            history.history['val_accuracy']
        )):
            mlflow.log_metrics({"val_loss": loss, "val_accuracy": acc}, step=i)

        mlflow.tensorflow.log_model(model, "bilstm_model")

    return model, tok
