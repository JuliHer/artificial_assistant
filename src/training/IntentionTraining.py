import os
import tensorflow as tf
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import src.utils.helper as hp

class IntentTraining:
    def __init__(self):
        current_file = Path(__file__).resolve()
        self.project_root = current_file.parent.parent.parent
        with open(str(self.project_root / "src" / "configs" / "config.yaml"), 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def create_neuronal(self, map):
        embedding_dim = self.config["intention_settings"]["embedding_dim"]
        num_intents = len(map)

        input = tf.keras.layers.Input(shape=(embedding_dim,))
        n = tf.keras.layers.Dense(512, activation="relu")(input)
        n = tf.keras.layers.BatchNormalization()(n)
        n = tf.keras.layers.Dropout(0.3)(n)

        n = tf.keras.layers.Dense(256, activation="relu")(n)
        n = tf.keras.layers.BatchNormalization()(n)
        n = tf.keras.layers.Dropout(0.2)(n)

        n = tf.keras.layers.Dense(128, activation="relu")(n)
        n2 = tf.keras.layers.Dense(num_intents, activation="softmax")(n)
        model = tf.keras.models.Model(inputs=input, outputs=n2)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        model.compile(
            optimizer=optimizer, 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
            metrics=['accuracy']
        )
        model.summary()

        return model
    
    def fit(self, model, X, y, plot=False):
        version = self.config["intention_settings"]["version"]
        checkpoint_path = str(self.project_root / "models" / "intent_classifier" / f"KORA_{version}_best.keras")

        # Configurar el Checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',        
            verbose=1,                 
            save_best_only=True,       
            mode='min'                 
        )

        weights = hp.get_weights(y)
        
        history = model.fit(
            X, y, 
            epochs=50, 
            class_weight=weights,
            batch_size=32,
            validation_split=0.2,
            callbacks=[checkpoint],
            verbose=1
        )
        model.save(str(self.project_root / "models" / "intent_classifier" / f"KORA_{version}.keras"))

        if plot:
            self.plotHistory(history)

    def plotHistory(self, history):
        history_dict = history.history

        # Obtener precisión y pérdida de entrenamiento
        acc = history_dict.get('accuracy') or history_dict.get('acc')
        loss = history_dict.get('loss')
        
        # Obtener métricas de validación, si existen
        val_acc = history_dict.get('val_accuracy') or history_dict.get('val_acc')
        val_loss = history_dict.get('val_loss')
        
        # Número de épocas
        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(14, 5))
        
        # Gráfico de precisión
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'bo-', label='Precisión Entrenamiento')
        if val_acc is not None:
            plt.plot(epochs, val_acc, 'ro-', label='Precisión Validación')
        else:
            print("No se encontró la métrica 'val_accuracy' o 'val_acc' en el history.")
        plt.title('Precisión de Entrenamiento y Validación')
        plt.xlabel('Épocas')
        plt.ylabel('Precisión')
        plt.legend()
        
        # Gráfico de pérdida
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'bo-', label='Pérdida Entrenamiento')
        if val_loss is not None:
            plt.plot(epochs, val_loss, 'ro-', label='Pérdida Validación')
        else:
            print("No se encontró la métrica 'val_loss' en el history.")
        plt.title('Pérdida de Entrenamiento y Validación')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()
        
        plt.tight_layout()
        plt.show()







    