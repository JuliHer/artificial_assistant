import os
import tensorflow as tf
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
class NERTraining:
    def __init__(self):
        current_file = Path(__file__).resolve()
        self.project_root = current_file.parent.parent.parent
        with open(str(self.project_root / "src" / "configs" / "config.yaml"), 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def create_neuronal(self):
        embedding_dim = self.config["ner_settings"]["embedding_dim"]
        n_types = self.config["ner_settings"]["types"]
        max_seq = self.config["ner_settings"]["max_words"]

        n_classes = (n_types*2)+1

        input = tf.keras.layers.Input(shape=(max_seq, embedding_dim))

        x = tf.keras.layers.Masking(mask_value=0.0)(input)
        
        attention_out = tf.keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=embedding_dim // 8
        )(x, x)
        x = tf.keras.layers.Add()([x, attention_out]) # Conexión residual
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_classes, activation="softmax"))(x)

        model = tf.keras.models.Model(inputs=input, outputs=output)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        model.compile(
            optimizer=optimizer, 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
            metrics=['accuracy']
        )
        model.summary()
        self.model = model
        print("Modelo reado correctamente")
    
    def fit(self, X, y, plot=False):
        version = self.config["ner_settings"]["version"]

        checkpoint_path = str(self.project_root / "models" / "ner" / f"KORA_{version}_best.keras")

        # Configurar el Checkpoint
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',        
            verbose=1,                 
            save_best_only=True,       
            mode='min'                 
        )

        weights = np.ones(y.shape)
        weights[y > 0] = 4.0      # Peso general para entidades
        weights[y % 2 != 0] = 12.0 # Peso extra para etiquetas B (Inicios)

        history = self.model.fit(
            X, y, 
            epochs=25,
            sample_weight=weights,
            batch_size=64,
            validation_split=0.2,
            callbacks=[checkpoint],
            verbose=1
        )
        self.model.save(str(self.project_root / "models" / "ner" / f"KORA_{version}.keras"))

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
    
    def getModel(self):
        return self.model
    
    






    