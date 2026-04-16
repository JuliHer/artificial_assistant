import numpy as np
import tensorflow as tf
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

class WakeWordTraining:
    def __init__(self):
        current_file = Path(__file__).resolve()
        self.project_root = current_file.parent.parent.parent
        with open(str(self.project_root / "src" / "configs" / "config.yaml"), 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

    def create_neuronal(self):

        x_in = tf.keras.layers.Input(shape=(64, 40, 1))

        # Primera capa
        x = tf.keras.layers.Conv2D(16, (3,3), padding='same', activation='relu')(x_in)
        x = tf.keras.layers.BatchNormalization()(x)

        # Bloques DS-CNN
        for i, filters in enumerate([16, 32, 32, 64]):
            stride = (2,2) if i == 1 else (1,1) # Reduce a la mitad el tamaño en el segundo bloque
            x = tf.keras.layers.DepthwiseConv2D((3,3), strides=stride, padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

            x = tf.keras.layers.Conv2D(filters, (1,1), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

        x = tf.keras.layers.GlobalMaxPooling2D()(x)

        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        out = tf.keras.layers.Dense(1, activation='sigmoid')(x)



        model = tf.keras.models.Model(inputs=x_in, outputs=out)


        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name="prec"),
            tf.keras.metrics.Recall(name="rec"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.FalseNegatives(name="fn")
        ]
        
        

        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4)
        
        model.compile(
            
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=metrics
        )
        model.summary()

        self.model = model
        
        print("Modelo reado correctamente")
    
    def fit(self, X_train, X_test, Y_train, Y_test, plot=False):
        version = self.config["wakeword_settings"]["KORA"]["version"]
        name = self.config["wakeword_settings"]["KORA"]["wake_word"]
        checkpoint_path = str(self.project_root / "models" / "wakeword_detector" / f"{name}_{version}_best.keras")

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_prec',        
            verbose=0,
            save_best_only=True,       
            mode='max'                 
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=3, 
            min_lr=1e-6
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_prec',
            verbose=1,
            patience=7,
            mode='max'
        )
       
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(Y_train),
            y=Y_train
        )

        print(f"pesos: 0 {weights[0]} |  1 {weights[1]}")

        class_weights = {0: weights[0], 1: weights[1]}
        
        history = self.model.fit(
            X_train, Y_train, 
            epochs=25, 
            class_weight=class_weights,
            batch_size=64,
            callbacks=[checkpoint, reduce_lr, early_stop],
            validation_data=(X_test, Y_test),
            verbose=1
            
        )
        self.model.save(str(self.project_root / "models" / "wakeword_detector" / f"{name}_{version}.keras"))

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







    