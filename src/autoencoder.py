import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc


def load_dataset(path_train):
    X_train = pd.read_csv(path_train)
    return X_train


def create_autoencoder(dataset):
    input_dim = dataset.shape[1]
    input_layer = tf.keras.Input(shape=(input_dim,))
    encoder1 = tf.keras.layers.Dense(input_dim, activation='relu')(input_layer)
    encoder2 = tf.keras.layers.Dense(int(input_dim / 2), activation='relu')(encoder1)
    encoder3 = tf.keras.layers.Dense(int(input_dim / 4), activation='relu')(encoder2)
    decoder1 = tf.keras.layers.Dense(int(input_dim / 4), activation='relu')(encoder3)
    decoder2 = tf.keras.layers.Dense(int(input_dim / 2), activation='relu')(decoder1)
    decoder3 = tf.keras.layers.Dense(input_dim, activation='relu')(decoder2)
    output_layer = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoder3)
    autoencoder = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    optimizer = tf.keras.optimizers.Adam()
    autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
    return autoencoder


def training(autoencoder, X_train):
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    autoencoder.fit(X_train, X_train,
                    epochs=100,
                    batch_size=256,
                    shuffle=True,
                    validation_split=0.2,  # Usa il 20% dei dati di training per la validazione
                    callbacks=[early_stop])


def calculate_loss(autoencoder, X_test):
    reconstructions = autoencoder.predict(X_test)
    loss = np.mean(np.square(reconstructions - X_test), axis=1)
    return loss


def curva_roc(y_test, loss, figures_path):
    fpr, tpr, thresholds_roc = roc_curve(y_test, loss)

    roc_auc = auc(fpr, tpr)

    optimal_threshold_index = np.argmax(tpr - fpr)  # tpr-fpr è l'indice di Youden
    best_threshold_roc = thresholds_roc[optimal_threshold_index]

    print("AUC-ROC:", roc_auc)
    print("Best Threshold (ROC):", best_threshold_roc)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC Curve (AUC-ROC = %0.2f)' % roc_auc, color='b')
    plt.scatter(fpr[optimal_threshold_index], tpr[optimal_threshold_index], color='red', marker='o',
                label='Optimal Threshold')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)

    figure_path = os.path.join(figures_path, 'roc_curve.png')
    print(f"Saving roc curve figure in {figure_path}")
    plt.savefig(figure_path)
    return best_threshold_roc


def detect_anomaly(new_data, autoencoder, threshold):
    reconstructions = autoencoder.predict(new_data)
    loss = np.mean(np.square(reconstructions - new_data), axis=1)
    anomalies = loss >= threshold
    return anomalies


def real_labels(y_test):
    real_labels = y_test.to_numpy().astype(bool).flatten()
    return real_labels


def metrics(real_labels, predicted_labels, figures_path):
    cm = confusion_matrix(real_labels, predicted_labels)
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    print(classification_report(real_labels, predicted_labels))
    print("\nAccuracy:", accuracy_score(real_labels, predicted_labels))

    figure_path = os.path.join(figures_path, 'confusion_matrix.png')
    print(f"Saving confusion matrix figure in {figure_path}")
    plt.savefig(figure_path)


def feature_importance(autoencoder, X_test, figures_path):
    reconstructions = autoencoder.predict(X_test)
    reconstruction_errors = np.square(reconstructions - X_test)
    mean_feature_error = np.mean(reconstruction_errors, axis=0)
    feature_names = X_test.columns
    feature_error_dict = dict(zip(feature_names, mean_feature_error))
    sorted_feature_error = sorted(feature_error_dict.items(), key=lambda x: x[1], reverse=True)
    plt.figure(figsize=(10, 15))
    plt.barh([x[0] for x in sorted_feature_error], [x[1] for x in sorted_feature_error])
    plt.gca().invert_yaxis()
    plt.xlabel('Mean Reconstruction Error')
    plt.ylabel('Feature Name')
    plt.title('Mean Reconstruction Error for Each Feature')

    figure_path = os.path.join(figures_path, 'feature_importance.png')
    print(f"Saving reconstruction error figure in {figure_path}")
    plt.savefig(figure_path)
    plt.savefig(figure_path)
