import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import random


def load_data(file_path):
    """
    Load dataset from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        tuple: Feature matrix and target vector.
    """
    try:
        data = pd.read_csv(file_path)
        X = data.drop('label', axis=1).values
        y = data['label'].values
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


def preprocess_data(X, y):
    """
    Normalize, reshape, and convert labels to categorical format.
    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
    Returns:
        tuple: Preprocessed feature matrix and labels.
    """
    try:
        X = X / 255.0
        X = X.reshape(-1, 28, 28, 1)
        y = to_categorical(y)
        return X, y
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        return None, None


def create_cnn_model(input_shape, num_classes):
    """
    Define a CNN model architecture.
    Args:
        input_shape (tuple): Shape of input data.
        num_classes (int): Number of output classes.
    Returns:
        model: Compiled CNN model.
    """
    try:
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])
        return model
    except Exception as e:
        print(f"Error creating model: {e}")
        return None


def flatten_weights(weights):
    """
    Flatten a list of weight arrays for optimization.
    Args:
        weights (list): List of weight arrays from the model.
    Returns:
        np.ndarray: Flattened weight array.
    """
    return np.concatenate([w.flatten() for w in weights])


def reshape_weights(flat_weights, model):
    """
    Reshape flat weight array back to model-compatible format.
    Args:
        flat_weights (np.ndarray): Flattened weight array.
        model (Sequential): Model to retrieve weight shapes.
    Returns:
        list: Reshaped weights.
    """
    shapes = [w.shape for w in model.get_weights()]
    reshaped = []
    idx = 0
    for shape in shapes:
        size = np.prod(shape)
        reshaped.append(flat_weights[idx:idx + size].reshape(shape))
        idx += size
    return reshaped


class Particle:
    """
    Represents a single particle in the Particle Swarm Optimization (PSO).
    """
    def __init__(self, n_weights):
        self.position = np.random.uniform(-1, 1, n_weights)
        self.velocity = np.zeros(n_weights)
        self.best_position = self.position.copy()
        self.best_fitness = float('-inf')

    def update_velocity(self, global_best_position, c1, c2):
        """
        Update the velocity of the particle.
        Args:
            global_best_position (np.ndarray): Global best position in swarm.
            c1 (float): Cognitive coefficient.
            c2 (float): Social coefficient.
        """
        r1 = random.random()
        r2 = random.random()
        self.velocity += (c1 * r1 * (self.best_position - self.position) +
                          c2 * r2 * (global_best_position - self.position))

    def update_position(self, bounds):
        """
        Update the position of the particle within bounds.
        Args:
            bounds (tuple): Min and max bounds for position values.
        """
        self.position = np.clip(self.position + self.velocity,
                                bounds[0], bounds[1])

    def evaluate_fitness(self, model, X_val, y_val):
        """
        Evaluate the fitness of the particle using the validation set.
        Args:
            model (Sequential): Keras model to evaluate fitness.
            X_val (np.ndarray): Validation feature matrix.
            y_val (np.ndarray): Validation labels.
        Returns:
            float: Fitness score.
        """
        try:
            model.set_weights(reshape_weights(self.position, model))
            _, fitness = model.evaluate(X_val, y_val, verbose=0)
            return fitness
        except Exception as e:
            print(f"Error evaluating fitness: {e}")
            return float('-inf')


def train_pso(model, particles, X_val, y_val, epochs, c1, c2, bounds):
    """
    Train a neural network using Particle Swarm Optimization (PSO).
    Args:
        model (Sequential): Keras model to optimize.
        particles (list): List of Particle instances.
        X_val (np.ndarray): Validation feature matrix.
        y_val (np.ndarray): Validation labels.
        epochs (int): Number of iterations for PSO.
        c1 (float): Cognitive coefficient.
        c2 (float): Social coefficient.
        bounds (tuple): Min and max bounds for position values.
    """
    global_best_position = np.zeros_like(particles[0].position)
    global_best_fitness = float('-inf')

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for idx, particle in enumerate(particles):
            fitness = particle.evaluate_fitness(model, X_val, y_val)
            print(f"  Particle {idx + 1}: Fitness = {fitness:.4f}")
            if fitness > particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position.copy()
        
        for particle in particles:
            particle.update_velocity(global_best_position, c1, c2)
            particle.update_position(bounds)
    
    model.set_weights(reshape_weights(global_best_position, model))
    print(f"Training complete. Best fitness: {global_best_fitness:.4f}")


def evaluate_model(model, X_val, y_val):
    """
    Evaluate the model using accuracy and F1-score.
    """
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)

    acc = accuracy_score(y_true_classes, y_pred_classes)
    f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    return y_pred_classes, y_true_classes


def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Visualize the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]}", horizontalalignment="center")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def fine_tune_model(model, X_train, y_train, X_val, y_val):
    """
    Fine-tune the model using backpropagation after PSO.
    """
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5)


def main():
    """
    Main function to run the CNN training with PSO optimization.
    """
    print("Loading data...")
    X, y = load_data('sign_mnist_train.csv')
    if X is None or y is None:
        print("Data loading failed.")
        return

    print("Preprocessing data...")
    X, y = preprocess_data(X, y)
    if X is None or y is None:
        print("Data preprocessing failed.")
        return

    print("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Creating CNN model...")
    model = create_cnn_model((28, 28, 1), 25)
    if model is None:
        print("Model creation failed.")
        return

    print("Initializing particles...")
    particles = [Particle(len(flatten_weights(model.get_weights())))
                 for _ in range(10)]
    print("Training model with PSO...")
    train_pso(model, particles, X_val, y_val, epochs=1, c1=2, c2=2,
              bounds=(-1, 1))

    print("Evaluating model on validation set...")
    y_pred_classes, y_true_classes = evaluate_model(model, X_val, y_val)

    print("Visualizing confusion matrix...")
    plot_confusion_matrix(y_true_classes, y_pred_classes, classes=range(25))

    print("Fine-tuning model...")
    fine_tune_model(model, X_train, y_train, X_val, y_val)

    print("Final evaluation...")
    y_pred_classes, y_true_classes = evaluate_model(model, X_val, y_val)
    plot_confusion_matrix(y_true_classes, y_pred_classes, classes=range(25))


if __name__ == "__main__":
    main()
