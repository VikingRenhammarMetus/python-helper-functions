import datetime
import pytz

import matplotlib.pyplot as plt

import tensorflow_hub as hub

from tensorflow.keras.callbacks import TensorBoard as TensorBoardCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_tensorboard_callback(dir_name, experiment_name, timezone = "Europe/Stockholm"):
    """Construct a TensorBoard callback to be used in the TensorFlow fit() function.

    Args:
        dir_name (str): Directory for TensorBoard to use for logging data.
        experiment_name (str): Sub-directory for saving individual runs. All experiments under the same dir_name will be shown together in TensorBoard.
        timezone (str, optional): Time zone to use when appending the datetime to the experiment folder. Defaults to "Europe/Stockholm".

    Returns:
        tf.keras.callbacks.TensorBoard: The TensorBoard callback.
    """
    dt = datetime.datetime.now(datetime.timezone.utc).astimezone(pytz.timezone(timezone)).strftime("%Y_%m_%d-%H:%M:%S")
    log_dir = f"{dir_name}/{experiment_name}/{dt}/"
    tensorboard_callback = TensorBoardCallback(log_dir = log_dir)
    print(f"Saving TensorBoard log files to {log_dir}.")
    return tensorboard_callback

def create_feature_extraction_model(feature_extractor_url, n_classes, input_shape):
    """Creates a model, using the given feature extractor as the first N layers and adds a Dense output layer with the given number of classes.

    Args:
        feature_extractor_url (str): The TensorFlow Hub URL to fetch the model from.
        n_classes (int, optional): The number of neurons in the output layer. Defaults to 10.
        input_shape (list[int], optional): Shape of the input data to the feature extractor.

    Returns:
        tensorflow.keras.models.Sequential: The Sequential model combining the feature extractor with a Dense output layer.
    """
    feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                            trainable = False,
                                            name = "feature_extraction_layer",
                                            input_shape = input_shape)
    model = Sequential([feature_extractor_layer,
                        Dense(n_classes, activation = "softmax", name = "output_layer")])
    return model

def plot_metrics(history, metrics = ['loss', 'accuracy'], include_validation_data = True):
    """Plots each metric in a separate figure.

    Args:
        history (dict[list[Number]]): 
        metrics (list, optional): _description_. Defaults to ['loss', 'accuracy'].
        include_validation_data (bool, optional): _description_. Defaults to True.
    """
    epochs = range(len(history[[metrics[0]]]))

    for metric in metrics:
        plt.figure()
        plt.plot(epochs, history[metric], label = f"Training {metric}")
        if include_validation_data:
            plt.plot(epochs, history[f"val_{metric}"], label = f"Validation {metric}")
        plt.xlabel("Epochs")
        plt.legend()
    
    plt.show()