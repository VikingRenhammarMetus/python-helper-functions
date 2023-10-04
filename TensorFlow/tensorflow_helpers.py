import datetime
import pytz

import matplotlib.pyplot as plt

import itertools
import numpy as np
from sklearn.metrics import confusion_matrix

from tensorflow.keras.callbacks import TensorBoard as TensorBoardCallback

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

def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
  """
  From: https://github.com/mrdbourke/tensorflow-deep-learning/blob/main/extras/helper_functions.py
  
  Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """  
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)
  
  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Save the figure to the current working directory
  if savefig:
    fig.savefig("confusion_matrix.png")

def plot_metrics(history, metrics = ['loss', 'accuracy'], include_validation_data = True):
    """Plots each metric in a separate figure.

    Args:
        history (dict[list[Number]]): 
        metrics (list, optional): _description_. Defaults to ['loss', 'accuracy'].
        include_validation_data (bool, optional): _description_. Defaults to True.
    """
    epochs = range(len(history[metrics[0]]))

    for metric in metrics:
        plt.figure()
        plt.plot(epochs, history[metric], label = f"Training {metric}")
        if include_validation_data:
            plt.plot(epochs, history[f"val_{metric}"], label = f"Validation {metric}")
        plt.xlabel("Epochs")
        plt.legend()
    
    plt.show()