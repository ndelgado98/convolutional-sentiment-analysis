from keras.callbacks.callbacks import History
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

### NLP Exploratory Visualisations ###




### Neural Net Visualisations ###

def plot_loss(history_dict: dict or History) -> None:
    '''Takes a dictionary or keras history object and plots training and validation loss over epochs
    '''
    
    if isinstance(history_dict, History):
        history_dict = history_dict.history
    elif isinstance(history_dict, dict): 
        pass
    else:
        raise TypeError
    
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    
    epochs = range(1, len(loss_values) + 1)
    
    plt.plot(epochs, loss_values, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()
    
    
def plot_accuracy(history_dict: dict or History) -> None:
    '''Takes a dictionary or keras history object and plots training and validation accuracy over epochs
    '''
    
    if isinstance(history_dict, History):
        history_dict = history_dict.history
    elif isinstance(history_dict, dict): 
        pass
    else:
        raise TypeError
    
    accuracy_values = history_dict['categorical_accuracy']
    val_accuracy_values = history_dict['val_categorical_accuracy']
    
    epochs = range(1, len(accuracy_values) + 1)
    
    plt.plot(epochs, accuracy_values, 'ro', label='Training Accuracy')
    plt.plot(epochs, val_accuracy_values, 'r', label='Validation Accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()