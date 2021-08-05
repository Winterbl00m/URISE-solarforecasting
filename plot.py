from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from functools import partial
from tkinter import ttk

def plot_loss(history):
    # Plot training and validation loss over epochs
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)

    plt.figure()

    plt.plot(epochs, loss, color = 'blue', label='Training loss')
    plt.plot(epochs, val_loss, color = 'green', label='Validation loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def plot_train_rmse(history, list_of_outputs):
    # Plot training root-mean-squared error over epochs
    rmse = 'root_mean_squared_error'
    plt.figure()
    if len(list_of_outputs) > 1:
        epochs = range(1, len(history.history[list_of_outputs[0] + '_' + rmse]) + 1)
        for output in list_of_outputs:
            output_rmse = history.history[output + '_' + rmse]
            plt.plot(epochs, output_rmse, label= output + ' rmse')

    else:
        epochs = range(1, len(history.history[rmse]) + 1)
        for output in list_of_outputs:
            output_rmse = history.history[rmse]
            plt.plot(epochs, output_rmse, label= output + ' rmse')

    plt.title('Training RMSE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()


def plot_val_rmse(history, list_of_outputs):
    # Plot validation root-mean-squared error over epochs
    rmse = 'root_mean_squared_error'
    plt.figure()

    if len(list_of_outputs) > 1:
        epochs = range(1, len(history.history['val' + '_' + list_of_outputs[0] + '_' + rmse]) + 1)
        for output in list_of_outputs:
            output_rmse = history.history['val' + '_' + output + '_' + rmse]
            plt.plot(epochs, output_rmse, label= output + ' rmse')

    else:
        epochs = range(1, len(history.history['val' +  '_' + rmse]) + 1)
        for output in list_of_outputs:
            output_rmse = history.history['val' + '_' + rmse]
            plt.plot(epochs, output_rmse, label= output + ' rmse')


    plt.title('Validation RMSE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()

    plt.show()


def show_plots(window, history, list_of_outputs):
    # https://www.pythontutorial.net/tkinter/tkinter-notebook/ 
    # 
    # # the main Tkinter window
    # window = Tk()
    # # dimensions of the main window
    # window.geometry("750x750")
    # window.title('Test')

    notebook = ttk.Notebook(window)
    notebook.pack(pady=10, expand=True)
  
    # create frames
    frame1 = ttk.Frame(notebook, width=750, height=700)
    frame2 = ttk.Frame(notebook, width=750, height=700)
    frame3 = ttk.Frame(notebook, width=750, height=700)

    frame1.pack(fill='both', expand=True)
    frame2.pack(fill='both', expand=True)
    frame3.pack(fill='both', expand=True)

    # add frames to notebook
    notebook.add(frame1, text='Loss')
    notebook.add(frame2, text='Train RMSE')
    notebook.add(frame3, text='Val RMSE')
    
    # the figures that will contain the plots
    fig1 = Figure(figsize = (10, 10))
    fig2 = Figure(figsize = (10, 10))
    fig3 = Figure(figsize = (10, 10))

    # adding the subplots
    loss_plot = fig1.add_subplot(111)
    train_rmse_plot = fig2.add_subplot(111)
    val_rmse_plot = fig3.add_subplot(111)

    # Plot training and validation loss over epochs
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    rmse = 'root_mean_squared_error'

    # plotting loss
    loss_plot.plot(epochs, loss, color = 'blue', label='Training loss')
    loss_plot.plot(epochs, val_loss, color = 'green', label='Validation loss')
    loss_plot.title.set_text('Training and Validation Loss over Epochs')
    loss_plot.set_xlabel('Epochs')
    loss_plot.set_ylabel('Loss')
    loss_plot.legend()

	# plotting RMSE
    if len(list_of_outputs) > 1:
        for output in list_of_outputs:
            output_rmse = history.history[output + '_' + rmse]
            train_rmse_plot.plot(epochs, output_rmse, label= output + ' rmse')
            val_output_rmse = history.history['val' + '_' + output + '_' + rmse]
            val_rmse_plot.plot(epochs, val_output_rmse, label= output + ' rmse')
    else:
        for output in list_of_outputs:
            output_rmse = history.history[rmse]
            train_rmse_plot.plot(epochs, output_rmse, label= output + ' rmse')
            val_output_rmse = history.history['val' + '_' + rmse]
            val_rmse_plot.plot(epochs, val_output_rmse, label= output + ' rmse')

    train_rmse_plot.title.set_text('Training RMSE over Epochs')
    train_rmse_plot.set_xlabel('Epochs')
    train_rmse_plot.set_ylabel('Loss')
    train_rmse_plot.legend()

    val_rmse_plot.title.set_text('Validation RMSE over Epochs')
    val_rmse_plot.set_xlabel('Epochs')
    val_rmse_plot.set_ylabel('Loss')
    val_rmse_plot.legend()

    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas1 = FigureCanvasTkAgg(fig1, master = frame1)
    canvas2 = FigureCanvasTkAgg(fig2, master = frame2)
    canvas3 = FigureCanvasTkAgg(fig3, master = frame3)
    canvas1.draw()
    canvas2.draw()
    canvas3.draw()

    # placing the canvas on the Tkinter window
    canvas1.get_tk_widget().pack()
    canvas2.get_tk_widget().pack()
    canvas3.get_tk_widget().pack()

    # # run the gui
    # window.mainloop()