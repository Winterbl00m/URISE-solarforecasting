from LSTM_model import * 

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


def plot(history, list_of_outputs):
    # the figure that will contain the plot
    fig = Figure(figsize = (5, 5))

    # adding the subplots
    loss_plot = fig.add_subplot(111)
	train_rmse_plot = fig.add_subplot(112)

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

	# plotting training rmse
	if len(list_of_outputs) > 1:
        for output in list_of_outputs:
            output_rmse = history.history[output + '_' + rmse]
            train_rmse_plot.plot(epochs, output_rmse, label= output + ' rmse')
			val_output_rmse = history.history['val' + '_' + output + '_' + rmse]
            plt.plot(epochs, val_output_rmse, label= output + ' rmse')
    else:
        for output in list_of_outputs:
            output_rmse = history.history[rmse]
            train_rmse_plot.plot(epochs, output_rmse, label= output + ' rmse')

	train_rmse_plot.title.set_text('Training RMSE over Epochs')
    train_rmse_plot.set_xlabel('Epochs')
    train_rmse_plot.set_ylabel('Loss')
    train_rmse_plot.legend()


    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig, master = window)
    canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack()


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




# the main Tkinter window
window = Tk()

# setting the title
window.title('Plotting in Tkinter')

# dimensions of the main window
window.geometry("500x500")

# button that displays the plot
plot_button = Button(master = window,
					command = plot,
					height = 2,
					width = 10,
					text = "Plot")

# place the button
# in main window
plot_button.pack()

# run the gui
window.mainloop()
