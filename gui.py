from tkinter import *
from LSTM_model import *



def make_model(list_of_outputs):
    #reads data from the preprocessed csv file
    df = pd.read_csv('solar_load_weatherdata.csv')
    # list_of_outputs = ['air1', 'clotheswasher1', 'dishwasher1', 'furnace1', 'refrigerator1', 'solar']
    train_frac = .6
    val_frac = .2

    #Splits the data into train, val, and test
    train_indexes, val_indexes, test_indexes = split_data(df, train_frac, val_frac)

    #Creates the Datasets
    train_power, train_temp, train_y = create_dataset(df, indexes = train_indexes, list_of_outputs = list_of_outputs)
    val_power, val_temp, val_y = create_dataset(df, indexes = val_indexes, list_of_outputs = list_of_outputs)

    # Create Model
    model = create_LSTM_model(list_of_outputs)

    # Specify the optimizer, and compile the model with loss functions for both outputs
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    loss_dict = {}
    metrics_dict = {}
    for output in list_of_outputs:
        loss_dict[output] = 'mse'
        metrics_dict[output] = tf.keras.metrics.RootMeanSquaredError()
    model.compile(optimizer=optimizer, loss = loss_dict, metrics = metrics_dict)

    # Train the model for 100 epochs
    history = model.fit([train_power, train_temp], train_y,
                        epochs=100, batch_size=10, validation_data=([val_power, val_temp], val_y))

    # Print model summary and export to take_two_modelsummary.txt
    print(model.summary())
    with open('take_two_modelsummary.txt', 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # Save model
    # model.save_weights('./model.ckpt')

def displaySelected():
    appliances = []
    applianceSelected = lb.curselection()
    for i in applianceSelected:
        op = lb.get(i)
        appliances.append(op)
    for val in appliances:
        print(val)
    make_model(appliances)


if __name__ == "__main__":
    window = Tk() 
    window.title('Appliance Selection') 
    window.geometry('500x300')


    display = Label(window, text = "Please select which appliances you would like to be disaggregated.", font = ("Times", 14), padx = 10, pady = 10)
    display.pack() 

    lb = Listbox(window, selectmode = "multiple")
    lb.pack(padx = 10, pady = 10, expand = YES, fill = "both") 

    x =["air1", "solar", "clotheswasher1", "refrigerator1", "furnace1", "dishwasher1"]

    for item in range(len(x)): 
        lb.insert(END, x[item]) 
        # lb.itemconfig(item, bg="#bdc1d6") 

    Button(window, text="Create Model", command=displaySelected).pack()
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

    window.mainloop()





