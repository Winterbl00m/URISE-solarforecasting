from tkinter import *
from LSTM_model import *


def displaySelected(window, lb):
    appliances = []
    applianceSelected = lb.curselection()
    for i in applianceSelected:
        op = lb.get(i)
        appliances.append(op)
    frame1.destroy()   
    history = make_model(appliances)
    show_plots(window, history, appliances)


if __name__ == "__main__":
    window = Tk() 
    # window.title('Appliance Selection') 
    window.geometry('750x750')

    frame1 = Frame(window)
    frame1.pack(side="top", expand=True, fill="both")

    label1 = Label(frame1, text = "Please select which appliances you would like to be disaggregated.", font = ("Times", 14), padx = 10, pady = 10)
    label1.pack() 

    lb = Listbox(frame1, selectmode = "multiple")
    lb.pack(padx = 10, pady = 10, expand = YES, fill = "both") 

    x =["air1", "solar", "clotheswasher1", "refrigerator1", "furnace1", "dishwasher1"]
    for item in x: 
        lb.insert(END, item) 

    button1 = Button(frame1, text="Create Model", command=partial(displaySelected, window, lb)).pack()

    window.mainloop()





