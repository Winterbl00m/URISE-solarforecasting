
from tkinter import *

window = Tk() 
window.title('Appliance Selection') 
window.geometry('500x300')

var = StringVar()

def displaySelected():
    appliances = []
    applianceSelected = lb.curselection()
    for i in applianceSelected:
        op = lb.get(i)
        appliances.append(op)
    for val in appliances:
        print(val)

display = Label(window, text = "Please select which appliances you would like to be disaggregated.", font = ("Times", 14), padx = 10, pady = 10)
display.pack() 
lb = Listbox(window, selectmode = "multiple")
lb.pack(padx = 10, pady = 10, expand = YES, fill = "both") 

x =["air1", "solar", "clotheswasher1", "refrigerator1", "furnace1", "dishwasher1"]
for item in range(len(x)): 
	lb.insert(END, x[item]) 
	lb.itemconfig(item, bg="#bdc1d6") 

Button(window, text="Create Model", command=displaySelected).pack()
window.mainloop()

"""""
from tkinter import *

window = Tk()
window.title('Appliance Selection')
window.geometry('400x300')
window.config(bg='white')

def showSelected():
    itm = lb.get(lb.curselection())
    var.set(itm)


var =StringVar()
lb = Listbox(window)
lb.pack()

lb.insert(0, 'air1')
lb.insert(1, 'solar')
lb.insert(2, 'clotheswasher1')
lb.insert(3, 'refrigerator1')
lb.insert(4, 'furnace1')
lb.insert(5, 'dishwasher1')

disp = Label(window, textvariable=var)
disp.pack(pady=20)
Button(window, text='Create Model', command=showSelected).pack()

window.mainloop()






import tkinter as tk

#Open window
window = tk.Tk()
window.title("Appliance Selection")
#Asking Question
#label = tk.Label(text='Which appliances are you inquiring about?', fg='black', bg='white')


def showSelected():
    itm = lb.get(lb.curselection())
    var.set(itm)

lb = tk.Listbox(window)
lb.pack()
lb.insert(0, 'red')
lb.insert(1, 'green')
lb.insert(2, 'yellow')
lb.insert(3, 'blue')

disp = tk.Label(window, textvariable=var)
tk.Button(window, text='Show Selected', command=showSelected).pack(pady=20)
show = tk.Label(window)
show.pack()



button = tk.Button(
    text="air1",
    width=25,
    height=5,
    bg="blue",
    fg="white",
)
button = tk.Button(
    text="solar!",
    width=25,
    height=5,
    bg="blue",
    fg="white",
)
button = tk.Button(
    text="clotheswasher1",
    width=25,
    height=5,
    bg="blue",
    fg="white",
)
button = tk.Button(
    text="refrigerator1",
    width=25,
    height=5,
    bg="blue",
    fg="white",
)
button = tk.Button(
    text="furnace1",
    width=25,
    height=5,
    bg="blue",
    fg="white",
)
button = tk.Button(
    text="dishwasher1",
    width=25,
    height=5,
    bg="blue",
    fg="white",
)
button = tk.Button(
    text="Create Model",
    width=25,
    height=5,
    bg="green",
    fg="white",
)

#Listbox
l = tk.Listbox( parent , height=10)

choices = ["air1", "solar", "clotheswasher1", "refrigerator1", "furnace1", "dishwasher1"]
choicesvar = tk.StringVar(value=choices)
l = tk.Listbox( parent , listvariable=choicesvar)
choicesvar.set(choices)

#if Listbox.selection_includes("air1")

Listbox.curselection
"""

