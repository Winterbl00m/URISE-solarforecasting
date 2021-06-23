# https://dataport.pecanstreet.org/static/static_files/New_York/1minute_data_newyork.tar.gz

# https://dataport.pecanstreet.org/static/static_files/California/1minute_data_california.tar.gz
import gzip
import requests
import os 
import pandas as pd
import tkinter as tk

def download_data(url, filename):
    #Download timeseries data
    data = requests.get(url)
    with open(filename, 'wb')as file:
        file.write(data.content)
    
def download_metadata():
    #Download the metadata
    data = requests.get("https://dataport.pecanstreet.org/static/metadata.csv")
    with open('metadata.csv', 'wb')as file:
        file.write(data.content)


def unzip_data(filename):
    #Unzip data
    with gzip.open(filename, 'rb') as f:
        file_content = f.read()
    return file_content

def process_data():
    #Step 3 process data
    #TODO
    pass

def get_parameters():
        window = tk.Tk() 
    window.mainloop() 
    button = tk.Button(
    text="Click me!",
    width=25,
    height=5,
    bg="blue",
    fg="yellow",
    )
    def handle_click(event):
    print("The button was clicked!")

button = tk.Button(text="Click me!")

button.bind("<Button-1>", handle_click)


if __name__ == "__main__": 



    city = "New_York"
    cityname = "newyork"
    time_frame = "15minute"
    url =  "https://dataport.pecanstreet.org/static/static_files/" + city + "/" + time_frame + "_data_" + cityname + ".tar.gz"
    filename = time_frame + "_data_" + cityname + ".tar.gz"


    #If data is not already downloaded, downloads data
    if not os.path.isfile('metadata.csv'):
        download_metadata()
    if not os.path.isfile(filename):
        download_data(url, filename)


