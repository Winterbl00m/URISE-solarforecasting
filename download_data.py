import gzip
import requests
import os 
import pandas as pd

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


def unzip_data(filename,csv_filename):
    #Unzip data
    file_content = pd.read_csv(filename, compression='gzip', header=0, sep=' ', quotechar='"', error_bad_lines=False, encoding='latin1')
    # os.remove(filename)
    file_content.to_csv(csv_filename, index=False)
    return file_content


if __name__ == "__main__": 
    #Change these for different Cities or times scales
    city = "Austin"
    time_frame = "15minute"

    cityname = city.lower()
    cityname = cityname.replace('_', '') 
    # print(cityname)
    
    url =  "https://dataport.pecanstreet.org/static/static_files/" + city + "/" + time_frame + "_data_" + cityname + ".tar.gz"
    filename = time_frame + "_data_" + cityname + ".tar.gz"
    csv_filename = time_frame + "_data_" + cityname + ".csv"


    #If data is not already downloaded, downloads data
    # if not os.path.isfile('metadata.csv'):
    #     download_metadata()
    if not os.path.isfile(filename):
        download_data(url, filename)

    print(unzip_data(filename, csv_filename).head())


