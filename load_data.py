# https://dataport.pecanstreet.org/static/static_files/New_York/1minute_data_newyork.tar.gz

# https://dataport.pecanstreet.org/static/static_files/California/1minute_data_california.tar.gz
import gzip
import requests
import os 


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


if __name__ == "__main__": 
    #Change these for different Cities or times scales
    city = "New_York"
    time_frame = "15minute"

    cityname = city.lower()
    cityname = cityname.replace('_', '') 
    # print(cityname)
    
    url =  "https://dataport.pecanstreet.org/static/static_files/" + city + "/" + time_frame + "_data_" + cityname + ".tar.gz"
    filename = time_frame + "_data_" + cityname + ".tar.gz"


    #If data is not already downloaded, downloads data
    if not os.path.isfile('metadata.csv'):
        download_metadata()
    if not os.path.isfile(filename):
        download_data(url, filename)
        print(unzip_data(filename))


