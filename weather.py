import requests
from bs4 import BeautifulSoup
from darksky import forecast
from datetime import date, timedelta


#response = requests.get('https://api.darksky.net/forecast/87bfb3fd4a1cc0c04f833cd36a607d0c/17.387140,78.491684')
#soup = BeautifulSoup(response)
#print (response.json())


API_KEY = '87bfb3fd4a1cc0c04f833cd36a607d0c'
hyd = forecast(API_KEY,17.387140,78.491684)

print ( ((hyd['currently']['temperature']-32)*5)/9)


