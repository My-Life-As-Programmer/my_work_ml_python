from darksky import forecast
from datetime import date, timedelta

HYD = 17.387140,78.491684
API_KEY = '87bfb3fd4a1cc0c04f833cd36a607d0c'
weekday = date.today()
with forecast(API_KEY, *HYD) as boston:
    print(boston.daily.summary, end='\n---\n')
    for day in boston.daily:
        day = dict(day = date.strftime(weekday, '%a'),
                   sum = day.summary,
                   tempMin = day.temperatureMin,
                   tempMax = day.temperatureMax
                   )
        print('{day}: {sum} Temp range: {tempMin} - {tempMax}'.format(**day))
        weekday += timedelta(days=1)
