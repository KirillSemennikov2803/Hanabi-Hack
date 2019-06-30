from flask import Flask
from flask import render_template
from pandas import read_csv, DataFrame
import statsmodels.api as sm
import statsmodels as statsmodels
from statsmodels.iolib.table import SimpleTable
from sklearn.metrics import r2_score
import ml_metrics as metrics
import matplotlib as plt
from tbats import TBATS, BATS
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
#Метод Holt#########
def setGraf12(data):
    Graf12 ="var ctx = document.getElementById('12').getContext('2d');" \
        " var myChart = new Chart(ctx, { type: 'doughnut'," \
        " data: { labels: ['2011', '2012', '2013', '2014', '2015', '2016','2017', '2018', '2019', '2020', '2021', '2022', '2023']," \
        " datasets: [{ data: [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]," \
        " backgroundColor: [ 'rgba(0, 0, 255, 0.7)', 'rgba(0, 40 , 255, 0.7)', 'rgba(0, 80, 255, 0.7)'," \
        " 'rgba(0, 120, 255, 0.7)', 'rgba(0, 160, 255, 0.7)', 'rgba(0, 200, 255, 0.7)', 'rgba(0, 240, 255, 0.7)'," \
        " 'rgba(0, 255, 255, 0.7)', 'rgba(0, 255, 0, 0.4)', 'rgba(0, 255, 0, 0.5)', 'rgba(0, 255, 0, 0.6)', 'rgba(0, 255, 0, 0.7)', 'rgba(0, 255, 0, 0.8)' ], borderWidth: 1 }] }, options: { scales: { yAxes: [{" \
            " ticks: { beginAtZero: true } }] } } });"%(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9],data[10],data[11],data[12])
    f = open("static/js/12.js",'w')
    f.write(Graf12)
def setGraf13(data):
    Graf13 ="var ctx = document.getElementById('13').getContext('2d'); var myChart = new Chart(ctx, { type: 'bar', data: { labels: ['2011', '2012', '2013', '2014', '2015', '2016','2017', '2018', '2019', '2020', '2021', '2022', '2023'], datasets: [{ label: 'WT Average Wages', " \
            "data: [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]," \
            " backgroundColor: [ 'rgba(0, 0, 255, 0.7)', 'rgba(0, 40 , 255, 0.7)', 'rgba(0, 80, 255, 0.7)', 'rgba(0, 120, 255, 0.7)', 'rgba(0, 160, 255, 0.7)', 'rgba(0, 200, 255, 0.7)', 'rgba(0, 240, 255, 0.7)', 'rgba(0, 255, 255, 0.7)', 'rgba(0, 255, 0, 0.4)', 'rgba(0, 255, 0, 0.5)', 'rgba(0, 255, 0, 0.6)', 'rgba(0, 255, 0, 0.7)', 'rgba(0, 255, 0, 0.8)' ], borderWidth: 1 }] }, options: { scales: { yAxes: [{ ticks: { beginAtZero: true } }] } }" \
            " });"%(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9],data[10],data[11],data[12])
    f = open("static/js/13.js",'w')
    f.write(Graf13)
def setGraf14(data):
    Graf14 =" var ctx = document.getElementById('14').getContext('2d'); var myChart = new Chart(ctx, { type: 'polarArea', data: { labels: ['2011', '2012', '2013', '2014', '2015', '2016','2017', '2018', '2019', '2020', '2021', '2022', '2023']," \
            " datasets: [{ data: [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]" \
            ", backgroundColor: [ 'rgba(0, 0, 255, 0.7)', 'rgba(0, 40 , 255, 0.7)', 'rgba(0, 80, 255, 0.7)', 'rgba(0, 120, 255, 0.7)', 'rgba(0, 160, 255, 0.7)', 'rgba(0, 200, 255, 0.7)', 'rgba(0, 240, 255, 0.7)', 'rgba(0, 255, 255, 0.7)', 'rgba(0, 255, 0, 0.4)', 'rgba(0, 255, 0, 0.5)', 'rgba(0, 255, 0, 0.6)', 'rgba(0, 255, 0, 0.7)', 'rgba(0, 255, 0, 0.8)' ], borderWidth: 1 }] }, options: { scales: { yAxes: [{ ticks: { beginAtZero: true } }] } } })" \
            ";"%(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9],data[10],data[11],data[12])
    f = open("static/js/14.js",'w')
    f.write(Graf14)
app = Flask(__name__)


@app.route('/')
def hello_world():
    dataset = pd.read_csv('count_people.csv')
    train = dataset
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
    data = []
    for i in dataset['col']:
        data.append(int(i))
    test = dataset[-5:]
    y_hat_avg = test.copy()
    fit1 = Holt(np.asarray(train['col'])).fit(smoothing_level=0.3, smoothing_slope=0.1)
    y_hat_avg['Holt_linear'] = fit1.forecast(len(test))
    for i in y_hat_avg['Holt_linear']:
        data.append(int(i))
    setGraf12(data)

    #кОЛИЧЕСВО ЧЕЛОВЕК
    dataset = pd.read_csv('money.csv')
    train = dataset
    data = []
    for i in dataset['col']:
        data.append(int(i))
    test = dataset[-5:]
    y_hat_avg = test.copy()
    fit1 = Holt(np.asarray(train['col'])).fit(smoothing_level=0.3, smoothing_slope=0.1)
    y_hat_avg['Holt_linear'] = fit1.forecast(len(test))
    for i in y_hat_avg['Holt_linear']:
        data.append(int(i))
    setGraf13(data)
    #Деньги
    dataset = pd.read_csv('passagers.csv')
    train = dataset
    data = []
    for i in dataset['col']:
        data.append(int(i))
    test = dataset[-5:]
    y_hat_avg = test.copy()
    fit1 = Holt(np.asarray(train['col'])).fit(smoothing_level=0.3, smoothing_slope=0.1)
    y_hat_avg['Holt_linear'] = fit1.forecast(len(test))
    for i in y_hat_avg['Holt_linear']:
        data.append(int(i))
    print(data)
    setGraf14(data)
    return render_template('index.html',first_graf_link = "/",second_graf_link ="/second",title = "Holt")

@app.route('/autorization')
def autorize():
    return render_template('autorization.html')

@app.route('/second')
def second():
    return render_template('graph2.html ',first_graf_link = "/",second_graf_link ="/second",title = "Holt")
@app.route('/Winter_first')
def winter_first():
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
    dataset = pd.read_csv('count_people.csv')
    train = dataset
    data = []
    for i in dataset['col']:
        data.append(int(i))
    test = dataset[-5:]
    y_hat_avg = test.copy()
    fit1 = ExponentialSmoothing(np.asarray(train['col']), trend='add', ).fit()
    y_hat_avg['Holt_linear'] = fit1.forecast(len(test))
    for i in y_hat_avg['Holt_linear']:
        data.append(int(i))
    setGraf12(data)
    dataset = pd.read_csv('money.csv')
    train = dataset
    data = []
    for i in dataset['col']:
        data.append(int(i))
    test = dataset[-5:]
    y_hat_avg = test.copy()
    fit1 = ExponentialSmoothing(np.asarray(train['col']), trend='add', ).fit()
    y_hat_avg['Holt_linear'] = fit1.forecast(len(test))
    for i in y_hat_avg['Holt_linear']:
        data.append(int(i))
    setGraf13(data)
    dataset = pd.read_csv('passagers.csv')
    train = dataset
    data = []
    for i in dataset['col']:
        data.append(int(i))
    test = dataset[-5:]
    y_hat_avg = test.copy()
    fit1 = ExponentialSmoothing(np.asarray(train['col']), trend='add', ).fit()
    y_hat_avg['Holt_linear'] = fit1.forecast(len(test))
    for i in y_hat_avg['Holt_linear']:
        data.append(int(i))
    setGraf14(data)
    return render_template('index.html',first_graf_link = "/Winter_first",second_graf_link ="/Winter_second",title = "Holt-Winter")
@app.route('/Winter_second')
def winter_second():
    return render_template('graph2.html ',first_graf_link = "/Winter_first",second_graf_link ="/Winter_second",title = "Holt-Winter")

@app.route('/Tbat_first')
def Tbat_first():
    from tbats import TBATS, BATS
    dataset = pd.read_csv('count_people.csv')
    train = dataset
    data = []
    for i in dataset['col']:
        data.append(int(i))
    test=dataset[-5:]
    estimator = TBATS(seasonal_periods=(2, 2))
    model = estimator.fit(train['col'])
    y_forecast = model.forecast(steps=5)
    for i in y_forecast:
        data.append(int(i))
    setGraf12(data)
    dataset = pd.read_csv('money.csv')
    train = dataset
    data = []
    for i in dataset['col']:
        data.append(int(i))
    test = dataset[-5:]
    estimator = TBATS(seasonal_periods=(2, 2))
    model = estimator.fit(train['col'])
    y_forecast = model.forecast(steps=5)
    for i in y_forecast:
        data.append(int(i))
        print(int(i))
    setGraf13(data)
    dataset = pd.read_csv('passagers.csv')
    train = dataset
    data = []
    for i in dataset['col']:
        data.append(int(i))
    test = dataset[-5:]
    estimator = TBATS(seasonal_periods=(2, 2))
    model = estimator.fit(train['col'])
    y_forecast = model.forecast(steps=5)
    for i in y_forecast:
        data.append(int(i))
    setGraf14(data)
    return render_template('index.html',first_graf_link = "/Tbat_first",second_graf_link ="/Tbat_second",title = "Tbat")
@app.route('/Tbat_second')
def Tbat_second():
    return render_template('graph2.html ',first_graf_link = "/Tbat_first",second_graf_link ="/Tbat_second",title = "Tbat")
if __name__ == '__main__':
    app.run()
