# weathers.py

from flask import Flask,jsonify, render_template
from flask_restful import Api
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from dateutil.parser import parse
from sklearn.externals import joblib
import traceback
import os
from resources.todo import Todo
import itertools
import firebase_admin
from firebase_admin import credentials
import concurrent.futures
import urllib.request
from pprint import pprint
from datetime import date 
from firebase_admin import db
from ratelimit import limits, sleep_and_retry




app = Flask(__name__)
api = Api(app)

api.add_resource(Todo, "/todo/<int:id>")

@app.route('/', methods=['GET','POST'])
def view():
  
    
    clf = joblib.load("nn_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
    if clf:

        URLS = ['http://dataservice.accuweather.com/forecasts/v1/daily/5day/311399?apikey=W13g0DmUf6BbxnYDkG1mRmITeCGSBtiJ&metric=true',#Colombo
                'http://dataservice.accuweather.com/forecasts/v1/daily/5day/309721?apikey=W13g0DmUf6BbxnYDkG1mRmITeCGSBtiJ&metric=true',#Galle
                'http://dataservice.accuweather.com/forecasts/v1/daily/5day/307303?apikey=W13g0DmUf6BbxnYDkG1mRmITeCGSBtiJ&metric=true',#Kandy
                'http://dataservice.accuweather.com/forecasts/v1/daily/5day/311445?apikey=W13g0DmUf6BbxnYDkG1mRmITeCGSBtiJ&metric=true',#Kelaniya
                'http://dataservice.accuweather.com/forecasts/v1/daily/5day/311466?apikey=W13g0DmUf6BbxnYDkG1mRmITeCGSBtiJ&metric=true']#Wattala



    # Retrieve a single page and report the URL and contents
        def load_url(url, timeout):
            with urllib.request.urlopen(url, timeout=timeout) as conn:
                return conn.read()

        json_content = []
    # We can use a with statement to ensure threads are cleaned up promptly
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    # Start the load operations and mark each future with its URL
            future_to_url = {executor.submit(load_url, url, 60): url for url in URLS}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (url, exc))
                else:
                    print('%r page is %d bytes ' % (url, len(data)))
                    FIFTEEN_MINUTES = 900
            
                    @sleep_and_retry
                    @limits(calls=5, period=FIFTEEN_MINUTES)
                    def call_api(url):
                        response = requests.get(url)

                        if response.status_code != 200:
                            raise Exception('API response: {}'.format(response.status_code))
                        return response
                    
                    resp = call_api(url)
                    json_response = resp.json()
                    json_content.append(json_response)
                    pprint(json_content)
            with open('tmp/wea.json', 'w') as outfile:
                json.dump(json_content, outfile)


        with open('tmp/wea.json','r') as f_r:
          
          dataw = json.loads(f_r.read())
          cur = dir(os)
          count = 1
          content_obj = []
          content_arr = []
          
          for item in dataw:
            arr ={}
            link = item['Headline']['Link']
            #print(link)
            data = link.split("/")
            key = data[6]
            #print(key)
            arr[key] = link
            #print("arr:",arr)
            content_arr.append(arr)
            
            date1 = parse(item['Headline']['EffectiveDate'])
            dt = date1.date().strftime("%Y-%m-%d")
            global date
            today = date.today()
            global td
            td = today.strftime("%Y-%m-%d")
        
            if dt > td:
                print("data set is old")
                open('data.json', 'w').close()
            else:
                print("data set is new")

            for k in item.get('DailyForecasts'):

                    with open('tmp/locate.json','r') as json_file:
                        data = json.loads(json_file.read())
                        content_loc_arr = []
                        for i in range(len(data)):
                            arr_loc = {}
                            j=0
                            arr_loc["lat"] = data[i][j].get('GeoPosition').get('Latitude')
                            arr_loc["lon"] = data[i][j].get('GeoPosition').get('Longitude')
                            arr_loc["city_code"] = data[i][j].get('Key')
                            j=j+1
                            content_loc_arr.append(arr_loc)
        


                    dic_w_obj={}
                
                    date = parse(k['Date'])
                    dt = date.date().strftime("%m/%d/%Y")
                    wkdt = parse((date.date()).strftime("%m/%d/%Y"))
                    yr = wkdt.year
                    weekofyr= wkdt.isocalendar()[1]
                    dic_w_obj["id"]= count
                    dic_w_obj["year"] = yr
                    dic_w_obj["weekofyear"] = weekofyr
                    dic_w_obj["week_start_date"] = dt
                    link = k['Link']
                    #print(link)
                    data = link.split("/")
                    key = data[6]
                    
                    #dic_w_obj["latitude"] = lat
                    #dic_w_obj["longitude"] = lng
                    #dic_w_obj["precipitation_amt_mm"]=float(k['Rain'])
                    dic_w_obj["reanalysis_air_temp_k"] = k['Temperature']['Maximum']['Value'] +  273.15
                    #dic_w_obj["reanalysis_relative_humidity_percent"] =float(k['humidity'])
                    dic_w_obj["station_avg_temp_c"]= (float(k['Temperature']['Maximum']['Value']) + float(k['Temperature']['Minimum']['Value']))/2
                    dic_w_obj["station_diur_temp_rng_c"] = float(k['Temperature']['Maximum']['Value']) - float(k['Temperature']['Minimum']['Value'])
                    dic_w_obj["station_max_temp_c"] = float(k['Temperature']['Maximum']['Value'])
                    dic_w_obj["station_min_temp_c"] = float(k['Temperature']['Minimum']['Value'])
                    count += 1

  
                    for loc in content_loc_arr:
                        #print(loc)
                        if loc.get('city_code')==key:
                            #print(key)
                            dic_w_obj["latitude"] = loc.get('lat')
                            dic_w_obj["longitude"] = loc.get('lon')
                            break

                    #print(dic_w_obj)
                    content_obj.append(dic_w_obj)
    
        try:
            def Merge(dict1, dict2):
                res = {**dict1, **dict2}
                return res

            print("Type of content_obj", type(content_obj))

            query = pd.get_dummies(pd.DataFrame(content_obj))
            print(query)
            #query = query.reindex(columns=model_columns, fill_value=0)
            m,n=np.shape(query)
            print(m,n)
            query=query.values.reshape(75,5)
            prediction = clf.predict(np.array(query))
            predict = np.abs(prediction).astype(int)
            print(predict.shape)
       
            d = dict(enumerate(predict.flatten(), 1))
            print(d)
            n_items = dict(itertools.islice(d.items(), m))

            content_list = []
            for w_item in content_obj:
               # print("Type of w_item", type(w_item))
                dict_p = {}
                id = w_item["id"]
                dict_p["date"] = w_item["week_start_date"]
                dict_p["lat"] = w_item["latitude"]
                dict_p["lng"] = w_item["longitude"]
                dict_p["count"] = str(predict[id])
                content_list.append(dict_p)
                #print(dict_p)

            content_dic = {}
            area_dic = {}
            content_dic["week"] = dt
            content_dic["areas"] = content_list
            area_dic["area"] = content_dic
            print(area_dic)
            #print("Type of dict_p", type(dict_p))

            with open('tmp/prediction.json', 'w') as json_file:
                json.dump(area_dic, json_file)

            if (not len(firebase_admin._apps)):
            #cred = credentials.ApplicationDefault()
                cred = credentials.Certificate('./test.json')
                firebase_admin.initialize_app(cred, {
                'projectId': "medidocsl2019",
                'databaseURL' : "https://medidocsl2019.firebaseio.com/"})

            prediction_sample = db.reference("prediction")

            

            with open('tmp/prediction.json','r') as json_filep:
                datap = json.loads(json_filep.read())
                prediction_sample.push().set(datap)
                #print(datap)
                #for itemp in datap:
                    #print(itemp)
                    
            

            return jsonify({'area':[content_dic]})
        except:
            return jsonify({'trace':traceback.format_exc()})
    else:
            print('Train the model first')
            return('No model here to use')
   

	
if __name__ == '__main__':
  clf = joblib.load("nn_model.pkl")
  print('Model loaded')
  model_columns = joblib.load("model_columns.pkl")
  print('Model columns loaded')
  app.run()
