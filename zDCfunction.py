
import json
import requests
import time
import time
from datetime import datetime
import json
import random
import pytz
import calendar

def instancePrint(str):
    print("zDC:",str)

def avg(a,b):
    return (a+b)/2
def ma(lst,interval):
    if interval == 0:
        return lst
    output = []
    ma_lst = []
    for item in lst:
        ma_lst.append(item)
        if len(ma_lst) > interval:
            ma_lst.pop(0)
        result = 0
        for val in ma_lst:
            result+= float(val)
        result = result/len(ma_lst)
        output.append(result)
    return output
def access_api(msg):
    # API endpoint URL
    url = 'http://3.82.22.194:5000/api/receive'

    start_time = time.time()
    # Example 1: Sending "hello" message
    payload = {'message': msg}
    response = requests.post(url, json=payload)
    # Print the response*
    msg = response.json()['response']
    elapsed_time = (time.time() - start_time) * 1000
    instancePrint(f"Task completed in {elapsed_time} milliseconds")
    return msg
def current_time(timezone):
    time = pytz.timezone(timezone) 
    time = datetime.now(time)
    
    day = calendar.day_name[time.weekday()]
    dt = time.strftime("%Y-%m-%d %H:%M:%S")
    time = time.strftime("%H:%M:%S")

    return dt, time, day

#-----------------------------------
#I N I T I A L I Z E
#-----------------------------------

#-----------------------------------
#B O D Y
#-----------------------------------

    




def DC():
    #-----------------------------------------
    #S E T U P
    #-----------------------------------------
    startIndex = 540
    endIndex = 4140#2340
    instancePrint(access_api('access:'+str(startIndex)+":"+str(endIndex)))
    symbols = []
    with open('zDC_text_files//baseline_prices.txt', 'r') as f:
        baseline_prices = json.loads(f.read())
    prices = {}
    max_ln = 540
    with open('zDC_text_files//tickers.txt', 'r') as f:
        tickers = json.loads(f.read())
    with open('zDC_text_files//stocks_to_tickers.txt', 'r') as f:
        stocks_to_tickers = json.loads(f.read())
    with open('zDC_text_files//tickers_to_stocks.txt', 'r') as f:
        tickers_to_stocks = json.loads(f.read())
    symbols = ["dt_list"]+tickers
    timer = 5
    save = 5
    default_interval = 600
    flag = True
    #-----------------------------------------
    #I N I T I A L I Z E   P R I C E S
    #-----------------------------------------
    removed = [] 
    msg = ""
    for symbol in symbols:
        msg += ","+symbol
    msg = msg[1:]
    responses = access_api('price_list:'+msg).split(",")
    initial = {}
    for response in responses:
        response = response.split(":")
        symbol = response[0]
        if response[1] == "None":
            removed.append(symbol)
        else:
            if symbol == "dt_list":
                initial[symbol] = response[1]
            else:
                initial[symbol] = float(response[2])
    for symbol in removed:
        symbols.remove(symbol)
    for symbol in symbols:
        lst = []
        if symbol == "dt_list":
            for item in baseline_prices[symbol][:540]:
                lst.append(item)
            lst = [str(item) for item in lst]
        else:
            for item in baseline_prices[symbol][:540]:
                lst.append(item[0])
            lst = ma(lst,1)
        prices[symbol] = lst

    with open('zODworkspace//prices.txt', 'w') as f:
        json.dump(prices, f)
    #-----------------------------------------
    #B O D Y
    #-----------------------------------------
    while flag:
        dt,now,day = current_time("America/New_York")
        instancePrint(dt)
        with open('zODworkspace//prices.txt', 'r') as f:
            prices = json.loads(f.read())
        msg = ""
        for symbol in symbols:
            msg+= ","+symbol
        msg = msg[1:]
        responses = access_api('price_list:'+msg).split(",")

        for response in responses:
            response = response.split(":")
            symbol = response[0]
            lst = prices[symbol]
            if symbol == "dt_list":
                dt = response[1]
                lst.append(dt)
                if len(lst) > max_ln:
                    for i in range(len(lst)-max_ln):
                        lst.pop(0)
                prices[symbol] = lst
            else:

                bid,ask,volume = float(response[2]),float(response[4]),float(response[6])
                price = (bid+ask)/2
                lst.append(price)
                if len(lst) > max_ln:
                    for i in range(len(lst)-max_ln):
                        lst.pop(0)
                prices[symbol] = lst
        
        with open('zODworkspace//prices.txt', 'w') as f:
                json.dump(prices, f)
        time.sleep(2)