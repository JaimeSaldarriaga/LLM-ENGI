from pycoingecko import CoinGeckoAPI
import datetime
import pandas as pd 

def get_coin_data(days):
    cg = CoinGeckoAPI()
    bitcoin_data = cg.get_coin_market_chart_by_id(id='bitcoin', vs_currency='usd', days=days)
    return bitcoin_data['prices']


def transform_data(data):
    prices = data
    list_dicts = []
    for timestamp, price in prices:
        date_hour = datetime.datetime.fromtimestamp(timestamp / 1000)
        modify_date = date_hour.strftime('%Y-%m-%d %H:%M:%S')
        list_dicts.append({'date': modify_date, 'pricing': price})
        
    return pd.DataFrame(list_dicts)