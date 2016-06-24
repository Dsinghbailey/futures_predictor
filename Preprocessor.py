import pandas as pd
import time
from datetime import datetime


def get_mds():
    # Read
    mds = pd.read_csv('DailySummaries.csv')

    # Select columns
    mds = mds[['Date', 'Market', 'Open', 'High', 'Low', 'Close', 'Volume', 'ATR10']]

    # Drop some markets
    mds = mds[(mds['Market'] != 'VX') & (mds['Market'] != 'FOAT')]

    # Reformat Date
    mds['Date'] = mds['Date'].map(lambda x: datetime.strptime(x, '%m/%d/%Y'))

    # Limit date
    mds = mds[mds['Date'] > datetime(2006, 1, 1)]
    mds = mds[mds['Date'] < datetime(2016, 1, 1)]

    # Sort
    mds = mds.sort_values(by=['Market', 'Date'])

    '''Carry forward data when there is a holiday by reindexing dataframe
    according to market with most dates'''
    # Get lookup table for dates and market index
    val_counts = pd.value_counts(mds['Market'].values, sort=True)
    mkt_alldays = val_counts[val_counts == val_counts.max()].index[0]
    all_dates = mds[mds['Market'] == mkt_alldays]['Date']
    all_dates = all_dates.reset_index(drop=True)
    all_dates = all_dates.reset_index(['Date'], drop=False)
    all_dates['Date2Num'] = [time.mktime(date.timetuple()) for date in all_dates.Date]
    all_dates.set_index(['Date2Num'], drop=False, inplace=True)
    all_dates = all_dates['index']
    all_dates = all_dates.to_dict()

    # Apply lookup table across df
    mds.set_index(['Date'], drop=False, inplace=True)
    mds['DayIndex'] = [time.mktime(date.timetuple()) for date in mds.Date]
    mds['DayIndex'] = mds['DayIndex'].replace(all_dates)

    # Drop dates not contained in lookup
    mds = mds[mds['DayIndex'] < 20000]

    # Get master index
    mkt_names = list(mds.Market.unique())
    mds['MasterIndex'] = mds['DayIndex'] + len(all_dates)*mds.Market.map(lambda x : mkt_names.index(x))
    mds.set_index('MasterIndex', drop=True, inplace=True)
    mds = mds.reset_index('MasterIndex')
    # fill holidays
    # OPCL = yesterday's close; volume = 0
    mds['Volume'].fillna(0, inplace=True)
    mds['Close'].fillna(method='ffill', inplace=True)
    mds['ATR10'].fillna(method='ffill', inplace=True)
    mds['Open'].fillna(mds['Close'], inplace=True)
    mds['High'].fillna(mds['Close'], inplace=True)
    mds['Low'].fillna(mds['Close'], inplace=True)

    return mds


def get_simple_mds():
    mds = get_mds()

    # Select Columns and return
    mds = mds[['Market', 'DayIndex', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return mds


def get_full_mds():

    mds = get_mds()

    # Select Columns
    mds = mds[['Market', 'DayIndex', 'Open', 'High',
               'Low', 'Close', 'Volume', 'ATR10']]

    # Add Features
    mds['YOpen'] = mds.groupby(by='Market')['Open'].shift(1)
    mds['YHigh'] = mds.groupby(by='Market')['High'].shift(1)
    mds['YLow'] = mds.groupby(by='Market')['Low'].shift(1)
    mds['YClose'] = mds.groupby(by='Market')['Close'].shift(1)
    mds['YVolume'] = mds.groupby(by='Market')['Volume'].shift(1)
    mds['Gap'] = (mds['Open'] - mds['YClose'])/mds['ATR10']
    mds['YChange'] = (mds['YClose'] - mds['YOpen'])/mds['ATR10']
    mds['YRange'] = (mds['YHigh'] - mds['YLow'])/mds['ATR10']
    mds['OpenMA10'] = (mds['Open'] - mds['Open'].rolling(window=10, min_periods=1).mean())/mds['ATR10']
    mds['OpenMA30'] = (mds['Open'] - mds['Open'].rolling(window=30, min_periods=1).mean())/mds['ATR10']
    mds['OpenMA50'] = (mds['Open'] - mds['Open'].rolling(window=50, min_periods=1).mean())/mds['ATR10']

    # Add labels
    mds['Up'] = (mds['Close'] - mds['Open']) > 0

    # Remove perfect knowledge
    mds.drop(['Close', 'High', 'Low', 'Volume'], axis=1, inplace=True)

     return mds


def get_scaled_mds():
    mds = get_full_mds()
    mds = mds[['Market', 'DayIndex', 'Gap', 'YChange', 'YRange',
               'OpenMA10', 'OpenMA30', 'OpenMA50', 'ATR10', 'Up']]
    return mds


def get_test_mds():
    mds = get_mds()
    # Select Columns
    mds = mds[['Market', 'DayIndex', 'Open', 'High',
               'Low', 'Close', 'Volume', 'ATR10']]
    # Add Features
    mds['Change'] = (mds['Close'] - mds['Open'])/mds['Open']
    mds['YOpen'] = mds.groupby(by='Market')['Open'].shift(1)
    mds = mds.dropna()
    # Add labels
    mds['Up'] = (mds['Close'] - mds['Open']) > 0

    # Remove perfect knowledge
    return mds


# Unit Test
def unit_test():
    print get_simple_mds().head(10)
