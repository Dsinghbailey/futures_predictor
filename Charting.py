import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.finance import candlestick_ochl
from Preprocessor import get_simple_mds
import time

def chart_market(mkt, mktdata):
    # Get data and format
    mktdata = mktdata[mktdata['Market'] == mkt]
    DOCHLV = zip(mktdata.DayIndex, mktdata.Open, mktdata.Close,
                 mktdata.High, mktdata.Low, mktdata.Volume)

    # Graph
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.1)
    ax = fig.add_subplot(111)
    plt.title(mkt)
    ax.autoscale_view()
    matplotlib.finance.candlestick_ochl(ax, DOCHLV, width=0.6, colorup='g', colordown='r', alpha=1.0)
    plt.show()
