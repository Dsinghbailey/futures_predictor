import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np


# Function for drawing Candlesticks from simple mds
def draw_mkt_chart(mkt, mktdata):
    # Get data and format
    mktdata = mktdata[mktdata['Market'] == mkt]
    DOCHLV = zip(mktdata.DayIndex, mktdata.Open, mktdata.Close,
                 mktdata.High, mktdata.Low, mktdata.Volume)
    # Graph
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.1)
    ax = fig.add_subplot(111)
    plt.title(mkt, fontsize=18)
    plt.xlabel('Day Index', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    ax.autoscale_view()
    matplotlib.finance.candlestick_ochl(ax, DOCHLV, width=0.6,
                                        colorup='g', colordown='r', alpha=1.0)
    plt.show()


# Function for drawing histograms from simple mds
def draw_mkt_histo(mkt, mktdata):
    # Get data and format
    mktdata = mktdata[mktdata['Market'] == mkt]
    mkt_change = mktdata.Close - mktdata.Open

    # the histogram of the data
    n, bins, patches = plt.hist(mkt_change, 50,
                                normed=1, facecolor='blue', alpha=0.75)

    # add a 'best fit' line
    y = mlab.normpdf(bins, np.mean(mkt_change),
                     np.std(mkt_change))
    plt.plot(bins, y, 'r--', linewidth=1)
    # Plot hist
    plt.xlabel('Size', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Histogram of %s' % mkt)
    plt.grid(True)
    plt.show()
