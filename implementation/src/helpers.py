import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def get_data(eth_csv_path: str, btc_csv_path: str) -> pd.DataFrame:
    """
    Reads ETH and BTC prices data from given CSV files and returns one DataFrame, where index is a
    date and columns are "ETH" and "BTC". Columns contain average of low and high prices.
    Date is day.
    """
    df_eth = pd.read_csv(eth_csv_path, skiprows=1)
    df_eth['Date'] = pd.to_datetime(df_eth['Date'])
    df_eth['price'] = (df_eth['High'] + df_eth['Low']) / 2
    df_eth = df_eth.sort_values(by='Date').reset_index(drop=True)
    df_eth = df_eth[['Date', 'price']]
    df_eth.rename(columns={'price': 'ETH'}, inplace=True)

    df_btc = pd.read_csv(btc_csv_path, skiprows=1)
    df_btc['Date'] = pd.to_datetime(df_btc['Date'])
    df_btc['price'] = (df_btc['High'] + df_btc['Low']) / 2
    df_btc = df_btc.sort_values(by='Date', ).reset_index(drop=True)
    df_btc = df_btc[['Date', 'price']]
    df_btc.rename(columns={'price': 'BTC'}, inplace=True)

    df = pd.merge(df_eth, df_btc, how='left', on='Date')
    df.dropna(inplace=True)
    df.set_index(keys='Date', inplace=True)
    return df


def plot_prices(
        df: pd.DataFrame,
        column_1: str,
        column_2: str,
        plot_type: str = 'scatter',
        save: bool = False):
    """Makes scatter plot of column_1 and column_2 prices."""
    plt.figure(figsize=(20, 8))
    if plot_type == 'scatter':
        sns.scatterplot(data=df, x=column_1, y=column_2)
    elif plot_type == 'line':
        sns.lineplot(data=df, x=df.index, y=column_1, label=column_1)
        sns.lineplot(data=df, x=df.index, y=column_2, label=column_2)
        plt.legend()
    else:
        raise NotImplementedError
    plt.xlabel(column_1, size=16)
    plt.ylabel(column_2, size=16)
    plt.title(f'{column_1} / {column_2}', size=20)
    plt.tight_layout()
    if save:
        plt.savefig(f'{column_1}_{column_2}_price_history')

    plt.show()


def plot_prices_over_time(df: pd.DataFrame):
    """Plots the correlation between ETH and BTC prices over time."""
    n = df.shape[0]
    dates = [str(p.date()) for p in df[::n // 9].index]
    colors = np.linspace(0.1, 1, n)

    plt.figure(figsize=(15, 8))
    sc = plt.scatter(
        df['ETH'], df['BTC'],
        s=30,
        c=colors,
        cmap=plt.cm.get_cmap('jet'),
        edgecolor='k',
        alpha=0.7
    )
    cb = plt.colorbar(sc)
    cb.ax.set_yticklabels(dates)
    plt.xlabel('ETH')
    plt.ylabel('BTC')
    plt.show()


def plot_regression_lines(prices_df: pd.DataFrame, regression_df: pd.DataFrame, step: int = 50):
    """Plots the correlation between ETH and BTC prices over time."""
    cm = plt.cm.get_cmap('jet')

    n = prices_df.shape[0]
    dates = [str(p.date()) for p in prices_df[::n // 9].index]
    colors = np.linspace(0.1, 1, n)

    plt.figure(figsize=(15, 8))
    sc = plt.scatter(
        prices_df['ETH'], prices_df['BTC'],
        s=30,
        c=colors,
        cmap=plt.cm.get_cmap('jet'),
        edgecolor='k',
        alpha=0.7
    )
    cb = plt.colorbar(sc)
    cb.ax.set_yticklabels(dates)
    plt.xlabel('ETH')
    plt.ylabel('BTC')

    xi = np.linspace(prices_df['ETH'].min(), prices_df['ETH'].max(), 2)
    colors_l = np.linspace(0.1, 1, len(regression_df[::step]))
    for i, row in enumerate(regression_df[::step].iterrows()):
        _, coefficients = row
        intercept, slope = coefficients['intercept'], coefficients['slope']
        plt.plot(xi, intercept + slope * xi, alpha=.8, lw=1, c=cm(colors_l[i]))

    plt.ylim(top=25000)

    plt.show()
