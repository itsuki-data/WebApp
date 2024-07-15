# %% Import libraries
import altair as alt
import pandas as pd
import streamlit as st
import yfinance as yf

# %% Set the page title
st.title("Stock Price App")

# %% Set sidebar
st.sidebar.header("User Input")
st.sidebar.write("Enter the stock ticker and date range:")
days = st.sidebar.slider("Number of days", 1, 365, 1)


# %% Get the stock price data
tickers = {
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "netflix": "NFLX",
}
companies = st.multiselect(
    "Select company", list(tickers.keys()), default=["apple", "microsoft"]
)


@st.cache_data
def get_data(days, tickers):
    df = pd.DataFrame()
    for company in tickers:
        tkr = yf.Ticker(tickers[company])
        hist = tkr.history(period=f"{days}mo")
        hist.index = hist.index.strftime("%Y-%m-%d")
        hist = hist[["Close"]]
        hist.columns = [company]
        hist = hist.T
        hist.index.name = "Name"
        df = pd.concat([df, hist])
    return df


try:
    df = get_data(days, tickers)
    if not companies:
        st.error("Please select at least one company.")
    else:
        data = df.loc[companies]
        st.write("Stock prices(USD)", data.sort_index())
        data = data.T.reset_index()
        data = pd.melt(data, id_vars=["Date"]).rename(
            columns={"value": "Stock Prices(USD)"}
        )
        y_max = data["Stock Prices(USD)"].max()
        chart = (
            alt.Chart(data)
            .mark_line(opacity=0.8, clip=True)
            .encode(
                x="Date:T",
                y=alt.Y(
                    "Stock Prices(USD):Q",
                    stack=None,
                    scale=alt.Scale(domain=(0, y_max)),
                ),
                color="Name:N",
            )
        )
        st.altair_chart(chart, use_container_width=True)
except:
    st.error("Error fetching data")
