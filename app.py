import dash
import yfinance as yf
import pandas as pd
import plotly.express as px
from dash import dcc, html, Input, Output, State
from model import train_arima

app = dash.Dash(__name__)
server = app.server

# ---------------- LAYOUT ----------------
app.layout = html.Div([

    # ========= LEFT PANEL (GREEN) =========
    html.Div([

        html.H3("Welcome to the Stock Dash App!", style={"color": "white"}),

        html.Label("Input stock code:", style={"color": "white"}),
        dcc.Input(id="stock-code", type="text"),

        html.Br(), html.Br(),

        html.Button("Submit", id="submit-btn"),

        html.Hr(),

        html.Label("Select Date Range:", style={"color": "white"}),
        dcc.DatePickerRange(id="date-range"),

        html.Br(), html.Br(),

        html.Button("Stock Price", id="stock-price-btn"),
        html.Button("Indicators", id="indicators-btn"),

        html.Br(), html.Br(),

        dcc.Input(
            id="forecast-days",
            type="number",
            placeholder="No. of days"
        ),

        html.Br(), html.Br(),

        html.Button("Forecast", id="forecast-btn"),

    ], style={
        "width": "25%",
        "padding": "20px",
        "backgroundColor": "#0f766e",
        "minHeight": "100vh"
    }),

    # ========= RIGHT PANEL =========
    html.Div([

        html.Div(id="description"),
        html.Hr(),

        html.Div(id="price-graph"),
        html.Div(id="ema-graph"),
        html.Div(id="forecast-content"),

    ], style={
        "width": "75%",
        "padding": "40px"
    })

], style={"display": "flex"})


# ---------------- COMPANY INFO ----------------
@app.callback(
    Output("description", "children"),
    Input("submit-btn", "n_clicks"),
    State("stock-code", "value")
)
def update_company_info(n_clicks, stock_code):
    if not n_clicks or not stock_code:
        return ""

    try:
        info = yf.Ticker(stock_code).get_info()
        return html.Div([
            html.H2(info.get("shortName", stock_code)),
            html.P(info.get("longBusinessSummary", "No description available"))
        ])
    except:
        return "Unable to fetch company info"


# ---------------- STOCK PRICE GRAPH ----------------
@app.callback(
    Output("price-graph", "children"),
    Input("stock-price-btn", "n_clicks"),
    State("stock-code", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date")
)
def show_stock_price(n, stock, start, end):
    if not n or not stock:
        return ""

    df = yf.download(stock, start=start, end=end)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.reset_index(inplace=True)

    fig = px.line(
        df,
        x="Date",
        y=["Open", "Close"],
        title="Opening and Closing Price vs Date"
    )

    return dcc.Graph(figure=fig)


# ---------------- EMA GRAPH ----------------
@app.callback(
    Output("ema-graph", "children"),
    Input("indicators-btn", "n_clicks"),
    State("stock-code", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date")
)
def show_ema(n, stock, start, end):
    if not n or not stock:
        return ""

    df = yf.download(stock, start=start, end=end)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.reset_index(inplace=True)

    df["EMA_20"] = df["Close"].ewm(span=20).mean()

    fig = px.line(
        df,
        x="Date",
        y="EMA_20",
        title="Exponential Moving Average (EMA 20)"
    )

    return dcc.Graph(figure=fig)


# ---------------- FORECAST GRAPH ----------------
@app.callback(
    Output("forecast-content", "children"),
    Input("forecast-btn", "n_clicks"),
    State("stock-code", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    State("forecast-days", "value")
)
def show_forecast(n, stock, start, end, days):
    if not n or not stock or not days:
        return ""

    df = yf.download(stock, start=start, end=end)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df.reset_index(inplace=True)

    forecast_df = train_arima(df.set_index("Date")["Close"], days)

    fig = px.line(
        forecast_df,
        x="Date",
        y="Forecast",
        title=f"Forecast for next {days} days"
    )

    return dcc.Graph(figure=fig)


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
