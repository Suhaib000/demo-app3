import plotly.graph_objs as go
import random
import plotly.graph_objects as px
import pandas as pd
import numpy as np
import ssl


def f(time_period , df1):
    #you can here select the the period to change the data depending on what you select
    df1 = df1.iloc[0:-1:time_period]
    df1 = df1.reset_index()
    #calculate OBV
    OBV=[]
    OBV.append(0)
    #Loop through the data set (close price)
    for i in range(1, len(df1.close)):
        if  df1.close[i] > df1.close[i-1]:
            OBV.append(OBV[-1] + df1.Volume[i])
        elif df1.close[i] < df1.close[i-1]:
            OBV.append(OBV[-1] - df1.Volume[i])
        else:
            OBV.append(OBV[-1])
    
    df1['OBV']=OBV
    
    OBVEMA = df1['OBV'].ewm(span=24).mean()    #number of days??
    Delta = (df1['OBV'] - OBVEMA) * 0.000001
    Signal =Delta.ewm( span=24).mean()
    
    
    #EMAs & DEMAs of price
    emav1= df1['close'].ewm(span=8).mean()  #???
    emav12 =  emav1.ewm(span=8).mean()      #???
    demav1 = 2 * emav1 - emav12
    
    emav2= df1['close'].ewm(span=24).mean()  #???
    emav22 =  emav2.ewm(span=24).mean()      #???
    demav2 = 2 * emav2 - emav22
    
    #Resulting Condition on Bull or Bear
    bull = Delta > Signal 
    bear = Delta < Signal 
    superbull = bull & (demav1 > demav2 ) 
    superbear = bear & (demav1 < demav2 )  
    
    df_copy = df1.copy()
    df_copy['color'] = 'color'
    
    
    #Chop
    #=============================================================================
    
    #=============================================================================
    
    
    #Now decide colour based on result
    for index, row in df_copy.iterrows():
        if superbull[index]:
            df_copy.at[index, 'color'] = 'green'
        else:
            if superbear[index]:
                df_copy.at[index, 'color'] = 'red'
            else:
                if bear[index]:
                    df_copy.at[index, 'color'] = 'grey'
                else:
                    df_copy.at[index, 'color'] = 'teal'  
    df_green = df_copy[df_copy['color']=='green']
    df_red = df_copy[df_copy['color']=='red']
    df_grey = df_copy[df_copy['color']=='grey']
    df_teal = df_copy[df_copy['color']=='teal']
    return df_green , df_red , df_grey , df_teal





def traces(df_green , df_red , df_grey , df_teal):
    trace1 = go.Candlestick(
        x=df_green['date'],
        open=df_green['open'], high=df_green['high'],
        low=df_green['low'], close=df_green['close'],
        increasing_line_color= 'green', decreasing_line_color= 'green',name='Super Bull', hoverinfo='name+x'
    )


    trace2 = go.Candlestick(
        x=df_red['date'],
        open=df_red['open'], high=df_red['high'],
        low=df_red['low'], close=df_red['close'],
        increasing_line_color= 'red', decreasing_line_color= 'red',name='Super Bear', hoverinfo='name+x'
    )
    trace3 = go.Candlestick(
        x=df_grey['date'],
        open=df_grey['open'], high=df_grey['high'],
        low=df_grey['low'], close=df_grey['close'],
        increasing_line_color= 'grey', decreasing_line_color= 'grey',name='Bear', hoverinfo='name+x'
    )  

    trace4 = go.Candlestick(
        x=df_teal['date'],
        open=df_teal['open'], high=df_teal['high'],
        low=df_teal['low'], close=df_teal['close'],
        increasing_line_color= 'teal', decreasing_line_color= 'teal',name='Bull', hoverinfo='name+x'
    )
    return trace1 ,trace2 ,trace3 ,trace4


def plot(trace):
    
    data =trace
    
    fig=px.Figure(data=data)

    if len(trace) == 8:
        
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=list([
                    dict(label = "1 Days",
                         method = "update",
                         args = [{"visible": [True, True, True, True,False,False,False,False]}]), # hide trace(5,6,7,8)
                    dict(label = "3 Day",
                         method = "update",
                         args = [{"visible": [False, False, False, False, True, True, True, True]}]), # hide trace(1,2,3,4)

                    ])
    #                 ,bgcolor='red'
                )
            ])
    else:
        pass


    # update
    fig.update_layout(template='plotly_dark',
                      autosize=False,
#                       title="Dropdown",
                      hovermode="x unified",
                  showlegend=True,
                  xaxis=dict(title="Date"),
                  yaxis=dict(title="Price"),
                     width = 1000,
                     )
    return fig


new_path = "Historical NFT Floors - 1.xlsx"
mfers_df = pd.read_excel(new_path,sheet_name = 'mfers')
broda_df = pd.read_excel(new_path,sheet_name = 'broda')
Cool_df = pd.read_excel(new_path,sheet_name = 'Cool')

#change Fate format

mfers_df['date'] = pd.to_datetime(mfers_df.Date, format="%A, %b %d, %Y")
broda_df['date'] = pd.to_datetime(broda_df.Date, format="%A, %b %d, %Y")
Cool_df['date'] = pd.to_datetime(Cool_df.Date, format="%A, %b %d, %Y")

#find open column.

def open_(df):
    open_ = df['Floor Price']
    open_ = open_.drop(0).reset_index(drop=True)
    open_[len(open_) + 1]=df['Floor Price'].iloc[-1]
    return open_

mfers_df['open'] = open_(mfers_df)
broda_df['open'] = open_(broda_df)
Cool_df['open'] = open_(Cool_df)


#there are some spaeces between Volume valus so we need to remove them before convert this column to be float
def vol_to_folat(Volume):
    volume = []
    for i in Volume:
        if type(i)==str:
            volume.append(float(i.replace(' ', '')))
        else:
            volume.append(i)
    return volume

#mfers_df
mfers_df['low'], mfers_df['high'] ,mfers_df['close']= mfers_df['Floor Price'],mfers_df['Floor Price'],mfers_df['Floor Price']
mfers_df['Volume'] =vol_to_folat(mfers_df['Volume'])

#broda_df
broda_df['low'], broda_df['high'] ,broda_df['close']= broda_df['Floor Price'],broda_df['Floor Price'],broda_df['Floor Price']
broda_df['Volume'] =vol_to_folat(broda_df['Volume'])

#Cool_df
Cool_df['low'], Cool_df['high'] ,Cool_df['close']= Cool_df['Floor Price'],Cool_df['Floor Price'],Cool_df['Floor Price']
Cool_df['Volume'] =vol_to_folat(Cool_df['Volume'])



mfers_green , mfers_red, mfers_grey , mfers_teal = f(1,mfers_df)
trace1_mfers ,trace2_mfers ,trace3_mfers ,trace4_mfers =traces(mfers_green , mfers_red, mfers_grey , mfers_teal)
trace_mfers = [trace1_mfers ,trace2_mfers ,trace3_mfers ,trace4_mfers]
fig_mfers = plot(trace_mfers)


broda_green , broda_red, broda_grey , broda_teal = f(1,broda_df.dropna())
trace1_broda ,trace2_broda ,trace3_broda ,trace4_broda =traces(broda_green , broda_red, broda_grey , broda_teal)
trace_broda = [trace1_broda ,trace2_broda ,trace3_broda ,trace4_broda ]
fig_broda= plot(trace_broda)

Cool_green , Cool_red, Cool_grey , Cool_teal = f(1,Cool_df.dropna())
trace1_Cool ,trace2_Cool,trace3_Cool ,trace4_Cool =traces(Cool_green , Cool_red, Cool_grey , Cool_teal)
trace_Cool = [trace1_Cool ,trace2_Cool,trace3_Cool ,trace4_Cool ]
fig_Cool=plot(trace_Cool)



import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import joblib
import nltk



    
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',dbc.themes.DARKLY]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div(children=[
dash.html.Br(),
dash.html.Br(),
    html.Div([
        
        html.H1('plot mfers',style={'textAlign':'center','color':'white','font-size': '30px'}),
     html.Div([dcc.Graph(figure=fig_mfers)], className='row'),
        
        
     html.H1('plot Cool',style={'textAlign':'center','color':'white','font-size': '30px'}),   
     html.Div([dcc.Graph(figure=fig_Cool)], className='row' ),
        
        
     html.H1('plot broda',style={'textAlign':'center','color':'white','font-size': '30px'}),   
     html.Div([dcc.Graph(figure=fig_broda)], className='row'),
        ] ,className='row'),
    ])

#======================================================================================================   
app.run_server()
# if __name__ == '__main__':
#     app.run_server(debug=True)