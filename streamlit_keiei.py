#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 17:51:47 2022

@author: harukiyoshida
"""

import streamlit as st
import sys
from pandas_datareader.stooq import StooqDailyReader
import pandas as pd
import numpy as np
from collections import deque
import datetime as dt
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide")

def main():
  st.title('Monte Carlo simulation')
  st.write(sys.version)
  
  #st.snow()
  
  #全銘柄リスト、xlsファイル読み込み
  path = 'data_j_2023.6.xls'
  df_all_company_list = path_to_df_all_company_list(path)
  st.write('All stocks 全銘柄')
  st.dataframe(df_all_company_list)
  
  #銘柄選択
  st.write('Please select stocks : 銘柄を選択してください')
  selections = st.multiselect('',df_all_company_list['コード&銘柄名'],)
  st.write('Selected stocks : 選択した銘柄')
  
  #選択した銘柄表示
  st.dataframe(selections_to_selected_company_list_and_selected_company_list_hyouji(df_all_company_list,selections)[0])
  selected_company_list = selections_to_selected_company_list_and_selected_company_list_hyouji(df_all_company_list,selections)[1]
  selected_company_list_hyouji = selections_to_selected_company_list_and_selected_company_list_hyouji(df_all_company_list,selections)[2]
  selected_company_list_hyouji_datenashi = selections
  
  #パラメータ設定
  duration = st.slider('Years? : 株価取得期間は？(年)',1,5,1,)
  N = st.slider('Trial times of MC? : モンテカルロ法回数は？',10000,50000,10000,)
  
  #ボタン部分
  if st.button("Submit and get csv"):
    
    df_price_merged = selected_company_list_to_get_df(selected_company_list,selected_company_list_hyouji,duration)[0]
    df_tourakuritu_merged = selected_company_list_to_get_df(selected_company_list,selected_company_list_hyouji,duration)[1]

    st.write('株価データ : Stock price data')
    st.dataframe(df_price_merged)
    
    a=df_price_merged
    fig = go.Figure()
    for i in range(len(selected_company_list_hyouji_datenashi)):
      fig.add_trace(go.Scatter(x=a['Date'],y=a.iloc[:,i+1],name=selected_company_list_hyouji_datenashi[i]))
    fig.update_traces(hovertemplate='%{y}')
    fig.update_layout(hovermode='x')
    fig.update_layout(height=500,width=1000,
                      title='株価 : Stock Price',
                      xaxis={'title': 'Date'},
                      yaxis={'title': 'price/円'})                  
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig)
    
    
    standard_date_tentative  = (0,0)    
    standard_date_tentative2 = len(df_price_merged) -1  -standard_date_tentative[0]
    standard_date = df_price_merged.iat[standard_date_tentative2,0]
    df_price_100 = df_price_merged
    for i in range(len(selected_company_list_hyouji_datenashi)):
      df_price_100[selected_company_list_hyouji_datenashi[i]]=100*df_price_100[selected_company_list_hyouji_datenashi[i]]/df_price_100.at[df_price_100.index[standard_date_tentative2], selected_company_list_hyouji_datenashi[i]]
    
    #100に揃えた価格推移
    b=df_price_100
    fig = go.Figure()
    for i in range(len(selected_company_list_hyouji_datenashi)):
      fig.add_trace(go.Scatter(x = b['Date'],y = b.iloc[:,i+1],name = selected_company_list_hyouji_datenashi[i]))
    fig.update_traces(hovertemplate='%{y}')
    fig.update_layout(hovermode='x')
    fig.update_layout(height=500,width=1000,
                      title='株価 : Stock Price({}=100)'.format(standard_date.date()),
                      xaxis={'title': 'Date'},
                      yaxis={'title': 'price'})
    fig.update_layout(showlegend=True)
    #fig.add_shape(type="line",x0=standard_date, y0=0, x1=standard_date, y1=100, line=dict(color="black",width=1))
    st.plotly_chart(fig)

    #st.dataframe(df_tourakuritu_merged)

    #ヒストグラム
    fig = go.Figure()
    for i in range(len(selected_company_list_hyouji_datenashi)):
        fig.add_trace(go.Histogram(x=df_tourakuritu_merged.iloc[:,i+1],
                                   xbins=dict(start=-0.5, end=0.5, size=0.002),
                                   opacity=0.5,
                                   name='{}'.format(selected_company_list_hyouji_datenashi[i]),
                                   #nbinsx=50
                                   #histnorm='probability',
                                   #hovertext='date{}'.df_tourakuritu_merged.iloc[:,i+1]
                                   ))
        fig.update_layout(height=500,width=1500,
                          title='Histogram of return per day : 日時収益率のヒストグラム',
                          xaxis={'title': '日時収益率'},
                          yaxis={'title': '度数'})

    fig.update_layout(barmode='overlay')
    st.plotly_chart(fig)
        
    #correlation
    fig_corr = px.imshow(df_tourakuritu_merged.drop('Date', axis=1).corr(), text_auto=True, 
                         zmin=-1,zmax=1,
                         color_continuous_scale=['blue','white','red'])
    fig_corr.update_layout(height=500,width=1000,
                           title='Correlation of return per day : 日時収益率の相関係数'
                           )
    st.plotly_chart(fig_corr)
    
    
    #MC------------------------------------------------------------------------------------
    df=df_tourakuritu_merged
    df=df.drop('Date', axis=1)
    company_list_hyouji_datenashi=df.columns.values
    #st.write(company_list_hyouji_datenashi)

    n=len(df.columns)
    #st.write(n)

    def get_portfolio(array1,array2,array3):
        rp = np.sum(array1*array2)
        sigmap = array1 @ array3 @ array1
        return array1.tolist(), rp, sigmap

    df_vcm=df.cov()

    a=np.ones((n,n))
    np.fill_diagonal(a,len(df))
    np_vcm=df_vcm.values@a

    a=np.ones((n,n))
    np.fill_diagonal(a,len(df))

    df_mean=df.mean()
    np_mean=df_mean.values
    np_mean=np_mean*len(df)

    x=np.random.uniform(size=(N,n))
    x/=np.sum(x, axis=1).reshape([N, 1])

    temp=np.identity(n)
    x=np.append(x,temp, axis=0)

    squares = [get_portfolio(x[i],np_mean,np_vcm) for i in range(x.shape[0])]
    df2 = pd.DataFrame(squares,columns=['投資比率','収益率', '収益率の分散'])

    df2['分類']='PF{}資産で構成'.format(len(company_list_hyouji_datenashi))
    for i in range(x.shape[0]-n,x.shape[0]):
      df2.iat[i, 3] = company_list_hyouji_datenashi[i-x.shape[0]]
      #print(i,company_list_hyouji_datenashi[i-x.shape[0]])

    #st.dataframe(df2)
    
    #result
    fig = px.scatter(df2, x='収益率の分散', y='収益率',hover_name='投資比率',color='分類')
    fig.update_layout(height=500,width=1000,
                      title='Result of MC : モンテカルロシミュレーション結果',
                      xaxis={'title': 'Variance of expected return : 期待収益率の分散'},
                      yaxis={'title': 'Expected return : 期待収益率'},
                      )
    st.plotly_chart(fig)

        
def path_to_df_all_company_list(path):
    df_all_company_list = pd.read_excel(path)
    df_all_company_list = df_all_company_list.replace('-', np.nan)
    df_all_company_list['コード&銘柄名'] = df_all_company_list['コード'].astype(str)+df_all_company_list['銘柄名']
    return df_all_company_list
  
def selections_to_selected_company_list_and_selected_company_list_hyouji(df_all_company_list,selections):
    df_meigarasenntaku_temp = df_all_company_list[df_all_company_list['コード&銘柄名'].isin(selections)]
    selected_company_list = [str(i)+'.JP' for i in df_meigarasenntaku_temp['コード']]
    d = deque(selections)
    d.appendleft('Date')
    selected_company_list_hyouji = list(d)
    return df_meigarasenntaku_temp, selected_company_list, selected_company_list_hyouji

def selected_company_list_to_get_df(selected_company_list,selected_company_list_hyouji,duration):
    end = dt.datetime.now()
    start = end-dt.timedelta(days=duration*365)
    for i in range(len(selected_company_list)):
        code = selected_company_list[i]

        stooq = StooqDailyReader(code, start=start, end=end)
        df = stooq.read()  # pandas.core.frame.DataFrame

        df_price = df['Close']
        df_price = df_price.reset_index()

        df_tourakuritu = df['Close']
        df_tourakuritu = df_tourakuritu.pct_change(-1)
        df_tourakuritu = df_tourakuritu.reset_index()
        df_tourakuritu = df_tourakuritu.dropna()
        df_tourakuritu = df_tourakuritu.reset_index(drop=True)

        if i ==0:
          df_price_merged = df_price
          df_tourakuritu_merged = df_tourakuritu
        else:
          df_price_merged=pd.merge(df_price_merged, df_price, on='Date')
          df_tourakuritu_merged=pd.merge(df_tourakuritu_merged, df_tourakuritu, on='Date')
          
    df_price_merged = df_price_merged.set_axis(selected_company_list_hyouji, axis='columns')
    df_tourakuritu_merged = df_tourakuritu_merged.set_axis(selected_company_list_hyouji, axis='columns')
    df_price_merged['Date'] = df_price_merged['Date'].dt.round("D")
    df_tourakuritu_merged['Date'] = df_tourakuritu_merged['Date'].dt.round("D")
    return df_price_merged, df_tourakuritu_merged
  
  
if __name__ == "__main__":
    main()
