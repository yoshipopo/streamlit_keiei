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
import math 
from collections import deque
import datetime as dt
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(layout="wide")

def main():
  st.title('経営管理論　模範解答')
  st.write(sys.version)
  
  #################################################
  st.header('銘柄選択')
  
  #全銘柄リスト、xlsファイル読み込み
  path = 'data_j_2023.6.xls'
  df_all_company_list = path_to_df_all_company_list(path)
  st.write('全銘柄 : All stocks')
  st.dataframe(df_all_company_list)
  
  #銘柄選択
  selections_temp = st.multiselect('銘柄を選択してください : Please select stocks',df_all_company_list['コード&銘柄名'])
  selections = sorted(selections_temp)
  with st.expander('for developer'):
      st.write('並べ替え前(selections_temp):',selections_temp)
      st.write('並べ替え後(selections):',selections)
  
  st.write('選択した銘柄 : Selected stocks')
  
  #選択した銘柄.表示用に整理
  st.dataframe(selections_to_selected_company_list_and_selected_company_list_hyouji(df_all_company_list,selections)[0])
  selected_company_list = selections_to_selected_company_list_and_selected_company_list_hyouji(df_all_company_list,selections)[1]
  selected_company_list_hyouji = selections_to_selected_company_list_and_selected_company_list_hyouji(df_all_company_list,selections)[2]
  selected_company_list_hyouji_datenashi = selections
  with st.expander('for developer'):
      st.write('selected_company_list_hyouji:', selected_company_list_hyouji)
      st.write('selected_company_list_hyouji_datenashi:', selected_company_list_hyouji_datenashi)
  
  #パラメータ設定
  duration_start = st.date_input("データ開始日", dt.date(2022, 4, 20))
  duration_end = st.date_input("データ終了日", dt.date(2023, 4, 21))
  #st.write('データ収集一番最初の日', duration_start)
  
  #duration = st.slider('Years? : 株価取得期間は？(年)',1,5,1,)
  
  N = st.slider('Trial times of MC? : モンテカルロ法回数は？',10000,50000,10000,)

  #################################################
  #ボタン部分
  if st.button("Submit and get csv"):
    
    #################
    st.header('課題1')
    st.subheader('課題1.1')
    st.write('崩壊してなければOK')

    #################ここで株価取得．
    st.subheader('課題1.2')
    df_price_merged = selected_company_list_to_get_df(selected_company_list,selected_company_list_hyouji,duration_start,duration_end)[0]
    df_tourakuritu_merged = selected_company_list_to_get_df(selected_company_list,selected_company_list_hyouji,duration_start,duration_end)[1]
    
    st.write('取得した株価データ　： Stock Price data')
    st.dataframe(df_price_merged)

    ######株価グラフ
    a=df_price_merged
    temp_forshow=df_price_merged
    fig = go.Figure()
    for i in range(len(selected_company_list_hyouji_datenashi)):
      fig.add_trace(go.Scatter(x=a['Date'],y=a.iloc[:,i+1],name=selected_company_list_hyouji_datenashi[i]))
    fig.update_traces(hovertemplate='%{y}')
    fig.update_layout(hovermode='x')
    fig.update_layout(height=500,width=800,
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

    #with st.expander('元データ(df_price_merged)'):
      #st.dataframe(temp_forshow)
    
    _ = """
    #100に揃えた価格推移
    b=df_price_100
    fig = go.Figure()
    for i in range(len(selected_company_list_hyouji_datenashi)):
      fig.add_trace(go.Scatter(x = b['Date'],y = b.iloc[:,i+1],name = selected_company_list_hyouji_datenashi[i]))
    fig.update_traces(hovertemplate='%{y}')
    fig.update_layout(hovermode='x')
    fig.update_layout(height=500,width=800,
                      title='株価 : Stock Price({}=100)'.format(standard_date.date()),
                      xaxis={'title': 'Date'},
                      yaxis={'title': 'price'})
    fig.update_layout(showlegend=True)
    #fig.add_shape(type="line",x0=standard_date, y0=0, x1=standard_date, y1=100, line=dict(color="black",width=1))
    st.plotly_chart(fig)
    """

    ######対数収益率グラフ
    c=df_tourakuritu_merged
    fig = go.Figure()
    for i in range(len(selected_company_list_hyouji_datenashi)):
      fig.add_trace(go.Scatter(x=c['Date'],y=c.iloc[:,i+1],name=selected_company_list_hyouji_datenashi[i]))
    fig.update_traces(hovertemplate='%{y}')
    fig.update_layout(hovermode='x')
    fig.update_layout(height=500,width=800,
                      title='log-return : 対数収益率',
                      xaxis={'title': 'Date'},
                      yaxis={'title': 'log-return'})                  
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig)

    with st.expander('元データ(df_tourakuritu_merged)'):
      st.dataframe(df_tourakuritu_merged)

    ######対数収益率ヒストグラム
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
        fig.update_layout(height=500,width=800,
                          title='対数収益率のヒストグラム : Histogram of log-return',
                          xaxis={'title': 'log-return'},
                          yaxis={'title': '度数'})
    fig.update_layout(barmode='overlay')
    st.plotly_chart(fig)

    with st.expander('元データ(df_tourakuritu_merged)'):
      st.dataframe(df_tourakuritu_merged)

    
    #################
    st.subheader('課題1.3')
    #データ元は..1年使うということで,半年を1期間とする期待収益率，標準偏差、相関係数の計算．
    st.write('半年（125日)に対応する期待収益率，標準偏差、相関係数')

    #期待収益率
    #st.dataframe(df_tourakuritu_merged.drop(columns='Date')) #Date落とす.
    df_temp_expreturn = df_tourakuritu_merged.drop(columns='Date').mean()*125
    #df_temp_expreturn.rename(columns=['期待収益率'], inplace=True)
    #df_temp_expreturn.columns=['期待収益率'] #ここで列名設定している
    st.write('期待収益率 : expected log-return')
    st.dataframe(df_temp_expreturn)

    #標準偏差
    df_temp_stdev = df_tourakuritu_merged.drop(columns='Date').std()*math.sqrt(125)
    st.write('標準偏差 : standard deviation')
    st.dataframe(df_temp_stdev)

    #相関係数　correlation
    df_temp_corr = df_tourakuritu_merged.drop(columns='Date').corr()
    st.write('相関係数 : correlation')
    st.dataframe(df_temp_corr)
    
    _ = """　色付きで，綺麗に相関係数描画
    with st.expander('メモ'):
      fig_corr = px.imshow(df_temp_corr, text_auto=True, 
                           zmin=-1,zmax=1,
                           color_continuous_scale=['blue','white','red'])
      fig_corr.update_layout(height=500,width=800,
                             title='Correlation of log-return: 対数収益率の相関係数')
      st.plotly_chart(fig_corr)
      
      st.dataframe(pd.concat([df_temp_expreturn, df_temp_stdev, df_temp_corr], axis=1))
    """
    
    #あとポートフォリオ
    st.write('ポートフォリオの投資比率が100%になっていることを確認していればよい．')
    #st.number_input(selections[0],value=30)
    #st.slider('{selections[0]}の投資比率',0,100,1,)


    
    
    #################
    st.subheader('課題1.4')
    st.write('3銘柄のシャープレシオ:')
    st.write('とりあえず安全利子率を0で計算.Excelで，この値に近い計算を行なっていればOK')
    st.dataframe(df_temp_expreturn/df_temp_stdev)


    st.write('ポートフォリオのシャープレシオ:')
    st.write('ポートフォリオの期待収益率，標準偏差の計算が間違っていなければOK')
    
    

    


    ##################################
    st.header('課題2')
    
    #################
    st.subheader('課題2.1')
    st.write('課題2.2と整合的であれば良い')
    
    #################
    st.subheader('課題2.2')
    df=df_tourakuritu_merged
    df=df.drop('Date', axis=1)
    company_list_hyouji_datenashi=df.columns.values
    
    with st.expander('for developer(company_list_hyouji_datenashi)'):
      st.write('company_list_hyouji_datenashi',company_list_hyouji_datenashi)

    n=len(df.columns)
    with st.expander('for developer(n)'):
      st.write('n',n)

    def get_portfolio(array1,array2,array3):
        rp = np.sum(array1*array2)
        sigmap = array1 @ array3 @ array1
        return array1.tolist(), rp, sigmap #tolistは，nparrayをlistに変換

    df_vcm=df.cov()
    with st.expander('for developer(df_vcm)'):
      st.write('df_vcm',df_vcm) 

    a=np.ones((n,n)) #n*nの1の行列 array([[1., 1., 1.],[1., 1., 1.],[1., 1., 1.]])
    np.fill_diagonal(a,125) #np.fill_diagonal(a,len(df))
    np_vcm=df_vcm.values@a

    a=np.ones((n,n))
    np.fill_diagonal(a,125) #np.fill_diagonal(a,len(df))

    df_mean=df.mean()
    np_mean=df_mean.values
    np_mean=np_mean*125 #np_mean=np_mean*len(df)

    x=np.random.uniform(size=(N,n))   #Nは，モンテカルロ試行回数
    x/=np.sum(x, axis=1).reshape([N, 1])
    temp=np.identity(n)
    x=np.append(x,temp, axis=0) #xは3銘柄のランダムな投資比率.[0.3868,	0.4789,	0.1343]がN行存在する
    
    with st.expander('for developer(x)'):
      st.write('x',x)

    squares = [get_portfolio(x[i],np_mean,np_vcm) for i in range(x.shape[0])]
    df2 = pd.DataFrame(squares,columns=['投資比率','収益率', '収益率の分散'])
    #st.dataframe('g',df2)
    
    df2['分類']='PF{}資産で構成'.format(len(company_list_hyouji_datenashi))  #x.shape[0]は，行列xの行数を返す．[1]は列数．
    for i in range(x.shape[0]-n,x.shape[0]):
      df2.iat[i, 3] = company_list_hyouji_datenashi[i-x.shape[0]]
      #print(i,company_list_hyouji_datenashi[i-x.shape[0]])
    
    df2['収益率の標準偏差'] = np.sqrt(df2['収益率の分散'])
    #df2.drop(columns='収益率の分散', inplace=True)

    with st.expander('for developer(df2)'):
      st.write('df2',df2)
    
    #result
    fig = px.scatter(df2, x='収益率の標準偏差', y='収益率',hover_name='投資比率',color='分類')
    fig.update_layout(height=500,width=800,
                      title='Result of MC : モンテカルロシミュレーション結果',
                      xaxis={'title': 'Standard deviation of expected return : 収益率の標準偏差'},
                      yaxis={'title': 'Expected return : 期待収益率'},
                      )
    st.plotly_chart(fig)

    st.write('上図に，自己のポートフォリオの点が"正しい計算の上で"描画されていれば良い．')

    #################
    st.subheader('課題2.3')
    st.write('課題2.2と整合的であれば良い．')

    #################
    st.subheader('課題2.4')
    st.write('自己のポートフォリオがポートフォリオ理論の観点から好ましいものであったか，議論ができていれば良い．')


    #################
    st.subheader('課題2.5')
    st.write('課題2.2のグラフより，リスク（標準偏差）が減少しているはずである．そうなっていればOK')
    df=df_tourakuritu_merged
    df=df.drop('Date', axis=1)

    
    df_temp25=df.cov()
    with st.expander('for developer(df_temp25)'):
      st.write('df_temp25',df_temp25) 

    #相関係数が0ということは，共分散が0ということ．分散共分散行列(df_vcmの，対角成分以外を0にすればよい)
    diagonal_df_temp25 = np.diag(df_temp25) #df_temp25の対角成分取得
    st.write('diagonal_df_temp25',diagonal_df_temp25) 
    
    df_vcm = np.zeros((n, n),int) #全部0のn*n（銘柄数*銘柄数）の行列作成
    np.fill_diagonal(df_vcm, diagonal_df_temp25) #ここで上書き
    #st.write('df_zero',df_zero)
    #df_vcm = df_zero
    with st.expander('for developer(df_vcm)'):
      st.write('df_vcm',df_vcm) 
    
    

    a=np.ones((n,n)) #n*nの1の行列 array([[1., 1., 1.],[1., 1., 1.],[1., 1., 1.]])
    np.fill_diagonal(a,125) #np.fill_diagonal(a,len(df))
    np_vcm=df_vcm.values@a

    a=np.ones((n,n))
    np.fill_diagonal(a,125) #np.fill_diagonal(a,len(df))

    df_mean=df.mean()
    np_mean=df_mean.values
    np_mean=np_mean*125 #np_mean=np_mean*len(df)

    x=np.random.uniform(size=(N,n))   #Nは，モンテカルロ試行回数
    x/=np.sum(x, axis=1).reshape([N, 1])
    temp=np.identity(n)
    x=np.append(x,temp, axis=0) #xは3銘柄のランダムな投資比率.[0.3868,	0.4789,	0.1343]がN行存在する
    with st.expander('for developer(x)'):
      st.write('x',x)

    squares = [get_portfolio(x[i],np_mean,np_vcm) for i in range(x.shape[0])]
    df2 = pd.DataFrame(squares,columns=['投資比率','収益率', '収益率の分散'])
    #st.dataframe('g',df2)

  
    df2['分類']='PF{}資産で構成'.format(len(company_list_hyouji_datenashi))  #x.shape[0]は，行列xの行数を返す．[1]は列数．
    for i in range(x.shape[0]-n,x.shape[0]):
      df2.iat[i, 3] = company_list_hyouji_datenashi[i-x.shape[0]]
      #print(i,company_list_hyouji_datenashi[i-x.shape[0]])
    
    df2['収益率の標準偏差'] = np.sqrt(df2['収益率の分散'])
    #df2.drop(columns='収益率の分散', inplace=True)

    with st.expander('for developer(df2)'):
      st.write('df2',df2)
    
    #result
    fig = px.scatter(df2, x='収益率の標準偏差', y='収益率',hover_name='投資比率',color='分類')
    fig.update_layout(height=500,width=800,
                      title='Result of MC : モンテカルロシミュレーション結果',
                      xaxis={'title': 'Standard deviation of expected return : 収益率の標準偏差'},
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

def selected_company_list_to_get_df(selected_company_list_local,selected_company_list_hyouji_local,duration_1,duration_2):
    #スタートの日付
    end = duration_2 #dt.datetime.now()
    start = duration_1 #end-dt.timedelta(days=duration*365)
    for i in range(len(selected_company_list_local)):
        code = selected_company_list_local[i]

        stooq = StooqDailyReader(code, start=start, end=end)
        df_local = stooq.read()  # pandas.core.frame.DataFrame

        df_price_local = df_local['Close']
        df_price_local = df_price_local.reset_index()

        df_tourakuritu_local = df_local['Close']
        #これは対数収益率ではない,
        #df_tourakuritu_local = df_tourakuritu_local.pct_change(-1)
        #対数収益率
        df_tourakuritu_local = np.log(df_local['Close']) - np.log(df_local['Close'].shift(-1))
        df_tourakuritu_local = df_tourakuritu_local.reset_index()
        df_tourakuritu_local = df_tourakuritu_local.dropna()
        df_tourakuritu_local = df_tourakuritu_local.reset_index(drop=True)

        if i ==0:
          df_price_merged_local = df_price_local
          df_tourakuritu_merged_local = df_tourakuritu_local
        else:
          df_price_merged_local=pd.merge(df_price_merged_local, df_price_local, on='Date')
          df_tourakuritu_merged_local=pd.merge(df_tourakuritu_merged_local, df_tourakuritu_local, on='Date')
          
    df_price_merged_local = df_price_merged_local.set_axis(selected_company_list_hyouji_local, axis='columns')
    df_tourakuritu_merged_local = df_tourakuritu_merged_local.set_axis(selected_company_list_hyouji_local, axis='columns')
    df_price_merged_local['Date'] = df_price_merged_local['Date'].dt.round("D")
    df_tourakuritu_merged_local['Date'] = df_tourakuritu_merged_local['Date'].dt.round("D")
    return df_price_merged_local, df_tourakuritu_merged_local
  
  
if __name__ == "__main__":
    main()
