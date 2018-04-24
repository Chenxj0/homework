# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 11:08:10 2018

@author: xinger
"""

#--------------------------------------------------------
#import

import os
import numpy as np
import pandas as pd
import jaqs_fxdayu
jaqs_fxdayu.patch_all()
from jaqs.data import DataView
from jaqs_fxdayu.data.dataservice import LocalDataService

import warnings
warnings.filterwarnings("ignore")

#--------------------------------------------------------
#define

start = 20170101
end = 20180101
factor_list  = ['BBI','RVI','Elder','ChaikinVolatility','EPS','PE','PS','ACCA','CTOP','MA10RegressCoeff12','AR','BR','ARBR','np_parent_comp_ttm','total_share','bps']
check_factor = ','.join(factor_list)

dataview_folder = r'E:/data'
ds = LocalDataService(fp = dataview_folder)

SH_id = ds.query_index_member("000001.SH", start, end)
SZ_id = ds.query_index_member("399106.SZ", start, end)
stock_symbol = list(set(SH_id)|set(SZ_id))

dv_props = {'start_date': start, 'end_date': end, 'symbol':','.join(stock_symbol),
         'fields': check_factor,
         'freq': 1,
         "prepare_fields": True}

dv = DataView()
dv.init_from_config(dv_props, data_api=ds)
dv.prepare_data()

def alpha_154():
    alpha_154 = dv.add_formula('alpha_154',
                            " (((vwap- Min(vwap, 16))) < (Correlation(vwap, Ts_Mean(volume,180), 18)))".format(6)
                            , is_quarterly=False, add_data=True)
    return alpha_154


def InvtRate():
    dv.add_field('tot_oper_cost', ds)
    dv.add_field('inventories', ds)
    InvtRate=dv.add_formula('InvtRate', "TTM(tot_oper_cost)/Ts_Mean(inventories,4)", is_quarterly=True, add_data=True)
    return InvtRate

def ATR6():
    high = dv.get_ts("high")
    low = dv.get_ts("low")
    close = dv.get_ts("close")
    ATR6=dv.add_formula('ATR6', "Ta('ATR',0,close,low,high,volume,6)", is_quarterly=False, add_data=True)
    return ATR6

def dividendps():
    dv.add_field('distributable_profit_shrhder', ds)
    dv.add_field('total_share', ds)
    dividendps=dv.add_formula('dividendps', "distributable_profit_shrhder/total_share", is_quarterly=False,
                   add_data=True)  # 未直接找到数据，故用定义股东股利/总股份数来计算
    return dividendps

def DAVOL20():
    dv.add_field('turnover_ratio', ds)
    DAVOL20=dv.add_formula('DAVOLA20', "Ts_Mean(turnover,20)-Ts_Mean(turnover,120)", is_quarterly=False, add_data=True)
    return DAVOL20

def ROA():
    dv.add_field('tot_assets', ds)
    dv.add_field('ebit', ds)
    ROA=dv.add_formula('ROA', "TTM(ebit)/Ts_Mean(TTM(tot_assets),4)", is_quarterly=True, add_data=True)
    return ROA





def Forward_PE
    dv.add_field('ebit', ds)
    dv.add_field('total_share', ds)
    total_share=dv.get_ts('total_share')
    date=list(total_share.index)
    for i in range(0,488):
        date[i]=parse(str(date[i]))
        date[i]=date[i].month
        if date[i]%4==0:
            date[i]=4
        else:
            date[i] =  (date[i]%4)*4
    datetime_index=DataFrame.copy(total_share)
    for i in range(0,981):
        datetime_index.iloc[:,i]=date
    datetime_index = datetime_index.astype('float64')
    Forward_PE=dv.append_df(datetime_index,'datetime')
    dv.add_formula('Forward_PE',"total_share*close/TTM(ebit)*datetime",is_quarterly=True,add_data=True)

def SGRO_cpt():#此因子要求时间为5年以上当时间没有足够长式，会产生大量的NAN
    dv.add_field('total_oper_rev', ds)
    total_oper_rev=dv.get_ts('total_oper_rev')
    time_list=list(range(1,1221))
    time=DataFrame.copy(total_oper_rev)
    for i in range(0,981):
        time.iloc[:,i]=time
    time = time.astype('float64')
    dv.append_df(time, 'time')
    SGRO_cpt=dv.add_formula('SGRO_cpt',"Covariance(TTM(total_oper_rev),time,60)/Covariance(time,time,60)/Abs(Ts_Mean(TTM(total_oper_rev),60))",is_quarterly=True,add_data=True)
    return SGRO_cpt

factor_list=['alpha_154','InvtRate','ATR6','dividendps','DAVOL20','ROA','SGRO_cpt','Forward_PE']


def test(factor, data):
    if not isinstance(data, pd.core.frame.DataFrame):
        raise TypeError('On factor {} ,output must be a pandas.DataFrame!'.format(factor))
    else:
        try:
            index_name = data.index.names[0]
            columns_name = data.index.names[0]
        except:
            if not (index_name in ['trade_date', 'report_date'] and columns_name == 'symbol'):
                raise NameError(
                    '''Error index name,index name must in ["trade_date","report_date"],columns name must be "symbol" ''')

        index_dtype = data.index.dtype_str
        columns_dtype = data.columns.dtype_str

        if columns_dtype not in ['object', 'str']:
            raise TypeError('error columns type')

        if index_dtype not in ['int32', 'int64', 'int']:
            raise TypeError('error index type')


test_factor = False

if test_factor:
    for factor in factor_list[5:]:
        data = globals()[factor]()
        test(factor, data)
