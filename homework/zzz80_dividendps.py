def run_formula(dv,param=None):
    dv.add_field('distributable_profit_shrhder', ds)
    dv.add_field('total_share', ds)
    dividendsps=dv.add_formula('dividendps',"distributable_profit_shrhder/total_share",is_quarterly=False,add_data=True)#未直接找到数据，故用定义股东股利/总股份数来计算
    return dividendsps
