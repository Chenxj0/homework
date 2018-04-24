def run_formula(dv,param=None):
    high = dv.get_ts("high")
    low = dv.get_ts("low")
    close = dv.get_ts("close")
    ATR6=dv.add_formula('ATR6',"Ta('ATR',0,close,low,high,volume,6)",is_quarterly=False)
    return ATR6
