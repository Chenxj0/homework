

def run_formula(dv, param=None):
    dv.add_field('turnover_ratio', ds)
    DAVOL20=dv.add_formula('DAVOL20',"Ts_Mean(turnover,20)-Ts_Mean(turnover,120)",is_quarterly=False)
    return DAVOL20