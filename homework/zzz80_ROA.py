def run_formula(dv,param=None):
    dv.add_field('tot_assets', ds)
    dv.add_field('ebit', ds)
    ROA=dv.add_formula('ROA',"TTM(ebit)/Ts_Mean(TTM(tot_assets),4)",is_quarterly=True,add_data=True)
    return ROA
