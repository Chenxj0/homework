def run_formula(dv,param=None):
    dv.add_field('tot_oper_cost', ds)
    dv.add_field('inventories', ds)
    InvTRate=dv.add_formula('InvTRate',"TTM(tot_oper_cost)/Ts_Mean(inventories,4)",is_quarterly=True,add_data=True)
    return InvTRate
