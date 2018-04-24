

def run_formula(dv, param=None):#此因子要求时间为5年以上当时间没有足够长式，会产生大量的NAN
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


""""""