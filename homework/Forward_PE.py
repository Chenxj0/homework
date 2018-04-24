

def run_formula(dv,param=None):
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

"""datetime=DataFrame(columns=(0,980))
for i in range(0,980):
    datetime[i]=date"""