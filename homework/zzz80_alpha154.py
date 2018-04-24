


def run_formula(dv, param=None):
       alpha_154 = dv.add_formula('alpha_154',
                            " (((vwap- Min(vwap, 16))) < (Correlation(vwap, Ts_Mean(volume,180), 18)))".format(6)
                            , is_quarterly=False)
       return alpha_154