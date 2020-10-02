import numpy as np
import pandas as pd


class time_seq:
    def make_idx_mat_dict(pmat_dict):
        idx_mat_dict = {t: np.full((1, pmat_dict[t].shape[0]), True)
                        for t in pmat_dict.keys()}
        return(idx_mat_dict)

    def format_read_umi_sum(filename, t):
        umi_sum_df = pd.read_csv(filename,
                                 usecols=['gene_id', 'umi_sum'],
                                 index_col='gene_id')
        umi_sum_df = umi_sum_df.rename(columns={'umi_sum': t})
        return(umi_sum_df)
