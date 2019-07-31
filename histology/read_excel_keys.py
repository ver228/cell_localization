#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:28:38 2019

@author: avelinojaver
"""
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

if __name__ == '__main__':
    excel_file = '/Users/avelinojaver/OneDrive - Nexus365/bladder_cancer_tils/TMA_counts/Eosinophil+and+neutrophil+counts+final+in+TMA+2.xlsx'
    excel_file = Path(excel_file)
    save_dir = excel_file.parent
    
    save_names = [
            '2Afirst104834_GT-eosinophils.csv',
            '2Bfirst104868_GT-eosinophils.csv',
            
            '2Afirst104834_GT-neutrophils.csv',
            '2Bfirst104868_GT-neutrophils.csv',
            
            
            ]
    
    for isheet in range(4):
        save_name = save_dir / save_names[isheet]
        
        df = pd.read_excel(excel_file, sheet_name=isheet, index_col=0)
    
        valid_data = defaultdict(list)
        
        prev_irow = None
        for irow, row in df.iterrows():
            if irow == irow:
                prev_irow = irow
            
            for icol, v in zip(row.index, row.values):
                k = (int(prev_irow), int(icol))
                valid_data[k].append(v)
        print(max(valid_data.keys()))
        #%%
        expected_cols = [tuple(v) for k,v in valid_data.items() if (k[1] == 1) & (len(v) > 1)]
        expected_cols = set(expected_cols)
        assert len(expected_cols) == 1
        counts_cols = list(expected_cols)[0][2:]
        
        
        #%%
        columns = ['row', 'col', 'id', *counts_cols]
        dum_vals = len(counts_cols)*[np.nan]
        
        processed_data = []
        for k, val in valid_data.items():
            if len(val) == 1 or (val[1] != val[1]):
                ss = val[0]
                ss = ss.title() if isinstance(ss, str) else 'Null'
                row2add = (*k, ss, *dum_vals)
            else:
                pid = f'{val[0]}:{val[1]}'
                
                if val[2] == 'Necrosis':
                    pid += '-necrosis'
                    row2add = (*k, pid, *dum_vals)
                else:
                    def _process_count(x):
                        if isinstance(x, str):
                            x = x.partition(' ')[0]
                        return float(x)
                    
                    row2add = (*k, pid, *[_process_count(x) for x in val[2:]])
                
                
                
        
            processed_data.append(row2add)
            
        df2save = pd.DataFrame(processed_data, columns = columns)
        df2save.to_csv(save_name, index = False)
    
