import numpy as np
import os

def exportLatexTableSZ2(filename, data, variables, SD, dp, columns=None, 
                        panel_names=None, panel_ind=None, notes=None, 
                        note_labels=None, noteSE=None, stars=0):
    """
    Create LaTeX table from data
    
    Args:
        filename: output filename (without extension)
        data: data matrix
        variables: row labels
        SD: whether to include standard errors (list or scalar)
        dp: decimal places (scalar or matrix)
        columns: column headers (optional)
        panel_names: panel names (optional)
        panel_ind: panel indices (optional)
        notes: notes data (optional)
        note_labels: note labels (optional)
        noteSE: note standard error indices (optional)
        stars: whether to include significance stars (0 or 1)
    """
    # Check conditions and general setup
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # Panel/Notes Setup
    if panel_names is None or len(panel_names) == 0:
        panel_count = 1
        p_ind_ref = [0]
        panel_ind = [data.shape[0]]
    else:
        # Check panel conditions
        if len(panel_names) != len(panel_ind):
            raise ValueError('Variables panel_names and panel_ind must be same length')
        
        if isinstance(SD, (int, float)):
            SD = [SD] * len(panel_ind)
        elif len(SD) != len(panel_ind):
            raise ValueError('SD must be scalar or equal to variable panel_ind in length')
        
        # Create panel indexes
        panel_cum = np.cumsum(panel_ind)
        p_ind_ref = np.concatenate([[0], panel_cum[:-1]])
        panel_count = len(panel_names)
    
    # Conditions
    if len(variables) != data.shape[0]:
        raise ValueError('Number of variables must match data rows')
    
    if isinstance(SD, (int, float)) and SD == 1 and data.shape[0] % 2 == 1:
        raise ValueError('Object "data" must have an even number of rows if it contains standard deviations')
    
    # Set decimal places
    if isinstance(dp, (int, float)):
        dp = np.full(data.shape, dp)
    elif dp.shape != data.shape:
        raise ValueError('variable dp must match data dimensions')
    
    # Set empty rows
    empty = '&' * (data.shape[1] - 1) + '\\\\'
    
    # Create stars (simplified version)
    star_nums = None
    if stars == 1:
        star_nums = np.zeros(data.shape, dtype=int)
        for kk in range(panel_count):
            if SD[kk] == 1:
                # Calculate t-statistics and significance stars
                start_idx = p_ind_ref[kk]
                end_idx = start_idx + panel_ind[kk]
                
                for i in range(start_idx, end_idx - 1, 2):
                    if i + 1 < data.shape[0]:
                        tstats = np.abs(data[i, :] / data[i + 1, :])
                        star_nums[i, :] = (tstats > 2.58).astype(int) + \
                                         (tstats > 1.96).astype(int) + \
                                         (tstats > 1.645).astype(int)
    
    # Initialize output file
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(f"{filename}.tex", 'w') as fileID:
        # Begin tabular
        fileID.write('\\begin{tabular}{l')
        for ii in range(data.shape[1]):
            fileID.write('c')
        fileID.write('}\n')
        fileID.write('\\toprule\n')
        
        # Column numbering
        for jj in range(data.shape[1]):
            fileID.write(f'& ({jj+1})')
        fileID.write('\\\\\n')
        
        # Column titles
        if columns is not None:
            for ii in range(len(columns)):
                for jj in range(data.shape[1]):
                    fileID.write(f'& {columns[ii][jj] if jj < len(columns[ii]) else ""}')
                fileID.write('\\\\\n')
        fileID.write('\\midrule\n')
        
        # Cycle over panels
        for kk in range(panel_count):
            if panel_count > 1:
                fileID.write(panel_names[kk])
                for ii in range(data.shape[1]):
                    fileID.write('&')
                fileID.write('\\\\\n')
            
            # Fill Data
            p_count = 0
            for ii in range(p_ind_ref[kk], p_ind_ref[kk] + panel_ind[kk]):
                p_count += 1
                fileID.write(variables[ii])
                
                for jj in range(data.shape[1]):
                    if np.isnan(data[ii, jj]):
                        fileID.write('&')
                    elif SD[kk] == 1 and p_count % 2 == 0:
                        # Standard errors in parentheses
                        decimal_places = int(dp[ii, jj])
                        fileID.write(f'&({data[ii, jj]:.{decimal_places}f})')
                    else:
                        # Regular values
                        decimal_places = int(dp[ii, jj])
                        value_str = f'{data[ii, jj]:.{decimal_places}f}'
                        
                        # Add significance stars if needed
                        if stars == 1 and star_nums is not None:
                            stars_str = '*' * star_nums[ii, jj]
                            value_str += stars_str
                        
                        fileID.write(f'& {value_str}')
                
                fileID.write('\\\\\n')
            
            # Empty row between panels
            fileID.write(empty + '\n')
            fileID.write('\\\\\n')
        
        # Notes
        if notes is not None and note_labels is not None:
            if len(note_labels) != len(notes):
                raise ValueError('Note labels must match note inputs')
            
            if noteSE is None:
                noteSE = []
            
            # Pad noteSE if needed
            if len(noteSE) < len(notes):
                noteSE = list(noteSE) + [np.nan] * (len(notes) - len(noteSE))
            
            counter = 0
            for ii in range(len(note_labels)):
                fileID.write(note_labels[ii])
                
                if ii in noteSE:
                    counter += 1
                    for jj in range(len(notes[ii])):
                        fileID.write(f'&({notes[ii][jj]})')
                else:
                    for jj in range(len(notes[ii])):
                        fileID.write(f'& {notes[ii][jj]}')
                
                fileID.write('\\\\\n')
            
            fileID.write('\\midrule\n')
        
        # Tear down
        fileID.write('\\bottomrule\n')
        fileID.write('\\end{tabular}\n')

    print(f"LaTeX table exported to {filename}.tex")
