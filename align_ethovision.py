import pandas as pd
import numpy as np

datapath = '~/Downloads/ethovision.csv'
save_folder = '~/Downloads/'

time_offset = 0
time_column = 'Trial time'

# Reads the dataframe to figure out how many rows to skip
header_df = pd.read_csv(datapath, header=None)
lines_to_skip = int(header_df.iloc[0, 1])
# Gets the animal name
animal_name = header_df.loc[header_df[0] == 'Animal ID', 1].values[0]

# read the data again
data = pd.read_csv(datapath, skiprows=[x for x in range(lines_to_skip) if x != lines_to_skip-2])

# Get the zone columns
zone_columns = [x for x in data.columns if 'In zone' in x]

results_df = pd.DataFrame(columns=['Bout type', 'Bout start', 'Bout end'])

for column in zone_columns:
    zone_df = pd.DataFrame(columns=['Bout type', 'Bout start', 'Bout end'])
    # Separate by zone type
    zone_series = data.loc[:, column]
    # Mask for entering/exiting
    change_mask = zone_series.diff().abs() > 0
    # Entries and exits indices
    entries_index = zone_series.loc[change_mask & (zone_series == 1)].index
    exits_index = zone_series.loc[change_mask & (zone_series == 0)].index
    # Entries and exits time
    entry_times = data.loc[entries_index, time_column].values + time_offset
    exit_times = data.loc[exits_index, time_column].values + time_offset
    # Add final time if mouse never does a last exit
    if len(exit_times) < len(entry_times):
        exit_times = np.append(exit_times, data.loc[data.index[-1], time_column])
    elif len(entry_times) > len(exit_times):
        entry_times = np.append(data.loc[0, time_column], entry_times)
    # Add to zone_df
    zone_df['Bout start'] = entry_times
    zone_df['Bout end'] = exit_times
    zone_df['Bout type'] = column

    results_df = pd.concat([results_df, zone_df], axis=0)

results_df = results_df.sort_values('Bout start')
results_df.reset_index(drop=True, inplace=True)
results_df.to_csv(save_folder + animal_name + '.csv')
