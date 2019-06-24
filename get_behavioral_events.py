import pandas as pd
import numpy as np
import os
import re
import argparse

class GetBehavioralEvents(object):

    def __init__(self, datapath=None, savefolder=None, time_offset=0, 
        time_column='Trial time', minimum_bout_time=1, dtype='ethovision',
        name_match=r'\d{5,}-\d*', zone_match=r'zone'):
        # Timing info
        self.time_offset = time_offset
        self.time_column = time_column
        self.minimum_bout_time = minimum_bout_time
        # Datapaths
        self.savefolder = savefolder
        self.datapath = datapath
        self.dtype = dtype
        # Animal info
        self.name_match = name_match

    def set_datapath(self):
        if self.datapath is None:
            self.datapath = './data/ethovision.csv'

    def set_savefolder(self):
        # Will save to the base folder of the datapath (example above saves to ~/Downloads/)
        self.set_datapath()
        if self.savefolder is None:
            self.savefolder = os.sep.join(self.datapath.split(os.sep)[:-1]) + os.sep

    def save_files(self, dataset):
        for data in dataset:
            animal_name = data[0]
            df = data[1]

            df.to_csv(self.savefolder + self.dtype + '_' + animal_name + '.csv', 
                index=False)

    def prune_minimum_bouts(self, df):
        return df.loc[df['Bout end'] - df['Bout start'] >= self.minimum_bout_time]

    @staticmethod
    def clean_and_strip_string(string, sep=' '):
        sep.join(string.split())

    def process_ethovision(self):
        # Reads the dataframe to figure out how many rows to skip
        header_df = pd.read_csv(self.datapath, header=None)
        lines_to_skip = int(header_df.iloc[0, 1])
        # Gets the animal name
        animal_name = header_df.loc[header_df[0] == 'Animal ID', 1].values[0]

        # read the data again
        data = pd.read_csv(self.datapath, skiprows=[x for x in range(lines_to_skip) if x != lines_to_skip-2])

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
            entry_times = data.loc[entries_index, self.time_column].values + self.time_offset
            exit_times = data.loc[exits_index, self.time_column].values + self.time_offset
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

        # Combine all results and sort them
        results_df = results_df.sort_values('Bout start')
        results_df.reset_index(drop=True, inplace=True)

        # Make sure bouts are above minimum threshold
        results_df = self.prune_minimum_bouts(results_df)

        return [(animal_name, results_df)]

    def process_anymaze(self):
        with open(self.datapath, 'r') as fp:
            contents = fp.readlines()

        animal_regex = re.compile(self.name_match)
        zone_regex = re.compile(r'zone')
        milkshake_loc_regex = re.compile(r'^(Left|Right)')
        time_regex = re.compile(r'secs')

        animals = set([x.split()[0] for x in contents if bool(animal_regex.search(x))])
        datadict = {}
        for animal in animals:
            datadict[animal] = pd.DataFrame(columns=['Bout type', 'Bout start', 'Bout end', 'Milkshake Location'])

        zone = None
        milkshake_loc = None
        animal = None
        time = None

        for line in contents:
            # Check zone
            if bool(zone_regex.search(line)):
                zone = self.clean_and_strip_string(line)

            # Check milkshake location
            elif bool(milkshake_loc_regex.search(line)):
                milkshake_loc = self.clean_and_strip_string(line)

            # Check animal location
            elif bool(animal_regex.search(line)):
                animal = self.clean_and_strip_string(line)

    def run(self):
        self.set_datapath()
        self.set_savefolder()
        if self.dtype == 'ethovision':
            dataset = self.process_ethovision()
        elif self.dtype == 'anymaze':
            dataset = self.process_anymaze()
        self.save_files(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datapath', type=str, default='./data/ethovision.csv',
        help='Path to ethnovision csv.'
    )
    parser.add_argument(
        '--savefolder', type=str, default=None,
        help='Folder to save data to'
    )
    parser.add_argument(
        '--time-column', type=str, default='Trial time',
        help='Column in dataframe with times',
    )
    parser.add_argument(
        '--time-offset', type=float, default=0,
        help='How much time to offset all bouts (to align with imaging data)',
    )
    parser.add_argument(
        '--minimum-bout-time', type=float, default=0,
        help='minimum length of time for a bout',
    )
    parser.add_argument(
        '--datatype', type=str, default='ethovision',
        help='datatype (ethovision, anymaze, etc.)'
        )
    args = parser.parse_args()
    datapath = args.datapath 
    savefolder = args.savefolder 
    time_offset = args.time_offset 
    time_column = args.time_column 
    minimum_bout_time = args.minimum_bout_time
    dtype = args.datatype
    
    event_parser = GetBehavioralEvents(
                                        datapath=datapath, 
                                        savefolder=savefolder, 
                                        time_offset=time_offset, 
                                        time_column=time_column, 
                                        minimum_bout_time=minimum_bout_time,
                                        dtype=dtype
                                    )
    event_parser.run()