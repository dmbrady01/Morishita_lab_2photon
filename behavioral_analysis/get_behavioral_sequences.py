import pandas as pd
import argparse
import os
import numpy as np

SEQUENCE_DICT = [
    {
        'name': 'chamber_to_social',
        'sequence': ['social_chamber', 'social'],
        'Bout duration': ['>=0', '>=0'],
        'Latency to next bout start': ['>=0', '>=0'],
        'Bout start': ('1', 'Bout start'),
        'Bout end': ('1', 'Bout end')
    },
    {
        'name': 'chamber_to_object',
        'sequence': ['object_chamber', 'object'],
        'Bout duration': ['>=0', '>=0'],
        'Latency to next bout start': ['>=0', '>=0'],
        'Bout start': ('1', 'Bout start'),
        'Bout end': ('1', 'Bout end')
    },
    {
        'name': 'proper_first_social',
        'sequence': ['social_chamber', 'social'],
        'Bout duration': ['>=0', '>=5'],
        'Latency to next bout start': ['<3', '>=0'],
        'Bout start': ('1', 'Bout start'),
        'Bout end': ('2', 'Bout end')
    },
    {
        'name': 'proper_first_object',
        'sequence': ['object_chamber', 'object'],
        'Bout duration': ['>=0', '>=3'],
        'Latency to next bout start': ['<=4', '>=0'],
        'Bout start': ('1', 'Bout start'),
        'Bout end': ('2', 'Bout end')
    },
    {
        'name': 'proper_entire_social',
        'sequence': ['social_chamber', 'social'],
        'Bout duration': ['>=0', '>=5'],
        'Latency to next bout start': ['<3', '>=0'],
        'Bout start': ('1', 'Bout start'),
        'Bout end': ('last', 'Bout end')
    },
    {
        'name': 'proper_entire_object',
        'sequence': ['object_chamber', 'object'],
        'Bout duration': ['>=0', '>=3'],
        'Latency to next bout start': ['<=4', '>=0'],
        'Bout start': ('1', 'Bout start'),
        'Bout end': ('last', 'Bout end')
    }
]


class GetBehavioralSequences(object):

    def __init__(self, datapath=None, savefolder=None, sequences=SEQUENCE_DICT):
        # Setting datapath and savefolder
        self.datapath = datapath
        self.savefolder = savefolder
        self.set_savefolder()
        # simple sequence analysis
        self.sequences = sequences

    def set_datapath(self):
        if self.datapath is None:
            self.datapath = './data/ethovision.csv'

    @staticmethod
    def get_animal_name(datapath):
        return datapath.split('_')[-1].split('.')[0]

    def set_savefolder(self):
        # Will save to the base folder of the datapath (example above saves to ~/Downloads/)
        self.set_datapath()
        if self.savefolder is None:
            self.savefolder = os.sep.join(self.datapath.split(os.sep)[:-1]) + os.sep
        elif self.savefolder[-1] != os.sep:
            self.savefolder = self.savefolder + os.sep

    def save_files(self, dataset):
        for data in dataset:
            animal_name = data[0]
            df = data[1]
            df.to_csv(self.savefolder + 'behavioral_sequences_' + animal_name + '.csv',
                index=False)

    @staticmethod
    def load_data(datapath):
        return pd.read_csv(datapath)

    @staticmethod
    def sort_by_bout_start(df):
        # Combine all results and sort them
        df = df.sort_values(['Bout start', 'Bout type'], ascending=[True, False])
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def chain_columns_to_list(df, column, length, dtype=str):
        return pd.concat([df[column].astype(dtype).shift(-1*x) for x in range(length)], axis=1).values.tolist()

    @staticmethod
    def map_and_eval(value_list, eval_list):
        zipped_list = zip(value_list, eval_list)
        return all([eval(x+y) if pd.notna(x) else False for x, y in zipped_list])

    @staticmethod
    def equivalent(value_list, eval_list):
        return value_list == eval_list

    def match_sequence(self, df, column, sequence, fnc='map_and_eval'):
        iterated_sequences = self.chain_columns_to_list(df, column, len(sequence))
        if fnc == 'map_and_eval':
            mask = map(lambda x: self.map_and_eval(x, sequence), iterated_sequences)
        else:
            mask = map(lambda x: self.equivalent(x, sequence), iterated_sequences)
        return pd.Series(mask)

    def get_bout_time(self, df, column, position, sequence=None):
        if position == 'last':
            position = len(sequence)
            df = self.get_last_timing_from_bout_type(df)

        position = int(position) - 1
        return df[column].shift(-1*position)

    def get_last_timing_from_bout_type(df):
        change_bout_type_mask = df['Bout type'].ne(df['Bout type'].shift().bfill())
        change_bout_type_mask[0] = True
        grouper = change_bout_type_mask.astype(int).cumsum()
        agg_fns = {'Bout start': 'last', 'Bout end': 'last'}
        new_df = df.groupby(grouper).agg(agg_fns)
        recast = pd.merge(grouper, df.groupby(grouper).agg(agg_fns), left_on='Bout type', right_index=True).drop('Bout type', axis=1)
        return recast

    def find_sequences(df):
        pass

    # def find_simple_sequences(self, df):
    #     found_sequences = [df]
    #     for sequence in self.simple_sequences:
    #         name = sequence['name']
    #         template = sequence['sequence']
    #         num_events = len(template)
    #         mask = pd.Series(map(lambda x: x == template, pd.concat([df['Bout type'].shift(-1*x) for x in range(num_events)], axis=1).values.tolist()))
    #         new_df = df.loc[mask, :].copy()
    #         new_df['Bout type'] = name
    #         found_sequences.append(new_df)
    #     df = pd.concat(found_sequences, axis=0)
    #     df = self.sort_by_bout_start(df)
    #     return df

    def run(self):
        df = self.load_data(self.datapath)
        animal_name = self.get_animal_name(self.datapath)
        df = self.find_sequences(df)
        dataset = [(animal_name, df)]
        self.save_files(dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datapath', type=str, default='./data/ethovision.csv',
        help='Path to ethovision csv.'
    )
    parser.add_argument(
        '--savefolder', type=str, default=None,
        help='Folder to save data to'
    )
    args = parser.parse_args()
    datapath = args.datapath 
    savefolder = args.savefolder 
    
    event_parser = GetBehavioralSequences(
                                        datapath=datapath, 
                                        savefolder=savefolder
                                    )
    event_parser.run()