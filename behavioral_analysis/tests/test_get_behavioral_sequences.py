import unittest
from mock import patch
import os
import numpy as np
import pandas as pd
from behavioral_analysis.get_behavioral_sequences import GetBehavioralSequences, SEQUENCE_DICT

class TestSimpleSequenceDict(unittest.TestCase):

    def test_class_of_bout_type_dict(self):
        # Make sure BOUT_TYPE_DICT is a list of dictionaries
        self.assertIsInstance(SEQUENCE_DICT, list)
        self.assertTrue(all([isinstance(x, dict) for x in SEQUENCE_DICT]))

    def test_keys_of_each_dict(self):
        self.assertTrue(all(['sequence' in x.keys() for x in SEQUENCE_DICT]))
        self.assertTrue(all(['name' in x.keys() for x in SEQUENCE_DICT]))
        self.assertTrue(all(['Bout duration' in x.keys() for x in SEQUENCE_DICT]))
        self.assertTrue(all(['Latency to next bout start' in x.keys() for x in SEQUENCE_DICT]))
        self.assertTrue(all(['Bout start' in x.keys() for x in SEQUENCE_DICT]))
        self.assertTrue(all(['Bout end' in x.keys() for x in SEQUENCE_DICT]))

class TestGetBehavioralSequences(unittest.TestCase):

    def test_initialize_properly(self):
        e = GetBehavioralSequences(
                                datapath='a', 
                                savefolder='b',
                                sequences={'a'} 
                                )
        self.assertEqual(e.datapath, 'a')
        self.assertEqual(e.savefolder, 'b/')
        self.assertEqual(e.sequences, {'a'})

    def test_set_datapath(self):
        default_datapath = './data/ethovision.csv'
        self.assertEqual(GetBehavioralSequences().datapath, default_datapath)

    def test_get_animal_name(self):
        datapath = './etho_123.csv'
        self.assertEqual(GetBehavioralSequences().get_animal_name(datapath), '123')

    def test_set_save_folder_default(self):
        e = GetBehavioralSequences(datapath=None)
        e.set_savefolder()
        default_datapath = './data/ethovision.csv'
        default_save_folder = './data/'
        self.assertEqual(e.datapath, default_datapath)
        self.assertEqual(e.savefolder, default_save_folder)

    def test_set_save_folder_add_os_seop(self):
        e = GetBehavioralSequences(savefolder='/abc')
        e.set_savefolder()
        default_save_folder = '/abc/'
        self.assertEqual(e.savefolder, default_save_folder)

    @patch('pandas.DataFrame.to_csv')
    def test_save_files(self, mock_to_csv):
        e = GetBehavioralSequences(
                                savefolder='/path'
                                )
        data = [
                ('123', pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})), 
                ('456', pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]}))
                ]
        e.save_files(data)
        check_file = '/path' + os.sep + 'behavioral_sequences_123.csv'
        check_file2 = '/path' + os.sep + 'behavioral_sequences_456.csv'
        mock_to_csv.assert_any_call(check_file, index=False)
        mock_to_csv.assert_any_call(check_file2, index=False)

    @patch('pandas.read_csv')
    def test_load_data(self, mock_read_csv):
        mock_read_csv.return_value = 123
        df = GetBehavioralSequences().load_data('/path/to.csv')
        self.assertEqual(df, 123)
        mock_read_csv.assert_called_with('/path/to.csv')

    def test_sort_by_bout_start(self):
        data = {
                    'Bout start': [2, 1, 2],
                    'Bout type': ['a', 'b', 'c'],
                    'Bout end': [1, 4, 10]
        }
        df = pd.DataFrame(data, index=[0,1,2])
        ordered_data = {
                    'Bout start': [1, 2, 2],
                    'Bout type': ['b', 'c', 'a'],
                    'Bout end': [4, 10, 1]
        }
        ordered_df = pd.DataFrame(ordered_data, index=[0, 1,2])
        pd.testing.assert_frame_equal(GetBehavioralSequences().sort_by_bout_start(df), ordered_df)

    def test_chain_columns_to_list(self):
        data = {
            'Bout type': ['1', '2', '3', '4']
        }
        df = pd.DataFrame(data)
        check = [
            [1., 2., 3.],
            [2., 3., 4.],
            [3., 4., np.nan],
            [4., np.nan, np.nan]
        ]
        results = GetBehavioralSequences().chain_columns_to_list(df, 'Bout type', 3, dtype=float)
        # Need to test frames because nans don't match for lists
        pd.testing.assert_frame_equal(pd.DataFrame(check), pd.DataFrame(results))

    def test_map_and_eval(self):
        eval_list = ['>=3', '<5', '>100', '==10']
        value_list = ['4', '1', '10', '10']
        value2_list = ['4', '4', '101', '10']
        value3_list = ['4', '4', np.nan, '4']
        self.assertEqual(GetBehavioralSequences().map_and_eval(value_list, eval_list),
            False)
        self.assertEqual(GetBehavioralSequences().map_and_eval(value2_list, eval_list),
            True)
        self.assertEqual(GetBehavioralSequences().map_and_eval(value3_list, eval_list),
            False)

    def test_equivalent(self):
        self.assertTrue(GetBehavioralSequences().equivalent(['a', 'b'], ['a', 'b']))
        self.assertFalse(GetBehavioralSequences().equivalent(['a', 'b'], ['a', 'c']))

    def test_match_sequence_for_bout_types(self):
        data = {
            'Bout type': ['a', 'b', 'c', 'd', 'b', 'c']
        }
        df = pd.DataFrame(data)
        check = pd.Series([False, True, False, False, False, False])
        results = GetBehavioralSequences().match_sequence(df, 'Bout type', 
            ['b', 'c', 'd'], fnc='equivalent')
        pd.testing.assert_series_equal(results, check)

    def test_match_sequence_for_durations(self):
        data = {
            'Bout duration': [4, 5, 6, 2, 1, 3]
        }
        df = pd.DataFrame(data)
        check = pd.Series([False, True, False, False, False, False])
        results = GetBehavioralSequences().match_sequence(df, 'Bout duration', 
            ['==5', '==6', '==2'], 'map_and_eval')
        pd.testing.assert_series_equal(results, check)

    def test_get_bout_time(self):
        data = {
            'duration': [1, 1, 2, 3, 5]
        }
        df = pd.DataFrame(data)
        results = GetBehavioralSequences().get_bout_time(df, 'duration', 2)
        pd.testing.assert_series_equal(results, pd.Series([1, 2, 3, 5, np.nan], name='duration'))

    # def test_find_simple_sequences(self):
    #     sequence_dict = [
    #         {
    #             'sequence': ['a', 'a'],
    #             'name': 'aa'
    #         },
    #         {
    #             'sequence': ['a', 'b', 'c'],
    #             'name': 'abc'
    #         }
    #     ]
    #     data = {
    #         'Bout type': ['a', 'a', 'b', 'c', 'd'],
    #         'Bout start': [1, 2, 3, 4, 5],
    #         'Bout end': [2, 3, 4, 5, 6]
    #     }
    #     check_data = {
    #         'Bout type': ['aa', 'a', 'abc', 'a', 'b', 'c', 'd'],
    #         'Bout start': [1, 1, 2, 2, 3, 4, 5],
    #         'Bout end': [2, 2, 3, 3, 4, 5, 6]
    #     }
    #     df = pd.DataFrame(data)
    #     check = pd.DataFrame(check_data)
    #     e = GetBehavioralSequences(simple_sequences=sequence_dict)
    #     results = e.find_simple_sequences(df)
    #     pd.testing.assert_frame_equal(results, check)

    @patch.object(GetBehavioralSequences, 'load_data', return_value='a')
    @patch.object(GetBehavioralSequences, 'get_animal_name', return_value='b')
    @patch.object(GetBehavioralSequences, 'find_sequences', return_value='c')
    @patch.object(GetBehavioralSequences, 'save_files', return_value='c')
    def test_run(self, mock_save, mock_seq, mock_name, mock_load):
        e = GetBehavioralSequences()
        e.run()
        mock_load.assert_called_with(e.datapath)
        mock_name.assert_called_with(e.datapath)
        mock_seq.assert_called_with('a')
        mock_save.assert_called_with([('b', 'c')])