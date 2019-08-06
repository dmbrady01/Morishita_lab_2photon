import unittest
import tempfile
import shutil
import pandas as pd
import os
from mock import patch
from collections import OrderedDict

from behavioral_analysis.get_behavioral_events import GetBehavioralEvents, BOUT_TYPE_DICT, STIMULUS_NAME_SET

class TestBoutTypeDict(unittest.TestCase):

    def test_class_of_bout_type_dict(self):
        # Make sure BOUT_TYPE_DICT is a list of dictionaries
        self.assertIsInstance(BOUT_TYPE_DICT, list)
        self.assertTrue(all([isinstance(x, dict) for x in BOUT_TYPE_DICT]))

    def test_keys_of_each_dict(self):
        self.assertTrue(all(['location' in x.keys() for x in BOUT_TYPE_DICT]))
        self.assertTrue(all(['zone' in x.keys() for x in BOUT_TYPE_DICT]))
        self.assertTrue(all(['name' in x.keys() for x in BOUT_TYPE_DICT]))

class TestStimulusNameSet(unittest.TestCase):

    def test_stimulus_name_set_is_a_set(self):
        self.assertIsInstance(STIMULUS_NAME_SET, set)

class TestGetBehavioralEvents(unittest.TestCase):
    "Tests for the GetBehavorialEvents object."

    def setUp(self):
        self.dirpath = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dirpath)

    def test_initialization_attributes(self):
        e = GetBehavioralEvents(
                                datapath='a', 
                                savefolder='b', 
                                time_offset='c', 
                                time_column='d', 
                                minimum_bout_time='e', 
                                datatype='f', 
                                name_match='g', 
                                max_session_time='h', 
                                label_dict='i', 
                                offset_datapath='j', 
                                fp_datapath='k', 
                                stimulus_name_set='l', 
                                latency_threshold='m'
                                )
        self.assertEqual(e.datapath, 'a')
        self.assertEqual(e.savefolder, 'b/')
        self.assertEqual(e.time_offset, 'c')
        self.assertEqual(e.time_column, 'd')
        self.assertEqual(e.minimum_bout_time, 'e')
        self.assertEqual(e.datatype, 'f')
        self.assertEqual(e.name_match, 'g')
        self.assertEqual(e.max_session_time, 'h')
        self.assertEqual(e.label_dict, 'i')
        self.assertEqual(e.offset_datapath, 'j')
        self.assertEqual(e.fp_datapath, 'k')
        self.assertEqual(e.stimulus_name_set, 'l')
        self.assertEqual(e.latency_threshold, 'm')

    def test_set_datapath(self):
        e = GetBehavioralEvents(datapath=None)
        e.set_datapath()
        default_datapath = './data/ethovision.csv'
        self.assertEqual(e.datapath, default_datapath)

    def test_set_save_folder_default(self):
        e = GetBehavioralEvents(datapath=None)
        e.set_savefolder()
        default_datapath = './data/ethovision.csv'
        default_save_folder = './data/'
        self.assertEqual(e.datapath, default_datapath)
        self.assertEqual(e.savefolder, default_save_folder)

    def test_set_save_folder_add_os_seop(self):
        e = GetBehavioralEvents(savefolder='/abc')
        e.set_savefolder()
        default_save_folder = '/abc/'
        self.assertEqual(e.savefolder, default_save_folder)

    def test_save_files(self):
        e = GetBehavioralEvents(
                                savefolder=self.dirpath,
                                datatype='abc'
                                )
        data = [
                ('123', pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})), 
                ('456', pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]}))
                ]
        e.save_files(data)
        check_file = self.dirpath + os.sep + 'abc_123.csv'
        check_file2 = self.dirpath + os.sep + 'abc_456.csv'
        print(os.listdir(self.dirpath))
        print(self.dirpath)
        assert os.path.exists(check_file)
        assert os.path.exists(check_file2)

    def test_prune_minimum_bouts(self):
        e = GetBehavioralEvents(minimum_bout_time=1)
        data = {
                    'Bout start': [1, 2, 3],
                    'Bout end': [1, 4, 10]
        }
        df = pd.DataFrame(data, index=[0,1,2])
        reduced_data = {
                    'Bout start': [2, 3],
                    'Bout end': [4, 10]
        }
        reduced_df = pd.DataFrame(reduced_data, index=[1,2])
        pd.testing.assert_frame_equal(e.prune_minimum_bouts(df), reduced_df)

    def test_add_time_offset(self):
        e = GetBehavioralEvents(time_offset=1)
        data = {
                    'Bout start': [1, 2, 3],
                    'Bout end': [1, 4, 10]
        }
        df = pd.DataFrame(data, index=[0,1,2])
        pd.testing.assert_frame_equal(df + 1, e.add_time_offset(df))

    def test_sort_by_bout_start(self):
        e = GetBehavioralEvents()
        data = {
                    'Bout start': [2, 1, 3],
                    'Bout end': [1, 4, 10]
        }
        df = pd.DataFrame(data, index=[0,1,2])
        ordered_data = {
                    'Bout start': [1, 2, 3],
                    'Bout end': [4, 1, 10]
        }
        ordered_df = pd.DataFrame(ordered_data, index=[0, 1,2])
        pd.testing.assert_frame_equal(e.sort_by_bout_start(df), ordered_df)

    def test_relabel_bout_type(self):
        label_dict = [
            {
                'location': 'right',
                'zone': ['left interaction', 'left sniffing'],
                'name': 'object'
            },
            {
                'location': 'left',
                'zone': ['left interaction', 'left sniffing'],
                'name': 'social'
            }
        ]
        e = GetBehavioralEvents(label_dict=label_dict)
        self.assertEqual(e.relabel_bout_type('left sniffing', 'left'), 'social')
        self.assertEqual(e.relabel_bout_type('left sniffing', 'right'), 'object')
        self.assertEqual(e.relabel_bout_type('not in label dict', 'right'), 'not in label dict')
        self.assertEqual(e.relabel_bout_type('left sniffing', 'not in label dict'), 'left sniffing')

    def test_relabel_bout_type_for_df(self):
        label_dict = [
            {
                'location': 'right',
                'zone': ['left interaction', 'left sniffing'],
                'name': 'object'
            },
            {
                'location': 'left',
                'zone': ['left interaction', 'left sniffing'],
                'name': 'social'
            }
        ]
        data = {
            'Bout type': ['left interaction', 'right interaction', 'nothing'],
            'Stimulus Location': ['right', 'right', 'right']
        }
        df = pd.DataFrame(data)
        relabeled_data = {
            'Bout type': ['object', 'social', 'nothing'],
            'Stimulus Location': ['right', 'right', 'right']
        }
        relabeled_df = pd.DataFrame(df)
        e = GetBehavioralEvents(label_dict=label_dict)
        pd.testing.assert_frame_equal(relabeled_df, e.relabel_bout_type_for_df(df))

    def test_clean_and_strip_string(self):
        e = GetBehavioralEvents()
        self.assertEqual(e.clean_and_strip_string(' Hello   world again!', sep=','), 
            'Hello,world,again!')

    @patch('pandas.read_csv')
    def test_get_ethovision_header_info(self, mock_read_csv):
        data = [
            ['Header', 5],
            ['Animal ID', 'abc'],
            ['Stimulus Location', 'right']
        ]
        mock_read_csv.return_value = pd.DataFrame(data)
        print(pd.DataFrame(data))
        e = GetBehavioralEvents()
        a, s, l = e.get_ethovision_header_info('/path/', 
            stimulus_name_set={'stimulus location'})
        self.assertEqual(l, 5)
        self.assertEqual(a, 'abc')
        self.assertEqual(s, 'right')



