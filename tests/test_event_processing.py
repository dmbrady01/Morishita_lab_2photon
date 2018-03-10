#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_event_processing.py: Python script that contains tests for event_processing.py 
"""

__author__ = "DM Brady"
__datewritten__ = "01 Mar 2018"
__lastmodified__ = "10 Mar 2018"

# Import unittest modules and event_processing
import unittest
from neo.core import Event, Segment
import quantities as pq
import numpy as np
import pandas as pd
from collections import OrderedDict
from imaging_analysis.event_processing import LoadEventParams
from imaging_analysis.event_processing import TruncateEvent
from imaging_analysis.event_processing import TruncateEvents
from imaging_analysis.event_processing import ExtractEventsToList
from imaging_analysis.event_processing import ProcessEventList
from imaging_analysis.event_processing import ProcessEvents
from imaging_analysis.event_processing import ProcessTrials



class TestLoadEventParams(unittest.TestCase):
    "Code tests for LoadEventParams function"
    
    def setUp(self):
        self.dpath = 'imaging_analysis/event_params.json'
        self.evtdict = OrderedDict({
            'channels': ['ch1'], 
            'events': {
                'one': {
                    'code': [1],
                    'plot': '-k',
                    'type': 'results'
                },
                'two': {
                    'code': [0],
                    'plot': 'b',
                    'type': 'start'
                }
            },
            'startoftrial': ['start'],
            'endoftrial': ['results'],
            'epoch': ['results']
        })
        self.codedf = pd.DataFrame(data=[[1], [0]], index=['one', 'two'], 
            columns=['ch1'])
        self.codedf.index.name = 'event'
        self.plotdf = pd.DataFrame(data=[['-k'], ['b']], index=['one', 'two'],
            columns=['plot'])
        self.plotdf.index.name = 'event'
        self.typedf = pd.DataFrame(data=[['results'], ['start']], index=['one', 'two'],
            columns=['type'])
        self.typedf.index.name = 'event'

    def tearDown(self):
        del self.dpath
        del self.evtdict
        del self.codedf
        del self.plotdf
        del self.typedf

    def test_returns_a_dataframe(self):
        "Makes sure function returns a dataframe and checks dpath works"
        start, end, epoch, code, plot, typedf = LoadEventParams(dpath=self.dpath)
        self.assertIsInstance(code, pd.core.frame.DataFrame)

    def test_makes_sure_dpath_is_correct(self):
        "Makes sure dpath is correct is supplied"
        self.assertRaises(IOError, LoadEventParams, dpath='/not/path')

    def test_makes_sure_evtdict_is_correct(self):
        "Makes sure evtdict is correct is supplied"
        self.assertRaises(TypeError, LoadEventParams, evtdict='not a dict')

    def test_makes_sure_startoftrial_is_returned(self):
        "Makes sure start of trials is returned"
        start, end, epoch, code, plot, typedf = LoadEventParams(evtdict=self.evtdict)
        self.assertEqual(start, self.evtdict['startoftrial'])

    def test_makes_sure_endoftrial_is_returned(self):
        "Makes sure end of trials is returned"
        start, end, epoch, code, plot, typedf = LoadEventParams(evtdict=self.evtdict)
        self.assertEqual(end, self.evtdict['endoftrial'])

    def test_makes_sure_epoch_is_returned(self):
        "Makes sure epoch is returned"
        start, end, epoch, code, plot, typedf = LoadEventParams(evtdict=self.evtdict)
        self.assertEqual(epoch, self.evtdict['epoch'])

    def test_makes_sure_code_dataframe_has_right_structure(self):
        "Checks code dataframe has right structure"
        start, end, epoch, code, plot, typedf = LoadEventParams(evtdict=self.evtdict)
        code.sort_index(inplace=True)
        self.codedf.sort_index(inplace=True)
        pd.testing.assert_frame_equal(self.codedf, code)

    def test_makes_sure_plot_dataframe_has_right_structure(self):
        "Checks plot dataframe has right structure"
        start, end, epoch, code, plot, typedf = LoadEventParams(evtdict=self.evtdict)
        plot.sort_index(inplace=True)
        self.plotdf.sort_index(inplace=True)
        pd.testing.assert_frame_equal(self.plotdf, plot)

    def test_makes_sure_results_dataframe_has_right_structure(self):
        "Checks results dataframe has right structure"
        start, end, epoch, code, plot, typedf = LoadEventParams(evtdict=self.evtdict)
        typedf.sort_index(inplace=True)
        self.typedf.sort_index(inplace=True)
        pd.testing.assert_frame_equal(self.typedf, typedf)



class TestTruncateEvent(unittest.TestCase):
    "Code tests for the TruncateEvent function."

    def setUp(self):
        self.evt = Event(np.arange(0, 100 ,1)*pq.s, 
                        labels=np.repeat(np.array(['t0', 't1'], dtype='S'), 50))
        self.not_evt = np.random.randn(1000, 1)
        self.evt_start = 10
        self.evt_pre_start = self.evt_start - 5
        self.evt_end = 90
        self.evt_post_end = self.evt_end + 5

    def tearDown(self):
        del self.evt
        del self.not_evt
        del self.evt_start
        del self.evt_end
        del self.evt_pre_start
        del self.evt_post_end

    def test_makes_sure_events_are_passed(self):
        "Test to make sure event object is passed"
        self.assertRaises(TypeError, TruncateEvent, self.not_evt)

    def test_evt_start_works(self):
        "Test to make sure evt is truncated from start"
        trunc_sig = TruncateEvent(self.evt, start=self.evt_start)
        times = trunc_sig.times
        self.assertFalse(self.evt_pre_start in times)

    def test_evt_end_works(self):
        "Test to make sure evt is truncated from end"
        trunc_sig = TruncateEvent(self.evt, end=self.evt_end)
        times = trunc_sig.times
        self.assertFalse(self.evt_post_end in times)



class TestTruncateEvents(unittest.TestCase):
    "Code tests for TruncateEvents function."

    def setUp(self):
        self.evt = Event(np.arange(0, 100 ,1)*pq.s, 
                        labels=np.repeat(np.array(['t0', 't1'], dtype='S'), 50))
        self.evt2 = Event(np.arange(0, 100 ,1)*pq.s, 
                        labels=np.repeat(np.array(['t2', 't3'], dtype='S'), 50))
        self.events = [self.evt, self.evt2]
        self.evt_start = 10
        self.evt_pre_start = self.evt_start - 5
        self.evt_end = 90
        self.evt_post_end = self.evt_end + 5

    def tearDown(self):
        del self.evt
        del self.evt_start
        del self.evt_end
        del self.evt_pre_start
        del self.evt_post_end
        del self.events

    def test_makes_sure_list_is_passed(self):
        "Makes sure list is passed to function"
        self.assertRaises(TypeError, TruncateEvents, 100)

    def test_evt_start_works(self):
        "Test to make sure evt is truncated from start in event_list"
        trunc_sig = TruncateEvents(self.events, start=self.evt_start)
        times = [x.times for x in trunc_sig]
        self.assertFalse(all(self.evt_pre_start in x for x in times))

    def test_evt_end_works(self):
        "Test to make sure evt is truncated from end in event_list"
        trunc_sig = TruncateEvents(self.events, end=self.evt_end)
        times = [x.times for x in trunc_sig]
        self.assertFalse(all(self.evt_post_end in x for x in times))



class TestExtractEventsToList(unittest.TestCase):
    "Code tests for ExtractEventsToList"

    def setUp(self):
        self.evt = Event(times=np.arange(0, 100 , 1)*pq.s, name='Ch1', 
                        labels=np.repeat(np.array(['t0', 't1'], dtype='S'), 50))
        self.evt2 = Event(times=np.arange(0, 100 , 3)*pq.s, name='Ch2', 
                        labels=np.repeat(np.array(['t2', 't3'], dtype='S'), 17))
        self.segment = Segment()
        self.segment.events.append(self.evt)
        self.segment.events.append(self.evt2)
        self.df = pd.DataFrame(data=[[1, 0], [1, 1]], index=['start', 'stop'],
            columns=['Ch1', 'Ch2'])
        self.df2 = pd.DataFrame(data=[[1, 0], [1, 1]], index=['start', 'stop'],
            columns=['Ch3', 'Ch4'])
        self.df3 = pd.DataFrame(data=[[1, 0, 0], [1, 1, 1]], 
            index=['start', 'stop'], columns=['Ch1', 'Ch2', 'Ch3'])

    def tearDown(self):
        del self.evt
        del self.evt2 
        del self.segment 
        del self.df
        del self.df2
        del self.df3

    def test_seg_must_be_a_seg_object(self):
        "Makes sure seg is a seg object"
        self.assertRaises(TypeError, ExtractEventsToList, seg='not a seg object')
    
    def test_evtframe_must_be_a_dataframe_object(self):
        "Makes sure evtframe is a pandas datafrane object"
        self.assertRaises(TypeError, ExtractEventsToList, evtframe='not a dataframe')

    def test_seg_events_must_have_channel_names_that_match_evtframe_columns(self):
        """Makes sure at least one of the event objects has a name that matches
        the columns in dataframe."""
        self.assertRaises(ValueError, ExtractEventsToList, seg=self.segment, 
            evtframe=self.df2)

    def test_eventlist_length_matches_evtframe_columns_length(self):
        """Makes that event list length == evetframe.columns.shape[0]"""
        self.assertRaises(ValueError, ExtractEventsToList, seg=self.segment, 
            evtframe=self.df3)

    def test_list_is_returned(self):
        "Checks that list is returned"
        output = ExtractEventsToList(seg=self.segment, evtframe=self.df)
        self.assertIsInstance(output, list)

    def test_list_of_dicts_is_returned(self):
        "Checks that list of dicts is returned"
        output = ExtractEventsToList(seg=self.segment, evtframe=self.df)
        check = [isinstance(x, dict) for x in output]
        self.assertTrue(all(check))

    def test_time_name_works(self):
        "Checks that dictionary key is set to time_name"
        output = ExtractEventsToList(seg=self.segment, evtframe=self.df, 
            time_name='myname')
        dict_keys = [x.keys() for x in output]
        dict_check = ['myname' in x for x in dict_keys]
        self.assertTrue(all(dict_check))

    def test_ch_name_works(self):
        "Checks that dictionary key is set to ch_name"
        output = ExtractEventsToList(seg=self.segment, evtframe=self.df, 
            ch_name='myname')
        dict_keys = [x.keys() for x in output]
        dict_check = ['myname' in x for x in dict_keys]
        self.assertTrue(all(dict_check))

    def test_ch_values_are_integers_that_match_evtframe_column_indices(self):
        "Checks that values from ch_name match evtframe column indices"
        column_indices = [x for x in range(self.df.columns.shape[0])]
        output = ExtractEventsToList(seg=self.segment, evtframe=self.df)
        output_values = [x['ch'] for x in output]
        self.assertEqual(column_indices, output_values)

    def test_time_values_are_extracted_properly(self):
        "Checks that time values are extracted properly"
        output = ExtractEventsToList(seg=self.segment, evtframe=self.df)
        output_values = [x['times'] for x in output]
        check = [self.evt.times, self.evt2.times]
        zipped = zip(check, output_values)
        check_zip = [np.array_equal(x, y) for x, y in zipped]
        self.assertTrue(all(check_zip))



class TestProcessEventList(unittest.TestCase):
    "Code tests for ProcessEventList"

    def setUp(self):
        self.evt = Event(times=np.arange(0, 100 , 1)*pq.s, name='Ch1', 
                        labels=np.repeat(np.array(['t0', 't1'], dtype='S'), 50))
        self.evt2 = Event(times=np.arange(0, 100 , 3)*pq.s, name='Ch2', 
                        labels=np.repeat(np.array(['t2', 't3'], dtype='S'), 17))
        self.eventlist = [{'ch': 0, 'times': self.evt.times}, \
                        {'ch': 1, 'times': self.evt2.times}]
        self.df = pd.DataFrame(data=[[1, 0], [1, 1]], index=['start', 'stop'],
            columns=['Ch1', 'Ch2'])

    def tearDown(self):
        del self.evt
        del self.evt2 
        del self.df

    def test_tolerance_must_be_number(self):
        "Makes sure tolerance is a number"
        self.assertRaises(TypeError, ProcessEventList, tolerance='not a number')

    def test_eventlist_is_a_list(self):
        "Makes sure eventlist is a list"
        self.assertRaises(TypeError, ProcessEventList, eventlist='not a list')

    def test_ch_name_must_be_a_key(self):
        "Makes sure ch_name is a key in dictionaries in eventlist"
        self.assertRaises(ValueError, ProcessEventList, eventlist=self.eventlist, 
            tolerance=1, evtframe=self.df, ch_name='not_channel_name')

    def test_time_name_must_be_a_key(self):
        "Makes sure time_name is a key in dictionaries in eventlist"
        self.assertRaises(ValueError, ProcessEventList, eventlist=self.eventlist, 
            tolerance=1, evtframe=self.df, time_name='not_time_name')

    def test_returns_two_variables(self):
        "Checks that two variables are returned"
        output = ProcessEventList(eventlist=self.eventlist, tolerance=1, 
            evtframe=self.df)
        self.assertEqual(len(output), 2)

    def test_event_times_is_a_list(self):
        "Checks returned event_times is a list"
        eventtimes, eventlabels = ProcessEventList(eventlist=self.eventlist, 
            tolerance=1, evtframe=self.df)
        self.assertIsInstance(eventtimes, list)

    def test_event_labels_is_a_list(self):
        "Checks returned event_labels is a list"
        eventtimes, eventlabels = ProcessEventList(eventlist=self.eventlist, 
            tolerance=1, evtframe=self.df)
        self.assertIsInstance(eventlabels, list)

    def test_event_times_are_correct(self):
        "Checks that events times are correct"
        eventtimes, eventlabels = ProcessEventList(eventlist=self.eventlist, 
            tolerance=1, evtframe=self.df)
        equal = np.array_equal(np.array(eventtimes), np.arange(0, 100 , 1)*pq.s)
        self.assertTrue(equal)     

    def test_event_labels_are_correct(self):
        "Checks that events labels are correct"
        eventtimes, eventlabels = ProcessEventList(eventlist=self.eventlist, 
            tolerance=0.1, evtframe=self.df)
        test_array = np.arange(0, 100, 1)
        mask_array = np.arange(0, 100, 3)
        test_array = test_array.astype('S')
        test_array[mask_array] = 'stop'
        test_array[np.where(test_array != 'stop')[0]] = 'start'
        equal = np.array_equal(eventlabels, test_array)
        self.assertTrue(equal)     



class TestProcessEvents(unittest.TestCase):
    "Code tests for ProcessEvents function"

    def setUp(self):
        self.evt = Event(times=np.arange(0, 100 , 1)*pq.s, name='Ch1', 
                        labels=np.repeat(np.array(['t0', 't1'], dtype='S'), 50))
        self.evt2 = Event(times=np.arange(0, 100 , 3)*pq.s, name='Ch2', 
                        labels=np.repeat(np.array(['t2', 't3'], dtype='S'), 17))
        self.segment = Segment()
        self.segment.events.append(self.evt)
        self.segment.events.append(self.evt2)
        self.df = pd.DataFrame(data=[[1, 0], [1, 1]], index=['start', 'stop'],
            columns=['Ch1', 'Ch2'])

    def tearDown(self):
        del self.evt
        del self.evt2 
        del self.segment

    def test_seg_must_be_a_seg_object(self):
        "Makes sure seg is a seg object"
        self.assertRaises(TypeError, ProcessEvents, seg='not a segment')

    def test_event_object_is_added(self):
        "Makes sure event object is added to seg"
        init_len = len(self.segment.events)
        ProcessEvents(seg=self.segment, tolerance=1, evtframe=self.df)
        self.assertTrue(init_len + 1, len(self.segment.events))



class TestProcessTrials(unittest.TestCase):
    "Code tests for ProcessTrials function"

    def setUp(self):
        self.evt = Event(times=np.arange(0, 100 , 1)*pq.s, name='Ch1', 
                        labels=np.repeat(np.array(['t0', 't1'], dtype='S'), 50))
        self.evt2 = Event(times=np.arange(0, 100 , 3)*pq.s, name='Ch2', 
                        labels=np.repeat(np.array(['t2', 't3'], dtype='S'), 17))
        self.segment = Segment()
        self.segment.events.append(self.evt)
        self.segment.events.append(self.evt2)
        self.df = pd.DataFrame(data=[[1, 0], [1, 1]], index=['start', 'stop'],
            columns=['Ch1', 'Ch2'])
        self.staroftrial = ['start']

    def tearDown(self):
        del self.evt
        del self.evt2 
        del self.segment

    def test_startoflist_is_a_list(self):
        "Makes sure startoflist is a list"
        self.assertRaises(TypeError, ProcessTrials, startoftrial='not a segment')

    def test_typedf_is_a_dataframe(self):
        "Makes sure typedf is a dataframe"
        self.assertRaises(TypeError, ProcessTrials, startoftrial=self.staroftrial, 
            typedf='not a dataframe')

    def test_seg_is_a_seg_object(self):
        "Makes sure seg is a segment object"
        self.assertRaises(TypeError, ProcessTrials, startoftrial=self.staroftrial, 
            typedf=self.df, seg='not a seg object')





if __name__ == '__main__':
    unittest.main()