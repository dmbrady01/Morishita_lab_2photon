#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_event_processing.py: Python script that contains tests for event_processing.py 
"""

__author__ = "DM Brady"
__datewritten__ = "01 Mar 2018"
__lastmodified__ = "01 Mar 2018"

# Import unittest modules and event_processing
import unittest
from imaging_analysis.event_processing import EvtDict
from imaging_analysis.event_processing import ProcessEvents



class TestEvtDict(unittest.TestCase):
    "Code tests for EvtDict function"
    
    def setUp(self):
        self.list1 = ['event1', 'event2']
        self.dict1 = {'1': 'event1', '2': 'event2'}
        self.string = 'hello'

    def tearDown(self):
        del self.list1
        del self.dict1
        del self.string

    def test_returns_a_dict(self):
        "Makes sure EvtDict returns a dictionary"
        self.assertIsInstance(EvtDict(), dict)

    def test_dict_length_same_as_given_list(self):
        "Makes sure the dictionary length is equal to given list length"
        self.assertEqual(len(EvtDict(self.list1)), len(self.list1))

    def test_dict_key_begins_at_one(self):
        "Makes sure the first key is 1"
        results = EvtDict(self.list1)
        keys = [int(key) for key in results.keys()]
        self.assertEqual(min(keys), 1)

    def test_keys_are_strings(self):
        "Makes sure keys are strings (not integers or floats)"
        keys = EvtDict().keys()
        str_check = [isinstance(key, str) for key in keys]
        self.assertTrue(all(str_check))

    def test_argument_must_be_iterable(self):
        self.assertRaises(TypeError, EvtDict, self.string)

    def test_returns_properly_ordered_dict(self):
        "Makes sure that the list is mapped to the dictionary in the right order"
        self.assertEqual(EvtDict(self.list1), self.dict1)



class TestProcessEvents(unittest.TestCase):
    "Code tests for ProcessEvents function"

    def setUp(self):
        self.wrong_type = 'hello'

    def tearDown(self):
        del self.wrong_type

    def test_tolerance_must_be_number(self):
        "Makes sure tolerance is a number"
        self.assertRaises(TypeError, ProcessEvents, tolerance=self.wrong_type)

    def test_seg_must_be_a_seg_object(self):
        "Makes sure seg is a seg object"
        self.assertRaises(TypeError, ProcessEvents, seg=self.wrong_type)



if __name__ == '__main__':
    unittest.main()