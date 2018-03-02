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
from photon_analysis.utils import ReadNeoPickledObj
from photon_analysis.utils import ReadNeoTdt
from photon_analysis.utils import WriteNeoPickledObj



class TestReadNeoPickledObj(unittest.TestCase):

    def setUp(self):
        self.bad_path = 56
        self.bad_name = [100]
        self.not_found_path = '/hamburger'

    def tearDown(self):
        del self.bad_path
        del self.bad_name
        del self.not_found_path

    def test_makes_sure_path_is_a_string(self):
        "Makes sure path argument is a string"
        self.assertRaises(TypeError, ReadNeoPickledObj, path=self.bad_path)

    def test_makes_sure_name_is_a_string(self):
        "Makes sure name argument is a string"
        self.assertRaises(TypeError, ReadNeoPickledObj, path=self.bad_name)

    def test_makes_sure_dpath_exists(self):
        "Makes sure path to pickled object exists"
        self.assertRaises(IOError, ReadNeoPickledObj, path=self.not_found_path)



class TestReadNewTdt(unittest.TestCase):

    def setUp(self):
        self.bad_path = 56
        self.not_found_path = '/hamburger'

    def tearDown(self):
        del self.bad_path
        del self.not_found_path

    def test_makes_sure_path_is_a_string(self):
        "Makes sure path argument is a string"
        self.assertRaises(TypeError, ReadNeoTdt, self.bad_path)

    def test_makes_sure_path_exists(self):
        "Makes sure path to directory exists"
        self.assertRaises(IOError, ReadNeoTdt, self.not_found_path)



class WriteNeoPickledObj(unittest.TestCase):

    def setUp(self):
        self.bad_path = 56
        self.bad_name = [122]
        self.bad_block = {'a': 1}

    def tearDown(self):
        del self.bad_path
        del self.bad_name
        del self.bad_block

    def test_makes_sure_path_is_a_string(self):
        "Makes sure path argument is a string"
        self.assertRaises(TypeError, WriteNeoPickledObj, path=self.bad_path)

    def test_makes_sure_name_is_a_string(self):
        "Makes sure name argument is a string"
        self.assertRaises(TypeError, WriteNeoPickledObj, name=self.bad_name)

    def test_makes_sure_block_is_a_neo_block_object(self):
        "Makes sure block is a new.core.block.Block object"
        self.assertRaises(TypeError, WriteNeoPickledObj, block=self.bad_block)



if __name__ == '__main__':
    unittest.main()