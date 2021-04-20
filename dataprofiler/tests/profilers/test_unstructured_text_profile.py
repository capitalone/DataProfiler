import unittest
from unittest import mock
from collections import defaultdict

import pandas as pd

from dataprofiler.profilers.unstructured_text_profile import TextProfiler

class TestUnstructuredTextProfile(unittest.TestCase):
    
    
    def test_text_profile_update_and_name(self):
        text_profile = TextProfiler("Name")
        sample = pd.Series(["Hello my name is Grant",
                            "Bob and Grant are friends"])
        text_profile.update(sample)
        self.assertEqual("Name", text_profile.name)

        
    def test_vocab(self):
        text_profile = TextProfiler("Name")
        sample = pd.Series(["Hello my name is Grant",
                            "Bob and Grant are friends"])
        text_profile.update(sample)
        profile = text_profile.profile

        # Assert vocab is correct
        expected_vocab = ['y', 'e', 'G', 'H', 'r', 'b', 'l', 's', 'm',
                          't', 'i', 'B', 'd', 'f', 'o', ' ', 'n', 'a']
        self.assertListEqual(sorted(expected_vocab), sorted(profile['vocab']))

        # Update the data again
        sample = pd.Series(["Grant knows how to code",
                            "Grant will code with Bob"])
        text_profile.update(sample)
        profile = text_profile.profile

        # Assert vocab is correct
        expected_vocab = [' ', 'B', 'G', 'H', 'a', 'b', 'c', 'd', 'e',
                          'f', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'r',
                          's', 't', 'w', 'y']
        self.assertListEqual(sorted(expected_vocab), sorted(profile['vocab']))

 
    def test_words_and_word_count(self):
        text_profile = TextProfiler("Name")
        sample = pd.Series(["Hello my name is Grant",
                            "Bob and Grant are friends"])
        text_profile.update(sample)
        profile = text_profile.profile

        # Assert words is correct and stop words are not present
        expected_words = ['Hello', 'name', 'Grant', 'Bob', 'friends']
        self.assertListEqual(expected_words, profile['words'])
        self.assertNotIn("is", profile['words'])

        # Assert word counts are correct
        expected_word_count = {'Hello': 1, 'name': 1, 'Grant': 2, 'Bob': 1,
                               'friends': 1}
        self.assertDictEqual(expected_word_count, profile['word_count'])

        # Update the data again
        sample = pd.Series(["Grant knows how to code",
                            "Grant will code with Bob"])
        text_profile.update(sample)
        profile = text_profile.profile
        
        # Assert words is correct and stop words are not present
        expected_words = ['Hello', 'name', 'Grant', 'Bob', 'friends', 'knows',
                          'code']
        self.assertListEqual(expected_words, profile['words'])
        self.assertNotIn("with", profile['words'])

        # Assert word counts are correct
        expected_word_count = {'Hello': 1, 'name': 1, 'Grant': 4, 'Bob': 2,
                               'friends': 1, 'knows': 1, 'code': 2}
        self.assertDictEqual(expected_word_count, profile['word_count'])
        
    def test_sample_size(self):
        text_profile = TextProfiler("Name")
        sample = pd.Series(["Hello my name is Grant",
                            "Bob and Grant are friends"])
        text_profile.update(sample)

        # Assert sample size is accurate
        self.assertEqual(2, text_profile.sample_size)

        # Update the data again
        sample = pd.Series(["Grant knows how to code",
                            "Grant will code with Bob"])
        text_profile.update(sample)

        # Assert sample size is accurate
        self.assertEqual(4, text_profile.sample_size)
    
    def test_timing(self):
        text_profile = TextProfiler("Name")
        sample = pd.Series(["Hello my name is Grant",
                            "Bob and Grant are friends"])
        text_profile.update(sample)
        profile = text_profile.profile

        # Assert timing is occurring
        self.assertIn("vocab", profile["times"])
        self.assertIn("words", profile["times"])