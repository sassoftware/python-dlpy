#!/usr/bin/env python
# encoding: utf-8
#
# Copyright SAS Institute
#
#  Licensed under the Apache License, Version 2.0 (the License);
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import os

from dlpy.data_clean import DataClean

import swat
import swat.utils.testing as tm


class TestDataCleaning(tm.TestCase):

    @classmethod      
    def setUpClass(self):
        self.filename = os.path.join('datasources','metadata_for_audio.txt')
        self.project_path = os.path.dirname(os.path.abspath(__file__))
        self.full_filename = os.path.join(self.project_path, self.filename)        
        
        self.replacements_txt = os.path.join('datasources', 'data_clean_replacements_test.txt')
        self.tst_replacements = os.path.join(self.project_path, self.replacements_txt)         

        self.dc = DataClean()
        self.conn = swat.CAS()
  
    @classmethod
    def tearDownClass(self):
        pass

    def setUp(self):
        print(' ')
        print('=============================================================')          
        print('Test: %s ' % self.id())        
        print('Description: %s ' % self.shortDescription())
        print('=============================================================')
    
    def tearDown(self):
        pass                
    
    def test_dc_valid_file(self):
        '''read in a valid text file from datasource'''
        self.dc.readin_file_as_bytearray(self.full_filename)

    def test_dc_replacements(self):
        '''make sure we can read in various scenarios in dc_replacements
        
        using a test version of the replacement rules, and ensure any 
        invalid format of rule is not included.
        '''
        curr_dc=DataClean(replacements=self.tst_replacements)
        print ('-------------------------')
        for k,v in curr_dc.replacements.items():
            print('%s=%s' % (k,v))
            print('--------------------------------------------------------')
            print('testing for %s, to be replaced with %s' % (k,v))
            test_string = 'recording_7.wav %s Wow that is awesome' % k
            print('...test string >>%s' % test_string)   
            #set contents prior to processing,typically users would not do this
            #would set contents at instance. doing this just for testing
            curr_dc.contents=test_string                             
            curr_dc.process_contents(show_results=False)
            print('...processed as>>%s' % curr_dc.contents.decode())
            self.assertEqual(-1,curr_dc.contents.decode().find(k),'...seems the original token is still in the string')
            self.assertGreater(0, curr_dc.contents.decode().find(k),'...seems like the new token was not applied')            
        
    def test_dc_load_replacemetns(self):
        self.dc.load_dc_replacements()
        
    def test_dc_process_contents_clean_find_n_replace_start_of_string(self):
        '''ensure that we can replace at the start of the string'''
        curr_replacements = self.dc.replacements
        
        for k,v in curr_replacements.items():
            if k != 'SYMBOL_AS_SPACE':
                print('--------------------------------------------------------')
                print('testing for %s, to be replaced with %s' % (k,v))
                test_string = 'recording_7.wav %s Wow that is awesome' % k
                print('...test string >>%s' % test_string)
                tst_dc = DataClean(contents=test_string)            
                tst_dc.process_contents(show_results=False)
                print('...processed as>>%s' % tst_dc.contents.decode())
                self.assertEqual(-1,tst_dc.contents.decode().find(k),'...seems the original token is still in the string')
                self.assertGreater(0, tst_dc.contents.decode().find(k),'...seems like the new token was not applied')            

    def test_dc_process_contents_clean_find_n_replace_end_of_string(self):
        '''ensure that we can replace at the end of the string'''
        curr_replacements = self.dc.replacements
        
        for k,v in curr_replacements.items():
            if k != 'SYMBOL_AS_SPACE':            
                print('--------------------------------------------------------')
                print('testing for %s, to be replaced with %s' % (k,v))
                test_string = 'recording_7.wav Wow that is awesome %s' % k
                print('...test string >>%s' % test_string)
                tst_dc = DataClean(contents=test_string)            
                tst_dc.process_contents(show_results=False)
                print('...processed as>>%s' % tst_dc.contents.decode())
                self.assertEqual(-1,tst_dc.contents.decode().find(k),'...seems the original token is still in the string')
                self.assertGreater(0, tst_dc.contents.decode().find(k),'...seems like the new token was not applied')            
                    
    def test_dc_process_contents_clean_find_n_replace_middle_of_string(self):
        '''ensure that we can replace at the middle of the string'''
        curr_replacements = self.dc.replacements
        
        for k,v in curr_replacements.items():
            if k != 'SYMBOL_AS_SPACE':
                print('--------------------------------------------------------')
                print('testing for %s, to be replaced with %s' % (k,v))
                test_string = 'recording_7.wav Wow that is %s awesome' % k
                print('...test string >>%s' % test_string)
                tst_dc = DataClean(contents=test_string)            
                tst_dc.process_contents(show_results=False)
                print('...processed as>>%s' % tst_dc.contents.decode())
                self.assertEqual(-1,tst_dc.contents.decode().find(k),'...seems the original token is still in the string')
                self.assertGreater(0, tst_dc.contents.decode().find(k),'...seems like the new token was not applied')            
                     
    def test_dc_process_contents_clean_find_n_replace_multiple_instances_in_string(self):
        '''ensure that we can replace at the multiple locations within the string'''
        curr_replacements = self.dc.replacements
        
        for k,v in curr_replacements.items():
            if k != 'SYMBOL_AS_SPACE':            
                print('--------------------------------------------------------')
                print('testing for %s, to be replaced with %s' % (k,v))
                test_string = 'recording_7.wav %s Wow %s that is %s awesome %s' % (k,k,k,k)
                print('...test string >>%s' % test_string)
                tst_dc = DataClean(contents=test_string)            
                tst_dc.process_contents(show_results=False)
                print('...processed as>>%s' % tst_dc.contents.decode())
                self.assertEqual(-1,tst_dc.contents.decode().find(k),'...seems the original token is still in the string')
                self.assertGreater(0, tst_dc.contents.decode().find(k),'...seems like the new token was not applied')            
                      
    def test_dc_process_contents_clean_find_n_replace_symbols(self):
        '''ensure that we can replace at the multiple locations of symbols'''

        curr_replacements = self.dc.replacements
        
        for k,v in curr_replacements.items():
            if k == 'SYMBOL_AS_SPACE':                            
                ignore_sym=['[',']',',']
                for each in v:
                    if each not in ignore_sym: 
                        print('--------------------------------------------------------')
                        print('testing for symbol "%s", to be removed (i.e. replaced with ''' % each)
                        test_string = 'recording_7.wav %s Wow %s%s%s that is awesome %s' % (each,each,each,each,each)
                        print('...test string >>%s' % test_string)
                        tst_dc = DataClean(contents=test_string)            
                        tst_dc.process_contents(show_results=False)
                        print('...processed as>>%s' % tst_dc.contents.decode())
                        self.assertEqual(-1,tst_dc.contents.decode().find(k),'...seems the original token is still in the string')
                        self.assertGreater(0, tst_dc.contents.decode().find(k),'...seems like the new token was not applied')            
                                   
    def test_dc_process_search_non_speech_events_singlechar(self):
        '''make sure we can find the non speech event and ensure it's under the limit'''
        
        test_limit='[%s]'%('a'*1)        
        test_string = 'recording_7.wav %sWow that is awesome' % test_limit
        print('...test string >>%s' % test_string)
        tst_dc = DataClean(contents=test_string)            
        self.assertFalse(tst_dc._search_non_speech_events(),'...expected to within limit, confirm')

    def test_dc_process_search_non_speech_events_nochars(self):
        '''make sure we can find the non speech with no chars event and ensure it's under the limit'''
        
        test_limit='[]'        
        test_string = 'recording_7.wav %sWow that is awesome' % test_limit
        print('...test string >>%s' % test_string)
        tst_dc = DataClean(contents=test_string)            
        self.assertFalse(tst_dc._search_non_speech_events(),'...expected to within limit, confirm')      
        
    def test_dc_process_search_non_speech_events_one_less_limit(self):
        '''ensure that one less the limit will still pass (as False)'''
        
        test_limit='[%s]'%('a'*19)        
        test_string = 'recording_7.wav %sWow that is awesome' % test_limit
        print('...test string >>%s' % test_string)
        tst_dc = DataClean(contents=test_string)            
        self.assertFalse(tst_dc._search_non_speech_events(),'...expected to within limit, confirm')             
                            
    def test_dc_process_search_non_speech_events_at_limit(self):
        '''ensure that one the limit will still pass (as False)'''
        
        test_limit='[%s]'%('a'*20)        
        test_string = 'recording_7.wav %sWow that is awesome' % test_limit
        print('...test string >>%s' % test_string)
        tst_dc = DataClean(contents=test_string)            
        self.assertFalse(tst_dc._search_non_speech_events(),'...expected at limit of 20 chars, confirm')                                
         
    def test_dc_process_search_non_speech_events_over_limit(self):
        '''ensure that OVER the limit will still fail (as True)'''
        
        test_limit='[%s]'%('a'*30)        
        test_string = 'recording_7.wav %sWow that is awesome' % test_limit
        print('...test string >>%s' % test_string)
        tst_dc = DataClean(contents=test_string)            
        self.assertTrue(tst_dc._search_non_speech_events(),'...expected to be over the limit of 20 chars, confirm') 
        
    def test_dc_process_search_non_speech_events_multiple_instances_both_okay(self):
        '''ensure we can locate multiple non speech events'''        
        test_limit_1='[%s]'%('a'*10)
        test_limit_2='[%s]'%('b'*15)        
        test_string = 'recording_7.wav %sWow that is %s awesome' % (test_limit_1,test_limit_2)
        print('...test string >>%s' % test_string)
        tst_dc = DataClean(contents=test_string)            
        self.assertFalse(tst_dc._search_non_speech_events(),'...expected to be under the limit, confirm')   
        
    def test_dc_process_search_non_speech_events_multiple_instances_2nd_okay(self):
        '''ensure we can locate multiple non speech events, one instance is bad'''        
        test_limit_1='[%s]'%('a'*30)
        test_limit_2='[%s]'%('b'*15)        
        test_string = 'recording_7.wav %sWow that is %s awesome' % (test_limit_1,test_limit_2)
        print('...test string >>%s' % test_string)
        tst_dc = DataClean(contents=test_string)            
        self.assertTrue(tst_dc._search_non_speech_events(),'...expected to be under the limit, confirm')    
        
    def test_dc_process_search_non_speech_events_multiple_instances_1st_okay(self):
        '''ensure we can locate multiple non speech events, one instance is bad'''        
        test_limit_1='[%s]'%('a'*10)
        test_limit_2='[%s]'%('b'*40)        
        test_string = 'recording_7.wav %sWow that is %s awesome' % (test_limit_1,test_limit_2)
        print('...test string >>%s' % test_string)
        tst_dc = DataClean(contents=test_string)            
        self.assertTrue(tst_dc._search_non_speech_events(),'...expected to be under the limit, confirm')   
        
    def test_dc_process_search_non_speech_events_multiple_instances_both_over(self):
        '''ensure we can locate multiple non speech events, one instance is bad'''        
        test_limit_1='[%s]'%('a'*21)
        test_limit_2='[%s]'%('b'*21)        
        test_string = 'recording_7.wav %sWow that is %s awesome' % (test_limit_1,test_limit_2)
        print('...test string >>%s' % test_string)
        tst_dc = DataClean(contents=test_string)            
        self.assertTrue(tst_dc._search_non_speech_events(),'...expected to be under the limit, confirm')                                                                  

    def test_dc_process_consecspaces_beginning_one_double_and_one_single(self):
        '''ensure we can replace consecutive spaces for processing, 1 needs it 1 doesn't'''        
        test_limit_1='>>%s<<'%(' '*2)
        test_limit_2='>>%s<<'%(' '*1)        
        test_string = 'recording_7.wav %sWow that is %s awesome' % (test_limit_1,test_limit_2)
        print('...test string >>%s' % test_string)
        tst_dc = DataClean(contents=test_string)
        status,updated_string = tst_dc._search_replace()  
        print(updated_string)  
        self.assertLess(len(updated_string), len(test_string), '...expected string to less due to replacements')                              
        self.assertTrue(status,'...expected to be under the limit, confirm')    
        
    def test_dc_process_consecspaces_beginning_both_multiple(self):
        '''ensure we can replace consecutive spaces for processing, both need to be replaced'''        
        test_limit_1='>>%s<<'%(' '*30)
        test_limit_2='>>%s<<'%(' '*25)        
        test_string = 'recording_7.wav %sWow that is %s awesome' % (test_limit_1,test_limit_2)

        tst_dc = DataClean(contents=test_string)   
        status,updated_string = tst_dc._search_replace()  
        print('...replace status: %s' % status)
        print('...test string    >>:%s' % test_string)
        print('...updated string >>:%s' % updated_string) 
        self.assertLess(len(updated_string), len(test_string), '...expected string to less due to replacements')                    
        self.assertTrue(status,'...expected to be under the limit, confirm')             

    def test_dc_process_consecspaces_beginning_both_single(self):
        '''ensure we can replace consecutive spaces for processing, none should return false'''        
        test_limit_1='>>%s<<'%(' '*1)
        test_limit_2='>>%s<<'%(' '*1)        
        test_string = 'recording_7.wav %sWow that is %s awesome' % (test_limit_1,test_limit_2)
        print('...test string >>%s' % test_string)
        tst_dc = DataClean(contents=test_string) 
        status,updated_string = tst_dc._search_replace()  
        print(updated_string)  
        self.assertEqual(len(updated_string), len(test_string), '...expected string to equal since no replacement')                           
        self.assertFalse(status,'...nothing should have been replaced based on default regex')  
            
    def test_dc_process_consecspaces_beginning_one_single_and_double_single(self):
        '''ensure we can replace consecutive spaces for processing, second one should be replaced'''        
        test_limit_1='>>%s<<'%(' '*1)
        test_limit_2='>>%s<<'%(' '*2)        
        test_string = 'recording_7.wav %sWow that is %s awesome' % (test_limit_1,test_limit_2)
        print('...test string >>%s' % test_string)
        tst_dc = DataClean(contents=test_string)     
        status,updated_string = tst_dc._search_replace()  
        print(updated_string)          
        self.assertLess(len(updated_string), len(test_string), '...expected string to less due to replacements')                                        
        self.assertTrue(status,'...expected to be under the limit, confirm')  

    def test_dc_process_consecspaces_multiple_locations(self):
        '''ensure we can replace consecutive spaces for processing 2 locations, one at end'''        
        test_limit_1='%s'%(' '*1)
        test_limit_2='%s'%(' '*2)        
        test_string = 'recording_7.wav %sWow that is %s awesome%s' % (test_limit_1,test_limit_2,test_limit_2)
        print('...test string >>%s' % test_string)
        tst_dc = DataClean(contents=test_string)   
        status,updated_string = tst_dc._search_replace()  
        print(updated_string)        
        self.assertLess(len(updated_string), len(test_string), '...expected string to less due to replacements')                                            
        self.assertTrue(status,'...expected to be under the limit, confirm')          
        
    def test_dc_datasource_data(self):
        '''just process the test example and ensure everything works'''
        print ('---------------------------------------')
        print ('Metadata Prior to Cleaning')
        print ('---------------------------------------')        
#         curr_dc=DataClean()
#         pre_data = curr_dc.readin_file_as_bytearray(self.full_filename)
#         for each in pre_data.decode():
#             print (each)
        curr_dc=DataClean(contents_as_path=self.full_filename).process_contents(show_results=True)
                     
                     
    def test_dc_datasource_data_to_castable(self):
        '''process everything and build a castable from results'''
        
        curr_dc=DataClean(conn=self.conn,contents_as_path=self.full_filename)
        response = curr_dc.process_contents(show_results=False)        
        curr_dc.create_castable(response['results'],'cool_cas',replace=True,promote=True) 
        