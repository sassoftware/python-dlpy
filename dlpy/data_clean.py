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

import json
import os
import re
import swat
import sys
from decimal import Decimal


class DataClean():
    '''
    DataClean

    '''

    def __init__(self, conn=None, contents_as_path=None, contents=None, replacements=None):
        self.filename = os.path.join('datasources', 'data_clean_replacements.txt')
        self.project_path = os.path.dirname(os.path.abspath(__file__)) 
        self.replacements_file = os.path.join(self.project_path, self.filename)
        
        self.replacements = None
        self.contents = None        
        self.conn = conn 

        # Replacements
        if replacements: 
            print('...using user provided replacements')
            if isinstance(replacements, str):
                print('...using user provided replacements from provided file.')
                self.load_dc_replacements(replacements)                
            else:
                print('...error: replacements must be a file path to replacements file')            
        else:
            print('...using default replacements file from datasource dir')
            self.load_dc_replacements(self.replacements_file)            
        
        # Content
        if contents_as_path: # Read contents from a path
            self.contents = self.readin_file_as_bytearray(contents_as_path)
        elif contents:
            if not isinstance(contents, bytearray):
                self.contents = bytearray(contents.encode())
            else:
                self.contents = contents

    def load_dc_replacements(self, path=None):
        '''
        Read in the contents of the replacements file within the datasource directory by default

        You can bypass this file by passing in directly the fullpath a
        different file location.
        
        Parameters
        ----------
        full_path : string, optional
            Full path to the location of a replacements file.
            
        '''
        if not path:
            path = self.replacements_file
        
        _contents = None
        _replacements = {}
        try:
            print ('...reading in %s' % path)           
            if os.path.exists(path):
                with open(path, 'r') as content_file:
                    _contents = content_file.read().split('\n')
                    
                for each in _contents:
                    #if not a comment, read it in
                    if not each.startswith('#'): 
                        #ignore blank lines         
                        if len(each)>0:
                            #expect to find = with a min of ''=''                  
                            if each.find('=') and len(each)>5:                  
                                k,v = each.strip().split('=')
                                if len(k)>=2 and len(v)>=2:                            
                                    _replacements[k]=v  
                                else:
                                    print('...warning,invalid rule "%s" found, ignoring rule.... please adjust' % each)                                    
                            else:
                                print('...warning,invalid rule "%s" found, ignoring rule.... please adjust' % each)
                                                             
            else:
                print ("...unable to locate %s " % path) 
        except IOError as err:
            print ('...error reading file %s' % err)
        except: 
            print ('...error:', sys.exc_info()[0])          
                
        if _replacements:
            self.replacements = _replacements
            
    def readin_file_as_bytearray(self, path):
        '''
        Read in the contents of the file, it not found raise exception

        You can bypass this file by passing in directly the path a different
        file location.
        
        Parameters
        ----------
        path : string, optional
            Full path to the location of a content file.
            
        '''
        _contents = None
        try:
            print ('...reading in %s' % path)
            if os.path.exists(path):
                with open(path, 'r') as content_file:
                    _contents = content_file.read()
            else:
                print ("...unable to locate %s " % path) 
        except IOError as err:
            print ('...error reading file %s' % err)
        except: 
            print ('...error:', sys.exc_info()[0])              
        
        return bytearray(_contents.encode())
    
    def _search_non_speech_events(self, left_id='[', right_id=']', limit=20):
        '''
        Search for non speech events

        If beyond limit, return False. By default if non speech event of
        20 chars are found within the left_id [ and right_id of ], False
        will be returned.
        
        Parameters
        ----------
        left_id : string
            left delimiter of non speech event
            Default: '['
        right_id : string
            right delimiter of non speech event
            Default: ']'
        limit : int
            number of chars limit within left_id and right_id
            Default: 20
            
        Returns
        -------
        True
            If chars are beyond the limit
        False
            If the delimiter not found or number of chars below limit

        '''          
        found = False
        results = None
        if self.contents:
            if not isinstance(self.contents,bytearray):
                self.contents = bytearray(self.contents.encode())            
            
            #search for groups between the chars []
            regex_str = r'(\%s.*?\%s)' % (left_id,right_id)        
            regex=regex_str.encode("utf-8")
            exp = re.compile(regex) 
            results = re.finditer(exp, self.contents) 
                    
        if results:            
            for each in results:
                #ensure each group found, is less than the limit
                num_chars = each.end() - each.start() - 2
                if num_chars > limit:
                    print('...error, speech event beyond limit..found one where %s %s chars %s'% (left_id,num_chars,right_id))
                    found=True
                    break
        else:
            print('...no non speech events found over the limit of %s chars between %s and %s' %(limit,left_id,right_id))
            found=False

        return found  

    def _search_replace(self, replace_regex_str=r' ', search_regex_str=r'[ ]{1,}'):
        '''
        Do regular expression substitution on self.contents

        Parameters
        ----------
        replace_regex_str : string
            expression that will replace the search_regex_str if found in 
            self.contents
            Default: r' '
        search_regex_str : string
            expression that will search in self.contents, if found will be
            replaced with replace_regex_str. self.contents will be updated with 
            this new version.
            Default: r'[ ]{2,}'

        Returns
        -------
        (bool, string)
            Returns tuple of ("success in replacement","the update/non updated string")
            status of True means the string was found and updated
            status of False means the string was not found and/or noting updated

        ''' 
        exp_search = re.compile(search_regex_str.encode())
        results = re.sub(exp_search,replace_regex_str.encode(),self.contents) 
    
        #ensure our replaced expression can NOT be found in the result 
        #...and if results != original aka no match        
        if exp_search.findall(results):
            if results != self.contents:                
                print ('...replaced expression "%s" with expression "%s" in "%s"'% (search_regex_str,replace_regex_str,self.contents.decode()))
                print ('...replaced results: "%s"' % results.decode())
                self.contents = results
                return (True,results)  
            else:
                print('...did not find matching expression (nothing replaced)')
                return(False,results)
        else:        
            print ('...warning, expression "%s" was NOT replaced with "%s" in "%s"' % (search_regex_str,replace_regex_str,self.contents))        
            return (False,results)  
    
    def _search_replace_from_listing(self):
        ''' Apply replacements from self.replacements to self.contents '''
        def symbol_as_space(v):
            ignore_sym=['[', ']', ',']
            for each in v:
                if each not in ignore_sym: 
                    self.contents = self.contents.replace(bytearray(each.encode()),bytearray(''.encode()))            
        
        if self.contents and self.replacements:
            for k,v in self.replacements.items(): 
                if k == 'SYMBOL_AS_SPACE':
                    symbol_as_space(v)
                else:
                    #include a space around replacement, else replacement is 
                    #bunched up with near chars
                    v_mod=' %s'%v
                    self.contents = self.contents.replace(bytearray(k.encode()),bytearray(v_mod.encode()))

    def does_table_exist(self, table, caslib=None):
        ''' Return True or False indicating table exists '''
        caslib = caslib if caslib else None
        return bool(dict(self.conn.table.tableexists(table,caslib,_messagelevel=None))['exists'])     

    def drop_table(self, table, caslib=None):
        ''' Drop the specified table '''
        caslib = caslib if caslib else None
        return self.conn.droptable(table,caslib) 

    def create_castable(self, data, name, caslib=None, replace=False, promote=False,
                        has_geo=False, geo_fields=[], decimal_context=True,
                        col_names=None):
        '''
        Build a CASTable provided a dictionary
        
        If any value is a list type, it will be converted to JSON String
        This is considering the intended result to be a mv field for search
        based actions and processing

        '''

        if_needed = 0

        caslib = caslib if caslib != None else None
        tbl=None     
        if not self.conn:
            print('...ERROR: do not have a valid connection to existing server instance.')
        else:
            if replace:   
                if self.does_table_exist(name,caslib=caslib): #global
                    self.conn.droptable(name,caslib=caslib) 
                    if self.does_table_exist(name,caslib=caslib): #session
                        self.conn.droptable(name,caslib=caslib)    

            if if_needed:
                for each in data:
                    for k, v in each.items():
                        if isinstance(v,list):
                            print('...located mv field (%s) in targeted table (%s), converting value to JSON String' % (k,name))
                            each[k]=json.dumps(v)

                        #Handle special case with search geofields.
                        #Search expects the contents to be json string not json
                        if has_geo:
                            if k in geo_fields:
                                print('...found a geo field, adjusting for CASTable')
                                each[k]=json.dumps(v)

                if decimal_context:
                    #adjust context for floating point math (aka make Decimal the default)
                    #Adjust by transformation via the json loads call.
                    data = json.loads(json.dumps(data),
                                      parse_int=Decimal,
                                      parse_float=Decimal)
                                                            
            try:
                tbl = swat.CASTable.from_dict(self.conn, data,
                                              casout={'name': name, 'caslib': caslib, 'promote': promote})
                
                #reshape columns if needed
                if col_names:
                    tbl=tbl[col_names]
                                
                                
            except swat.SWATError:
                print("...error:action was not successful (confirm parameters and caslib)")
    
            if self.does_table_exist(name,caslib=caslib):
                print("...located newly created table %s in caslib %s " % (name,caslib))
            else:
                print("...something happened, unable to locate table %s in caslib %s " % (name,caslib))

            return name


    def process_contents(self, show_results=False, build_astable=False, audio_path=None):
        '''
        Main method that processes the self.contents

        All supporting functions are called within this function.
        
        Parameters
        ----------
        show_results : bool, optional
            If you will to see the final contents after processing.
        build_astable : bool, optional
            ???
        audio_path : string, optional
            ???

        Returns
        -------
        ???

        '''
        found = self._search_non_speech_events()
        response = {}
        results = []
        if not found:
            #replace spaces between words with &
            self._search_replace(replace_regex_str=r'&',search_regex_str=r'[ ]{1,}')
            #replace based on data_clean_replacements listing
            self._search_replace_from_listing() 

            #each line....
            for each in self.contents.decode().split('\n'):
                #split at the file extension
                _items = each.split('wav')      
                if len(_items) == 2:
                    #clean up contents
                    _items[1]=_items[1].strip('\t').strip()
                    results.append({'_filename_':'%s.wav' % _items[0],'ylen':_items[1]})
                else:
                    print ('...WARNING: did not find full results for line (empty or sparse line?):%s'%_items)

            if show_results:
                for each in self.contents.split(b'\n'):
                    print (each)
        else:            
            print('...WARNING: found speech events beyond limit')

        longest_line = max(self.contents.decode().split('\n'), key=len)
        actual_line=longest_line.split('wav')[1].strip('\t').strip()
        longest_line_len = len(actual_line)
        
        print ('...found to the longest line with text length of %s' % longest_line_len)
        print ('...%s'%actual_line)

        #Iterate over ylen items
        c_response = []
        for each_row in results:
            col_heading = [{'y%s'%x:' '} for x in range(longest_line_len)]
  
            for idx,ylen_c in list(enumerate(each_row['ylen'])):
                col_heading[idx]['y%s'%idx]=ylen_c                                   
            tmp_dict = {'_filename_':each_row['_filename_']}            
 
 
            for each in col_heading:               
                #pad any remaining char in the row with ~
                for k,v in each.items():
                    if v == ' ':
                        each[k]='~'
                tmp_dict.update(each) 
            c_response.append(tmp_dict)
           
            tmp_dict['ylen']=len(each_row['ylen'])
            if audio_path:            
                tmp_dict['_path_']=os.path.join(audio_path,each_row['_filename_'])
            else:
                tmp_dict['_path_']=None 
        
        col_names=['y%s'%x for x in range(longest_line_len)]
        col_names=['_filename_']+col_names 
        col_names.append('ylen')
        col_names.append('_path_')

        #add key to response
        response['longest_line'] = longest_line
        response['longest_line_len'] = longest_line_len
        response['col_names']= col_names
        response['results'] = c_response

        return response
