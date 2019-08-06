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

from swat import *
import pandas as pd
from dlpy.model import *
from dlpy.applications import *
from dlpy.images import ImageTable
import matplotlib.image as mpimg
from dlpy.utils import DLPyError
import random

def get_image_features(conn,model,image_table,dense_layer,target='_filename_0'):
    '''
    Generate CASTable of image features

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model: dlpy Model object
        Specifies CNN model to use for extracting features
    image_table: imageTable
        Specifies name of CASTable that contains images to be used for training
    dense_layer: string
        Specifies layer from CNN model to extract features from
    target: string, optional
        Specifies the name of the column containing the response variable
        Default: '_filename_0'

    Returns
    -------
    :class:`CASTable`

    '''
    width = model.summary['Output Size'][0][1]
    height = model.summary['Output Size'][0][0]
    image_table.resize(width=width,height=height)

    if dense_layer not in list(model.summary['Layer']):
        raise DLPyError('Specified dense_layer not a layer in model')

    X, y = model.get_features(data=image_table,dense_layer=dense_layer,target=target)

    # initialize dictionary with columns
    table_dict = {}
    for i in range(len(X[0])):
        table_dict['f{}'.format(i)] = list()
    # add filenames to dict
    table_dict[target] = list()
    for file in y:
        table_dict[target].append(file)
    # add features to dict
    for var in table_dict[target]:
        idx = list(y).index(var)
        X_list = X[idx]
        for i in range(len(X[0])):
            table_dict['f{}'.format(i)].append(X_list[i])
    features = CASTable.from_dict(conn,table_dict)

    return features


def create_captions_table(conn, captions_file, caption_col_name='Var', delimiter='\t'):
    '''
    Generate CASTable of captions and filenames

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    captions_file : string
        Specifies absolute path to file containing image filenames and captions.
        This file has to be accessible from the client.
    caption_col_name : string, optional
        Specifies base name of columns that contain captions
        Default : 'Var'
    delimiter : string, optional
        Specifies delimiter in the captions_file between captions
        Default : '\t'

    Returns
    -------
    :class:`CASTable`

    '''
    captions_dict = dict()
    line_list = []

    # read file lines into large list
    with open(captions_file, 'r') as readFile:
        for line in readFile:
            line_list.append(line)

    # find number of captions
    num_captions = len(line_list[0].split(delimiter)) - 1
    if num_captions == 0:
        raise DLPyError('Something went wrong with the captions file -'
                        ' most likely the wrong delimiter was specified or'
                        ' the captions file is incorrectly formatted')

    # initialize dictionary
    captions_dict['_filename_0'] = list()
    for i in range(num_captions):
        captions_dict['{}{}'.format(caption_col_name,i)] = list()

    # add filenames and captions to dictionary
    for line in line_list:
        items = line.split(delimiter)
        captions_dict['_filename_0'].append(items[0])
        for j in range(num_captions):
            captions_dict['{}{}'.format(caption_col_name,j)].append(items[j+1].strip())
    captions = CASTable.from_dict(conn, captions_dict)

    return captions


def create_embeddings_from_object_detection(conn, image_table, detection_model, word_embeddings_file,
                                            n_threads=None,gpu=None,max_objects=5, word_delimiter='\t'):
    '''
    Builds CASTable with objects detected in images as numeric data

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    image_table: imageTable
        Specifies name of CASTable that contains images to be used for training
    detection_model : CASTable or string
        Specifies CASTable containing model parameters for the object detection model
    word_embeddings_file : string
        Specifies full path to file containing pre-trained word vectors to be used for text generation
        This file should be accessible from the client.
    n_threads : int, optional
        Specifies the number of threads to use when scoring the table. All cores available used when
        nothing is set.
        Default : None
    gpu : Gpu, optional
        When specified, specifies which gpu to use when scoring the table. GPU=1 uses all available
        GPU devices and default parameters.
        Default : None
    max_objects : int, optional
        Specifies max number of objects detected if less than five
        Default : 5
    word_delimiter : string, optional
        Specifies delimiter used in word_embeddings file
        Default : '\t'
    Returns
    -------
    :class:`CASTable`

    '''
    if not os.path.exists(word_embeddings_file):
        raise DLPyError('word_embeddings_file does not exist')

    if not isinstance(image_table,ImageTable):
        raise DLPyError('image_table must be an ImageTable object')

    conn.loadactionset('deepLearn')
    conn.loadactionset('textparse')

    width = detection_model.summary['Output Size'][0][1]
    height = detection_model.summary['Output Size'][0][0]
    image_table.resize(width=width,height=height)
    scoring_error = False
    try:
        scored = detection_model.predict(data=image_table,n_threads=n_threads,gpu=gpu)
    except:
        scoring_error = True
    if scoring_error or scored is None:
        raise DLPyError('Something went wrong while scoring the data.')

    object_table = detection_model.valid_res_tbl
    # combine first n objects into single column
    first_objects = object_table.copy()

    first_objects['first_objects'] = first_objects['_Object0_'] + ","
    if max_objects>5:
        max_objects = 5
    for i in range(1,max_objects):
        objects = first_objects['_Object{}_'.format(i)] + ","
        first_objects['first_objects'] = first_objects['first_objects'].add(objects)

    objects_numeric = numeric_parse_text(conn,
                                         first_objects,
                                         word_embeddings_file,
                                         word_delimiter=word_delimiter)

    # merge objects table and numeric table
    df1 = objects_numeric.to_frame()
    df2 = first_objects.to_frame()
    objects = pd.merge(df1, df2, left_on='_id_', right_on='_id_', how='left')

    objects = conn.upload_frame(objects,casout=dict(name='objects',replace=True))
    # remove unnecessary columns
    useful_vars = list(objects_numeric.columns)
    useful_vars.append('_filename_0')
    useful_vars.append('first_objects')
    bad_columns = set(list(objects.columns)) - set(useful_vars)
    final_objects = objects.drop(bad_columns, axis=1)

    return final_objects


def numeric_parse_text(conn,table,word_embeddings_file,word_delimiter='\t',parse_column='first_objects'):
    '''
    Parses text data into numeric data using a word-embeddings file

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    table: CASTable
        Specifies table containing data to be parsed
    word_embeddings_file : string
        Specifies path to file containing word embeddings
    word_delimiter : string, optional
        Specifies delimiter used in word embeddings file
        Default : '\t'
    parse_column : string, optional
        Specifies name of column containing text data to be parsed
        Default : 'first_objects'
    Returns
    -------
    :class:`CASTable`

    '''
    # parse object text into numeric data
    conn.upload_file(data=word_embeddings_file,
                     casout=dict(name='word_embeddings', replace=True),
                     importOptions=dict(fileType='delimited', delimiter=word_delimiter, getNames=True, guessRows=2,
                                        varChars=True)
                     )
    conn.tpParse(table=table, docId='_id_', entities='NONE', text=parse_column, nounGroups=False,
                 offset=dict(name='pos_output', replace=True))
    conn.applyWordVector(casout=dict(name='objects_all_numeric', replace=True),
                         modelTable=dict(name='word_embeddings'), table=dict(name='pos_output'))
    conn.altertable('objects_all_numeric', columns=[dict(name='_Document_', rename='_id_')])
    objects_numeric = conn.CASTable('objects_all_numeric')

    return objects_numeric

def reshape_caption_columns(conn,table,caption_col_name='Var',num_captions=5,):
    '''
    Reshapes table so there is only one caption per row of the table

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    table : CASTable or string
        Specifies name of CASTable containing the merged captions, features, and objects
    caption_col_name : string, optional
        Specifies basename of columns that contain captions
        Default : 'Var'
    num_captions : int, optional
        Specifies number of captions per image
        Default : 5

    Returns
    -------
    :class:`CASTable`

    '''
    # convert table to one caption per line
    columns = list(table.columns)
    if '{}0'.format(caption_col_name) not in columns:
        raise DLPyError('caption_col_name {} does not exist in the table'.format(caption_col_name))
    capt_idx_start = columns.index('{}0'.format(caption_col_name))

    # initialize new_tbl dictionary with columns
    new_tbl = dict()
    for c in columns:
        if caption_col_name not in c:
            new_tbl[c] = []
    new_tbl['caption'] = []

    # make list of rows containing only one caption each
    new_tbl_list = list()
    rows = (table.values).tolist()
    try:
        for row in rows:
            for i in range(num_captions):
                new_row = []
                for j in range(len(row)):
                    if j not in range(capt_idx_start, capt_idx_start + num_captions):
                        new_row.append(row[j])
                new_row.append(row[capt_idx_start + i])
                new_tbl_list.append(new_row)
    except IndexError:
        raise DLPyError("Wrong number of captions specified")

    # add values to dictionary
    for row in new_tbl_list:
        cnt = 0
        for key in new_tbl.keys():
            new_tbl[key].append(row[cnt])
            cnt += 1

    # create CASTable from dictionary
    rnn_input = CASTable.from_dict(conn,new_tbl)

    return rnn_input


def create_captioning_table(conn, image_table, features_model, captions_file,
                            obj_detect_model=None,word_embeddings_file=None,
                            num_captions=5,dense_layer='fc7',captions_delimiter='\t', 
                            caption_col_name='Var',embeddings_delimiter='\t',n_threads=None,gpu=None):
    '''
    Builds CASTable wtih all necessary info to train an image captioning model

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    image_table: imageTable
        Specifies name of CASTable that contains images to be used for training
    features_model : dlpy Model object
        Specifies CNN model to use for extracting features
    captions_file : string
        Specifies absolute path to file containing image filenames and captions
        Client should have access to this file.
    obj_detect_model : CASTable or string, optional
        Specifies CASTable containing model parameters for the object detection model
        Default : None
    word_embeddings_file : string, optional
        Specifies full path to file containing pre-trained word vectors to be used for text generation.
        This file should be accessible from the client.
        Required if obj_detect_model is not None
        Default : None
    num_captions : int, optional
        Specifies number of captions for each image in the captions file
        Default : 5
    dense_layer: string, optional
        Specifies layer from CNN model to extract features from
        Default : 'fc7'
    captions_delimiter : string, optional
        Specifies delimiter between filenames and captions in the image captions text file
        Default : '\t'
    caption_col_name : string, optional
        Specifies base name for column names for the columns containing captions
        Default : 'Var'
    embeddings_delimiter : string, optional
        Specifies delimiter used in word embeddings file
        Default : '\t'
    n_threads : int, optional
        Specifies the number of threads to use when scoring the table. All cores available used when
        nothing is set.
        Default : None
    gpu : Gpu, optional
        When specified, specifies which gpu to use when scoring the table. GPU=1 uses all available
        GPU devices and default parameters.
        Default : None

    Returns
    -------
    :class:`CASTable`

    '''
    # get all necessary tables
    image_features = get_image_features(conn,features_model,image_table,dense_layer)
    captions_table = create_captions_table(conn,captions_file,delimiter=captions_delimiter,caption_col_name=caption_col_name)

    # merge features and captions tables
    df1 = captions_table.to_frame()
    df2 = image_features.to_frame()
    captions_features = pd.merge(df1,df2,left_on='_filename_0',right_on='_filename_0',how='left')
    result = conn.upload_frame(captions_features,casout=dict(name='captions_features',replace=True))
    # conn.dljoin(table=captions_table,annotatedTable=image_features,
    #             id='_filename_0',casOut=dict(name='captions_features',replace=True))
    # result = conn.CASTable('captions_features')

    if obj_detect_model is not None:
        if word_embeddings_file is None:
            raise DLPyError("word_embeddings_file required for object detection")
        else:
            # resize images for object detection scoring
            detected_objects = create_embeddings_from_object_detection(conn,image_table,obj_detect_model,word_embeddings_file,word_delimiter=embeddings_delimiter,
                                                      n_threads=n_threads,gpu=gpu)
            # conn.dljoin(table=dict(name='captions_features'),annotatedTable=detected_objects,
            #             id='_filename_0',casOut=dict(name='obj_capt_feats',replace=True))
            df1 = detected_objects.to_frame()
            df2 = result.to_frame()
            obj_capt_feat = pd.merge(df1,df2,left_on='_filename_0',right_on='_filename_0',how='left')
            result = conn.upload_frame(obj_capt_feat,casout=dict(name='full_table',replace=True))

    final_table = reshape_caption_columns(conn,result,caption_col_name=caption_col_name,num_captions=num_captions)
    drop_columns = set(final_table.columns) - set(captions_table.columns) - set(image_features.columns)
    if obj_detect_model:
        drop_columns = set(drop_columns) - set(detected_objects.columns)
    drop_columns.remove('caption')
    final_table.drop(drop_columns,axis=1,inplace=True)

    return final_table

def ImageCaptioning(conn,model_name='image_captioning',num_blocks=3,neurons=50,
                    rnn_type='LSTM',max_output_len=15):
    '''
    Builds an RNN to be used for image captioning

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    model_name : string, optional
        Specifies output name of the model
        Default: 'image_captioning'
    num_blocks : int, optional
        Specifies number of samelength recurrent layers
        Default : 3
    neurons : int, optional
        Specifies number of neurons in each layer
        Default : 50
    rnn_type : string, optional
        Specifies the type of the rnn layer. Possible Values: RNN, LSTM, GRU
        Default: LSTM
    max_output_len : int, optional
        Specifies max number of tokens to generate in the final layer (i.e. max caption length)
        Default : 15
    Returns
    -------
    :class:`CASTable`

    '''
    if num_blocks < 1:
        raise DLPyError('num_blocks must be greater than 1')

    model = Sequential(conn, model_table=model_name)

    model.add(InputLayer(name='input'))
    print('InputLayer added named "input"')
    for i in range(num_blocks):
        model.add(Recurrent(n=neurons, init='msra', rnn_type=rnn_type, output_type='samelength'))

    model.add(Recurrent(n=neurons, init='msra', rnn_type=rnn_type, output_type='encoding'))

    model.add(Recurrent(n=neurons, init='msra', rnn_type=rnn_type, output_type='arbitrarylength',
                        max_output_length=max_output_len))

    model.add(OutputLayer(name='output'))
    print('OutputLayer added named "output"')
    return model

def display_predicted_image_captions(conn,result_tbl,npreds=2,ncol=2,img_path=None,figsize=None):
    '''
    Shows caption prediction for random images

    Parameters
    ----------
    conn : CAS
        Specifies the CAS connection object.
    result_tbl : CASResults object
        Table containing results from scoring the test data
    npreds : int, optional
       Specifies number of caption predictions to show
        Default : 2
    ncol : int, optional
        Specifies number of columns to display images in
        Default : 2
    img_path : string, optional
        If used, specifies path to wanted_file to show images along with captions and objects.
        If None, only shows captions and objects
        Default : None
    figsize : tuple of ints, optional
        Specifies size of images to be displayed
        Default : (16,(16 / ncol*nrow))

    '''
    results = scored_results_to_dict(result_tbl)

    nimages = min(npreds, len(results))

    if img_path is None:
        for i in range(nimages):
            r = random.randint(0, len(results) - 1)

            f_name = list(results.keys())[r]
            actual_caps = (conn.CASTable(result_tbl.name, where='''_filename_0="{}"'''.format(f_name)).iloc[:, 'caption']).values
            truth = "\n\t".join(actual_caps)
            objects = (conn.CASTable(result_tbl.name, where='''_filename_0="{}"'''.format(f_name)).iloc[:, 'first_objects']).values
            objects = "\n\t".join(objects[0].split(','))
            rand_row = results[f_name]
            prediction = rand_row[1]
            print("Filename: {}\nObjects: {}\nGround Truth: {}\nPredicted: {}\n".format(f_name, objects, truth,
                                                                                        prediction))
    else:
        if nimages > ncol:
            nrow = nimages // ncol + 1
        else:
            nrow = 1
            ncol = nimages
        if figsize is None:
            figsize = (16, 16 // ncol * nrow)
        fig = plt.figure(figsize=figsize)

        for i in range(nimages):
            r = random.randint(0, len(results) - 1)
            f_name = list(results.keys())[r]
            rand_row = results[f_name]
            actual_caps = (conn.CASTable(result_tbl.name, where='''_filename_0="{}"'''.format(f_name)).iloc[:, 'caption']).values
            truth = "\n".join(actual_caps)
            objects = (conn.CASTable(result_tbl.name, where='''_filename_0="{}"'''.format(f_name)).iloc[:, 'first_objects']).values
            objects = objects[0]
            caption = rand_row[1]
            if '/' in img_path:
                image = '{}/{}'.format(img_path, f_name)
            elif '\\' in img_path:
                image = '{}\{}'.format(img_path, f_name)
            else:
                raise DLPyError('img_path given is not a valid path')
            image = mpimg.imread(image)
            ax = fig.add_subplot(nrow, ncol, i + 1)
            ax.set_title('Objects: {}\nGround Truth: {}\nPredicted: {}'.format(objects, truth, caption))
            plt.imshow(image)
            plt.xticks([]), plt.yticks([])
        plt.show()

def scored_results_to_dict(result_tbl):
    '''
    Converts results in CASResults table to a dictionary of values

    Parameters
    ----------
    result_tbl : CASResults object
        Table containing results from scoring the test data

    Returns
    -------
    dict

    '''
    exists = True
    try:
        result_columns = list(result_tbl.columns)
    except:
        exists = False
    if exists is False:
        raise DLPyError('Specified result_tbl could not be located in the caslib')

    filename_idx = result_columns.index('_filename_0')
    caption_idx = result_columns.index('caption')
    prediction_idx = result_columns.index('_DL_Pred_')

    result_values = dict()
    for row in list(result_tbl.values):
        tuple1 = [row[caption_idx].strip(), row[prediction_idx].strip()]
        result_values[row[filename_idx]] = tuple(tuple1)

    return result_values


def get_max_capt_len(captions_file, delimiter='\t'):
    '''
    Finds maximum length of captions from file containing

    Parameters
    ----------
    captions_file : string
        Specifies physical path to file containing ground truth image captions. This has
        to be client accesible.
    delimiter : string, optional
        Specifies delimiter between captions and filenames in captions_file
        Default : '\t'

    Returns
    -------
    int

    '''
    max_cap_len = 0
    with open(captions_file, 'r') as readFile:
        for line in readFile:
            captions = line.split(delimiter)[1:]
            if len(captions) < 1:
                raise DLPyError("Error with captions file or delimiter")
            for cap in captions:
                if len(cap.split()) > max_cap_len:
                    max_cap_len = len(cap.split())
    return max_cap_len
