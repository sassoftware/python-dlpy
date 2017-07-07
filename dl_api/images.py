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
import matplotlib.pyplot as plt
import numpy as np
from .utils import random_name


class Image:
    '''
    An Image class, that support the following functions:
    load, train_test_split, display, crop, resize, patches, summary, freq


    Parameters:

    ----------
    sess :
        Specifies the session of the CAS connection.
    path : string
        Specifies the path of the image data.
        Note: images should be save in the form of parent_dir/label/image_files.
    blocksize : int
        Specifies the number of bytes to use for blocks that are read by threads.
        Default: 64


    Returns

    -------
    An image class in the session.
    '''

    def __init__(self, sess, path=None, blocksize=64):
        if not sess.queryactionset('image')['image']:
            sess.loadactionset('image')
        self.path = path
        self.sess = sess

        self.patch_level = 0

        CAS_TblName = random_name()
        self.tbl = dict(name=CAS_TblName)

        if path is not None:
            self.load(path, blocksize=blocksize)

    def load(self, path, blocksize=64, **kwargs):
        '''
        Function to load images

        Parameters:

        ----------
        path : string
            Specifies the path of the image data.
            Note: images should be save in the form of parent_dir/label/image_files.
        blocksize : int
            Specifies the number of bytes to use for blocks that are read by threads.
            Default: 64
        kwargs: dictionary, optional
            Specify the optional arguments for the loadimages action.


        Returns

        -------
        Load the images in the specified directory into the CAS table name by Image.tbl.
        '''

        sess = self.sess

        sess.image.loadimages(
            casout=dict(**self.tbl, replace=True, blocksize=blocksize),
            distribution=dict(type='random'), recurse=True, labelLevels=-1,
            path=path, **kwargs)

        computedvars = '_filename_0'
        SASCode = "length _filename_0 varchar(*);\
                   _loc1 = LENGTH(_path_) - INDEX(REVERSE(_path_),'/')+2;\
                   _filename_0 = SUBSTR(_path_,_loc1);"
        sess.partition(casout=dict(**self.tbl, replace=True),
                       table=dict(**self.tbl, computedVars=computedvars,
                                  computedVarsProgram=SASCode))
        self._summary()

    def save(self, path):
        '''
        Function to save the images to the specified directory.


        Parameters:

        ----------
        path: string.
            Specifies the directory to save the images.
            Note: the directory must be accessible from the CAS server.


        Returns

        -------
        '''

        sess = self.sess

        CAS_LibName = random_name('Caslib', 6)
        sess.addCaslib(name=CAS_LibName, path=path, activeOnAdd=False)

        file_name = '_filename_{}'.format(self.patch_level)
        sess.image.saveImages(caslib=CAS_LibName,
                              images=vl(table=dict(**self.tbl), path=file_name),
                              labelLevels=1)
        sess.dropcaslib(caslib=CAS_LibName)

    def copy(self):
        '''
        Function to create a copy the image object.
        '''

        sess = self.sess
        Image_new = Image(sess)
        Image_new.path = self.path
        Image_new.patch_level = self.patch_level

        sess.partition(casout=dict(**Image_new.tbl, replace=True),
                       table=dict(**self.tbl))
        return Image_new

    def drop(self):
        '''
        Function to remove the image object and delete the associated CAS table.
        '''

        sess = self.sess

        sess.droptable(**self.tbl)

        del self.sess, self.patch_level, self.tbl, self.path, self.summary, \
            self.channel_means, self.label_freq

    def two_way_split(self, test_rate=20, blocksize=64):
        '''
        Function to split image data into training and testing sets.


        Parameters:

        ----------
        test_rate : double, between 0 and 100, optional.
            Specify the proportion of the testing data set, e.g. 20 mean 20% of the images will be in the testing set.
            Default : 20.
        blocksize : int
            Specifies the number of bytes to use for blocks that are read by threads.
            Default: 64


        Returns

        -------
        image_tr : an Image object containing the training data
        image_te : an Image object containing the testing data
        '''

        sess = self.sess

        if not sess.queryactionset('sampling')['sampling']:
            sess.loadactionset('sampling')

        partindname = random_name(name='PartInd_', lenth=2)
        sess.stratified(output=dict(casOut=dict(**self.tbl, blocksize=blocksize,
                                                replace=True),
                                    copyVars="ALL", PARTINDNAME=partindname),
                        samppct=test_rate, samppct2=100 - test_rate,
                        partind=True,
                        table=dict(**self.tbl, groupby='_label_'))

        image_tr = Image(sess)
        image_tr.path = self.path
        image_tr.patch_level = self.patch_level
        sess.partition(casout=dict(**image_tr.tbl, replace=True),
                       table=dict(**self.tbl, where='{}=2'.format(partindname),
                                  groupby='_label_'))

        image_te = Image(sess)
        image_te.path = self.path
        image_te.patch_level = self.patch_level

        sess.partition(casout=dict(**image_te.tbl, replace=True),
                       table=dict(**self.tbl, where='{}=1'.format(partindname),
                                  groupby='_label_'))
        image_tr._summary()
        image_te._summary()

        return image_tr, image_te

    def three_way_split(self, valid_rate=20, test_rate=20, blocksize=64):
        '''
        Function to split image data into training and testing sets.


        Parameters:

        ----------
        valid_rate : double, between 0 and 100, optional.
            Specify the proportion of the validation data set,
            e.g. 20 mean 20% of the images will be in the validation set.
            Default : 20.
        test_rate : double, between 0 and 100, optional.
            Specify the proportion of the testing data set,
            e.g. 20 mean 20% of the images will be in the testing set.
            Default : 20.
            Note: the total of valid_rate and test_rate cannot be exceed 100
        blocksize : int
            Specifies the number of bytes to use for blocks that are read by threads.
            Default: 64


        Returns

        -------
        image_tr : an Image object containing the training data
        image_te : an Image object containing the testing data
        '''

        sess = self.sess

        if not sess.queryactionset('sampling')['sampling']:
            sess.loadactionset('sampling')

        partindname = random_name(name='PartInd_', length=2)
        sess.stratified(output=dict(casOut=dict(**self.tbl, blocksize=blocksize,
                                                replace=True),
                                    copyVars="ALL", PARTINDNAME=partindname),
                        samppct=valid_rate, samppct2=test_rate,
                        partind=True,
                        table=dict(**self.tbl, groupby='_label_'))

        image_tr = Image(sess)
        image_tr.path = self.path
        image_tr.patch_level = self.patch_level
        sess.partition(casout=dict(**image_tr.tbl, replace=True),
                       table=dict(**self.tbl, where='{}=0'.format(partindname),
                                  groupby='_label_'))

        image_valid = Image(sess)
        image_valid.path = self.path
        image_valid.patch_level = self.patch_level
        sess.partition(casout=dict(**image_valid.tbl, replace=True),
                       table=dict(**self.tbl, where='{}=1'.format(partindname),
                                  groupby='_label_'))

        image_te = Image(sess)
        image_te.path = self.path
        image_te.patch_level = self.patch_level
        sess.partition(casout=dict(**image_te.tbl, replace=True),
                       table=dict(**self.tbl, where='{}=2'.format(partindname),
                                  groupby='_label_'))
        image_tr._summary()
        image_valid._summary()
        image_te._summary()

        return image_tr, image_valid, image_te

    def display(self, nimages=5, ncol=8, random_flag=False):
        '''
        Function to display images.

        Parameters:

        ----------
        nimages : int, optional.
            Specify the number of images to be displayed. If nimage is greater than the maximum number of images in the
            table, it will be set to this maximum number.
            Default : 5.
            Note: large nimages costs a lot of memory to display.
        ncol : int, optional.
            Specifies the layout of the display, determine the number of columns in the plots.
            Default: 8.
        random_flag: boolean, optional.
            Specifies whether to randomly choose the images for display.
            Default: False


        Returns

        -------
        Plot the specified number of images.

        '''

        sess = self.sess

        nmax = sess.CASTable(**self.tbl).numrows().numrows
        nimages = min(nimages, nmax)

        if random_flag:
            temp_tbl = sess.fetchimages(imagetable=sess.CASTable(
                **self.tbl,
                computedvars=['random_index'],
                computedvarsprogram="call streaminit(-1);random_index=rand(\"UNIFORM\");"),
                sortby='random_index', to=nimages)
        else:
            temp_tbl = sess.fetchimages(imagetable=sess.CASTable(**self.tbl), to=nimages)

        if nimages > ncol:
            nrow = nimages // ncol + 1
        else:
            nrow = 1
            ncol = nimages
        fig = plt.figure(figsize=(16, 16 // ncol * nrow))

        for i in range(nimages):
            image = temp_tbl.Images.Image[i]
            label = temp_tbl.Images.Label[i]
            image = np.asarray(image)
            ax = fig.add_subplot(nrow, ncol, i + 1)
            ax.set_title('{}'.format(label))
            plt.imshow(image)
            plt.xticks([]), plt.yticks([])

    def crop(self, x=0, y=0, width=None, height=None, replace=True):
        '''
        Function to crop images.

        Parameters:

        ----------
        x : int, optional.
            Specify the x location of the top left corn of the cropped images.
            Default : 0.
        y : int, optional.
            Specify the y location of the top left corn of the cropped images.
            Default : 0.
        width : int, optional.
            Specify the width of the cropped images.
            Default : 224.
        height : int, optional.
            Specify the height of the cropped images.
            If not specified, height will be set to be equal to width.
            Default : None.
        replace: boolean, optional.
            Specifies whether to update the original images or create a new set of images.
            Default: True


        Returns

        -------
        If replace = Flase, it will return a new Image object with the cropped images.
        '''
        sess = self.sess

        if (width is None) and (height is None):
            width = 224
        if width is None:
            width = height
        if height is None:
            height = width

        image_table = self.tbl

        column_names = ['_filename_{}'.format(i) for i in range(self.patch_level + 1)]
        if replace:
            sess.processimages(
                imageTable=image_table,
                copyVars=column_names,
                casout=dict(**self.tbl, replace=True),
                imagefunctions=[dict(functionoptions=
                                     dict(functionType="GET_PATCH", x=x, y=y,
                                          w=width, h=height))])
            self._summary()
        else:
            Image_new = self.copy()
            Image_new.crop(x=x, y=x, width=width, height=height)

            return Image_new

    def resize(self, width=None, height=None, replace=True):
        '''
        Function to resize images.

        Parameters:

        ----------
        width : int, optional.
            Specify the target width of the resized images.
            Default : 224.
        height : int, optional.
            Specify the target height of the resized images.
            If not specified, height will be set to be equal to width.
            Default : None.
        replace: boolean, optional.
            Specifies whether to update the original images or create a new set of images.
            Default: True


        Returns

        -------
        If replace = Flase, it will return a new Image object with the resized images.
        '''

        sess = self.sess

        if (width is None) and (height is None):
            width = 224
        if width is None:
            width = height
        if height is None:
            height = width

        image_table = self.tbl

        column_names = ['_filename_{}'.format(i) for i in range(self.patch_level + 1)]

        if replace:
            sess.processimages(
                imageTable=image_table,
                copyVars=column_names,
                casout=dict(**self.tbl, replace=True),
                imagefunctions=[dict(functionoptions=
                                     dict(functionType="RESIZE",
                                          w=width, h=height))])
            self._summary()
        else:
            Image_new = self.copy()
            Image_new.resize(width=width, height=height)
            return Image_new

    def patches(self, x=0, y=0, width=None, height=None, stepSize=None,
                outputWidth=None, outputHeight=None, replace=True):
        '''
        Function to get patches from the images.

        Parameters:

        ----------
        x : int, optional.
            Specify the x location of the top left corn of the first patches.
            Default : 0.
        y : int, optional.
            Specify the y location of the top left corn of the first patches.
            Default : 0.
        width : int, optional.
            Specify the width of the patches.
            Default : 224.
        height : int, optional.
            Specify the width of the patches.
            If not specified, height will be set to be equal to width.
            Default : None.
        stepSize : int, optional.
            Specify the step size of the moving windows for extracting the patches.
            Default : None, meaning stepSize = width.
        outputWidth : int, optional.
            Specify the output width of the patches.
            If not equal to width, the patches will be resize to the output width.
            Default : None, meaning outputWidth = width.
        outputHeight : int, optional.
            Specify the output height of the patches.
            If not equal to height, the patches will be resize to the output height.
            Default : None, meaning outputHeight = height.
        replace: boolean, optional.
            Specifies whether to update the original images or create a new set of images.
            Default: True


        Returns

        -------
        If replace = Flase, it will return a new Image object with the patches of the images.
        '''

        sess = self.sess

        if (width is None) and (height is None):
            width = 224
        if width is None:
            width = height
        if height is None:
            height = width
        if stepSize is None:
            stepSize = width
        if outputWidth is None:
            outputWidth = width
        if outputHeight is None:
            outputHeight = height

        image_table = self.tbl
        croplist = [dict(sweepImage=True, x=x, y=y,
                         width=width, height=height,
                         stepSize=stepSize,
                         outputWidth=outputWidth,
                         outputHeight=outputHeight)]

        column_names = ['_filename_{}'.format(i) for i in range(self.patch_level + 1)]

        if replace:
            sess.augmentImages(imageTable=image_table,
                               copyVars=column_names,
                               casout=dict(**self.tbl, replace=True),
                               cropList=croplist)
            # The following code generate the latest file name according to the number of patches operations.
            computedvars = '_filename_{}'.format(self.patch_level + 1)
            SASCode = "length _filename_{} varchar(*); ".format(self.patch_level + 1) + \
                      "dot_loc = LENGTH(_filename_{0}) - INDEX(REVERSE(_filename_{0}),'.')+1; ".format(
                          self.patch_level) + \
                      "_filename_{1} = SUBSTR(_filename_{0},1,dot_loc-1)||".format(
                          self.patch_level, self.patch_level + 1) + \
                      "compress('_'||x||'_'||y||SUBSTR(_filename_{0},dot_loc)); ".format(
                          self.patch_level, self.patch_level + 1)
            sess.partition(casout=dict(**self.tbl, replace=True),
                           table=dict(**self.tbl,
                                      computedVars=computedvars,
                                      computedVarsProgram=SASCode))

            self.patch_level += 1
            self._summary()

        else:
            Image_new = self.copy()
            Image_new.patches(x=x, y=y, width=width, height=height, stepSize=stepSize,
                              outputWidth=outputWidth, outputHeight=outputHeight)

            return Image_new

    def random_patches(self, random_ratio=0.5, x=0, y=0, width=None, height=None,
                       stepSize=None, outputWidth=None, outputHeight=None, replace=True):
        '''
        Function to generate patches from the images randomly.

        Parameters:

        ----------
        random_ratio: double, optional
            Specifies the proportion of the generated pateches to output.
            Default : 0.5.
        x : int, optional.
            Specify the x location of the top left corn of the first patches.
            Default : 0.
        y : int, optional.
            Specify the y location of the top left corn of the first patches.
            Default : 0.
        width : int, optional.
            Specify the width of the patches.
            Default : 224.
        height : int, optional.
            Specify the width of the patches.
            If not specified, height will be set to be equal to width.
            Default : None.
        stepSize : int, optional.
            Specify the step size of the moving windows for extracting the patches.
            If not specified, it will be set to be equal to width.
            Default : None.
        outputWidth : int, optional.
            Specify the output width of the patches.
            If not specified, it will be set to be equal to width.
            Default : None.
        outputHeight : int, optional.
            Specify the output height of the patches.
            If not specified, it will be set to be equal to height.
            Default : None.
        replace: boolean, optional.
            Specifies whether to update the original images or create a new set of images.
            Default: True


        Returns

        -------
        If replace = Flase, it will return a new Image object with the patches of the images.
        '''

        sess = self.sess

        if (width is None) and (height is None):
            width = 224
        if width is None:
            width = height
        if height is None:
            height = width
        if stepSize is None:
            stepSize = width
        if outputWidth is None:
            outputWidth = width
        if outputHeight is None:
            outputHeight = height

        image_table = self.tbl
        croplist = [dict(sweepImage=True, x=x, y=y,
                         width=width, height=height,
                         stepSize=stepSize,
                         outputWidth=outputWidth,
                         outputHeight=outputHeight)]

        column_names = ['_filename_{}'.format(i) for i in range(self.patch_level + 1)]

        if replace:
            sess.augmentImages(imageTable=image_table,
                               copyVars=column_names,
                               casout=dict(**self.tbl, replace=True),
                               cropList=croplist,
                               randomRatio=random_ratio,
                               writeRandomly=True)
            # The following code generate the latest file name according to the number of patches operations.
            computedvars = '_filename_{}'.format(self.patch_level + 1)
            SASCode = "length _filename_{} varchar(*); ".format(self.patch_level + 1) + \
                      "dot_loc = LENGTH(_filename_{0}) - INDEX(REVERSE(_filename_{0}),'.')+1; ".format(
                          self.patch_level) + \
                      "_filename_{1} = SUBSTR(_filename_{0},1,dot_loc-1)||".format(
                          self.patch_level, self.patch_level + 1) + \
                      "compress('_'||x||'_'||y||SUBSTR(_filename_{0},dot_loc)); ".format(
                          self.patch_level, self.patch_level + 1)
            sess.partition(casout=dict(**self.tbl, replace=True),
                           table=dict(**self.tbl,
                                      computedVars=computedvars,
                                      computedVarsProgram=SASCode))
            self.patch_level += 1
            self._summary()

        else:
            Image_new = self.copy()
            Image_new.random_patches(random_ratio=random_ratio,
                                     x=x, y=y,
                                     width=width, height=height,
                                     stepSize=stepSize,
                                     outputWidth=outputWidth, outputHeight=outputHeight)

            return Image_new

    def _summary(self):
        '''
        Functioin to summarize the image table.
        '''
        sess = self.sess

        summary = sess.image.summarizeimages(imageTable=self.tbl)
        label_freq = sess.freq(table=self.tbl, inputs='_label_')
        channel_means = summary['Summary'].ix[0, ['mean1stChannel', 'mean2ndChannel', 'mean3rdChannel']].tolist()

        self.summary = summary
        self.label_freq = label_freq
        self.channel_means = channel_means
