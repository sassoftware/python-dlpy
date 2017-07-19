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

import matplotlib.pyplot as plt
import numpy as np
from swat.cas.table import CASTable
from .utils import random_name


class ImageTable(CASTable):

    @classmethod
    def load_path(cls, conn, path, casout=None, blocksize=64, **kwargs):
        '''
        Create a new ImageTable using images in `path`

        Parameters
        ----------
        conn : CAS
            The CAS connection object
        path : string
            The path to the image directory on the server
        casout : dict, optional
            The output table specifications
        blocksize : int, optional
            The output table blocksize
        **kwargs : keyword arguments, optional
            Additional keyword arguments to the `image.loadimages` action

        Returns
        -------
        :class:`ImageTable`

        '''
        conn.loadactionset('image', _messagelevel='error')

        if casout is None:
            casout = {}

        if 'blocksize' not in casout:
            casout['blocksize'] = blocksize
        if 'name' not in casout:
            casout['name'] = random_name()

        conn.retrieve('image.loadimages', _messagelevel='error',
                      casout=casout,
                      distribution=dict(type='random'),
                      recurse=True, labellevels=-1,
                      path=path, **kwargs)

        code = "length _filename_0 varchar(*); " + \
               "_loc1 = LENGTH(_path_) - INDEX(REVERSE(_path_),'/')+2; " + \
               "_filename_0 = SUBSTR(_path_,_loc1);"

        conn.retrieve('table.partition', _messagelevel='error',
                      casout=dict(**casout, replace=True),
                      table=dict(**casout,
                                 computedvars=['_filename_0'],
                                 computedvarsprogram=code))
        conn.retrieve('sampling.shuffle', _messagelevel='error',
                      casout=dict(**casout, replace=True), table=casout)

        out = cls(**casout)
        out.set_connection(conn)
        return out

    def __init__(self, name, blocksize=64, **table_params):
        CASTable.__init__(self, name, **table_params)
        self.blocksize = blocksize
        self.patch_level = 0

    def save_images(self, path):
        '''
        Function to save the images to the specified directory

        Parameters:
        ----------
        path : string
            Specifies the directory on the server to save the images

        '''
        caslib = random_name('Caslib', 6)
        self._retrieve('addcaslib', name=caslib, path=path, activeonadd=False)

        file_name = '_filename_{}'.format(self.patch_level)
        self._retrieve('image.saveimages', caslib=caslib,
                       images=dict(table=self, path=file_name),
                       labellevels=1)

        self._retrieve('dropcaslib', caslib=caslib)

    def copy_table(self, casout=None):
        '''
        Function to create a copy the image object

        Parameters
        ----------
        casout : dict, optional
            Output CAS table parameters

        Returns
        -------
        :class:`ImageTable`

        '''
        if casout is None:
            casout = {}

        out = self._retrieve('table.partition', casout=casout, table=self)['casTable']

        out = type(self)(**out.params)
        out.set_connection(self.get_connection())
        out.path = self.path
        out.patch_level = self.patch_level

        return out

    def two_way_split(self, test_rate=20, blocksize=None):
        '''
        Function to split image data into training and testing sets

        Parameters:
        ----------
        test_rate : double, optional.
            Specify the proportion of the testing data set,
            e.g. 20 mean 20% of the images will be in the testing set.
        blocksize : int
            Specifies the number of bytes to use for blocks that are read
            by threads.

        Returns
        -------
        ( training ImageTable, testing ImageTable )

        '''
        blocksize = blocksize or self.blocksize or 64

        self._retrieve('loadactionset', actionset='sampling')

        partindname = random_name(name='PartInd_', length=2)
        self._retrieve('sampling.stratified',
                       output=dict(casout=dict(**self.params,
                                               blocksize=blocksize,
                                               replace=True),
                                   copyvars='all', partindname=partindname),
                       samppct=test_rate, samppct2=100 - test_rate,
                       partind=True,
                       table=dict(**self.tbl, groupby='_label_'))

        train = self._retrieve('table.partition', 
                               table=dict(**self.to_table_params(),
                                          where='{}=2'.format(partindname),
                                          groupby='_label_'))['casTable']
        train = type(self)(**train.params)
        train.set_connection(self.get_connection())
        train.path = self.path
        train.patch_level = self.patch_level

        test = self._retrieve('table.partition', 
                              table=dict(**self.to_table_params(),
                                         where='{}=1'.format(partindname),
                                         groupby='_label_'))['casTable']
        test = type(self)(**test.params)
        test.set_connection(self.get_connection())
        test.path = self.path
        test.patch_level = self.patch_level

        return train, test

    def three_way_split(self, valid_rate=20, test_rate=20, blocksize=None):
        '''
        Function to split image data into training and testing sets.

        Parameters
        ----------
        valid_rate : double, optional
            Specify the proportion of the validation data set,
            e.g. 20 mean 20% of the images will be in the validation set.
        test_rate : double, optional
            Specify the proportion of the testing data set,
            e.g. 20 mean 20% of the images will be in the testing set.
            Note: the total of valid_rate and test_rate cannot be exceed 100
        blocksize : int
            Specifies the number of bytes to use for blocks that are read by threads.

        Returns
        -------
        ( train ImageTable, valid ImageTable, test ImageTable ) 

        '''
        blocksize = blocksize or self.blocksize or 64

        self._retrieve('loadactionset', actionset='sampling')

        partindname = random_name(name='PartInd_', length=2)
        self._retrieve('sampling.stratified', 
                       output=dict(casout=dict(**self.to_outtable_params(),
                                               blocksize=blocksize,
                                               replace=True),
                                    copyvars='all',
                                    partindname=partindname),
                        samppct=valid_rate, samppct2=test_rate,
                        partind=True,
                        table=dict(**self.to_table_params(), groupby='_label_'))

        train = self._retrieve('sampling.partition', 
                               table=dict(**self.to_table_params(),
                                          where='{}=0'.format(partindname),
                                          groupby='_label_'))['casTable']
        train = ImageTable(**train.params)
        train.set_connection(self.get_connection())
        train.path = self.path
        train.patch_level = train.patch_level

        valid = self._retrieve('sampling.partition', 
                               table=dict(**self.to_table_params(),
                                          where='{}=1'.format(partindname),
                                          groupby='_label_'))['casTable']
        valid = ImageTable(**valid.params)
        valid.set_connection(self.get_connection())
        valid.path = self.path
        valid.patch_level = train.patch_level

        test = self._retrieve('sampling.partition',
                              table=dict(**self.to_table_params(),
                                         where='{}=2'.format(partindname),
                                         groupby='_label_'))
        test.set_connection(self.get_connection())
        test.path = self.path
        test.patch_level = self.patch_level

        return train, valid, test

    def display_images(self, nimages=5, ncol=8, randomize=False):
        '''
        Display a grid of images

        Parameters:
        ----------
        nimages : int, optional
            Specifies the number of images to be displayed.
            If nimage is greater than the maximum number of images in the
            table, it will be set to this maximum number.
            Note: Specifying a large value for nimages can lead to slow
            performance.
        ncol : int, optional
            Specifies the layout of the display, determine the number of
            columns in the plots.
        randomize : boolean, optional
            Specifies whether to randomly choose the images for display.

        Returns
        -------
        matplotlib.figure

        '''
        nimages = min(nimages, len(self))

        if randomize:
            temp_tbl = self._retrieve('image.fetchimages', 
                                      imagetable=dict(**self.to_table_params(),
                                          computedvars=['random_index'],
                                          computedvarsprogram='call streaminit(-1);\
                                                               random_index=rand("UNIFORM");'),
                                      sortby='random_index', to=nimages)
        else:
            temp_tbl = self._retrieve('image.fetchimages', 
                                      imagetable=self.to_table_params(), to=nimages)

        if nimages > ncol:
            nrow = nimages // ncol + 1
        else:
            nrow = 1
            ncol = nimages

        fig = plt.figure(figsize=(16, 16 // ncol * nrow))

        for i in range(nimages):
            image = temp_tbl['Images']['Image'][i]
            label = temp_tbl['Images']['Label'][i]
            image = np.asarray(image)
            ax = fig.add_subplot(nrow, ncol, i + 1)
            ax.set_title('{}'.format(label))
            plt.imshow(image)
            plt.xticks([]), plt.yticks([])

        return fig

    def crop_images(self, x=0, y=0, width=None, height=None, inplace=True):
        '''
        Crop images in the table

        Parameters
        ----------
        x : int, optional
            Specify the x location of the top-left corner of the cropped images.
        y : int, optional
            Specify the y location of the top-left corner of the cropped images.
        width : int, optional
            Specify the width of the cropped images.
        height : int, optional
            Specify the height of the cropped images.
            If not specified, height will be set to be equal to width.
        inplace: boolean, optional
            Specifies whether to update the original table, or to create a new one.

        Returns
        -------
        ImageTable
            If `inplace=False`
        None
            If `inplace=True`

        '''
        if (width is None) and (height is None):
            width = 224
        if width is None:
            width = height
        if height is None:
            height = width

        image_table = self.tbl

        column_names = ['_filename_{}'.format(i) for i in range(self.patch_level + 1)]
        if inplace:
            self._retrieve('image.processimages',
                           imagetable=self.to_table_params(),
                           copyvars=column_names,
                           casout=dict(**self.to_outtable_params(), replace=True),
                           imagefunctions=[dict(functionoptions=
                                                dict(functiontype='GET_PATCH', x=x, y=y,
                                                     w=width, h=height))])

        else:
            out = self.copy_table()
            out.crop(x=x, y=x, width=width, height=height)
            return out

    def resize_images(self, width=None, height=None, inplace=True):
        '''
        Resize images in the table

        Parameters
        ----------
        width : int, optional
            Specify the target width of the resized images.
        height : int, optional
            Specify the target height of the resized images.
            If not specified, height will be set to be equal to width.
        inplace: boolean, optional
            Specifies whether to update the original table, or to create
            a new one.

        Returns
        -------
        ImageTable
            If `inplace=False`
        None
            If `inplace=True`

        '''
        if (width is None) and (height is None):
            width = 224
        if width is None:
            width = height
        if height is None:
            height = width

        column_names = ['_filename_{}'.format(i) for i in range(self.patch_level + 1)]

        if inplace:
            self._retrieve('image.processimages',
                           imagetable=self.to_table_params(),
                           copyvars=column_names,
                           casout=dict(**self.to_outtable_params(), replace=True),
                           imagefunctions=[dict(functionoptions=
                                                dict(functiontype='RESIZE',
                                                     w=width, h=height))])
            self._retrieve('image.partition', 
                           casout=dict(**self.to_outtable_params(), replace=True),
                           table=self.to_table_params())

        else:
            out = self.copy_table()
            out.resize_images(width=width, height=height)
            return out

    def get_patches(self, x=0, y=0, width=None, height=None, step_size=None,
                    output_width=None, output_height=None, inplace=True):
        '''
        Generate patches from images in the table

        Parameters
        ----------
        x : int, optional
            Specify the x location of the top-left corner of the first patches.
        y : int, optional
            Specify the y location of the top-left corner of the first patches.
        width : int, optional
            Specify the width of the patches.
        height : int, optional
            Specify the width of the patches.
            If not specified, height will be set to be equal to width.
        step_size : int, optional
            Specify the step size of the moving windows for extracting the patches.
            Default : None, meaning step_size=width.
        output_width : int, optional
            Specify the output width of the patches.
            If not equal to width, the patches will be resize to the output width.
            Default : None, meaning output_width=width.
        output_height : int, optional
            Specify the output height of the patches.
            If not equal to height, the patches will be resize to the output height.
            Default : None, meaning output_height=height.
        inplace: boolean, optional
            Specifies whether to update the original table, or create a new one.

        Returns
        -------
        ImageTable
            If `inplace=False`
        None
            If `inplace=True`

        '''

        if (width is None) and (height is None):
            width = 224
        if width is None:
            width = height
        if height is None:
            height = width

        if step_size is None:
            step_size = width

        if output_width is None:
            output_width = width
        if output_height is None:
            output_height = height

        croplist = [dict(sweepimage=True, x=x, y=y,
                         width=width, height=height,
                         stepsize=step_size,
                         outputwidth=output_width,
                         outputheight=output_height)]

        column_names = ['_filename_{}'.format(i) for i in range(self.patch_level + 1)]

        if inplace:
            self._retrieve('image.augmentimages',
                           imagetable=self.to_table_params(),
                           copyvars=column_names,
                           casout=dict(**self.to_outtable_params(), replace=True),
                           croplist=croplist)

            # The following code generate the latest file name according
            # to the number of patches operations.
            computedvars = '_filename_{}'.format(self.patch_level + 1)
            code = "length _filename_{1} varchar(*); " + \
                   "dot_loc = LENGTH(_filename_{0}) - INDEX(REVERSE(_filename_{0}),'.')+1; " + \
                   "_filename_{1} = SUBSTR(_filename_{0},1,dot_loc-1) || " + \
                   "compress('_'||x||'_'||y||SUBSTR(_filename_{0},dot_loc)); "
            code = code.format(self.patch_level, self.patch_level + 1)

            self._retrieve('table.partition', 
                           casout=dict(**self.to_outtable_params(), replace=True),
                           table=dict(**self.to_table_params(),
                                      computedvars=computedvars,
                                      computedvarsprogram=code))
            self._retrieve('sampling.shuffle',
                           casout=dict(**self.to_outtable_params(), replace=True),
                           table=self.to_table_params())
            self.patch_level += 1

        else:
            out = self.copy_table()
            out.get_patches(x=x, y=y, width=width, height=height, step_size=step_size,
                            output_width=output_width, output_height=output_height)
            return out

    def get_random_patches(self, random_ratio=0.5, x=0, y=0, width=None, height=None,
                           step_size=None, output_width=None, output_height=None,
                           inplace=True):
        '''
        Generate random patches from images in the table

        Parameters
        ----------
        random_ratio: double, optional
            Specifies the proportion of the generated pateches to output.
        x : int, optional
            Specifies the x location of the top-left corner of the first patches.
        y : int, optional
            Specifies the y location of the top-left corner of the first patches.
        width : int, optional
            Specifies the width of the patches.
        height : int, optional
            Specifies the width of the patches.
            If not specified, height will be set to be equal to width.
        step_size : int, optional
            Specifies the step size of the moving windows for extracting the patches.
            If not specified, it will be set to be equal to width.
        output_width : int, optional
            Specifies the output width of the patches.
            If not specified, it will be set to be equal to width.
        output_height : int, optional
            Specifies the output height of the patches.
            If not specified, it will be set to be equal to height.
        inplace: boolean, optional
            Specifies whether to update the original table, or create a new one.

        Returns
        -------
        ImageTable
            If `inplace=True`
        None
            If `inplace=False`

        '''
        if (width is None) and (height is None):
            width = 224
        if width is None:
            width = height
        if height is None:
            height = width

        if step_size is None:
            step_size = width

        if output_width is None:
            output_width = width
        if output_height is None:
            output_height = height

        croplist = [dict(sweepimage=True, x=x, y=y,
                         width=width, height=height,
                         stepsize=step_size,
                         outputwidth=output_width,
                         outputheight=output_height)]

        column_names = ['_filename_{}'.format(i) for i in range(self.patch_level + 1)]

        if inplace:
            self._retrieve('image.augmentimages', 
                           imagetable=self.to_table_params(),
                           copyvars=column_names,
                           casout=dict(**self.to_outtable_params(), replace=True),
                           croplist=croplist,
                           randomratio=random_ratio,
                           writerandomly=True)

            # The following code generate the latest file name according
            # to the number of patches operations.
            computedvars = '_filename_{}'.format(self.patch_level + 1)
            code = "length _filename_{1} varchar(*); " + \
                   "dot_loc = LENGTH(_filename_{0}) - INDEX(REVERSE(_filename_{0}),'.')+1; " + \
                   "_filename_{1} = SUBSTR(_filename_{0},1,dot_loc-1) || " + \
                   "compress('_'||x||'_'||y||SUBSTR(_filename_{0},dot_loc)); "
            code = code.format(self.patch_level, self.patch_level + 1)

            self._retrieve('table.partition',
                           casout=dict(**self.to_outtable_params(), replace=True),
                           table=dict(**self.to_table_params(),
                                      computedvars=computedvars,
                                      computedvarsprogram=code))
            self.patch_level += 1

        else:
            out = self.copy_table()
            out.get_random_patches(random_ratio=random_ratio,
                                   x=x, y=y,
                                   width=width, height=height,
                                   step_size=step_size,
                                   output_width=output_width,
                                   output_height=output_height)
            return out

    @property
    def image_summary(self):
        return self._retrieve('image.summarizeimages', imagetable=self)['Summary']

    @property
    def label_freq(self):
        return self._retrieve('simple.freq', table=self, inputs=['_label_'])['Freq']

    @property
    def channel_means(self):
        return self.image_summary.ix[0, ['mean1stChannel', 'mean2ndChannel',
                                         'mean3rdChannel']].tolist()
