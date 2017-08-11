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
from swat.cas.table import CASTable
from .utils import random_name, image_blocksize


def two_way_split(tbl, test_rate=20, stratify_by='_label_', casout=None):
    '''
    Function to split image data into training and testing sets

    Parameters:
    ----------
    tbl : CASTable
        The CAS table to split
    test_rate : double, optional
        Specify the proportion of the testing data set,
        e.g. 20 mean 20% of the images will be in the testing set.
    stratify_by : string, optional
        The variable to stratify by
    casout : dict, optional
        Additional parameters for statified output table

    Returns
    -------
    ( training CASTable, testing CASTable )

    '''
    if casout is None:
        casout = tbl.to_outtable_params()
    elif isinstance(casout, CASTable):
        casout = casout.to_outputtable_params()

    tbl._retrieve('loadactionset', actionset='sampling')

    partindname = random_name(name='PartInd_', length=2)

    tbl._retrieve('sampling.stratified',
                  output=dict(casout=casout, copyvars='all', partindname=partindname),
                  samppct=test_rate, samppct2=100 - test_rate,
                  partind=True,
                  table=dict(groupby=stratify_by, **tbl.to_table_params()))

    train = tbl.copy()
    out = tbl._retrieve('table.partition',
                        table=dict(where='{}=2'.format(partindname),
                                   groupby=stratify_by,
                                   **train.to_table_params()))['casTable']
    train.params.update(out.params)

    test = tbl.copy()
    out = tbl._retrieve('table.partition',
                        table=dict(where='{}=1'.format(partindname),
                                   groupby=stratify_by,
                                   **test.to_table_params()))['casTable']
    test.params.update(out.params)

    return train, test


def three_way_split(tbl, valid_rate=20, test_rate=20, stratify_by='_label_', casout=None):
    '''
    Function to split image data into training and testing sets.

    Parameters
    ----------
    tbl : CASTable
        The CAS table to split
    valid_rate : double, optional
        Specify the proportion of the validation data set,
        e.g. 20 mean 20% of the images will be in the validation set.
    test_rate : double, optional
        Specify the proportion of the testing data set,
        e.g. 20 mean 20% of the images will be in the testing set.
        Note: the total of valid_rate and test_rate cannot be exceed 100
    stratify_by : string, optional
        The variable to stratify by
    casout : dict, optional
        Additional parameters for statified output table

    Returns
    -------
    ( train CASTable, valid CASTable, test CASTable )

    '''
    if casout is None:
        casout = tbl.to_outtable_params()
    elif isinstance(casout, CASTable):
        casout = casout.to_outputtable_params()

    tbl._retrieve('loadactionset', actionset='sampling')

    partindname = random_name(name='PartInd_', length=2)

    tbl._retrieve('sampling.stratified',
                  output=dict(casout=casout, copyvars='all', partindname=partindname),
                  samppct=valid_rate, samppct2=test_rate,
                  partind=True,
                  table=dict(groupby=stratify_by, **tbl.to_table_params()))

    train = tbl.copy()
    out = tbl._retrieve('sampling.partition',
                        table=dict(where='{}=0'.format(partindname),
                                   groupby=stratify_by,
                                   **train.to_table_params()))['casTable']
    train.params.update(out.params)

    valid = tbl.copy()
    out = tbl._retrieve('sampling.partition',
                        table=dict(where='{}=1'.format(partindname),
                                   groupby=stratify_by,
                                   **valid.to_table_params()))['casTable']
    valid.params.update(out.params)

    test = tbl.copy()
    out = tbl._retrieve('sampling.partition',
                        table=dict(where='{}=2'.format(partindname),
                                   groupby=stratify_by,
                                   **test.to_table_params()))
    test.params.update(out.params)

    return train, valid, test


class ImageTable(CASTable):
    @classmethod
    def from_table(cls, tbl):

        '''
        Create an ImageTable from a CASTable

        Parameters
        ----------
        tbl : CASTable
            The CASTable object to use as the source

        Returns
        -------
        :class:`ImageTable`

        '''
        out = cls(**tbl.params)

        conn = tbl.get_connection()
        conn.loadactionset('image', _messagelevel='error')
        out.set_connection(conn)
        return out

    @classmethod
    def load_files(cls, conn, path, casout=None, **kwargs):
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
        **kwargs : keyword arguments, optional
            Additional keyword arguments to the `image.loadimages` action

        Returns
        -------
        :class:`ImageTable`

        '''
        conn.loadactionset('image', _messagelevel='error')

        if casout is None:
            casout = {}
        elif isinstance(casout, CASTable):
            casout = casout.to_outtable_params()

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

        conn.retrieve('table.shuffle', _messagelevel='error',
                      table=dict(computedvars=['_filename_0'],
                                 computedvarsprogram=code,
                                 **casout),
                      casout=dict(replace=True, **casout))

        out = cls(**casout)
        out.set_connection(conn)
        return out

    def __init__(self, name, **table_params):
        CASTable.__init__(self, name, **table_params)
        self.patch_level = 0

    def __copy__(self):
        out = CASTable.__copy__(self)
        out.patch_level = self.patch_level
        return out

    def __deepcopy__(self, memo):
        out = CASTable.__deepcopy__(self, memo)
        out.patch_level = self.patch_level
        return out

    def to_files(self, path):
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
                       images=dict(table=self.to_table_params(), path=file_name),
                       labellevels=1)

        self._retrieve('dropcaslib', caslib=caslib)

    def to_sashdat(self, path=None, name=None, **kwargs):
        '''
        Function to save the image table to a sashdat file.

        Parameters:
        ----------
        path : string
            Specifies the directory on the server to save the images

        '''
        caslib = random_name('Caslib', 6)
        self._retrieve('addcaslib', name=caslib, path=path, activeonadd=False,
                       datasource=dict(srcType="DNFS"))
        if name is None:
            name = self.to_params()['name'] + '.sashdat'

        self._retrieve('table.save', caslib=caslib, name=name, table=self.to_params(), **kwargs)
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

        out = self.copy()
        res = self._retrieve('table.partition', casout=casout, table=self)['casTable']
        out.params.update(res.params)

        return out

    def show(self, nimages=5, ncol=8, randomize=False, figsize=None):
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

        '''
        nimages = min(nimages, len(self))

        if randomize:
            temp_tbl = self._retrieve('image.fetchimages', _messagelevel='error',
                                      imagetable=dict(
                                          computedvars=['random_index'],
                                          computedvarsprogram='call streaminit(-1);\
                                                              random_index=rand("UNIFORM");',
                                          **self.to_table_params()),
                                      sortby='random_index', to=nimages)
        else:
            temp_tbl = self._retrieve('image.fetchimages', _messagelevel='error',
                                      imagetable=self.to_table_params(), to=nimages)

        if nimages > ncol:
            nrow = nimages // ncol + 1
        else:
            nrow = 1
            ncol = nimages
        if figsize is None:
            figsize = (16, 16 // ncol * nrow)
        fig = plt.figure(figsize=figsize)

        for i in range(nimages):
            image = temp_tbl['Images']['Image'][i]
            label = temp_tbl['Images']['Label'][i]
            ax = fig.add_subplot(nrow, ncol, i + 1)
            ax.set_title('{}'.format(label))
            plt.imshow(image)
            plt.xticks([]), plt.yticks([])

    def crop(self, x=0, y=0, width=None, height=None, inplace=True):
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
        blocksize = image_blocksize(width, height)

        column_names = ['_filename_{}'.format(i) for i in range(self.patch_level + 1)]

        if inplace:
            self._retrieve('image.processimages',
                           copyvars=column_names,
                           casout=dict(replace=True, blocksize=blocksize,
                                       **self.to_outtable_params()),
                           imagefunctions=[dict(functionoptions=
                                                dict(functiontype='GET_PATCH', x=x, y=y,
                                                     w=width, h=height))])

        else:
            out = self.copy_table()
            out.crop(x=x, y=x, width=width, height=height)
            return out

    def resize(self, width=None, height=None, inplace=True):
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
        blocksize = image_blocksize(width, height)
        column_names = ['_filename_{}'.format(i) for i in range(self.patch_level + 1)]

        if inplace:
            self._retrieve('image.processimages',
                           copyvars=column_names,
                           casout=dict(replace=True, blocksize=blocksize,
                                       **self.to_outtable_params()),
                           imagefunctions=[dict(functionoptions=
                                                dict(functiontype='RESIZE',
                                                     w=width, h=height))])
            # self._retrieve('table.partition',
            #                casout=dict(replace=True, blocksize=blocksize,
            #                           **self.to_outtable_params())

        else:
            out = self.copy_table()
            out.resize_images(width=width, height=height)
            return out

    def as_patches(self, x=0, y=0, width=None, height=None, step_size=None,
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

        blocksize = image_blocksize(output_width, output_height)
        croplist = [dict(sweepimage=True, x=x, y=y,
                         width=width, height=height,
                         stepsize=step_size,
                         outputwidth=output_width,
                         outputheight=output_height)]

        column_names = ['_filename_{}'.format(i) for i in range(self.patch_level + 1)]

        if inplace:
            self._retrieve('image.augmentimages',
                           copyvars=column_names,
                           casout=dict(replace=True, **self.to_outtable_params()),
                           croplist=croplist)

            # The following code generate the latest file name according
            # to the number of patches operations.
            computedvars = '_filename_{}'.format(self.patch_level + 1)
            code = "length _filename_{1} varchar(*); " + \
                   "dot_loc = LENGTH(_filename_{0}) - INDEX(REVERSE(_filename_{0}),'.')+1; " + \
                   "_filename_{1} = SUBSTR(_filename_{0},1,dot_loc-1) || " + \
                   "compress('_'||x||'_'||y||SUBSTR(_filename_{0},dot_loc)); "
            code = code.format(self.patch_level, self.patch_level + 1)

            self._retrieve('table.shuffle',
                           casout=dict(replace=True, blocksize=blocksize,
                                       **self.to_outtable_params()),
                           table=dict(computedvars=computedvars,
                                      computedvarsprogram=code,
                                      **self.to_table_params()))
            self.patch_level += 1

        else:
            out = self.copy_table()
            out.get_patches(x=x, y=y, width=width, height=height, step_size=step_size,
                            output_width=output_width, output_height=output_height)
            return out

    def as_random_patches(self, random_ratio=0.5, x=0, y=0, width=None, height=None,
                          step_size=None, output_width=None, output_height=None,
                          inplace=True):
        '''
        Generate random patches from images in the table

        Parameters
        ----------
        random_ratio: double, optional
            Specifies the proportion of the generated patches to output.
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

        blocksize = image_blocksize(output_width, output_height)

        croplist = [dict(sweepimage=True, x=x, y=y,
                         width=width, height=height,
                         stepsize=step_size,
                         outputwidth=output_width,
                         outputheight=output_height)]

        column_names = ['_filename_{}'.format(i) for i in range(self.patch_level + 1)]

        if inplace:
            self._retrieve('image.augmentimages',
                           copyvars=column_names,
                           casout=dict(replace=True, **self.to_outtable_params()),
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

            self._retrieve('table.shuffle',
                           casout=dict(replace=True, blocksize=blocksize,
                                       **self.to_outtable_params()),
                           table=dict(computedvars=computedvars,
                                      computedvarsprogram=code,
                                      **self.to_table_params()))

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
        out = self._retrieve('image.summarizeimages')['Summary']
        out = out.T.drop(['Column'])[0]
        out.name = None
        return out

    @property
    def label_freq(self):
        out = self._retrieve('simple.freq', table=self, inputs=['_label_'])['Frequency']
        out = out[['FmtVar', 'Level', 'Frequency']]
        out = out.set_index('FmtVar')
        # out.index.name = 'Label'
        out.index.name = None
        out = out.astype('int64')
        return out

    @property
    def channel_means(self):
        return self.image_summary[['mean1stChannel', 'mean2ndChannel',
                                   'mean3rdChannel']].tolist()

    @property
    def uid(self):
        file_name = '_filename_{}'.format(self.patch_level)
        uid = self[['_label_', file_name]].to_frame()
        # uid = uid.rename(columns={file_name: '_uid_'})
        return uid
