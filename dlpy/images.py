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

''' Special functionality for CAS tables containing image data '''

import matplotlib.pyplot as plt
import numpy as np
from swat.cas.table import CASTable

from .utils import random_name, image_blocksize


class ImageTable(CASTable):
    '''
    Specialized CASTable for Image Data

    Parameters
    ----------
    name : string
        The name of the CAS table
    **table_params : keyword arguments, optional
        Parameters to the :class:`CASTable` constructor

    Attributes
    ----------
    image_summary : pandas.Series
        The summary of the images contained in the image table.
    label_freq : pandas.Series
        The count of images in different categories.
    channel_means : tuple of double
        The mean of the image intensities in each channels.


    Returns
    -------
    :class:`ImageTable`

    '''

    def __init__(self, name, **table_params):
        CASTable.__init__(self, name, **table_params)
        self.patch_level = 0

    @classmethod
    def from_table(cls, tbl, image_col='_image_', label_col='_label_',
                   path_col=None, columns=None, casout=None):
        '''
        Create an ImageTable from a CASTable

        Parameters
        ----------
        tbl : CASTable
            The CASTable object to use as the source.
        image_col : str, optional
            Specifies the column name for the image data.
            Default = '_image_'
        label_col : str, optional
            Specifies the column name for the labels.
            Default = '_label_'
        path_col : str, optional
            Specifies the column name that stores the path for each image.
            Default = None, and the unique image ID will be generated from the labels.
        columns : list of str, optional
            Specifies the extra columns in the image table.
            Default = None
        casout : dict
            Specifies the output CASTable parameters.
            Default = None.
            Note : the options of replace=True, blocksize=32 will be automatically
            added to the casout option.

        Returns
        -------
        :class:`ImageTable`

        '''
        out = cls(**tbl.params)

        conn = tbl.get_connection()
        conn.loadactionset('image', _messagelevel='error')

        if casout is None:
            casout = {}
        elif isinstance(casout, CASTable):
            casout = casout.to_outtable_params()

        if 'name' not in casout:
            casout['name'] = random_name()

        if '_filename_0' in tbl.columninfo().ColumnInfo.Column.tolist():
            computedvars = []
            code = []
        else:
            computedvars = ['_filename_0']
            code = ['length _filename_0 varchar(*);']
            if path_col is not None:
                code.append(('_loc1 = LENGTH({0}) - '
                             'INDEX(REVERSE({0}),\'/\')+2;').format(path_col))
                code.append('_filename_0 = SUBSTR({},_loc1);'.format(path_col))
            else:
                code.append('call streaminit(-1);shuffle_id=rand("UNIFORM")*10**10;')
                code.append(('_filename_0=cats({},"_",put(put(shuffle_id,z10.)'
                             ',$char10.),".jpg");').format(label_col))

        if image_col != '_image_':
            computedvars.append('_image_')
            code.append('_image_ = {};'.format(image_col))

        if label_col != '_label_':
            computedvars.append('_label_')
            code.append('_label_ = {};'.format(label_col))

        code = '\n'.join(code)

        if computedvars:
            table_opts = dict(computedvars=computedvars,
                              computedvarsprogram=code,
                              **tbl.params)
        else:
            table_opts = dict(**tbl.params)

        # This will generate the '_image_' and '_label_' columns.
        conn.retrieve('table.shuffle', _messagelevel='error',
                      table=table_opts,
                      casout=dict(replace=True, blocksize=32, **casout))

        column_names = ['_image_', '_label_', '_filename_0', '_id_']
        if columns is not None:
            if not isinstance(columns, list):
                columns = list(columns)
            column_names += columns

        # Remove the unwanted columns.
        conn.retrieve('table.partition', _messagelevel='error',
                      table=dict(Vars=column_names, **casout),
                      casout=dict(replace=True, blocksize=32, **casout))

        out = cls(**casout)
        out.set_connection(conn)

        return out

    @classmethod
    def load_files(cls, conn, path, casout=None, columns=None, **kwargs):
        '''
        Create ImageTable from files in `path`

        Parameters
        ----------
        conn : CAS
            The CAS connection object
        path : string
            The path to the image directory on the server
        casout : dict, optional
            The output table specifications
        columns : list of str, optional
            Specifies the extra columns in the image table.
        **kwargs : keyword arguments, optional
            Additional keyword arguments to the `image.loadimages` action

        Returns
        -------
        :class:`ImageTable`

        '''
        conn.loadactionset('image', _messagelevel='error')

        if casout is None:
            casout = dict(name=random_name())
        elif isinstance(casout, CASTable):
            casout = casout.to_outtable_params()

        if 'name' not in casout:
            casout['name'] = random_name()

        conn.retrieve('image.loadimages', _messagelevel='error',
                      casout=casout,
                      distribution=dict(type='random'),
                      recurse=True, labellevels=-1,
                      path=path, **kwargs)

        code = []
        code.append('length _filename_0 varchar(*);')
        code.append('_loc1 = LENGTH(_path_) - INDEX(REVERSE(_path_),\'/\')+2;')
        code.append('_filename_0 = SUBSTR(_path_,_loc1);')
        code = '\n'.join(code)
        column_names = ['_image_', '_label_', '_filename_0', '_id_']
        if columns is not None:
            column_names += columns
        conn.retrieve('table.partition', _messagelevel='error',
                      table=dict(Vars=column_names,
                                 computedvars=['_filename_0'],
                                 computedvarsprogram=code,
                                 **casout),
                      casout=dict(replace=True, blocksize=32, **casout))

        out = cls(**casout)
        out.set_connection(conn)
        return out

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
        Save the images in the original format under the specified directory

        Parameters
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
        Save the ImageTable to a sashdat file

        Parameters
        ----------
        path : string
            Specifies the directory on the server to save the images

        '''
        caslib = random_name('Caslib', 6)
        self._retrieve('addcaslib', name=caslib, path=path, activeonadd=False,
                       datasource=dict(srcType='DNFS'))
        if name is None:
            name = self.to_params()['name'] + '.sashdat'

        self._retrieve('table.save', caslib=caslib, name=name,
                       table=self.to_params(), **kwargs)
        self._retrieve('dropcaslib', caslib=caslib)

    def copy_table(self, casout=None):
        '''
        Create a copy of the ImageTable

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
            casout['name'] = random_name()

        res = self._retrieve('table.partition', casout=casout, table=self)['casTable']
        out = ImageTable.from_table(tbl=res)
        out.params.update(res.params)

        return out

    def show(self, nimages=5, ncol=8, randomize=False, figsize=None):
        '''
        Display a grid of images

        Parameters
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
            temp_tbl = self.retrieve('image.fetchimages', _messagelevel='error',
                                     table=dict(
                                         computedvars=['random_index'],
                                         computedvarsprogram='call streaminit(-1);'
                                                             'random_index='
                                                             'rand("UNIFORM");',
                                         **self.to_table_params()),
                                     sortby='random_index', to=nimages)
        else:
            temp_tbl = self._retrieve('image.fetchimages', to=nimages)

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
            if len(image.size) == 2:
                plt.imshow(np.array(image), cmap='Greys_r')
            else:
                plt.imshow(image)
            plt.xticks([]), plt.yticks([])
        plt.show()

    def crop(self, x=0, y=0, width=None, height=None, inplace=True):
        '''
        Crop the images in the ImageTable

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
        :class:`ImageTable`
            If `inplace=False`
        `None`
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
                           imagefunctions=[
                               dict(functionoptions=dict(functiontype='GET_PATCH',
                                                         x=x, y=y,
                                                         w=width, h=height))])

        else:
            out = self.copy_table()
            out.crop(x=x, y=x, width=width, height=height)
            return out

    def resize(self, width=None, height=None, inplace=True):
        '''
        Resize the images in the ImageTable

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
        :class:`ImageTable`
            If `inplace=False`
        `None`
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
                           imagefunctions=[
                               dict(functionoptions=dict(functiontype='RESIZE',
                                                         w=width, h=height))])
            # self._retrieve('table.partition',
            #                casout=dict(replace=True, blocksize=blocksize,
            #                           **self.to_outtable_params())

        else:
            out = self.copy_table()
            out.resize(width=width, height=height)
            return out

    def as_patches(self, x=0, y=0, width=None, height=None, step_size=None,
                   output_width=None, output_height=None, inplace=True):
        '''
        Generate patches from the images in the ImageTable

        Parameters
        ----------
        x : int, optional
            Specify the x location of the top-left corner of the
            first patches.
        y : int, optional
            Specify the y location of the top-left corner of the
            first patches.
        width : int, optional
            Specify the width of the patches.
        height : int, optional
            Specify the width of the patches.
            If not specified, height will be set to be equal to width.
        step_size : int, optional
            Specify the step size of the moving windows for extracting
            the patches.
            Default : None, meaning step_size=width.
        output_width : int, optional
            Specify the output width of the patches.
            If not equal to width, the patches will be resize to the
            output width.
            Default : None, meaning output_width=width.
        output_height : int, optional
            Specify the output height of the patches.
            If not equal to height, the patches will be resize to the
            output height.
            Default : None, meaning output_height=height.
        inplace: boolean, optional
            Specifies whether to update the original table, or create a
            new one.

        Notes
        -----
        By creating crops with fixed window size and moving the window
        along the images.

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
            code = []
            code.append('length _filename_{1} varchar(*);')
            code.append('dot_loc = LENGTH(_filename_{0}) - '
                        'INDEX(REVERSE(_filename_{0}), \'.\')+1;')
            code.append('_filename_{1} = SUBSTR(_filename_{0}, 1, dot_loc-1) || '
                        'compress(\'_\'||x||\'_\'||y||SUBSTR(_filename_{0},dot_loc));')
            code = '\n'.join(code)
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
            out.as_patches(x=x, y=y, width=width, height=height, step_size=step_size,
                           output_width=output_width, output_height=output_height)
            return out

    def as_random_patches(self, random_ratio=0.5, x=0, y=0, width=None, height=None,
                          step_size=None, output_width=None, output_height=None,
                          inplace=True):
        '''
        Generate random patches from the images in the ImageTable

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
        :class:`ImageTable`
            If `inplace=True`
        `None`
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
            code = []
            code.append('length _filename_{1} varchar(*);')
            code.append('dot_loc = LENGTH(_filename_{0}) - '
                        'INDEX(REVERSE(_filename_{0}),\'.\')+1;')
            code.append('_filename_{1} = SUBSTR(_filename_{0},1,dot_loc-1) || '
                        'compress(\'_\'||x||\'_\'||y||SUBSTR(_filename_{0},dot_loc));')
            code = '\n'.join(code)
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
            out.as_random_patches(random_ratio=random_ratio,
                                  x=x, y=y,
                                  width=width, height=height,
                                  step_size=step_size,
                                  output_width=output_width,
                                  output_height=output_height)
            return out

    def random_mutations(self, color_jitter=True, color_shift=True, darken=False, 
                         horizontal_flip=True, invert_pixels=False, lighten=False, pyramid_down=False,
                         pyramid_up=False, rotate_left=False, rotate_right=False, sharpen=False,
                         vertical_flip=True, inplace=True):
        '''
        Generate random mutations from the images in the ImageTable

        Parameters
        ----------
        color_jitter : boolean, optional
            Specifies whether to apply color jittering to an input image.
        color_shift : boolean, optional
            Specifies whether to randomly change pixel intensity values of an input image.
        darken : boolean, optional
            Specifies whether to darken the input image.
        horizontal_flip : boolean, optional
            Specifies whether to flip the input image horizontally.
        invert_pixels : boolean, optional
            Specifies whether to invert all pixels in the input image.
        lighten : boolean, optional
            Specifies whether to lighten the input image.
        pyramid_down : boolean, optional
            Specifies whether to downsample and then blur the input image.
        pyramid_up : boolean, optional
            Specifies whether to upsample and then blur the input image.
        rotate_left : boolean, optional
            Specifies whether to rotate the input image to the left.
        rotate_right : boolean, optional
            Specifies whether to rotate the input image to the right.
        sharpen : boolean, optional
            Specifies whether to sharpen the input image.
        vertical_flip : boolean, optional
            Specifies whether to vertically flip the input image.

        Returns
        -------
        :class:`ImageTable`
            If `inplace=True`
        `None`
            If `inplace=False`

        '''

        croplist = [{'mutations':dict(colorjittering=color_jitter, 
                         colorshifting=color_shift,
                         darken=darken, lighten=lighten,
                         horizontalflip=horizontal_flip,
                         invertpixels=invert_pixels,
                         pyramiddown=pyramid_down,
                         pyramidup=pyramid_up,
                         rotateleft=rotate_left,
                         rotateright=rotate_right,
                         sharpen=sharpen, 
                         verticalflip=vertical_flip),
                      'usewholeimage':True}]

        column_names = ['_filename_{}'.format(i) for i in range(self.patch_level + 1)]

        if inplace:
            self._retrieve('image.augmentimages',
                           copyvars=column_names,
                           casout=dict(replace=True, **self.to_outtable_params()),
                           croplist=croplist,
                           writerandomly=True)

            # The following code generate the latest file name according
            # to the number of patches and mutation (_m) operations.
            computedvars = '_filename_{}'.format(self.patch_level + 1)
            code = []
            code.append('length _filename_{1} varchar(*);')
            code.append('dot_loc = LENGTH(_filename_{0}) - '
                        'INDEX(REVERSE(_filename_{0}),\'.\')+1;')
            code.append('_filename_{1} = SUBSTR(_filename_{0},1,dot_loc-1) || '
                        'compress(\'_\'||\'m{0}\'||SUBSTR(_filename_{0},dot_loc));')
            code = '\n'.join(code)
            code = code.format(self.patch_level, self.patch_level + 1)

            self._retrieve('table.shuffle',
                           casout=dict(replace=True, 
                                       **self.to_outtable_params()),
                           table=dict(computedvars=computedvars,
                                      computedvarsprogram=code,
                                      **self.to_table_params()))

            self.patch_level += 1

        else:
            out = self.copy_table()
            out.random_mutations(color_jitter=color_jitter, 
                                 color_shift=color_shift, 
                                 darken=darken, 
                                 horizontal_flip=horizontal_flip, 
                                 invert_pixels=invert_pixels, 
                                 lighten=lighten, 
                                 pyramid_down=pyramid_down,
                                 pyramid_up=pyramid_up, 
                                 rotate_left=rotate_left, 
                                 rotate_right=rotate_right, 
                                 sharpen=sharpen,
                                 vertical_flip=vertical_flip, 
                                 inplace=True)
            return out

    @property
    def image_summary(self):
        '''
        Summarize the images in the ImageTable

        Returns
        -------
        :class:`pd.Series`

        '''
        out = self._retrieve('image.summarizeimages')['Summary']
        out = out.T.drop(['Column'])[0]
        out.name = None
        return out

    @property
    def label_freq(self):
        '''
        Summarize the distribution of different classes (labels) in the ImageTable

        Returns
        -------
        :class:`pd.Series`

        '''
        out = self._retrieve('simple.freq', table=self, inputs=['_label_'])['Frequency']
        out = out[['FmtVar', 'Level', 'Frequency']]
        out = out.set_index('FmtVar')
        # out.index.name = 'Label'
        out.index.name = None
        out = out.astype('int64')
        return out

    @property
    def channel_means(self):
        '''
        A list of the means of the image intensities in each color channel.

        Returns
        -------
        ( first-channel-mean, second-channel-mean, third-channel-mean )

        '''
        return self.image_summary[['mean1stChannel', 'mean2ndChannel',
                                   'mean3rdChannel']].tolist()

    @property
    def uid(self):
        '''
        A unique ID for each image.

        Returns
        -------


        '''
        file_name = '_filename_{}'.format(self.patch_level)
        uid = self[['_label_', file_name]].to_frame()
        # uid = uid.rename(columns={file_name: '_uid_'})
        return uid
