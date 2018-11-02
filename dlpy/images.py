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
from .utils import random_name, image_blocksize, caslibify, DLPyError, input_table_check


def create_new_filename(image_table, mutation):
    code = None
    computedvars = None
    current_filename = image_table.filename_col
    if current_filename is not None:
        if image_table.patch_level == 0:
            computedvars = current_filename + '_0'
        else:
            index = current_filename[::-1].find('_')
            computedvars = current_filename[: -index] + str(image_table.patch_level)
        code = []
        code.append('length {1} varchar(*);')
        code.append('dot_loc = LENGTH({0}) - '
                    'INDEX(REVERSE({0}),\'.\')+1;')
        if mutation:
            code.append('{1} = SUBSTR({0},1,dot_loc-1) || '
                        'compress(\'_\'||\'m{2}\'||SUBSTR({0},dot_loc));')
            code = '\n'.join(code)
            code = code.format(current_filename, computedvars, image_table.patch_level)
        else:
            code.append('{1} = SUBSTR({0}, 1, dot_loc-1) || '
                        'compress(\'_\'||x||\'_\'||y||SUBSTR({0},dot_loc));')
            code = '\n'.join(code)
            code = code.format(current_filename, computedvars)

    return computedvars, code


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
        self.image_cols = '_image_'
        self.id_col = None
        self.filename_col = None
        self.cls_cols = None
        self.keypoints_cols = None
        self.obj_det_cols = None

    @classmethod
    def from_table(cls, tbl, id_col = None, filename_col=None, cls_cols=None, keypoints_cols = None, obj_det_cols = None):
        '''
        Create an ImageTable from a CASTable

        Parameters
        ----------
        tbl : CASTable
            The CASTable object to use as the source.
        id_col : str, optional
            Specifies the column name for id.
        filename_col : str, optional
            Specifies the column name for the labels.
        cls_cols : str, optional
            Specifies the column name that stores the labels for classification.
            Default = None
        keypoints_cols : list of str, optional
            Specifies the column names that stores the labels for key points detection or multiple regression.
            Default = None
        obj_det_cols : list of str, optional
            Specifies the column names that stores the labels for object detection.
            Default = None

        Returns
        -------
        :class:`ImageTable`

        '''
        out = cls(**tbl.params)

        conn = tbl.get_connection()
        conn.loadactionset('image', _messagelevel = 'error')
        # for vb015 the image column can only be _image_
        if '_image_' not in tbl.columns:
            raise DLPyError('The table that has _image_ as image column can be converted to ImageTable.')

        # if image_col not in tbl.columns:
        #     raise ValueError('{} is not in the table'.format(image_col))
        if id_col is not None and id_col not in tbl.columns:
            raise ValueError('{} is not in the table'.format(id_col))
        if filename_col is not None and filename_col not in tbl.columns:
            raise ValueError('{} is not in the table'.format(filename_col))
        # TODO: support multiple-task with list of classification columns
        if cls_cols is not None and cls_cols not in tbl.columns:
            raise ValueError('cls_cols contains the column not in the table')
        if keypoints_cols is not None and any(keypoints_cols not in tbl.columns):
            raise ValueError('keypoints_cols contains the column not in the table')
        if obj_det_cols is not None and any(obj_det_cols not in tbl.columns):
            raise ValueError('obj_det_cols contains the column not in the table')

        # out.image_cols = image_col
        out.id_col = id_col
        out.filename_col = filename_col
        out.cls_cols = cls_cols
        out.keypoints_cols = keypoints_cols
        out.obj_det_cols = obj_det_cols

        out.set_connection(conn)

        return out

    @classmethod
    def load_files(cls, conn, path, casout=None, columns=None, caslib=None, **kwargs):
        '''
        Create ImageTable from files in `path`

        Parameters
        ----------
        conn : CAS
            The CAS connection object
        path : string
            The path to the image directory on the server.
            Path may be absolute, or relative to caslib root if specified.
        casout : dict, optional
            The output table specifications
        columns : list of str, optional
            Specifies the extra columns in the image table.
        caslib : string, optional
            The name of the caslib containing the images.
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

        if caslib is None:
            caslib, path = caslibify(conn, path, task='load')

        conn.retrieve('image.loadimages', _messagelevel='error',
                      casout=casout,
                      distribution=dict(type='random'),
                      recurse=True, labellevels=-1,
                      path=path, caslib=caslib, **kwargs)

        code = []
        code.append('length _filename_ varchar(*);')
        code.append('_loc1 = LENGTH(_path_) - INDEX(REVERSE(_path_),\'/\')+2;')
        code.append('_filename_ = SUBSTR(_path_,_loc1);')
        code = '\n'.join(code)
        column_names = ['_image_', '_label_', '_filename_', '_id_']
        if columns is not None:
            column_names += columns
        conn.retrieve('table.partition', _messagelevel='error',
                      table=dict(Vars=column_names,
                                 computedvars=['_filename_'],
                                 computedvarsprogram=code,
                                 **casout),
                      casout=dict(replace=True, blocksize=32, **casout))

        out = cls(**casout)
        out.cls_cols = '_label_'
        out.id_col = '_id_'
        out.filename_col = '_filename_'
        out.set_connection(conn)
        return out

    def assign_property(self, castable):
        castable = ImageTable(**castable.to_outtable_params())
        castable.set_connection(self.get_connection())
        castable.patch_level = self.patch_level
        castable.id_col = self.id_col
        castable.filename_col = self.filename_col
        castable.cls_cols = self.cls_cols
        castable.keypoints_cols = self.keypoints_cols
        castable.obj_det_cols = self.obj_det_cols
        return castable

    def __copy__(self):
        out = CASTable.__copy__(self)
        out.patch_level = self.patch_level
        out.id_col = self.id_col
        out.filename_col = self.filename_col
        out.cls_cols = self.cls_cols
        out.keypoints_cols = self.keypoints_cols
        out.obj_det_cols = self.obj_det_cols
        return out

    def __deepcopy__(self, memo):
        out = CASTable.__deepcopy__(self, memo)
        out.patch_level = self.patch_level
        out.image_cols = self.image_cols
        out.id_col = self.id_col
        out.filename_col = self.filename_col
        out.cls_cols = self.cls_cols
        out.keypoints_cols = self.keypoints_cols
        out.obj_det_cols = self.obj_det_cols
        return out

    def to_files(self, path):
        '''
        Save the images in the original format under the specified directory

        Parameters
        ----------
        path : string
            Specifies the directory on the server to save the images

        '''

        if self.filename_col is None or self.filename_col not in self.columns:
            raise DLPyError("Cannot save the images since filename_col of the ImageTable "
                            "doesn't exist or is not defined")
        caslib = random_name('Caslib', 6)
        self._retrieve('addcaslib', name=caslib, path=path, activeonadd=False)

        rt = self._retrieve('image.saveimages', caslib=caslib,
                       images=dict(table=self.to_table_params(), path=self.filename_col),
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
        # todo
        out = ImageTable.from_table(tbl=res)
        out.params.update(res.params)

        return out

    def show(self, n_images=5, ncol=8, randomize=False, figsize=None):
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
        randomize : bool, optional
            Specifies whether to randomly choose the images for display.

        '''
        n_images = min(n_images, len(self))
        if self.numrows().numrows == 0:
            raise ValueError('The image table is empty.')

        if randomize:
            temp_tbl = self.retrieve('image.fetchimages', _messagelevel='error',
                                     image = self.image_cols,
                                     table=dict(
                                         computedvars=['random_index'],
                                         computedvarsprogram='call streaminit(-1);'
                                                             'random_index='
                                                             'rand("UNIFORM");',
                                         **self.to_table_params()),
                                     sortby='random_index', to=n_images)
        else:
            temp_tbl = self._retrieve('image.fetchimages', image = self.image_cols, to=n_images)

        if n_images > ncol:
            nrow = n_images // ncol + 1
        else:
            nrow = 1
            ncol = n_images
        if figsize is None:
            figsize = (16, 16 // ncol * nrow)
        fig = plt.figure(figsize=figsize)

        for i in range(n_images):
            image = temp_tbl['Images']['Image'][i]
            label_col = 'Label' if '_label_' == self.cls_cols else self.cls_cols
            label = temp_tbl['Images'][label_col][i]
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
        inplace : bool, optional
            Specifies whether to update the original table, or to create a new one.

        Returns
        -------
        :class:`ImageTable`
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
        block_size = image_blocksize(width, height)
        column_names = None
        if self.filename_col is not None:
            filename_prefix = self.filename_col[:-self.filename_col[::-1].find('_')-1]
            column_names = [col for col in self.columns if col.startswith(filename_prefix)]

        if inplace:
            self._retrieve('image.processimages',
                           image=self.image_cols,
                           copyvars=column_names,
                           casout=dict(replace=True, blocksize=block_size,
                                       **self.to_outtable_params()),
                           imagefunctions=[
                               dict(functionoptions=dict(functiontype='GET_PATCH',
                                                         x=x, y=y,
                                                         w=width, h=height))])

        else:
            out = self.copy_table()
            out.crop(x=x, y=y, width=width, height=height)
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
        inplace : bool, optional
            Specifies whether to update the original table, or to create
            a new one.

        Returns
        -------
        :class:`ImageTable`
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
        block_size = image_blocksize(width, height)
        column_names = None
        if self.filename_col is not None:
            filename_prefix = self.filename_col[:-self.filename_col[::-1].find('_')-1]
            column_names = [col for col in self.columns if col.startswith(filename_prefix)]

        if inplace:
            self._retrieve('image.processimages',
                           image = self.image_cols,
                           copyvars=column_names,
                           casout=dict(replace=True, blocksize=block_size,
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
        inplace : bool, optional
            Specifies whether to update the original table, or create a
            new one.

        Notes
        -----
        By creating crops with fixed window size and moving the window
        along the images.

        Returns
        -------
        :class:`ImageTable`
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

        block_size = image_blocksize(output_width, output_height)
        crop_list = [dict(sweepimage=True, x=x, y=y,
                         width=width, height=height,
                         stepsize=step_size,
                         outputwidth=output_width,
                         outputheight=output_height)]
        column_names = None
        if self.filename_col is not None:
            filename_prefix = self.filename_col[:-self.filename_col[::-1].find('_')-1]
            column_names = [col for col in self.columns if col.startswith(filename_prefix)]

        if inplace:
            self._retrieve('image.augmentimages',
                           image = self.image_cols,
                           copyvars=column_names,
                           casout=dict(replace=True, **self.to_outtable_params()),
                           croplist=crop_list)

            # The following code generate the latest file name according
            # to the number of patches operations.
            computedvars, code = create_new_filename(self, False)

            self._retrieve('table.shuffle',
                           casout=dict(replace=True, blocksize=block_size,
                                       **self.to_outtable_params()),
                           table=dict(computedvars=computedvars,
                                      computedvarsprogram=code,
                                      **self.to_table_params()))
            self.patch_level += 1
            self.filename_col = computedvars
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
        random_ratio : double, optional
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
        inplace : bool, optional
            Specifies whether to update the original table, or create a new one.

        Returns
        -------
        :class:`ImageTable`
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
        column_names = None
        if self.filename_col is not None:
            filename_prefix = self.filename_col[:-self.filename_col[::-1].find('_')-1]
            column_names = [col for col in self.columns if col.startswith(filename_prefix)]

        if inplace:
            self._retrieve('image.augmentimages',
                           image = self.image_cols,
                           copyvars=column_names,
                           casout=dict(replace=True, **self.to_outtable_params()),
                           croplist=croplist,
                           randomratio=random_ratio,
                           writerandomly=True)

            # The following code generate the latest file name according
            # to the number of patches operations.
            computedvars, code = create_new_filename(self, False)

            self._retrieve('table.shuffle',
                           casout=dict(replace=True, blocksize=blocksize,
                                       **self.to_outtable_params()),
                           table=dict(computedvars=computedvars,
                                      computedvarsprogram=code,
                                      **self.to_table_params()))

            self.patch_level += 1
            self.filename_col = computedvars
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
        color_jitter : bool, optional
            Specifies whether to apply color jittering to an input image.
        color_shift : bool, optional
            Specifies whether to randomly change pixel intensity values of an input image.
        darken : bool, optional
            Specifies whether to darken the input image.
        horizontal_flip : bool, optional
            Specifies whether to flip the input image horizontally.
        invert_pixels : bool, optional
            Specifies whether to invert all pixels in the input image.
        lighten : bool, optional
            Specifies whether to lighten the input image.
        pyramid_down : bool, optional
            Specifies whether to downsample and then blur the input image.
        pyramid_up : bool, optional
            Specifies whether to upsample and then blur the input image.
        rotate_left : bool, optional
            Specifies whether to rotate the input image to the left.
        rotate_right : bool, optional
            Specifies whether to rotate the input image to the right.
        sharpen : bool, optional
            Specifies whether to sharpen the input image.
        vertical_flip : bool, optional
            Specifies whether to vertically flip the input image.

        Returns
        -------
        :class:`ImageTable`
            If `inplace=True`
        None
            If `inplace=False`

        '''

        crop_list = [{'mutations':dict(colorjittering=color_jitter,
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
        column_names = None
        if self.filename_col is not None:
            filename_prefix = self.filename_col[:-self.filename_col[::-1].find('_')-1]
            column_names = [col for col in self.columns if col.startswith(filename_prefix)]

        if inplace:
            self._retrieve('image.augmentimages',
                           image = self.image_cols,
                           copyvars=column_names,
                           casout=dict(replace=True, **self.to_outtable_params()),
                           croplist=crop_list,
                           writerandomly=True)

            # The following code generate the latest file name according
            # to the number of patches and mutation (_m) operations.
            computedvars, code = create_new_filename(self, True)
            self._retrieve('table.shuffle',
                           casout=dict(replace=True,
                                       **self.to_outtable_params()),
                           table=dict(computedvars=computedvars,
                                      computedvarsprogram=code,
                                      **self.to_table_params()))

            self.patch_level += 1
            self.filename_col = computedvars
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
        out = self._retrieve('image.summarizeimages', image=self.image_cols)['Summary']
        out = out.T.drop(['Column'])[0]
        out.name = None
        return out

    @property
    def label_freq(self, label = None):
        '''
        Summarize the distribution of different classes (labels) in the ImageTable

        Returns
        -------
        :class:`pd.Series`

        '''
        if label is None:
            label = self.cls_cols
        out = self._retrieve('simple.freq', table=self, inputs=[label])['Frequency']
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
