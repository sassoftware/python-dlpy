import sys


# ResNet101 model definition
def ResNet101_Model(s, model_name='RESNET101', inputCropType=None, inputChannelOffset=None, include_top=True):
    # quick error-checking and default setting
    if (inputCropType == None):
        inputCropType = "NONE"
    else:
        if (inputCropType.upper() != "NONE") and (inputCropType.upper() != "UNIQUE"):
            sys.exit("ERROR: inputCropType can only be NONE or UNIQUE")

    if (inputChannelOffset == None):
        inputChannelOffset = [103.939, 116.779, 123.68]

    # instantiate model
    s.buildModel(model=dict(name=model_name, replace=True), type='CNN')

    # input layer
    s.addLayer(model=model_name, name='data',
               layer=dict(type='input', nchannels=3, width=224, height=224,
                          randomcrop=inputCropType, offsets=inputChannelOffset))

    # -------------------- Layer 1 ----------------------

    # conv1 layer: 64 channels, 7x7 conv, stride=2; output = 112 x 112 */
    s.addLayer(model=model_name, name='conv1',
               layer=dict(type='convolution', nFilters=64, width=7, height=7,
                          stride=2, act='identity'),
               srcLayers=['data'])

    # conv1 batch norm layer: 64 channels, output = 112 x 112 */
    s.addLayer(model=model_name, name='bn_conv1',
               layer=dict(type='batchnorm', act='relu'), srcLayers=['conv1'])

    # pool1 layer: 64 channels, 3x3 pooling, output = 56 x 56 */
    s.addLayer(model=model_name, name='pool1',
               layer=dict(type='pooling', width=3, height=3, stride=2, pool='max'),
               srcLayers=['bn_conv1'])

    # ------------------- Residual Layer 2A -----------------------

    # res2a_branch1 layer: 256 channels, 1x1 conv, output = 56 x 56
    s.addLayer(model=model_name, name='res2a_branch1',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['pool1'])

    # res2a_branch1 batch norm layer: 256 channels, output = 56 x 56
    s.addLayer(model=model_name, name='bn2a_branch1',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res2a_branch1'])

    # res2a_branch2a layer: 64 channels, 1x1 conv, output = 56 x 56
    s.addLayer(model=model_name, name='res2a_branch2a',
               layer=dict(type='convolution', nFilters=64, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['pool1'])

    # res2a_branch2a batch norm layer: 64 channels, output = 56 x 56
    s.addLayer(model=model_name, name='bn2a_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res2a_branch2a'])

    # res2a_branch2b layer: 64 channels, 3x3 conv, output = 56 x 56
    s.addLayer(model=model_name, name='res2a_branch2b',
               layer=dict(type='convolution', nFilters=64, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn2a_branch2a'])

    # res2a_branch2b batch norm layer: 64 channels, output = 56 x 56
    s.addLayer(model=model_name, name='bn2a_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res2a_branch2b'])

    # res2a_branch2c layer: 256 channels, 1x1 conv, output = 56 x 56
    s.addLayer(model=model_name, name='res2a_branch2c',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn2a_branch2b'])

    # res2a_branch2c batch norm layer: 256 channels, output = 56 x 56
    s.addLayer(model=model_name, name='bn2a_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res2a_branch2c'])

    # res2a residual layer: 256 channels, output = 56 x 56
    s.addLayer(model=model_name, name='res2a',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn2a_branch2c', 'bn2a_branch1'])

    # ------------------- Residual Layer 2B -----------------------

    # res2b_branch2a layer: 64 channels, 1x1 conv, output = 56 x 56
    s.addLayer(model=model_name, name='res2b_branch2a',
               layer=dict(type='convolution', nFilters=64, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res2a'])

    # res2b_branch2a batch norm layer: 64 channels, output = 56 x 56
    s.addLayer(model=model_name, name='bn2b_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res2b_branch2a'])

    # res2b_branch2b layer: 64 channels, 3x3 conv, output = 56 x 56
    s.addLayer(model=model_name, name='res2b_branch2b',
               layer=dict(type='convolution', nFilters=64, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn2b_branch2a'])

    # res2b_branch2b batch norm layer: 64 channels, output = 56 x 56
    s.addLayer(model=model_name, name='bn2b_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res2b_branch2b'])

    # res2b_branch2c layer: 256 channels, 1x1 conv, output = 56 x 56
    s.addLayer(model=model_name, name='res2b_branch2c',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn2b_branch2b'])

    # res2b_branch2c batch norm layer: 256 channels, output = 56 x 56
    s.addLayer(model=model_name, name='bn2b_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res2b_branch2c'])

    # res2b residual layer: 256 channels, output = 56 x 56
    s.addLayer(model=model_name, name='res2b',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn2b_branch2c', 'res2a'])

    # ------------------- Residual Layer 2C ----------------------- 

    # res2c_branch2a layer: 64 channels, 1x1 conv, output = 56 x 56
    s.addLayer(model=model_name, name='res2c_branch2a',
               layer=dict(type='convolution', nFilters=64, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res2b'])

    # res2c_branch2a batch norm layer: 64 channels, output = 56 x 56
    s.addLayer(model=model_name, name='bn2c_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res2c_branch2a'])

    # res2c_branch2b layer: 64 channels, 3x3 conv, output = 56 x 56
    s.addLayer(model=model_name, name='res2c_branch2b',
               layer=dict(type='convolution', nFilters=64, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn2c_branch2a'])

    # res2c_branch2b batch norm layer: 64 channels, output = 56 x 56
    s.addLayer(model=model_name, name='bn2c_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res2c_branch2b'])

    # res2c_branch2c layer: 256 channels, 1x1 conv, output = 56 x 56
    s.addLayer(model=model_name, name='res2c_branch2c',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn2c_branch2b'])

    # res2c_branch2c batch norm layer: 256 channels, output = 56 x 56
    s.addLayer(model=model_name, name='bn2c_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res2c_branch2c'])

    # res2c residual layer: 256 channels, output = 56 x 56
    s.addLayer(model=model_name, name='res2c',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn2c_branch2c', 'res2b'])

    # ------------- Layer 3A --------------------

    # res3a_branch1 layer: 512 channels, 1x1 conv, output = 28 x 28
    s.addLayer(model=model_name, name='res3a_branch1',
               layer=dict(type='convolution', nFilters=512, width=1, height=1,
                          stride=2, includebias=False, act='identity'),
               srcLayers=['res2c'])

    # res3a_branch1 batch norm layer: 512 channels, output = 28 x 28
    s.addLayer(model=model_name, name='bn3a_branch1',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res3a_branch1'])

    # res3a_branch2a layer: 128 channels, 1x1 conv, output = 28 x 28
    s.addLayer(model=model_name, name='res3a_branch2a',
               layer=dict(type='convolution', nFilters=128, width=1, height=1,
                          stride=2, includebias=False, act='identity'),
               srcLayers=['res2c'])

    # res3a_branch2a batch norm layer: 128 channels, output = 28 x 28
    s.addLayer(model=model_name, name='bn3a_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res3a_branch2a'])

    # res3a_branch2b layer: 128 channels, 3x3 conv, output = 28 x 28
    s.addLayer(model=model_name, name='res3a_branch2b',
               layer=dict(type='convolution', nFilters=128, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn3a_branch2a'])

    # res3a_branch2b batch norm layer: 128 channels, output = 28 x 28
    s.addLayer(model=model_name, name='bn3a_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res3a_branch2b'])

    # res3a_branch2c layer: 512 channels, 1x1 conv, output = 28 x 28
    s.addLayer(model=model_name, name='res3a_branch2c',
               layer=dict(type='convolution', nFilters=512, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn3a_branch2b'])

    # res3a_branch2c batch norm layer: 512 channels, output = 28 x 28
    s.addLayer(model=model_name, name='bn3a_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res3a_branch2c'])

    # res3a residual layer: 512 channels, output = 28 x 28
    s.addLayer(model=model_name, name='res3a',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn3a_branch2c', 'bn3a_branch1'])

    # ------------------- Residual Layer 3B1 -----------------------

    # res3b1_branch2a layer: 128 channels, 1x1 conv, output = 28 x 28
    s.addLayer(model=model_name, name='res3b1_branch2a',
               layer=dict(type='convolution', nFilters=128, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res3a'])

    # res3b1_branch2a batch norm layer: 128 channels, output = 28 x 28
    s.addLayer(model=model_name, name='bn3b1_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res3b1_branch2a'])

    # res3b1_branch2b layer: 128 channels, 3x3 conv, output = 28 x 28
    s.addLayer(model=model_name, name='res3b1_branch2b',
               layer=dict(type='convolution', nFilters=128, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn3b1_branch2a'])

    # res3b1_branch2b batch norm layer: 128 channels, output = 28 x 28
    s.addLayer(model=model_name, name='bn3b1_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res3b1_branch2b'])

    # res3b1_branch2c layer: 512 channels, 1x1 conv, output = 28 x 28
    s.addLayer(model=model_name, name='res3b1_branch2c',
               layer=dict(type='convolution', nFilters=512, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn3b1_branch2b'])

    # res3b1_branch2c batch norm layer: 512 channels, output = 28 x 28
    s.addLayer(model=model_name, name='bn3b1_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res3b1_branch2c'])

    # res3b1 residual layer: 512 channels, output = 28 x 28
    s.addLayer(model=model_name, name='res3b1',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn3b1_branch2c', 'res3a'])

    # ------------------- Residual Layer 3B2 -----------------------

    # res3b2_branch2a layer: 128 channels, 1x1 conv, output = 28 x 28
    s.addLayer(model=model_name, name='res3b2_branch2a',
               layer=dict(type='convolution', nFilters=128, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res3b1'])

    # res3b2_branch2a batch norm layer: 128 channels, output = 28 x 28
    s.addLayer(model=model_name, name='bn3b2_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res3b2_branch2a'])

    # res3b2_branch2b layer: 128 channels, 3x3 conv, output = 28 x 28
    s.addLayer(model=model_name, name='res3b2_branch2b',
               layer=dict(type='convolution', nFilters=128, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn3b2_branch2a'])

    # res3b2_branch2b batch norm layer: 128 channels, output = 28 x 28
    s.addLayer(model=model_name, name='bn3b2_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res3b2_branch2b'])

    # res3b2_branch2c layer: 512 channels, 1x1 conv, output = 28 x 28
    s.addLayer(model=model_name, name='res3b2_branch2c',
               layer=dict(type='convolution', nFilters=512, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn3b2_branch2b'])

    # res3b2_branch2c batch norm layer: 512 channels, output = 28 x 28
    s.addLayer(model=model_name, name='bn3b2_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res3b2_branch2c'])

    # res3b2 residual layer: 512 channels, output = 28 x 28
    s.addLayer(model=model_name, name='res3b2',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn3b2_branch2c', 'res3b1'])

    # ------------------- Residual Layer 3B3 -----------------------

    # res3b3_branch2a layer: 128 channels, 1x1 conv, output = 28 x 28
    s.addLayer(model=model_name, name='res3b3_branch2a',
               layer=dict(type='convolution', nFilters=128, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res3b2'])

    # res3b3_branch2a batch norm layer: 128 channels, output = 28 x 28
    s.addLayer(model=model_name, name='bn3b3_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res3b3_branch2a'])

    # res3b3_branch2b layer: 128 channels, 3x3 conv, output = 28 x 28
    s.addLayer(model=model_name, name='res3b3_branch2b',
               layer=dict(type='convolution', nFilters=128, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn3b3_branch2a'])

    # res3b3_branch2b batch norm layer: 128 channels, output = 28 x 28
    s.addLayer(model=model_name, name='bn3b3_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res3b3_branch2b'])

    # res3b3_branch2c layer: 512 channels, 1x1 conv, output = 28 x 28
    s.addLayer(model=model_name, name='res3b3_branch2c',
               layer=dict(type='convolution', nFilters=512, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn3b3_branch2b'])

    # res3b3_branch2c batch norm layer: 512 channels, output = 28 x 28
    s.addLayer(model=model_name, name='bn3b3_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res3b3_branch2c'])

    # res3b3 residual layer: 512 channels, output = 28 x 28
    s.addLayer(model=model_name, name='res3b3',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn3b3_branch2c', 'res3b2'])

    # ------------- Layer 4A --------------------

    # res4a_branch1 layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4a_branch1',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=2, includebias=False, act='identity'),
               srcLayers=['res3b3'])

    # res4a_branch1 batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4a_branch1',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4a_branch1'])

    # res4a_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4a_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=2, includebias=False, act='identity'),
               srcLayers=['res3b3'])

    # res4a_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4a_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4a_branch2a'])

    # res4a_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4a_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4a_branch2a'])

    # res4a_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4a_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4a_branch2b'])

    # res4a_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4a_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4a_branch2b'])

    # res4a_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4a_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4a_branch2c'])

    # res4a residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4a',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4a_branch2c', 'bn4a_branch1'])

    # ------------------- Residual Layer 4B1 -----------------------

    # res4b1_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b1_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4a'])

    # res4b1_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b1_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b1_branch2a'])

    # res4b1_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b1_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b1_branch2a'])

    # res4b1_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b1_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b1_branch2b'])

    # res4b1_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b1_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b1_branch2b'])

    # res4b1_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b1_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4b1_branch2c'])

    # res4b1 residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4b1',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4b1_branch2c', 'res4a'])

    # ------------------- Residual Layer 4B2 -----------------------

    # res4b2_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b2_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4b1'])

    # res4b2_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b2_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b2_branch2a'])

    # res4b2_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b2_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b2_branch2a'])

    # res4b2_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b2_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b2_branch2b'])

    # res4b2_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b2_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b2_branch2b'])

    # res4b2_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b2_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4b2_branch2c'])

    # res4b2 residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4b2',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4b2_branch2c', 'res4b1'])

    # ------------------- Residual Layer 4B3 -----------------------

    # res4b3_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b3_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4b2'])

    # res4b3_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b3_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b3_branch2a'])

    # res4b3_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b3_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b3_branch2a'])

    # res4b3_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b3_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b3_branch2b'])

    # res4b3_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b3_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b3_branch2b'])

    # res4b3_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b3_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4b3_branch2c'])

    # res4b3 residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4b3',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4b3_branch2c', 'res4b2'])

    # ------------------- Residual Layer 4B4 ----------------------- */

    # res4b4_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b4_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4b3'])

    # res4b4_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b4_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b4_branch2a'])

    # res4b4_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b4_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b4_branch2a'])

    # res4b4_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b4_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b4_branch2b'])

    # res4b4_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b4_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b4_branch2b'])

    # res4b4_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b4_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4b4_branch2c'])

    # res4b4 residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4b4',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4b4_branch2c', 'res4b3'])

    # ------------------- Residual Layer 4B5 -----------------------

    # res4b5_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b5_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4b4'])

    # res4b5_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b5_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b5_branch2a'])

    # res4b5_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b5_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b5_branch2a'])

    # res4b5_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b5_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b5_branch2b'])

    # res4b5_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b5_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b5_branch2b'])

    # res4b5_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b5_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4b5_branch2c'])

    # res4b5 residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4b5',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4b5_branch2c', 'res4b4'])

    # ------------------- Residual Layer 4B6 -----------------------

    # res4b6_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b6_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4b5'])

    # res4b6_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b6_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b6_branch2a'])

    # res4b6_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b6_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b6_branch2a'])

    # res4b6_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b6_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b6_branch2b'])

    # res4b6_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b6_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b6_branch2b'])

    # res4b6_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b6_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4b6_branch2c'])

    # res4b6 residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4b6',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4b6_branch2c', 'res4b5'])

    # ------------------- Residual Layer 4B7 -----------------------

    # res4b7_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b7_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4b6'])

    # res4b7_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b7_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b7_branch2a'])

    # res4b7_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b7_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b7_branch2a'])

    # res4b7_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b7_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b7_branch2b'])

    # res4b7_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b7_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b7_branch2b'])

    # res4b7_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b7_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4b7_branch2c'])

    # res4b7 residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4b7',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4b7_branch2c', 'res4b6'])

    # ------------------- Residual Layer 4B8 -----------------------

    # res4b8_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b8_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4b7'])

    # res4b8_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b8_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b8_branch2a'])

    # res4b8_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b8_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b8_branch2a'])

    # res4b8_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b8_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b8_branch2b'])

    # res4b8_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b8_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b8_branch2b'])

    # res4b8_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b8_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4b8_branch2c'])

    # res4b8 residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4b8',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4b8_branch2c', 'res4b7'])

    # ------------------- Residual Layer 4B9 -----------------------

    # res4b9_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b9_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4b8'])

    # res4b9_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b9_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b9_branch2a'])

    # res4b9_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b9_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b9_branch2a'])

    # res4b9_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b9_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b9_branch2b'])

    # res4b9_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b9_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b9_branch2b'])

    # res4b9_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b9_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4b9_branch2c'])

    # res4b9 residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4b9',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4b9_branch2c', 'res4b8'])

    # ------------------- Residual Layer 4B10 -----------------------

    # res4b10_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b10_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4b9'])

    # res4b10_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b10_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b10_branch2a'])

    # res4b10_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b10_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b10_branch2a'])

    # res4b10_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b10_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b10_branch2b'])

    # res4b10_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b10_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b10_branch2b'])

    # res4b10_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b10_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4b10_branch2c'])

    # res4b10 residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4b10',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4b10_branch2c', 'res4b9'])

    # ------------------- Residual Layer 4B11 -----------------------

    # res4b11_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b11_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4b10'])

    # res4b11_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b11_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b11_branch2a'])

    # res4b11_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b11_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b11_branch2a'])

    # res4b11_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b11_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b11_branch2b'])

    # res4b11_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b11_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b11_branch2b'])

    # res4b11_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b11_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4b11_branch2c'])

    # res4b11 residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4b11',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4b11_branch2c', 'res4b10'])

    # ------------------- Residual Layer 4B12 -----------------------

    # res4b12_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b12_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4b11'])

    # res4b12_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b12_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b12_branch2a'])

    # res4b12_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b12_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b12_branch2a'])

    # res4b12_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b12_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b12_branch2b'])

    # res4b12_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b12_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b12_branch2b'])

    # res4b12_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b12_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4b12_branch2c'])

    # res4b12 residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4b12',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4b12_branch2c', 'res4b11'])

    # ------------------- Residual Layer 4B13 -----------------------

    # res4b13_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b13_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4b12'])

    # res4b13_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b13_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b13_branch2a'])

    # res4b13_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b13_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b13_branch2a'])

    # res4b13_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b13_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b13_branch2b'])

    # res4b13_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b13_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b13_branch2b'])

    # res4b13_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b13_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4b13_branch2c'])

    # res4b13 residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4b13',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4b13_branch2c', 'res4b12'])

    # ------------------- Residual Layer 4B14 -----------------------

    # res4b14_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b14_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4b13'])

    # res4b14_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b14_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b14_branch2a'])

    # res4b14_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b14_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b14_branch2a'])

    # res4b14_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b14_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b14_branch2b'])

    # res4b14_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b14_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b14_branch2b'])

    # res4b14_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b14_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4b14_branch2c'])

    # res4b14 residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4b14',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4b14_branch2c', 'res4b13'])

    # ------------------- Residual Layer 4B15 -----------------------

    # res4b15_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b15_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4b14'])

    # res4b15_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b15_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b15_branch2a'])

    # res4b15_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b15_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b15_branch2a'])

    # res4b15_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b15_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b15_branch2b'])

    # res4b15_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b15_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b15_branch2b'])

    # res4b15_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b15_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4b15_branch2c'])

    # res4b15 residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4b15',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4b15_branch2c', 'res4b14'])

    # ------------------- Residual Layer 4B16 -----------------------

    # res4b16_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b16_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4b15'])

    # res4b16_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b16_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b16_branch2a'])

    # res4b16_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b16_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b16_branch2a'])

    # res4b16_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b16_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b16_branch2b'])

    # res4b16_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b16_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b16_branch2b'])

    # res4b16_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b16_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4b16_branch2c'])

    # res4b16 residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4b16',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4b16_branch2c', 'res4b15'])

    # ------------------- Residual Layer 4B17 -----------------------

    # res4b17_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b17_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4b16'])

    # res4b17_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b17_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b17_branch2a'])

    # res4b17_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b17_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b17_branch2a'])

    # res4b17_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b17_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b17_branch2b'])

    # res4b17_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b17_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b17_branch2b'])

    # res4b17_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b17_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4b17_branch2c'])

    # res4b17 residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4b17',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4b17_branch2c', 'res4b16'])

    # ------------------- Residual Layer 4B18 -----------------------

    # res4b18_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b18_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4b17'])

    # res4b18_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b18_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b18_branch2a'])

    # res4b18_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b18_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b18_branch2a'])

    # res4b18_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b18_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b18_branch2b'])

    # res4b18_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b18_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b18_branch2b'])

    # res4b18_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b18_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4b18_branch2c'])

    # res4b18 residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4b18',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4b18_branch2c', 'res4b17'])

    # ------------------- Residual Layer 4B19 -----------------------

    # res4b19_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b19_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4b18'])

    # res4b19_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b19_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b19_branch2a'])

    # res4b19_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b19_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b19_branch2a'])

    # res4b19_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b19_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b19_branch2b'])

    # res4b19_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b19_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b19_branch2b'])

    # res4b19_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b19_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4b19_branch2c'])

    # res4b19 residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4b19',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4b19_branch2c', 'res4b18'])

    # ------------------- Residual Layer 4B20 -----------------------

    # res4b20_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b20_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4b19'])

    # res4b20_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b20_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b20_branch2a'])

    # res4b20_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b20_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b20_branch2a'])

    # res4b20_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b20_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b20_branch2b'])

    # res4b20_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b20_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b20_branch2b'])

    # res4b20_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b20_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4b20_branch2c'])

    # res4b20 residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4b20',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4b20_branch2c', 'res4b19'])

    # ------------------- Residual Layer 4B21 -----------------------

    # res4b21_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b21_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4b20'])

    # res4b21_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b21_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b21_branch2a'])

    # res4b21_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b21_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b21_branch2a'])

    # res4b21_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b21_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b21_branch2b'])

    # res4b21_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b21_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b21_branch2b'])

    # res4b21_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b21_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4b21_branch2c'])

    # res4b21 residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4b21',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4b21_branch2c', 'res4b20'])

    # ------------------- Residual Layer 4B22 -----------------------

    # res4b22_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b22_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4b21'])

    # res4b22_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b22_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b22_branch2a'])

    # res4b22_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b22_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b22_branch2a'])

    # res4b22_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b22_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b22_branch2b'])

    # res4b22_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b22_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b22_branch2b'])

    # res4b22_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b22_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4b22_branch2c'])

    # res4b22 residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4b22',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4b22_branch2c', 'res4b21'])

    # ------------- Layer 5A -------------------- */

    # res5a_branch1 layer: 2048 channels, 1x1 conv, output = 7 x 7
    s.addLayer(model=model_name, name='res5a_branch1',
               layer=dict(type='convolution', nFilters=2048, width=1, height=1,
                          stride=2, includebias=False, act='identity'),
               srcLayers=['res4b22'])

    # res5a_branch1 batch norm layer: 2048 channels, output = 7 x 7
    s.addLayer(model=model_name, name='bn5a_branch1',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res5a_branch1'])

    # res5a_branch2a layer: 512 channels, 1x1 conv, output = 7 x 7
    s.addLayer(model=model_name, name='res5a_branch2a',
               layer=dict(type='convolution', nFilters=512, width=1, height=1,
                          stride=2, includebias=False, act='identity'),
               srcLayers=['res4b22'])

    # res5a_branch2a batch norm layer: 512 channels, output = 7 x 7
    s.addLayer(model=model_name, name='bn5a_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res5a_branch2a'])

    # res5a_branch2b layer: 512 channels, 3x3 conv, output = 7 x 7
    s.addLayer(model=model_name, name='res5a_branch2b',
               layer=dict(type='convolution', nFilters=512, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn5a_branch2a'])

    # res5a_branch2b batch norm layer: 512 channels, output = 7 x 7
    s.addLayer(model=model_name, name='bn5a_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res5a_branch2b'])

    # res5a_branch2c layer: 2048 channels, 1x1 conv, output = 7 x 7
    s.addLayer(model=model_name, name='res5a_branch2c',
               layer=dict(type='convolution', nFilters=2048, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn5a_branch2b'])

    # res5a_branch2c batch norm layer: 2048 channels, output = 7 x 7
    s.addLayer(model=model_name, name='bn5a_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res5a_branch2c'])

    # res5a residual layer: 2048 channels, output = 7 x 7
    s.addLayer(model=model_name, name='res5a',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn5a_branch2c', 'bn5a_branch1'])

    # ------------------- Residual Layer 5B -----------------------

    # res5b_branch2a layer: 512 channels, 1x1 conv, output = 7 x 7
    s.addLayer(model=model_name, name='res5b_branch2a',
               layer=dict(type='convolution', nFilters=512, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res5a'])

    # res5b_branch2a batch norm layer: 512 channels, output = 7 x 7
    s.addLayer(model=model_name, name='bn5b_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res5b_branch2a'])

    # res5b_branch2b layer: 512 channels, 3x3 conv, output = 7 x 7
    s.addLayer(model=model_name, name='res5b_branch2b',
               layer=dict(type='convolution', nFilters=512, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn5b_branch2a'])

    # res5b_branch2b batch norm layer: 512 channels, output = 7 x 7
    s.addLayer(model=model_name, name='bn5b_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res5b_branch2b'])

    # res5b_branch2c layer: 2048 channels, 1x1 conv, output = 7 x 7
    s.addLayer(model=model_name, name='res5b_branch2c',
               layer=dict(type='convolution', nFilters=2048, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn5b_branch2b'])

    # res5b_branch2c batch norm layer: 2048 channels, output = 7 x 7
    s.addLayer(model=model_name, name='bn5b_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res5b_branch2c'])

    # res5b residual layer: 2048 channels, output = 7 x 7
    s.addLayer(model=model_name, name='res5b',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn5b_branch2c', 'res5a'])

    # ------------------- Residual Layer 5C -----------------------

    # res5c_branch2a layer: 512 channels, 1x1 conv, output = 7 x 7
    s.addLayer(model=model_name, name='res5c_branch2a',
               layer=dict(type='convolution', nFilters=512, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res5b'])

    # res5c_branch2a batch norm layer: 512 channels, output = 7 x 7
    s.addLayer(model=model_name, name='bn5c_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res5c_branch2a'])

    # res5c_branch2b layer: 512 channels, 3x3 conv, output = 7 x 7
    s.addLayer(model=model_name, name='res5c_branch2b',
               layer=dict(type='convolution', nFilters=512, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn5c_branch2a'])

    # res5c_branch2b batch norm layer: 512 channels, output = 7 x 7
    s.addLayer(model=model_name, name='bn5c_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res5c_branch2b'])

    # res5c_branch2c layer: 2048 channels, 1x1 conv, output = 7 x 7
    s.addLayer(model=model_name, name='res5c_branch2c',
               layer=dict(type='convolution', nFilters=2048, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn5c_branch2b'])

    # res5c_branch2c batch norm layer: 2048 channels, output = 7 x 7
    s.addLayer(model=model_name, name='bn5c_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res5c_branch2c'])

    # res5c residual layer: 2048 channels, output = 7 x 7
    s.addLayer(model=model_name, name='res5c',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn5c_branch2c', 'res5b'])

    # ------------------- final layers ----------------------

    # pool5 layer: 2048 channels, 7x7 pooling, output = 1 x 1
    s.addLayer(model=model_name, name='pool5',
               layer=dict(type='pooling', width=7, height=7, stride=7, pool='mean'),
               srcLayers=['res5c'])
    if include_top:
        # fc1000 output layer: 1000 neurons */
        s.addLayer(model=model_name, name='fc1000',
                   layer=dict(type='output', n=1000, act='softmax'),
                   srcLayers=['pool5'])
