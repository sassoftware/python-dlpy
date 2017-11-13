import sys


# ResNet50 model definition
def ResNet50_Model(s, model_name='RESNET50', inputCropType=None, inputChannelOffset=None, include_top=True):
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

    # ------------------- Residual Layer 3B -----------------------

    # res3b_branch2a layer: 128 channels, 1x1 conv, output = 28 x 28
    s.addLayer(model=model_name, name='res3b_branch2a',
               layer=dict(type='convolution', nFilters=128, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res3a'])

    # res3b_branch2a batch norm layer: 128 channels, output = 28 x 28
    s.addLayer(model=model_name, name='bn3b_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res3b_branch2a'])

    # res3b_branch2b layer: 128 channels, 3x3 conv, output = 28 x 28
    s.addLayer(model=model_name, name='res3b_branch2b',
               layer=dict(type='convolution', nFilters=128, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn3b_branch2a'])

    # res3b_branch2b batch norm layer: 128 channels, output = 28 x 28
    s.addLayer(model=model_name, name='bn3b_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res3b_branch2b'])

    # res3b_branch2c layer: 512 channels, 1x1 conv, output = 28 x 28
    s.addLayer(model=model_name, name='res3b_branch2c',
               layer=dict(type='convolution', nFilters=512, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn3b_branch2b'])

    # res3b_branch2c batch norm layer: 512 channels, output = 28 x 28
    s.addLayer(model=model_name, name='bn3b_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res3b_branch2c'])

    # res3b residual layer: 512 channels, output = 28 x 28
    s.addLayer(model=model_name, name='res3b',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn3b_branch2c', 'res3a'])

    # ------------------- Residual Layer 3C -----------------------

    # res3c_branch2a layer: 128 channels, 1x1 conv, output = 28 x 28
    s.addLayer(model=model_name, name='res3c_branch2a',
               layer=dict(type='convolution', nFilters=128, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res3b'])

    # res3c_branch2a batch norm layer: 128 channels, output = 28 x 28
    s.addLayer(model=model_name, name='bn3c_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res3c_branch2a'])

    # res3c_branch2b layer: 128 channels, 3x3 conv, output = 28 x 28
    s.addLayer(model=model_name, name='res3c_branch2b',
               layer=dict(type='convolution', nFilters=128, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn3c_branch2a'])

    # res3c_branch2b batch norm layer: 128 channels, output = 28 x 28
    s.addLayer(model=model_name, name='bn3c_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res3c_branch2b'])

    # res3c_branch2c layer: 512 channels, 1x1 conv, output = 28 x 28
    s.addLayer(model=model_name, name='res3c_branch2c',
               layer=dict(type='convolution', nFilters=512, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn3c_branch2b'])

    # res3c_branch2c batch norm layer: 512 channels, output = 28 x 28
    s.addLayer(model=model_name, name='bn3c_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res3c_branch2c'])

    # res3c residual layer: 512 channels, output = 28 x 28
    s.addLayer(model=model_name, name='res3c',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn3c_branch2c', 'res3b'])

    # ------------------- Residual Layer 3D -----------------------

    # res3d_branch2a layer: 128 channels, 1x1 conv, output = 28 x 28
    s.addLayer(model=model_name, name='res3d_branch2a',
               layer=dict(type='convolution', nFilters=128, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res3c'])

    # res3d_branch2a batch norm layer: 128 channels, output = 28 x 28
    s.addLayer(model=model_name, name='bn3d_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res3d_branch2a'])

    # res3d_branch2b layer: 128 channels, 3x3 conv, output = 28 x 28
    s.addLayer(model=model_name, name='res3d_branch2b',
               layer=dict(type='convolution', nFilters=128, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn3d_branch2a'])

    # res3d_branch2b batch norm layer: 128 channels, output = 28 x 28
    s.addLayer(model=model_name, name='bn3d_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res3d_branch2b'])

    # res3d_branch2c layer: 512 channels, 1x1 conv, output = 28 x 28
    s.addLayer(model=model_name, name='res3d_branch2c',
               layer=dict(type='convolution', nFilters=512, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn3d_branch2b'])

    # res3d_branch2c batch norm layer: 512 channels, output = 28 x 28
    s.addLayer(model=model_name, name='bn3d_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res3d_branch2c'])

    # res3d residual layer: 512 channels, output = 28 x 28
    s.addLayer(model=model_name, name='res3d',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn3d_branch2c', 'res3c'])

    # ------------- Layer 4A --------------------

    # res4a_branch1 layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4a_branch1',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=2, includebias=False, act='identity'),
               srcLayers=['res3d'])

    # res4a_branch1 batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4a_branch1',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4a_branch1'])

    # res4a_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4a_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=2, includebias=False, act='identity'),
               srcLayers=['res3d'])

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

    # ------------------- Residual Layer 4B -----------------------

    # res4b_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4a'])

    # res4b_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b_branch2a'])

    # res4b_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b_branch2a'])

    # res4b_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4b_branch2b'])

    # res4b_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4b_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4b_branch2b'])

    # res4b_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4b_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4b_branch2c'])

    # res4b residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4b',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4b_branch2c', 'res4a'])

    # ------------------- Residual Layer 4C -----------------------

    # res4c_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4c_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4b'])

    # res4c_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4c_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4c_branch2a'])

    # res4c_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4c_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4c_branch2a'])

    # res4c_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4c_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4c_branch2b'])

    # res4c_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4c_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4c_branch2b'])

    # res4c_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4c_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4c_branch2c'])

    # res4c residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4c',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4c_branch2c', 'res4b'])

    # ------------------- Residual Layer 4D -----------------------

    # res4d_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4d_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4c'])

    # res4d_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4d_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4d_branch2a'])

    # res4d_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4d_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4d_branch2a'])

    # res4d_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4d_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4d_branch2b'])

    # res4d_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4d_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4d_branch2b'])

    # res4d_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4d_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4d_branch2c'])

    # res4d residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4d',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4d_branch2c', 'res4c'])

    # ------------------- Residual Layer 4E ----------------------- */

    # res4e_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4e_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4d'])

    # res4e_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4e_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4e_branch2a'])

    # res4e_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4e_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4e_branch2a'])

    # res4e_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4e_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4e_branch2b'])

    # res4e_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4e_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4e_branch2b'])

    # res4e_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4e_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4e_branch2c'])

    # res4e residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4e',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4e_branch2c', 'res4d'])

    # ------------------- Residual Layer 4F -----------------------

    # res4f_branch2a layer: 256 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4f_branch2a',
               layer=dict(type='convolution', nFilters=256, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['res4e'])

    # res4f_branch2a batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4f_branch2a',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4f_branch2a'])

    # res4f_branch2b layer: 256 channels, 3x3 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4f_branch2b',
               layer=dict(type='convolution', nFilters=256, width=3, height=3,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4f_branch2a'])

    # res4f_branch2b batch norm layer: 256 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4f_branch2b',
               layer=dict(type='batchnorm', act='relu'),
               srcLayers=['res4f_branch2b'])

    # res4f_branch2c layer: 1024 channels, 1x1 conv, output = 14 x 14
    s.addLayer(model=model_name, name='res4f_branch2c',
               layer=dict(type='convolution', nFilters=1024, width=1, height=1,
                          stride=1, includebias=False, act='identity'),
               srcLayers=['bn4f_branch2b'])

    # res4f_branch2c batch norm layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='bn4f_branch2c',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res4f_branch2c'])

    # res4f residual layer: 1024 channels, output = 14 x 14
    s.addLayer(model=model_name, name='res4f',
               layer=dict(type='residual', act='relu'),
               srcLayers=['bn4f_branch2c', 'res4e'])

    # ------------- Layer 5A -------------------- */

    # res5a_branch1 layer: 2048 channels, 1x1 conv, output = 7 x 7
    s.addLayer(model=model_name, name='res5a_branch1',
               layer=dict(type='convolution', nFilters=2048, width=1, height=1,
                          stride=2, includebias=False, act='identity'),
               srcLayers=['res4f'])

    # res5a_branch1 batch norm layer: 2048 channels, output = 7 x 7
    s.addLayer(model=model_name, name='bn5a_branch1',
               layer=dict(type='batchnorm', act='identity'),
               srcLayers=['res5a_branch1'])

    # res5a_branch2a layer: 512 channels, 1x1 conv, output = 7 x 7
    s.addLayer(model=model_name, name='res5a_branch2a',
               layer=dict(type='convolution', nFilters=512, width=1, height=1,
                          stride=2, includebias=False, act='identity'),
               srcLayers=['res4f'])

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
