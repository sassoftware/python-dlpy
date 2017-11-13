# LeNet with batch normalization model definition


# LeNet model definition (batch normalization version)
def LeNet_Model(s, model_name='LeNet', include_top=True):
    # instantiate model
    s.buildModel(model=dict(name=model_name, replace=True), type='CNN')

    # input layer
    s.addLayer(model=model_name, name='mnist',
               layer=dict(type='input', nchannels=1, width=28, height=28,
                          scale=0.00392156862745098039))

    # conv1: 5*5*20
    s.addLayer(model=model_name, name='conv1',
               layer=dict(type='convolution', nFilters=20, width=5, height=5,
                          stride=1, act='identity', noBias=True, init='xavier'),
               srcLayers=['mnist'])

    # conv1 batch normalization
    s.addLayer(model=model_name, name='conv1_bn',
               layer=dict(type='batchnorm', act='relu'), srcLayers=['conv1'])

    # pool1 2*2*2
    s.addLayer(model=model_name, name='pool1',
               layer=dict(type='pooling', width=2, height=2, stride=2, pool='max'),
               srcLayers=['conv1_bn'])

    # conv2: 5*5*50
    s.addLayer(model=model_name, name='conv2',
               layer=dict(type='convolution', nFilters=50, width=5, height=5,
                          stride=1, act='identity', noBias=True, init='xavier'),
               srcLayers=['pool1'])

    # conv2 batch normalization
    s.addLayer(model=model_name, name='conv2_bn',
               layer=dict(type='batchnorm', act='relu'), srcLayers=['conv2'])

    # pool2 2*2*2
    s.addLayer(model=model_name, name='pool2',
               layer=dict(type='pooling', width=2, height=2, stride=2, pool='max'),
               srcLayers=['conv2_bn'])

    # fully connected layer
    s.addLayer(model=model_name, name='ip1',
               layer=dict(type='fullconnect', n=500, init='xavier', act='relu'),
               srcLayers=['pool2'])
    if include_top:
        # output layer
        s.addLayer(model=model_name, name='ip2',
                   layer=dict(type='output', n=10, init='xavier', act='softmax'),
                   srcLayers=['ip1'])
