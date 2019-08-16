<div align="center">  
<h1>DLPy - SAS Viya Deep Learning API for Python</h1>
<img src="https://github.com/sassoftware/python-dlpy/raw/master/doc/images/sas-ai.jpg" alt="SAS Deep Learning Icon" height="150">
<p>An efficient way to apply deep learning methods to image, text, and audio data.</p>
  
  <a href="https://www.sas.com/en_us/software/viya.html">
    <img src="https://img.shields.io/badge/SAS%20Viya-3.4-blue.svg?&colorA=0b5788&style=for-the-badge&logoWidth=30&logo=data:image/png;base64,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"
        alt="SAS Viya Version"/>
  </a>
        
  <a href="https://www.sas.com/en_us/software/visual-data-mining-machine-learning.html">
    <img src="https://img.shields.io/badge/pip_install_sas_dlpy-blue.svg?&style=for-the-badge&colorA=254f73" alt="Python Version">
  
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3+-blue.svg?&style=for-the-badge&colorA=254f73" alt="Python Version">
  </a>
  
</div>

### Overview
DLPy is a high-level Python library for the SAS Deep learning features 
available in SAS Viya. DLPy is designed to provide an efficient way to 
apply deep learning methods to image, text, and audio data. DLPy APIs 
created following the [Keras](https://keras.io/) APIs with a touch 
of [PyTorch](https://pytorch.org/) flavor.

### What's Recently Added
* Text, audio, and time series support in addition to image
* New APIs for:
   * RNN based tasks: text classification, text generation, and sequence labeling
   * Object detection
   * Image segmentation
   * Time series processing and modeling
* Processing audio files and creating speech recognition models
* Additional pre-defined network architectures such as DenseNet, DarkNet, Inception, Yolo, 
FasterRCNN, U-Net, MobileNet, and ShuffleNet
* Enhanced data visualization and metadata handling

### Prerequisites
- Python version 3 or greater is required
- Install SAS [Scripting Wrapper for Analytics Transfer (SWAT)](https://github.com/sassoftware/python-swat) for Python using `pip install swat` or `conda install -c sas-institute swat`
- Access to a SAS Viya 3.4 environment with [Visual Data Mining and Machine Learning](https://www.sas.com/en_us/software/visual-data-mining-machine-learning.html) (VDMML) is required
- A user login to your SAS Viya back-end is required. See your system administrator for details if you do not have a SAS Viya account.
- It is recommended that you install the open source graph visualization software called [Graphviz](https://www.graphviz.org/download/) to enable graphic visualizations of the DLPy deep learning models
- Install DLPy using `pip install sas-dlpy` or `conda install -c sas-institute sas-dlpy`

#### SAS Viya and VDMML versions vs. DLPY versions
DLPy versions are aligned with the SAS Viya and VDMML versions. 
Below is the versions matrix.

<table>
<thead>
<tr>
<th align='left'>DLPy</th>
<th align='left'>SAS Viya</th>
<th align='left'>VDMML</th>
</thead>
<tbody>
<tr>
<td align='left'>1.1</td>
<td align='left'>3.4</td>
<td align='left'>8.4</td>
</tr>
<tr>
<td align='left'>1.0</td>
<td align='left'>3.4</td>
<td align='left'>8.3</td>
</tr>
</tbody>
</table>

The table above can be read as follows: DLPy versions between 1.0 (inclusive)
to 1.1 (exclusive) are designed to work with the SAS Viya 3.4 and VDMML 8.3.

#### External Libraries ####
The following versions of external libraries are supported:
- ONNX: versions >= 1.5.0
- Keras: versions >= 2.1.3


### Getting Started

To connect to a SAS Viya server, import SWAT and use the swat.CAS class to
create a connection:

Note: The default CAS port is 5570.

    >>> import swat
    >>> sess = swat.CAS('mycloud.example.com', 5570)

Next, import the DLPy package, and then build a simple convolutional 
neural network (CNN) model.

Import DLPy model functions:

    >>> from dlpy import Model, Sequential
    >>> from dlpy.layers import *

Use DLPy to create a sequential model and name it `Simple_CNN`:

    >>> model1 = Sequential(sess, model_table = 'Simple_CNN')

Define an input layer to add to `model1`:

    # The input shape contains RGB images (3 channels)
    # The model images are 224 px in height and 224 px in width

    >>> model1.add(InputLayer(3,224,224))

    NOTE: Input layer added.

Add a 2D convolution layer and a pooling layer:

    # Add 2-Dimensional Convolution Layer to model1
    # that has 8 filters and a kernel size of 7. 

    >>> model1.add(Conv2d(8,7))

    NOTE: Convolutional layer added.

    # Add Pooling Layer of size 2

    >>> model1.add(Pooling(2))

    NOTE: Pooling layer added.
    
Add an additional pair of 2D convolution and pooling layers:

    # Add another 2D convolution Layer that has 8 filters and a kernel size of 7 

    >>> model1.add(Conv2d(8,7))

    NOTE: Convolutional layer added.

    # Add a pooling layer of size 2 to # complete the second pair of layers. 
    
    >>> model1.add(Pooling(2))

    NOTE: Pooling layer added.
    
Add a fully connected layer:

    # Add Fully-Connected Layer with 16 units
    
    >>> model1.add(Dense(16))

    NOTE: Fully-connected layer added.
    
Finally, add the output layer:

    # Add an output layer that has 2 nodes and uses
    # the Softmax activation function 
    
    >>> model1.add(OutputLayer(act='softmax',n=2))

    NOTE: Output layer added.
    NOTE: Model compiled successfully 


### Additional Resources
- DLPy examples: https://github.com/sassoftware/python-dlpy/tree/master/examples
- DLPy API documentation [sassoftware.github.io/python-dlpy](https://sassoftware.github.io/python-dlpy/).
- [SAS SWAT for Python](http://github.com/sassoftware/python-swat/)
- [SAS ESPPy](https://github.com/sassoftware/python-esppy)
- Watch: Introduction to DLPy and examples (YouTube):
  * [Introduction Deep Learning with Python (DLPy) and SAS Viya](https://www.youtube.com/watch?v=RJ0gbsB7d_8)
  * [Image classification using CNNs](https://www.youtube.com/watch?v=RJ0gbsB7d_8&start=125)
  * [Object detection using TinyYOLOv2](https://www.youtube.com/watch?v=RJ0gbsB7d_8&start=390)
  * [Import and export deep learning models with ONNX](https://www.youtube.com/watch?v=RJ0gbsB7d_8&start=627)
  * [Text classification and text generation using RNNs](https://www.youtube.com/watch?v=RJ0gbsB7d_8&start=943)
  * [Time series forecasting using RNNs](https://www.youtube.com/watch?v=RJ0gbsB7d_8&start=1185)
- [SAS Deep Learning with Python made easy using DLPy](https://blogs.sas.com/content/subconsciousmusings/2019/03/13/sas-deep-learning-with-python-made-easy-using-dlpy/)

### Contributing
Have something cool to share? SAS gladly accepts pull requests on GitHub! See the [Contributor Agreement](https://github.com/sassoftware/python-dlpy/blob/master/ContributorAgreement.txt) for details.

### Licensing 
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at [LICENSE.txt](https://github.com/sassoftware/python-dlpy/blob/master/LICENSE.txt)

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License. 
