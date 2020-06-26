## [Use SAS DLPy to Create a SAS Viya Image Denoising Model](https://github.com/sassoftware/python-dlpy/tree/master/examples/image_denoising/Image_Denoising_Model.ipynb) 

### Overview
This example notebook uses SAS DLPy to create, train, and deploy a SAS Viya deep learning image denoising model. The image denoising model uses a pixel-wise CNN segmentation regression architecture to replace pixels in noisy images with predicted pixel values. The trained image denoise model can be used to cleanse and restore noisy images to prepare them for further analytic consumption.

CNN segmentation models are used for tasks such as object detection, image classification, and extracting information from low-level images. The model in this example structurally resembles a pixel-wise classification model, but the output task is different. This model uses regression to predict <i>values</i> for image pixels, rather than using classification to predict the <i>category</i> for image pixels.

The example data uses a set of images with generated salt-and-pepper noise. Salt-and-pepper noise is commonly seen when an image signal is disturbed, creating black-and-white pixel corruption in the output image.  Given an input data set of noisy images, the trained model up-samples input image pixels with predicted values, and outputs cleaned, denoised images. 

The pixel-wise segmentation regression model in this SAS DLPy notebook consists of an encoder network, a corresponding decoder network, and the final pixel-wise regression layer. The encoder network consists of convolutional and pooling layers. The decoder network consists of convolution and transpose convolution layers for up-sampling. The appropriate decoders use transposed convolution to perform a non-linear pixel-level up-sampling of the input feature maps.

### Contents

- [Important Note: Client and Server Definitions](https://github.com/sassoftware/python-dlpy/tree/master/examples/image_denoising/Image_Denoising_Model.ipynb#ClientServer)
- [Prepare Resources and Configure Computing Environment for Modeling](https://github.com/sassoftware/python-dlpy/tree/master/examples/image_denoising/Image_Denoising_Model.ipynb#getReady)
    - [Download the Image Data](https://github.com/sassoftware/python-dlpy/tree/master/examples/image_denoising/Image_Denoising_Model.ipynb#downloadData)
    - [Create Input Images with Salt-and-Pepper Noise](https://github.com/sassoftware/python-dlpy/tree/master/examples/image_denoising/Image_Denoising_Model.ipynb#AddSaltPepper)
    - [Import Required Python and SAS Modules](https://github.com/sassoftware/python-dlpy/tree/master/examples/image_denoising/Image_Denoising_Model.ipynb#importUtils)
- [Launch SAS CAS Session](https://github.com/sassoftware/python-dlpy/tree/master/examples/image_denoising/Image_Denoising_Model.ipynb#LaunchCAS)
- [Load Clean and Noisy Images into SAS CAS](https://github.com/sassoftware/python-dlpy/tree/master/examples/image_denoising/Image_Denoising_Model.ipynb#LoadTablesCAS)
- [Check New CAS Tables and Resize the Images](https://github.com/sassoftware/python-dlpy/tree/master/examples/image_denoising/Image_Denoising_Model.ipynb#ValidateResize)
- [Rename the Image Column in the Clean Image Table](https://github.com/sassoftware/python-dlpy/tree/master/examples/image_denoising/Image_Denoising_Model.ipynb#RenameColumns)
- [Merge Clean and Noisy Image Tables](https://github.com/sassoftware/python-dlpy/tree/master/examples/image_denoising/Image_Denoising_Model.ipynb#MergeTables)
- [Compare Noisy and Clean Images](https://github.com/sassoftware/python-dlpy/tree/master/examples/image_denoising/Image_Denoising_Model.ipynb#CompareImages)
- [Use SAS DLPy to Define the Image Denoising Model Architecture](https://github.com/sassoftware/python-dlpy/tree/master/examples/image_denoising/Image_Denoising_Model.ipynb#DefineDenoising)
- [Shuffle the Merged Data and Partition into Train and Validation Tables](https://github.com/sassoftware/python-dlpy/tree/master/examples/image_denoising/Image_Denoising_Model.ipynb#ShufflePartition)
- [Train the Image Denoising Model](https://github.com/sassoftware/python-dlpy/tree/master/examples/image_denoising/Image_Denoising_Model.ipynb#TrainModel)
- [Use the Image Denoising Model to Score Test Images](https://github.com/sassoftware/python-dlpy/tree/master/examples/image_denoising/Image_Denoising_Model.ipynb#ModelPredict)
- [Rename the Image Column in the Validation Table](https://github.com/sassoftware/python-dlpy/tree/master/examples/image_denoising/Image_Denoising_Model.ipynb#RenameValid)
- [Display Denoising Results Generated from Pixel Intensity Values](https://github.com/sassoftware/python-dlpy/tree/master/examples/image_denoising/Image_Denoising_Model.ipynb#PixelIntensity)
- [Display Output Table Image Columns](https://github.com/sassoftware/python-dlpy/tree/master/examples/image_denoising/Image_Denoising_Model.ipynb#DisplayOutputCols)
- [Display Output Table Prediction Images](https://github.com/sassoftware/python-dlpy/tree/master/examples/image_denoising/Image_Denoising_Model.ipynb#DisplayOutputs)
- [Display Scored Image Results](https://github.com/sassoftware/python-dlpy/tree/master/examples/image_denoising/Image_Denoising_Model.ipynb#DisplayFinal) 