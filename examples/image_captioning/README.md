### Overview
The following example demonstrates how to create a SAS Viya RNN image captioning model using the SAS DLPy API. The example begins by loading image and image caption data into SAS CAS. Two successive CNN models are created to perform feature extraction and object detection tasks. Using SAS DLPy, the imported pretrained VGG-16 CNN model outputs a vectorized extracted features table. Next, the imported pre-trained YoloV2 Multisize CNN model outputs a vectorized detected objects table for the same image data. 

The VGG-16 and YoloV2 Multisize model output tables (when combined with the original input image and captions tables) comprise the input data for the RNN image captioning model. (The example shows two ways to organize the data for RNN model consumption.) The SAS Viya RNN image captioning model architecture is defined in SAS DLPy, and then is trained using the merged image captions, features, and object tables as inputs. The trained RNN model is then used to display caption predictions for random images in the data. The scored output also includes the top detected objects in the image, as well as the associated ground truth captions.

### Data

The notebook begins with detailed instructions to download, access, or create all of the component resources required to complete the example. The toy input image data set and image captions are downloaded. Open-source components for pre-trained VGG-16 and pre-trained Yolo V2 Multisize CNN models are downloaded. Instructions are provided to download and format the user's choice of an open-source word embeddings (word vector) file for modeling use. The SAS DLPy API provides the architecture for the RNN image captioning model that is manually trained using the generated input data. 

The computing environment is configured by importing required Python analytic, mathematical, and plotting utilities before launching the SAS CAS cloud computing session. Performing all of the data preparation up front is intended to enable users to focus on the analytic modeling principles in this example, without data-wrangling interruptions. 

### Table of Contents 
- [Important Note: Client and Server Definitions](https://github.com/sassoftware/python-dlpy/blob/master/examples/image_captioning/ImageCaptioningExample.ipynb#ClientServer)
- [Prepare Resources and Configure Environment for Modeling](https://github.com/sassoftware/python-dlpy/blob/master/examples/image_captioning/ImageCaptioningExample.ipynb#getReady)
    - [Download the Image Data](https://github.com/sassoftware/python-dlpy/blob/master/examples/image_captioning/ImageCaptioningExample.ipynb#downloadData)
    - [Download the Image Captions File](https://github.com/sassoftware/python-dlpy/blob/master/examples/image_captioning/ImageCaptioningExample.ipynb#downloadCaptions)
    - [Download a Pre-Trained Word Vector File](https://github.com/sassoftware/python-dlpy/blob/master/examples/image_captioning/ImageCaptioningExample.ipynb#downloadEmbeddings)
    - [Format the Pre-Trained Word Vector File for Modeling](https://github.com/sassoftware/python-dlpy/blob/master/examples/image_captioning/ImageCaptioningExample.ipynb#addColHeadings)
    - [Download the VGG-16 Model and Weights](https://github.com/sassoftware/python-dlpy/blob/master/examples/image_captioning/ImageCaptioningExample.ipynb#VGG16)
    - [Download the YoloV2 Multisize Model and Weights](https://github.com/sassoftware/python-dlpy/blob/master/examples/image_captioning/ImageCaptioningExample.ipynb#YoloV2Multisize)
    - [Import Required Python and SAS DLPy Modules](https://github.com/sassoftware/python-dlpy/blob/master/examples/image_captioning/ImageCaptioningExample.ipynb#importPythonDLPy)
- [Connect to a SAS CAS Session](https://github.com/sassoftware/python-dlpy/blob/master/examples/image_captioning/ImageCaptioningExample.ipynb#connnectCAS)
- [Load and Process the Input Images in SAS CAS](https://github.com/sassoftware/python-dlpy/blob/master/examples/image_captioning/ImageCaptioningExample.ipynb#loadInputs)
- [Load Image Captions into SAS CAS](https://github.com/sassoftware/python-dlpy/blob/master/examples/image_captioning/ImageCaptioningExample.ipynb#captions)
- [Load VGG-16 Model and Extract Image Features](https://github.com/sassoftware/python-dlpy/blob/master/examples/image_captioning/ImageCaptioningExample.ipynb#features)
- [Load YoloV2 Model and Extract Detected Objects](https://github.com/sassoftware/python-dlpy/blob/master/examples/image_captioning/ImageCaptioningExample.ipynb#detection)
- [Merge Captions, Features, and Detected Objects Tables](https://github.com/sassoftware/python-dlpy/blob/master/examples/image_captioning/ImageCaptioningExample.ipynb#merge)
- [Create Image Captioning Model Training Data in a Single Step](https://github.com/sassoftware/python-dlpy/blob/master/examples/image_captioning/ImageCaptioningExample.ipynb#oneStep)
- [Use SAS DLPy to Create a SAS Viya Image Captioning Model](https://github.com/sassoftware/python-dlpy/blob/master/examples/image_captioning/ImageCaptioningExample.ipynb#RNNmodel)
- [Train the SAS Viya Image Captioning Model](https://github.com/sassoftware/python-dlpy/blob/master/examples/image_captioning/ImageCaptioningExample.ipynb#trainRNN)
- [Score the Image Captioning Table](https://github.com/sassoftware/python-dlpy/blob/master/examples/image_captioning/ImageCaptioningExample.ipynb#scoreCaptioningTable)
- [Display Image Captioning Predictions](https://github.com/sassoftware/python-dlpy/blob/master/examples/image_captioning/ImageCaptioningExample.ipynb#displayPredictions)

