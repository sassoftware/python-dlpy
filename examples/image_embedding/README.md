### Overview
These examples use DLPy's ImageEmbeddingTable API to show how to train image embedding models. When supplied with an input image, these models return the image's corresponding low-dimensional vector representation (it's embedding). An alternative perspective is that the model maps an image to a point in embedding space. The models are trained so that similar images are mapped close to one another in the embedding space and dissimilar images are mapped farther apart. The embedding space can then be used for further analysis such as clustering or classification.

During training, the embedding models utilize multiple input streams. A Siamese network uses two input streams, a Triplet network uses three input streams and a Quartet network uses four input streams. The model maps each image to the embedding space and updates to simultaneously shrink the distance between similar images and increase the distance between dissimilar images.

### Data
The models require data from multiple image tables. There are three main ways to prepare images for Siamese, Triplet, and Quartet networks:

* Use ImageEmbeddingTable to create a table of samples from a directory of images
* Generate samples on-the-fly from a directory of images
* Manually join image tables

The data we use for these examples comes from Google's [OpenImagesV4](https://arxiv.org/abs/1811.00982) using cats and birds for one example and dog breeds for the others. More instructrions for recreating the data set can be found within each notebook.
