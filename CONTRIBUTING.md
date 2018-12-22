# SAS DLPy Developer How-To

Developing DLPy is just like developing any 
other project on GitHub.  You clone the project, do your work, 
and submit a pull request.  


## Submitting a Pull Request

Submitting a pull request uses the standard process at GitHub.
Note that in the submitted changes, there must always be a unit test
for the code being contributed.  Pull requests that do not have a
unit test will not be accepted.

You also must include the text from the ContributerAgreement.txt file
along with your sign-off verifying that the change originated from you.

## Testing

For the most part, testing the SAS DLPy package is just like testing
any other Python package.  Tests are written using the standard unittest
package.  All test cases are subclasses of TestCase.  

SAS DLPy is centered around SAS' deep learning functionalities. Therefore
most of the tests include creating models and training some data. 
A number of resources are located under tests/datasources. Locate
these files and folders under your DLPY_DATA_DIR. If you are using
a Windows client then also locate the object detection folder under 
DLPY_DATA_DIR_LOCAL. These two (DLPY_DATA_DIR and DLPY_DATA_DIR_LOCAL)
are environment variables and should be set before running the unittests.
Further, there are a number of tests that require model files (ending with .h5).
They are not located as they do take some space. That said, those files can 
easily be downloaded online.

The below part is the configurations for the CAS Server.

Since CAS is a network resource and requires authentication, there is
some extra setup involved in getting your tests configured to run 
against your CAS server.  Normally this involves setting the following
environment variables.

* CASHOST - the hostname or IP address of your CAS server (Default: None)
* CASPORT - the port of your CAS server (Default: None)
* CASPROTOCOL - the protocol being using ('cas', 'http', 'https' or 'auto'; Default: 'cas')

* CASUSER - the CAS account username (Default: None)
* CASPASSWORD - the CAS account password (Default: None)

* CASDATALIB    - the CASLib where data sources are found (Default: CASTestTmp)
* CASMPPDATALIB - the CASLib on MPP servers where the data sources are found (Default: HPS)
* CASOUTLIB     - the CASLib to use for output CAS tables (Default: CASUSER)
* CASMPPOUTLIB  - the CASLib to use for output CAS tables on MPP servers (Default: CASUSER)

Some of these can alternatively be specified using configuration files.
The CASHOST, CASPORT, and CASPROTOCOL variables can be specified in a .casrc
in your home directory (or in any directory from the directory you are 
running from all the way up to your home directory).  It is actually written
in Lua, but the most basic form is as follows:

    cashost = 'myhost.com'
    casport = 5570
    casprotocol = 'cas'

The CASUSER and CASPASSWORD variables are usually extracted from your
`~/.authinfo` file automatically.  The only reason you should use environment
variables is if you have a generalized test running account that is
shared across various tools.

Finally, the CAS*DATALIB and CAS*OUTLIB variables configured where your
data sources and output tables reside.  Using the CASDATALIB location 
will make your tests run more efficiently since the test cases can load
the data from a server location.  If you don't specify a CASDATALIB (or
the specified one doesn't exist), the data files will be uploaded to the
server for each test (which will result in hundreds of uploads).  Most
people will likely set them all to CASUSER and create a directory called
`datasources` in their home directory with the contents of the 
`dlpy/tests/datasources/` directory.

Once you have these setup, you can use tools like nosetest to run the suite:

    nosetests -v dlpy.tests
