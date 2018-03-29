.. Copyright SAS Institute

Installation
============

DLPy can be installed using ``pip`` as follows::

    pip install dlpy

Releases can also be found on GitHub at  https://github.com/sassoftware/dlpy/releases.


Prerequisites
-------------

SWAT
****

DLPy does not do any analytic operations itself.  It is a wrapper around the SWAT
package which communicates with a SAS CAS server to do the analytics.  The SWAT
package can be installed from GitHub at the following URL.

https://github.com/sassoftware/python-swat/releases


Graphviz
********

When installing DLPy, the graphviz Python package will automatically be installed.
However, the graphviz Python package will not work without the graphviz command-line
tools from https://www.graphviz.org/download/.  You must install those for your
system, and add the location of the ``bin/`` subdirectory to your systems' path.

To verify that the utilities will work, run the following command::

    dot -V
