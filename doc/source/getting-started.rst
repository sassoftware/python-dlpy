
.. Copyright SAS Institute

.. currentmodule:: dlpy

***************
Getting Started
***************

.. ipython:: python
   :suppress:

   import os
   host = os.environ['CASHOST']
   port = os.environ['CASPORT']
   userid = None
   password = None

To connect to an ESP server, you just need a hostname and port number.
These are passed to the :class:`ESP` constructor.

.. ipython:: python

   import dlpy
