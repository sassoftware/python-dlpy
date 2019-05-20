Please print out the results of the code snippets below and add along with your issue to helps us better understand your environment.

```python
import os
import platform
import pandas as pd
import struct
import sys

def which(cmd):
    import os
    for path in os.environ['PATH'].split(os.pathsep):
        path = os.path.join(path, cmd)
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path
        path += '.exe'
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return path

print('Environment')
print('-----------')
print('HOME: %s' % os.environ.get('HOME', None))
print('HOMEDRIVE: %s' % os.environ.get('HOMEDRIVE', None))
print('HOMEPATH: %s' % os.environ.get('HOMEPATH', None))
print('PATH: %s' % os.environ.get('PATH', None))
print('PYTHONPATH: %s' % os.environ.get('PYTHONPATH', None))
print('TKPATH: %s' % os.environ.get('TKPATH', None))

print(' ')
print('Executables')
print('-----------')
print('PYTHON: %s' % which('python'))
print('PIP: %s' % which('pip'))
print('CONDA: %s' % which('conda'))
print('IPYTHON: %s' % which('ipython'))
print('JUPYTER: %s' % which('jupyter'))

print(' ')
print('SWAT')
print('----')
try:
    import swat
    print('Path: %s' % swat.__file__)
    print('Version: %s' % swat.__version__)
except ImportError:
    print('No SWAT installed')

print(' ')
print('ESPPy')
print('----')
try:
    import esppy
    print('Path: %s' % esppy.__file__)
    print('Version: %s' % esppy.__version__)
except ImportError:
    print('No ESPPy installed')

print(' ')
print('DLPy')
print('----')
try:
    import dlpy
    print('Path: %s' % dlpy.__file__)
    print('Version: %s' % dlpy.__version__)
except ImportError:
    print('No DLPy installed')

print(' ')
print('Pandas')
print('------')
print('Path: %s' % pd.__file__)
print('Version: %s' % pd.__version__)

print(' ')
print('System')
print('------')
print('Python: %s' % '.'.join(platform.python_version_tuple()))
print('Bits: %s' % (struct.calcsize('P') * 8))
print('Platform: %s' % platform.platform())
print('UnicodeWidth: %s' % ({1114111:'UCS4', 65535:'UCS2'}[sys.maxunicode]))
print('UserDir: %s' % os.path.expanduser('~'))
print('PythonPath:')
for item in sys.path:
    print('    %s' % item)
```