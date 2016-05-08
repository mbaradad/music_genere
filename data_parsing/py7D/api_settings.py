''' your local settings.
    Note that if you install using setuptools, change this
    module first before running setup.py install.
'''
from os.path import expanduser
home = expanduser("~")

lines = tuple(open(home + '.py7Dconfig', 'r'))

oauthkey = lines[0]
secret = lines[1]
country = lines[2]
