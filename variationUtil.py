from shutil import copyfile
import os

def setVariation(var):
    if os.path.exists('variations/cosfire'+var+'.py'):
        if os.path.exists('cosfire/cosfire.py'):
            os.remove('cosfire/cosfire.py')
        copyfile('variations/cosfire'+var+'.py', 'cosfire/cosfire.py')
