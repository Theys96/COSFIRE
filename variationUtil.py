from shutil import copyfile

def setVariation(var):
    copyfile('variations/cosfire'+var+'.py', 'cosfire/cosfire.py')
