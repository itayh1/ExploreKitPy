
from os import listdir, remove
from os.path import isdir, isfile, join, abspath

from Logger import Logger
'''
return list of the files in the directory
if there's no files, return empty list
if the dir doesn't exit, return None
'''
def listFilesInDir(directory: str) -> list:
    if not isdir(directory):
        return None
    files = [f for f in listdir(directory) if isfile(join(directory, f))]
    return files

def deleteFile(filePath: str):
    try:
        remove(filePath)
    except:
        Logger.Error('Failed delete file: ' + filePath)

def getAbsPath(filePath: str):
    try:
        return abspath(filePath)
    except:
        Logger.Error('Failed get file abs path for ' + filePath)
