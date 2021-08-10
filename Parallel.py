from joblib import Parallel, delayed

from Properties import Properties


def ParallelForEach(func, listOfListOfArgs):
    '''
    A simple example:

    >>> def add(x,y):
    ...     return (x+y)
    >>> ParallelForEach(add, [[1,2],[3,4],...])
    result=[3,7,...]
    :param func: function that will be executed in parallel
    :param listOfListOfArgs: list of arguments for every call
    :return: list of return from every call to func
    '''
    results = Parallel(n_jobs=Properties.numOfThreads)(delayed(func)(*arg) for arg in listOfListOfArgs)
    return results
