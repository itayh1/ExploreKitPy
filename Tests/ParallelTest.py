import time
import Parallel

def fibo(n):
    if n==1:
        return 1
    elif n==2:
        return 2
    return fibo(n-1) + fibo(n-2)

def awaiter():
    time.sleep(5)

def exp(a,b):
    time.sleep(5)
    return a**b

def main():
    # ParallelForEach(fibo, [[1],[2],[3],[4]])
    # ParallelForEach(awaiter, [[],[],[],[],[],[],[],[]])
    print(Parallel.ParallelForEach(exp, [[i, i+1] for i in range(1,9)]))
if __name__=='__main__':
    main()