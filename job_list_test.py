from joblib import Parallel, delayed
import time

# def funcc(i):
#     print(1-i/100)
#     time.sleep(1-i/100)
#     return(2*i)

data = Parallel(n_jobs=50)(delayed(funcc)(i) for i in range(50))

print(data)