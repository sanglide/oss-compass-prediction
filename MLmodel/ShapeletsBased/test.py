import numpy as np

if __name__=="__main__":
    lst1=[1,4,1,np.NAN,1]
    lst1 = [np.nanmean(lst1) if np.isnan(x) else x for x in lst1]
    print(lst1)