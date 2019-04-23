import numpy as np

def truncate(name):
    data = np.genfromtxt(name, delimiter=',')
    
    #print(np.shape(data))
    labels = data[10,:]
    
    #print(labels)
    indices = np.where(labels == 1)
    count= 0
    for kk in range(len(labels)):
        if labels[kk] == 1:
            count +=1
    print(count)
    
    #print(len(indices[0]))
    
    diff = np.zeros((1,len(indices[0])))
    diff = diff[0]
    #print("diff = ",diff)
    
    #print(indices[0][1])
    for i in range(1,len(indices[0])):
        diff[i] = indices[0][i] - indices[0][i-1]
       
    #print(diff)
    
    start_inds = indices[0][np.where(diff > 1)]
    #print(start_inds[0])
    
    start_inds = np.insert(start_inds,0,indices[0][0])
    #print(start_inds)
    mids = np.zeros(len(start_inds))
    #print(mids)
    for i in range(1,len(start_inds)):
        mids[i] = round((start_inds[i] + start_inds[i-1])/2)
    mids = np.delete(mids,0)
    
    mids = mids.astype(np.int64)
    
    
    window_inds = np.zeros(16*len(start_inds) - 1 - 3)
    count = 0
    i = 0
    j = 0
    #print(np.asarray(range(start_inds[i]-4,start_inds[i])))
    while count < (len(window_inds)):
        if i < len(start_inds):
            
            window_inds[count:count+4] = np.asarray(range(start_inds[i]-4,start_inds[i]))
            window_inds[count+4:count+8] = np.asarray(range(start_inds[i],start_inds[i]+4))
            window_inds[count+8:count+12] = np.asarray(range(start_inds[i]+4,start_inds[i]+8))
            i += 1
        if j < len(mids):
            window_inds[count+12:count+16] = np.asarray(range(mids[j],mids[j]+4))
            j+=1
        count += 16
    
    window_inds = window_inds.astype(np.int64)
    np.set_printoptions(threshold=np.nan)
    print(window_inds)
    print(start_inds)
    print(mids)
    print(indices)
    offset = 90
    EEG_desired = window_inds + offset
    print(EEG_desired)
    print(len(EEG_desired))
    
    truncated_data = data[:,EEG_desired]
    print(np.shape(truncated_data))
    return truncated_data, EEG_desired
