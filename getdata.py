import h5py
import numpy as np
import os
from sklearn.cluster import KMeans
import multiprocessing as mp
import scipy.io
def getdir():
    #BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BASE_DIR='./'
    DATA_DIR=os.path.join(BASE_DIR,'data')
    return DATA_DIR
def getspdir():
    BASE_DIR='./'
    DATA_DIR=os.path.join(BASE_DIR,'data')
    DATA_DIR=os.path.join(DATA_DIR,'hdf5_data')
    #print(DATA_DIR)
    return DATA_DIR
def load_h5(h5_filename):
    f=h5py.File(h5_filename)
    data=f['data'][:]
    return data
def load_h5label(h5_filename):
    f=h5py.File(h5_filename,'r')
    data=f['data'][:]
    label=f['label'][:]
    return data,label
def load_mat(matpath):
    f=scipy.io.loadmat(matpath)
    data=f['data'][:]
    label=f['label'][:]
    return data,label
def getfile(path):
    return [line.strip('\n') for line in open(path)]
def ocdev2(data,tnum=2):
    result=[]
    maxd=np.max(data,axis=0)
    mind=np.min(data,axis=0)
    interval=(maxd-mind)/tnum
    for i in range(tnum):
        for j in range(tnum):
            for k in range(tnum):
                #print((data[:,0]>mind[0]+i*interval[0]))
                xid=np.array(data[:,0]>=mind[0]+i*interval[0]) & np.array(data[:,0]<=mind[0]+(i+1)*interval[0])
                yid=np.array(data[:,1]>=mind[1]+j*interval[1]) & np.array(data[:,1]<=mind[1]+(j+1)*interval[1])
                zid=np.array(data[:,2]>=mind[2]+k*interval[2]) & np.array(data[:,2]<=mind[2]+(k+1)*interval[2])
                dataijk=data[xid & yid & zid]
                result.append(dataijk)
    return result
def ocdev(data,tnum=2):
    result=[]
    maxd=np.max(data,axis=0)
    mind=np.min(data,axis=0)
    interval=(maxd-mind)/tnum
    center=(maxd+mind)/2
    for i in range(tnum):
        for j in range(tnum):
            xid=np.array(data[:,0]>=mind[0]+i*interval[0]) & np.array(data[:,0]<=mind[0]+(i+1)*interval[0])
            yid=np.array(data[:,1]>=mind[1]+j*interval[1]) & np.array(data[:,1]<=mind[1]+(j+1)*interval[1])
            ks=1-np.int(bool(i)^bool(j))*2

            zid1=np.array(data[:,1]-center[1]<=ks*(data[:,0]-center[0]))
            zid2=np.array(data[:,1]-center[1]>=ks*(data[:,0]-center[0]))

            dataijk=data[xid & yid & zid1]
            result.append(dataijk)

            dataijk=data[xid & yid & zid2]
            result.append(dataijk)
    return result
def get_part_cens(data,tnum,ptnum,scale,cennum):
    datanum=np.shape(data)[0]
    #print(datanum)
    length=np.max(np.max(data,axis=0)-np.min(data,axis=0))
    cennum=int(cennum*scale)
    if cennum<int(datanum):
        onum=int(datanum/(16*cennum)) #if cennum<32 else datanum//128
        ndata=downsam(data,'u',num=onum,gridsize=length/32)
        cens=KMeans(n_clusters=cennum+1, random_state=1, n_init=10,max_iter=200, init='k-means++',tol=1e-4).fit(ndata)
        #cens,u,u0,d,jm,p,fpc=cmeans(np.array(ndata).T,m=1.1,c=cennum+1,error=1e-5,maxiter=500)
        cens=cens.cluster_centers_
    else:
        #assert False
        cens=np.array(downsam(data,'u',num=datanum//cennum,gridsize=length/512))
    return cens
def getcennum(datas,cennum):
    cennums=[]
    lengths=np.zeros(len(datas))
    for i in range(len(datas)):
        data=datas[i]
        if len(data)<1:
            lengths[i]=0.0
        else:
            center=np.mean(data,axis=1,keepdims=True)
            rdismat=np.sqrt(np.sum(np.square(data-center),axis=-1))#b*n
            lengths[i]=np.sum(rdismat,axis=-1,keepdims=True)

    cennums=cennum*lengths/sum(lengths)
    cennums=cennums.astype(np.int32)
    return cennums
def get_part_data(data,ptnum,split='train',centype='r',scale=2.5):
    stime=time.time()
    datanum=np.shape(data)[0]
    cennum=int(datanum/ptnum)+1
    tnum=8
    resdata=[]
    p0=mp.Pool(tnum)
    datas=ocdev(data,2)
    threads=[]
    cenlist=[]
    cennums=getcennum(datas,cennum)
    for i in range(len(datas)):
        if cennums[i]>0:
            thread=p0.apply_async(get_part_cens,args=(datas[i],tnum,ptnum,scale,cennums[i]))
            threads.append(thread)
    p0.close()
    p0.join()
    for t in threads:
        res=t.get()
        cenlist.append(res)
    etime1=time.time()
    cenlist=np.concatenate(cenlist,axis=0)
    #cennum=np.shape(cens)[0]
    result={}
    threads=[]
    tnum=6
    interval=datanum//(tnum-1)
    ttime=time.time()
    p=mp.Pool(tnum)
    for i in range(tnum):
        #if i<tnum-1:
        thread=p.apply_async(add2dict,args=(data[i*interval:(1+i)*interval],i,cenlist,1000*i))
        threads.append(thread)
    p.close()
    p.join()
    #ttime2=time.time()
    for t in threads:
        res=t.get()
        for k in list(res.keys()):
            if k not in result.keys():
                result[k]=res[k]
            else:
                result[k]=result[k]+res[k]
    ttime2=time.time()
    print(ttime2-ttime)

    for i in result.keys():
        datai=result[i]
        #print(np.shape(datai))
        if len(datai)>ptnum:
            #print(np.shape(datai))
            idx=np.random.choice(len(datai),ptnum,replace=False)
            datai=np.expand_dims(np.array(datai)[idx,:],axis=0)
            #datai=downsam(datai,num=ptnum,dtype='f')
        else:
            #print(np.shape(datai))
            idx=np.random.choice(len(datai),ptnum-len(datai),replace=True)
            idx=np.concatenate([np.array(range(len(datai))),idx],axis=0)
            datai=np.expand_dims(np.array(datai)[idx,:],axis=0)
        resdata.append(datai)
    data=np.concatenate(resdata,axis=0)[:,:,:3]
    dmax=np.max(data,axis=1,keepdims=True)
    dmin=np.min(data,axis=1,keepdims=True)
    length=(dmax-dmin)/2#+1e-8
    #length=np.max(length,axis=-1,keepdims=True)
    #length=np.ceil(length*0.5)/0.5
    #length=np.tile(np.max(length,axis=0,keepdims=True),[np.shape(data)[0],1,1])
    center=(dmax+dmin)/2
    data=(data-center)/(1e-8+length)#np.max(length,axis=-1,keepdims=True)
    etime2=time.time()
    print(etime2-stime,etime2-ttime2)
    return data,length,center
def add2list(result,ptnum):
    #print('>>>>>>>>>')
    resdata=[]
    #print(len(ikey))
    for datai in result:
        #datai=result[i]
        #print(np.shape(datai))
        if len(datai)>ptnum:
            #print(np.shape(datai))
            idx=np.random.choice(len(datai),ptnum,replace=False)
            datai=np.expand_dims(np.array(datai)[idx,:],axis=0)
        else:
            #print(np.shape(datai))
            idx=np.random.choice(len(datai),ptnum-len(datai),replace=True)
            idx=np.concatenate([np.array(range(len(datai))),idx],axis=0)
            datai=np.expand_dims(np.array(datai)[idx,:],axis=0)
        resdata.append(datai)
    redata=np.concatenate(resdata,axis=0)[:,:,:3]
    #print('___________')
    return redata
#data:b*n*3
def get_normal0(data,cir=True):
    if not cir:
        result=data
        dmax=np.max(result,axis=0,keepdims=True)
        dmin=np.min(result,axis=0,keepdims=True)
        #length=np.max((dmax-dmin)/2,axis=-1,keepdims=True)
        length=(dmax-dmin)/2
        center=(dmax+dmin)/2
        result=(result-center)/length
    else:
        center=np.mean(data,axis=0,keepdims=True)
        rdismat=np.sqrt(np.sum(np.square(data-center),axis=-1))#b*n
        r=np.max(rdismat,axis=-1,keepdims=True)
        length=np.expand_dims(r,axis=-1)
        #print(np.shape(para))
        result=(data-center)/length#+cen
    return result,length,center
#data:b*n*3
def get_normal(data,cir=True):
    if not cir:
        result=data
        dmax=np.max(result,axis=1,keepdims=True)
        dmin=np.min(result,axis=1,keepdims=True)
        #length=np.max((dmax-dmin)/2,axis=-1,keepdims=True)
        length=(dmax-dmin)/2
        center=(dmax+dmin)/2
        result=(result-center)/length
    else:
        center=np.mean(data,axis=1,keepdims=True)
        rdismat=np.sqrt(np.sum(np.square(data-center),axis=-1))#b*n
        r=np.max(rdismat,axis=-1,keepdims=True)
        length=np.expand_dims(r,axis=-1)
        #print(np.shape(para))
        result=(data-center)/length#+cen
    return result,length,center
