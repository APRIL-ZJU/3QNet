from qnet import encoder,progress_decode,chamfer_big,get_prob,sampling,word2code,code2word,get_topk
import os
import getdata#2 as getdata
import tensorflow as tf
from numpy import *
import numpy as np
import time
from tensorflow.python import pywrap_tensorflow
from tensorflow.contrib.coder.python.ops import coder_ops
import sys
import open3d as o3d
from bitarray import bitarray
from dealply import load_ply_data, write_ply_data, trans_ply, load_ply_alpha, write_ply_alpha, trans_alpha 
import random
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#data:n*3
def sort_coors(data):
    ind=np.lexsort(data.T)
    result=data[ind]
    return result,ind
#probposi:512,probs:512
def prob_binary(probposi,probs):
    poolen=len(probposi)
    posilen=np.log(poolen)/np.log(2)
    nonzeros=np.sum(probposi)
    spaposi=np.nonzero(probposi)[0]
    result=bitarray()
    result.frombytes(poolen.to_bytes(2,'big'))
    result.extend(probposi)
    for i in range(nonzeros):
        result.frombytes(int(32767*probs[spaposi[i]]).to_bytes(2,'big'))
    return result
def read_binary(f):
    #ftype=bool.from_bytes(f.read(1),'big')
    poolen=int.from_bytes(f.read(2),'big')
    result=np.zeros(poolen)
    probposi=bitarray()
    #print(poolen)
    probposi.fromfile(f,poolen//8)
    spaposi=np.nonzero(probposi.tolist())[0]
    nonzeros=len(spaposi)
    for i in range(nonzeros):
        result[spaposi[i]]=float(int.from_bytes(f.read(2),'big')/32767)
    return result
    
#prob:b*1*n*256
def predkprob(prob,knum=2):
    topkprob,topkidx=tf.nn.top_k(prob,knum)#batch*n*k,batch*n*k
    klimit=topkprob[:,:,:,-1:]#batch*n*1
    result=tf.where(tf.greater(prob,tf.tile(klimit,[1,1,1,tf.shape(prob)[-1]])),tf.ones_like(prob),tf.ones_like(prob))
    result=tf.reduce_sum(result,axis=2)
    result=result/tf.reduce_sum(result,axis=-1,keepdims=True)
    return result
def qcom(indir,filedir,outdir,level=6,dracodir='./',modelpath='./'):
    scalelist=[3.5,2.5,3.5,2.5,4.0,3.5,2.5,1.5]
    typelist=[1,1,2,2,3,3,3,3]
    klist=[256,256,256,256,256,256,256,256]
    #spalist=[2,1,0,2,1,0,2,1,0]
    level=len(klist)-level
    typenum=typelist[level]#level//2+1
    knum=klist[level]

    names=os.listdir(indir)

    start=0
    num=2048
    
    pointcloud_pl=tf.placeholder(tf.float32,[None,None,3],name='pointcloud_pl')
    pdata=pointcloud_pl
    cens,feats=encoder(pdata)
    cen1,cen2,cen3,cen4=cens
    rfeat1,rfeat2,rfeat3,rfeat4=feats

    
    idx1,featscode1,featpool1=word2code(rfeat1,1,32,None,None,True)
    idx2,featscode2,featpool2=word2code(rfeat2,2,128,None,None,True)
    idx3,featscode3,featpool3=word2code(rfeat3,3,256,None,None,True)
    idx4,featscode4,featpool4=word2code(rfeat4,4,256,None,None,True)

    ids=[idx1,idx2,idx3,idx4]
    pools=[featpool1,featpool2,featpool3,featpool4]
    codes=[featscode1,featscode2,featscode3,featscode4]

    #typenum=3
    cens,featscode,featpool=cens[typenum],codes[typenum],pools[typenum]
    cennum=int(num/np.power(4,typenum+1))#cens.get_shape()[1].value
    featidx0=ids[typenum]
    featidx0=tf.squeeze(featidx0,[-1])
    
    prob=tf.expand_dims(get_prob(tf.expand_dims(featidx0,axis=-1),featpool.get_shape()[0].value),axis=1)
    prob1=tf.placeholder(tf.float32,[1,1,featpool.get_shape()[0].value],name='prob1')

    rcens=tf.placeholder(tf.float32,[None,cennum,3],name='recover_cens')
    rcens1=rcens
    rpool=tf.placeholder(tf.float32,[None,featpool.get_shape()[1].value],name='featpool')
   
    pres=16
    cdf = coder_ops.pmf_to_quantized_cdf(
      prob, precision = pres)
    strings = coder_ops.range_encode(
      tf.cast(featidx0,tf.int16), cdf, precision=pres)
 
    rstring=tf.placeholder(tf.string,name='recover_str')
    pbsize=tf.placeholder(tf.int32,name='bsize')
    fcdf = coder_ops.pmf_to_quantized_cdf(prob1, precision = pres)
    ffeatidx = coder_ops.range_decode(rstring, [pbsize,cennum], fcdf, precision=pres)#b*32
    ridx=tf.cast(tf.expand_dims(ffeatidx,axis=-1),tf.int32)#b*32*1 
    
    ridx=tf.concat([tf.zeros_like(ridx),ridx],axis=-1)
    refeat=tf.gather_nd(tf.expand_dims(rpool,axis=0),ridx)#b*cennum*f

    rfeat=tf.placeholder(tf.float32,[None,cennum,featpool.get_shape()[1].value],name='rfeat')

    rrcens,rrfeats=code2word('codeword'+str(typenum+1),rcens1,rfeat,rpool)
    ptres=progress_decode(rrcens,rrfeats,typenum+1,0)
    ptres=ptres[-1]
    
    config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    config.gpu_options.allow_growth=True 
    with tf.Session(config=config) as sess:
        var=[v for v in tf.global_variables()]
        advar=[v for v in var if v.name.split('/')[0]!='Dinc_3' and v.name.split(':')[0]!='is_training' and 'decay' not in v.name]
        devar=[v for v in advar if v.name.split('/')[0]!='E' and 'decay' not in v.name.split('/')[0] and 'pool' not in v.name.split('/')[0]]
        print('***********************here load')

        tf.train.Saver(var_list=advar).restore(sess, tf.train.latest_checkpoint(modelpath))
        reptime=[]
        enctime=[]
        dectime=[]
        for name in names:
        #for name in [names[0],names[0]]:
            #print(name)
            path=os.path.join(indir,name)
            if not os.path.exists(filedir):
                os.makedirs(filedir)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            outpath=os.path.join(outdir,name)
            drcpath=os.path.join(filedir,name.split('.')[0]+'.drc')

            stime=time.time()
            pcd=o3d.io.read_point_cloud(path)
            data=np.array(pcd.points)
            indata=data

            result=data
            print('points number: ',np.shape(data)[0])
            allptnum=np.shape(data)[0]
            data,length,center=getdata.get_part_data(data,num,scale=scalelist[level])
            etime=time.time()
            reptime.append(etime-stime)
            bsize=shape(data)[0]

            workdir='./workbin'
            all_binname = drcpath

            plyname=os.path.join(workdir,'inter.ply')
            drcname=os.path.join(workdir,'inter.drc')
            os.system('rm '+plyname)
            os.system('rm '+drcname)
            #if True:
            bbsize=8
            itertime=int((bsize-1)/bbsize)+1

            censlist=[]
            problist=[]
            idxlist=[]
            itime=time.time()
            fpool=sess.run(featpool,feed_dict={pointcloud_pl:data[:1]})
            itime0=time.time()

            print(time.time()-itime0)
            for i in range(itertime):
                vcens,vprob,vidx=sess.run([cens,prob,featidx0],feed_dict={pointcloud_pl:data[i*bbsize:(i+1)*bbsize]})#b*N*3
                               
                vcnum=np.shape(vcens)[1]
                bbnum=np.shape(vcens)[0]
                label=np.ones([bbnum,vcnum,1])*i*bbsize+np.tile(np.reshape(np.array(range(bbnum)),[bbnum,1,1]),[1,vcnum,1])
                vcens=np.concatenate([vcens,label],axis=-1)
                censlist.append(vcens)
                problist.append(vprob)
                idxlist.append(vidx)

            vprob=np.concatenate(problist,axis=0)
            vprob=vprob.astype(bool).astype(int)
            vprob0=np.mean(vprob,axis=0,keepdims=True)
            vprob=((vprob0*32767).astype(int)/32767).astype(np.float16)
            probposi=vprob0.astype(bool).astype(int)

            bprob=prob_binary(np.squeeze(probposi).tolist(),np.squeeze(vprob0).tolist())

            vcens=np.concatenate(censlist,axis=0)
            vcens[:,:,:3]=vcens[:,:,:3]*length+center
            vcens=np.reshape(vcens,[-1,4])
            vidx=np.concatenate(idxlist,axis=0)
            
            vcens,ind=sort_coors(vcens)
            vidx=np.expand_dims(np.reshape(np.reshape(vidx,[-1,1])[ind],[bsize,cennum]),axis=-1)
            sstr=sess.run(strings,feed_dict={prob:vprob,featidx0:vidx.squeeze(-1)})
            
            write_ply_alpha(plyname,vcens)
            os.system(dracodir+'draco_encoder -point_cloud -i '+plyname+' -o '+drcname+' -cl 10 -qp 10 -qn 10')
            with open(drcname, 'rb') as f:
                censbit=f.read()
            
            drclen=os.path.getsize(drcname)
            with open(all_binname, 'wb') as f:
                bprob.tofile(f)
                f.write(drclen.to_bytes(4,'big'))
                f.write(censbit)
                f.write(sstr)
            encend=time.time()
            enctime.append(encend-itime0)

            
            dtime=time.time()
            with open(all_binname, 'rb') as f:
                fprob=np.reshape(read_binary(f),[1,1,-1]).astype(np.float16)
                #cennum=int.from_bytes(f.read(1),'big')
                censlen=int.from_bytes(f.read(4),'big')
                drcstr=f.read(censlen)
                fstr=f.read()
            fdrcname='drcply.drc'
            with open(fdrcname,'wb') as f:
                f.write(drcstr)
            
            os.system(dracodir+'draco_decoder -point_cloud -i '+fdrcname+' -o '+'draco_rec.ply')
            trans_alpha('draco_rec.ply')
            fcens=load_ply_alpha('draco_rec.ply')
            fcens,_=sort_coors(fcens)################
            fcens=fcens[:,:3]
             
            fcens=np.reshape(fcens[:,:3],[-1,cennum,3])
            result=[]
            fmins=np.min(fcens,axis=1)
            fmaxs=np.max(fcens,axis=1)
 
            flength=np.expand_dims((fmaxs-fmins)/2,axis=1)#+1e-8
            fcenter=np.expand_dims((fmaxs+fmins)/2,axis=1)
            cens1=(fcens-fcenter)/(flength+1e-8)
            outfeats=sess.run(refeat,feed_dict={prob1:fprob,rstring:fstr,rpool:fpool,pbsize:bsize})
            for i in range(itertime):
                rdata=sess.run(ptres,feed_dict={rcens:cens1[i*bbsize:(i+1)*bbsize],\
                        rfeat:outfeats[i*bbsize:(i+1)*bbsize],rpool:fpool,pbsize:bsize})#b*N*3
                result.append(rdata)
            result=np.concatenate(result,axis=0)
            result=result*flength+fcenter
            size0=np.shape(result)[0]
            result=np.reshape(result,[-1,3])
            size=np.reshape(np.array(range(size0)),[size0,1])
            size=np.reshape(np.tile(size,[1,2048]),[-1])

            write_ply_data(outpath,result)
            dtime=time.time()
            dectime.append(dtime-encend)
            print(np.mean(reptime[-1]),np.mean(enctime[-1]),np.mean(dectime))
    tf.reset_default_graph()
    return np.mean(reptime),np.mean(enctime),np.mean(dectime)
if __name__=='__main__':
    indir='./data'
    bindir='./bins'
    outdir='./outs'
    level=5#1 to 8
    dracodir='./draco/draco_build/'
    modelpath='./pretrained'
    qcom(indir,bindir,outdir,level,dracodir,modelpath)#compress models in indir to binary codes in bindir, and decompression them back to outdir
