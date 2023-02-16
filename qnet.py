import tensorflow as tf
from numpy import *
import numpy as np
import os
import getdata
import copy
import random

from tf_ops.CD import tf_nndistance
from tf_ops.emd import tf_auctionmatch
from pointnet_util import tf_sampling,tf_grouping, pointnet_sa_module_msg
from provider import shuffle_points,jitter_point_cloud,rotate_perturbation_point_cloud,random_scale_point_cloud
from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
from tf_ops.grouping.tf_grouping import query_ball_point, group_point

DATA_DIR=getdata.getspdir()
filelist=os.listdir(DATA_DIR)

trainfiles=getdata.getfile(os.path.join(DATA_DIR,'train_files.txt'))

EPOCH_ITER_TIME=200
BATCH_ITER_TIME=5000
BASE_LEARNING_RATE=0.001
REGULARIZATION_RATE=0.00001
BATCH_SIZE=16
DECAY_STEP=1000*BATCH_SIZE
DECAY_RATE=0.7
PT_NUM=2048
FILE_NUM=6
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def get_weight_variable(shape,stddev,name,regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)):
    weight = tf.get_variable(name=name,shape=shape,initializer=tf.contrib.layers.xavier_initializer())
    tf.summary.histogram(name+'/weights',weight)
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weight))
    return weight
def get_bias_variable(shape,value,name):
    bias = tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    tf.summary.histogram('/'+name,bias)
    return bias
def get_learning_rate(step):
    learning_rate = tf.train.exponential_decay(BASE_LEARNING_RATE, step,DECAY_STEP / BATCH_SIZE, DECAY_RATE, staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001)
    return learning_rate
def conv2d(scope,inputs,num_outchannels,kernel_size,stride=[1,1],padding='SAME',stddev=1e-3,use_bnorm=False,activation_func=tf.nn.relu):
    with tf.variable_scope(scope):
        kernel_h,kernel_w=kernel_size
        num_inchannels=inputs.get_shape()[-1].value
        kernel_shape=[kernel_h,kernel_w,num_inchannels,num_outchannels]
        kernel=get_weight_variable(kernel_shape,stddev,'weights')
        stride_h,stride_w=stride
        outputs=tf.nn.conv2d(inputs,kernel,[1,stride_h,stride_w,1],padding=padding)
        bias = get_bias_variable([num_outchannels],0,'biases')
        outputs=tf.nn.bias_add(outputs,bias)
        if use_bnorm:
            outputs=tf.contrib.layers.batch_norm(outputs,
                                      center=True, scale=True,
                                      updates_collections=None,
                                      scope='bn')
        if activation_func!=None:
            outputs=activation_func(outputs)
    return outputs
def fully_connect(scope,inputs,num_outputs,stddev=1e-3,activation_func=tf.nn.relu):
    num_inputs = inputs.get_shape()[-1].value
    # print(inputs,num_inputs)
    with tf.variable_scope(scope):
        weights=get_weight_variable([num_inputs,num_outputs],stddev=stddev,name='weights')
        bias=get_bias_variable([num_outputs],0,'bias')
        result=tf.nn.bias_add(tf.matmul(inputs,weights),bias)
    if(activation_func is not None):
        result=activation_func(result)
    return result
def deconv(scope,inputs,output_shape,kernel_size,stride=[1,1],padding='SAME',stddev=1e-3,activation_func=tf.nn.relu):
    with tf.variable_scope(scope) as sc:
        kernel_h,kernel_w=kernel_size
        num_outchannels=output_shape[-1]
        num_inchannels=inputs.get_shape()[-1].value
        kernel_shape=[kernel_h,kernel_w,num_outchannels,num_inchannels]
        kernel=get_weight_variable(kernel_shape,stddev,'weights')
        stride_h,stride_w=stride
        outputs=tf.nn.conv2d_transpose(inputs,filter=kernel,output_shape=output_shape,strides=[1,stride_h,stride_w,1],padding=padding)
        bias = get_bias_variable([num_outchannels],0,'bias')
        outputs = tf.nn.bias_add(outputs, bias)
        if activation_func != None:
            outputs = activation_func(outputs)
    return outputs
def sampling(npoint,xyz,use_type='f'):
    if use_type=='f':
        idx=tf_sampling.farthest_point_sample(npoint, xyz)
        new_xyz=tf_sampling.gather_point(xyz,idx)
    elif use_type=='r':
        bnum=tf.shape(xyz)[0]
        ptnum=xyz.get_shape()[1].value
        ptids=arange(ptnum)
        random.shuffle(ptids)
        ptid=tf.tile(tf.constant(ptids[:npoint],shape=[1,npoint,1],dtype=tf.int32),[bnum,1,1])
        bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1]),[1,npoint,1])
        idx=tf.concat([bid,ptid],axis=-1)
        new_xyz=tf.gather_nd(xyz,idx)
    return idx,new_xyz
def grouping(xyz,new_xyz, radius, nsample, points, knn=False, use_xyz=True):
    if knn:
        _,idx = tf_grouping.knn_point(nsample, xyz, new_xyz)
    else:
        _,id0 = tf_grouping.knn_point(1, xyz, new_xyz)
        valdist,idx = tf_grouping.knn_point(nsample, xyz, new_xyz)
        idx=tf.where(tf.greater(valdist,radius),tf.tile(id0,[1,1,nsample]),idx)
    grouped_xyz = tf_grouping.group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if points is not None:
        grouped_points = tf_grouping.group_point(points, idx) # (batch_size, npoint, nsample, channel)
        #print(grouped_points)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
        grouped_points=grouped_xyz

    return grouped_xyz,grouped_points
def global_layer(scope,pts,mlp=[64,64,128],use_bnorm=False):
    with tf.variable_scope(scope):
        tensor=tf.expand_dims(pts,axis=2)
        for i,outchannel in enumerate(mlp):
            tensor=conv2d('global_ptstate%d'%i,tensor,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
    return tf.squeeze(tensor,2)
def encoder(tensor,it=True):
    with tf.variable_scope('E'):
        l0_xyz=tensor
        l0_points=None
        ptnum=tensor.get_shape()[1].value
        globalfeat=global_layer('init_layer',l0_xyz,mlp=[64,128,256],use_bnorm=False)
        globalfeat=tf.reduce_max(globalfeat,axis=1,keepdims=True)
        gfeat=globalfeat

        cen1,feat1=pointnet_sa_module_msg(l0_xyz, l0_points, 512, [0.2], [8], [[32,256]], is_training=it, bn_decay=None, scope='layer1', use_nchw=False,use_knn=True)
        rcen1,rfeat1=global_fix('global1',cen1,feat1,gfeat,mlp=[256,256],mlp1=[256,256])

        cen2,feat2=pointnet_sa_module_msg(cen1, feat1, 128, [0.4], [8], [[64,256]], is_training=it, bn_decay=None, scope='layer2',use_nchw=False,use_knn=True)
        rcen2,rfeat2=global_fix('global2',cen2,feat2,gfeat,mlp=[256,256],mlp1=[256,256])

        cen3,feat3=pointnet_sa_module_msg(cen2, feat2, 32, [0.6], [8], [[128,256]], is_training=it, bn_decay=None, scope='layer3',use_nchw=False,use_knn=True)
        rcen3,rfeat3=global_fix('global3',cen3,feat3,gfeat,mlp=[256,256],mlp1=[256,256])

        cen4,feat4=pointnet_sa_module_msg(cen3, feat3, 8, [0.6], [8], [[128,256]], is_training=it, bn_decay=None, scope='layer4',use_nchw=False,use_knn=True)
        rcen4,rfeat4=global_fix('global4',cen4,feat4,gfeat,mlp=[256,256],mlp1=[256,256])

        tf.add_to_collection('cen3',cen3)
    return [rcen1,rcen2,rcen3,rcen4],[rfeat1,rfeat2,rfeat3,rfeat4]
def global_fix(scope,cens,feats,gfeat,mlp=[128,128],mlp1=[128,128]):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        #tensor0=tf.expand_dims(tf.concat([cens,feats],axis=-1),axis=2)
        tensor0=tf.expand_dims(tf.concat([cens,feats,tf.tile(gfeat,[1,tf.shape(feats)[1],1])],axis=-1),axis=2)
        tensor=tensor0
        for i,outchannel in enumerate(mlp):
            tensor=conv2d('global_ptstate%d'%i,tensor,outchannel,[1,1],padding='VALID',activation_func=tf.nn.relu)
        tensorword=tensor
        tensor=tf.reduce_max(tensor,axis=1,keepdims=True)#/PT_NUM
        tensor=tf.concat([tf.expand_dims(cens,axis=2),tf.expand_dims(feats,axis=2),tf.tile(tensor,[1,tf.shape(feats)[1],1,1])],axis=-1)
        for i,outchannel in enumerate(mlp1):
            tensor=conv2d('global_ptstate2%d'%i,tensor,outchannel,[1,1],padding='VALID',activation_func=tf.nn.relu)
        tensor=conv2d('global_ptout',tensor,3+feats.get_shape()[-1],[1,1],padding='VALID',activation_func=None)
        newcens=cens#+tf.squeeze(tensor[:,:,:,:3],[2])
        #newcens=tf.nn.tanh(newcens)
        newfeats=feats+tf.squeeze(tensor[:,:,:,3:],[2])
        tf.add_to_collection('cenex',tf.reduce_mean(tf.abs(tensor[:,:,:,:3])))
    return newcens,newfeats
def data_pool(name,ptnum,dimnum):
    data=tf.get_variable(name=name,shape=[ptnum,dimnum],initializer=tf.contrib.layers.xavier_initializer())
    return data
def get_topk(rawcode,codepool,knum):
    valdist,ptid = tf_grouping.knn_point(knum, codepool, rawcode)#batch*n*k
    bid=tf.tile(tf.reshape(tf.range(start=0,limit=tf.shape(rawcode)[0],dtype=tf.int32),[-1,1,1,1]),[1,tf.shape(rawcode)[1],knum,1])
    idx=tf.concat([bid,tf.expand_dims(ptid,axis=-1)],axis=-1)
    kcode=tf.gather_nd(codepool,idx)#batch*n*k*c
    kdist=tf.reduce_mean(tf.square(tf.expand_dims(rawcode,axis=2)-kcode),axis=-1)#/////////////////////
    return kdist,ptid,kcode
#rawcode:batch*n*c
#codepool:N*c
#decayratio:1
def code_transform(rawcode,codepool,decayratio,knum=16,distratio=0.001):
    codepool=tf.tile(tf.expand_dims(codepool,axis=0),[tf.shape(rawcode)[0],1,1])
    kdist,_,kcode=get_topk(rawcode,codepool,knum)#batch*n*k*c

    _,idx,code1=get_topk(rawcode,codepool,1)
    tf.add_to_collection('codeidx',idx)
    kmask=tf.exp(-kdist/(1e-8+decayratio))/(1e-8+tf.reduce_sum(tf.exp(-kdist/(1e-8+decayratio)),axis=-1,keepdims=True))

    tf.add_to_collection('kdist',kdist)
    tf.add_to_collection('kmask',kmask)
    kmindist=tf.reduce_mean(tf.reduce_min(kdist,axis=-1))
    tf.add_to_collection('kmindist',kmindist)
    kmask=tf.expand_dims(kmask,axis=-1)
    kcode=tf.reduce_sum(kmask*kcode,axis=-2,keepdims=True)
    code=tf.stop_gradient(code1-kcode)+kcode
    code=tf.squeeze(code,[2])
    return idx,code
#pts:batch*n*3
#codepool:N*c
def prob_predict(pts,codepool,mlp=[128,128],mlp1=[128,128],mlp2=[64,64]):
    with tf.variable_scope('probpre',reuse=tf.AUTO_REUSE):
        code=tf.expand_dims(pts,axis=2)
        for i, num_out_channel in enumerate(mlp):
            code = conv2d('pre_feat%d'%i, code, num_out_channel, [1,1])
        ptsfeat=tf.reduce_max(code,axis=1,keepdims=True)#b*1*1*c
        code=tf.concat([tf.tile(ptsfeat,[1,tf.shape(codepool)[0],1,1]),tf.tile(tf.expand_dims(tf.expand_dims(codepool,axis=1),axis=0),[tf.shape(pts)[0],1,1,1])],axis=-1)
        for i, num_out_channel in enumerate(mlp1):
            code = conv2d('all_feat%d'%i, code, num_out_channel, [1,1])
        allfeat=tf.reduce_max(code,axis=1,keepdims=True)#b*1*1*c
        code=tf.concat([tf.tile(allfeat,[1,tf.shape(codepool)[0],1,1]),tf.tile(tf.expand_dims(tf.expand_dims(codepool,axis=1),axis=0),[tf.shape(pts)[0],1,1,1])],axis=-1)
        for i, num_out_channel in enumerate(mlp2):
            code = conv2d('out_feat%d'%i, code, num_out_channel, [1,1])
        code = tf.squeeze(conv2d('out_predict', code, 1, [1,1],activation_func=None),[2])
        code=tf.square(code)
        code=code/(1e-5+tf.reduce_sum(code,axis=1,keepdims=True))#b*N*1
        #code=tf.exp(code)/tf.reduce_sum(tf.exp(code),axis=1,keepdims=True)#b*N*1
        code=tf.squeeze(code,[-1])
    return code
#idx:b*n*1
def get_prob(idx,pnum):
    pmat=tf.reshape(tf.range(pnum),[1,1,pnum])
    pmat=tf.tile(pmat,[tf.shape(idx)[0],tf.shape(idx)[1],1])
    dismat=tf.cast(tf.abs(idx-pmat),tf.float32)
    dismat=tf.where(tf.less_equal(dismat,1e-5),tf.ones_like(dismat),tf.zeros_like(dismat))
    dismat=tf.reduce_sum(dismat,axis=1)
    prob=dismat/tf.reduce_sum(dismat,axis=-1,keepdims=True)#b*N
    return prob

#ptscode:batch*n*3
#featcode:batch*n*c
def shape_recover(ptscode,featcode,mlp=[512,512],mlp2=[512,256]):
    ptslen=ptscode.get_shape()[-1].value
    featlen=featcode.get_shape()[-1].value
    code0=tf.concat([ptscode,featcode],axis=-1)
    code=tf.expand_dims(code0,axis=2)
    startcode=code
    for i, num_out_channel in enumerate(mlp):
        code = conv2d('recover_feat%d'%i, code, num_out_channel, [1,1],activation_func=tf.nn.relu)
    codemax=tf.reduce_max(code,axis=1,keepdims=True)#/PT_NUM
    code=tf.concat([startcode,tf.tile(codemax,[1,tf.shape(code)[1],1,1])],axis=-1)

    for i, num_out_channel in enumerate(mlp2):
        code = conv2d('recover_%d'%i,code, num_out_channel, [1,1],activation_func=tf.nn.relu)
    code=conv2d('recover_out',code,ptslen+featlen, [1,1], activation_func=None)
    code=tf.squeeze(code,[2])
    ptsmove=code[:,:,:ptslen]
    featmove=code[:,:,ptslen:]
    
    newpts=ptscode+ptsmove
    newfeat=featcode+featmove
    return newpts,newfeat
def get_pools(featnum,featlen,num):
    featpool=data_pool('featspool'+str(num),featnum,featlen)
    tf.add_to_collection('featspool',featpool)
    return featpool
def word2code(featsin,num=1,poolen=256,ptsdecay=None,featsdecay=None,infer=False):
    featlen=featsin.get_shape()[-1].value
    featpool=get_pools(poolen,featlen,num)
    #ubloss1=ubloss(featpool[:128],featpool)
    #ubloss2=ubloss(featpool[:64],featpool)
    #tf.add_to_collection('ubloss',ubloss1+ubloss2)
    if featsdecay is None:
        featsdecay=tf.square(tf.get_variable(name='featsdecay'+str(num),shape=[1],initializer=tf.constant_initializer(5.0)))
    
    tf.add_to_collection('decays',featsdecay)
    if infer:
        idx,featscode=code_transform(featsin,featpool,featsdecay,knum=1,distratio=1)
        return idx,featscode,featpool
    else:
        idx,featscode=code_transform(featsin,featpool,featsdecay,knum=poolen//2,distratio=1)
        return featscode,featpool
def code2word(scope,ptscode,featscode,featpool):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        newpts,newfeat=shape_recover(ptscode,featscode)
        #newpts,newfeat=ptscode,featscode
    return newpts,newfeat
def dcom_block(ptsin,featsin,featlen=128,up_ratio=4):
    mlp=[256,256]
    mlp2=[64,64,64]

    shortlen=int(featlen/up_ratio)
    ptnum=featsin.get_shape()[1].value
    newnum=up_ratio*ptnum
    infeats=tf.reshape(featsin,[-1,ptnum,up_ratio,shortlen])

    infeats=tf.concat([infeats,tf.tile(ptsin,[1,1,up_ratio,1])],axis=-1)
    for i,outchannel in enumerate(mlp):
        infeats=conv2d('dcomin%d'%i,infeats,outchannel,[1,1],padding='VALID')
    allfeat=tf.reduce_sum(infeats,axis=[1,2],keepdims=True)/PT_NUM
    infeats=tf.concat([infeats,tf.tile(allfeat,[1,ptnum,up_ratio,1])],axis=-1)


    netfeat=infeats
    for i,outchannel in enumerate(mlp):
        netfeat=conv2d('dcomstate%d'%i,netfeat,outchannel,[1,1],padding='VALID')
    netfeat=conv2d('dcomfeatout',netfeat,featlen,[1,1],padding='VALID',activation_func=tf.nn.leaky_relu)

    netpts=infeats
    for i,outchannel in enumerate(mlp2):
        netpts=conv2d('dcomptsfeat%d'%i,netpts,outchannel,[1,1],padding='VALID')
    move=conv2d('dcomptsout',netpts,3,[1,1],padding='VALID',activation_func=None)

    netfeat=tf.reshape(netfeat,[-1,newnum,1,featlen])+tf.reshape(tf.tile(featsin,[1,1,up_ratio,1]),[-1,up_ratio*ptnum,1,featlen])
    move_regular=tf.reduce_mean(tf.reduce_max(tf.reduce_sum(tf.square(move),axis=-1),axis=1))
    tf.add_to_collection('expand_dis',move_regular)
    newpts=move+ptsin
    newpts=tf.reshape(newpts,[-1,up_ratio*ptsin.get_shape()[1].value,1,3])
    #print(newpts,netfeat)
    return newpts, netfeat
def progress_block(pts0,feats0,cirnum,start=0,featlen=256):
    pts=tf.expand_dims(pts0,axis=2)
    feats=tf.expand_dims(feats0,axis=2)
    ptlist=[]
    #ptlist.append(pts0)
    featlist=[]
    for i in range(start,start+cirnum):
        with tf.variable_scope('Dinc',reuse=tf.AUTO_REUSE):
            infeats=feats
            pts,feats=dcom_block(pts,feats,featlen=featlen,up_ratio=4)
            ptlist.append(tf.squeeze(pts,axis=2))
            featlist.append(tf.reshape(feats,[-1,feats.get_shape()[1].value,featlen]))
    return ptlist,featlist
def progress_decode(cen3,feat3,cirnum=3,start=0):
    pt3list,feat3list=progress_block(cen3,feat3,cirnum,start)
    return pt3list
def chamfer_big(pcd1, pcd2):
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1),axis=1)#+0.1*tf.reduce_max(dist1,axis=1)
    dist2 = tf.reduce_mean(tf.sqrt(dist2),axis=1)#+0.1*tf.reduce_max(dist2,axis=1)
    dist=tf.reduce_mean((dist1+dist2)/2)
    return dist,idx1
#pcd1,pcd2:batch*n*k*3
def chamfer_local(pcda,pcdb):
    ptnum=pcda.get_shape()[1].value
    knum=pcda.get_shape()[2].value
    pcd1=tf.reshape(pcda,[-1,knum,3])
    pcd2=tf.reshape(pcdb,[-1,knum,3])

    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1=tf.sqrt(dist1)
    dist2=tf.sqrt(dist2)
    dist1=tf.reduce_mean(tf.reshape(dist1,[-1,ptnum,knum]),axis=2)#+0.1*tf.reduce_max(tf.reshape(dist1,[-1,ptnum,knum]),axis=1)#batch*k
    dist2=tf.reduce_mean(tf.reshape(dist2,[-1,ptnum,knum]),axis=2)#+0.1*tf.reduce_max(tf.reshape(dist2,[-1,ptnum,knum]),axis=1)#batch*k
    dist=tf.reduce_mean(tf.maximum(dist1,dist2))
    return dist,idx1
def multi_chamfer_func(cen,inputpts,output,n,k,r=0.2,theta1=0.5,theta2=0.5,use_frame=True,use_r=False,use_all=False):
    if use_all:
        in_cen=inputpts
    else:
        in_cen=cen
    if use_r:
        out_kneighbor,_=grouping(xyz=output,new_xyz=in_cen, radius=r, nsample=k, points=None, knn=False, use_xyz=False)
        in_kneighbor,_=grouping(xyz=inputpts,new_xyz=in_cen, radius=r, nsample=k, points=None, knn=False, use_xyz=False)
    else:
        out_kneighbor,_=grouping(output,new_xyz=in_cen, radius=r, nsample=k, points=None, knn=True, use_xyz=True)
        in_kneighbor,_=grouping(inputpts,new_xyz=in_cen, radius=r, nsample=k, points=None, knn=True, use_xyz=True)
    local_loss=chamfer_local(out_kneighbor,in_kneighbor)[0]
    if use_frame:
        frame_loss=chamfer_big(inputpts,output)[0]
        return theta1*frame_loss+theta2*local_loss
    else:
        return theta2*local_loss
def detail_multi_k_func(cen,input,output,n,klist,theta,useall=False,use_type='e'):
    detail_func=multi_chamfer_func
    loss=detail_func(cen,input,output,n,klist[0],theta[0],theta[1],use_r=False,use_all=useall)
    for i in range(1,len(klist)):
        loss=loss+detail_func(cen,input,output,n,klist[i],theta[i+1],use_frame=False,use_r=False,use_all=useall)
    return loss
#batch*n*3
def normal_trans(data):
    uplimit=tf.reduce_max(data,axis=1,keepdims=True)
    downlimit=tf.reduce_min(data,axis=1,keepdims=True)
    cen=(uplimit+downlimit)/2
    length=(uplimit-downlimit)/2
    result=(data-cen)/length#tf.reduce_max(length,axis=-1,keepdims=True)
    return result,cen,length
    #return data,0,1
def mydecode(scope,cen,featscode,featpool,cirnum,std=0.1):
    noisecen=cen+tf.random_normal(tf.shape(cen),mean=0.0,stddev=std,dtype=tf.float32)
    noiseprob=prob_predict(noisecen,featpool,mlp=[256,512],mlp1=[256,512],mlp2=[64,64])
    realprob=get_prob(tf.get_collection('codeidx')[0],featpool.get_shape()[0].value)
    probloss=tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(realprob-noiseprob),axis=-1)))
    cen1,word1=code2word(scope,cen,featscode,featpool)
    cenloss=tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(cen1-cen),axis=-1)))
    tf.add_to_collection('cenloss',cenloss)
    out=progress_decode(cen1,word1,cirnum)
    return out,probloss
def train():
    pointcloud_pl=tf.placeholder(tf.float32,[None,2048,3],name='pointcloud_pl')

    datain,_,_=normal_trans(pointcloud_pl)
    pt8=sampling(8,datain,use_type='f')[-1]
    pt32=sampling(32,datain,use_type='f')[-1]
    pt128=sampling(128,datain,use_type='f')[-1]
    pt512=sampling(512,datain,use_type='f')[-1]
    
    cens,feats=encoder(datain)
    global_step=tf.Variable(0,trainable=False)
    epoch_step=tf.cast(BATCH_SIZE*global_step/(2048*6),tf.int32)
    alpha0=tf.train.piecewise_constant(epoch_step,[50,100],[0.01,0.01,0.01],'alpha_op')
    
    cen1,cen2,cen3,cen4=cens
    rfeat1,rfeat2,rfeat3,rfeat4=feats

    featscode1,featpool1=word2code(rfeat1,1,32,None,None)
    featscode2,featpool2=word2code(rfeat2,2,128,None,None)
    featscode3,featpool3=word2code(rfeat3,3,256,None,None)
    featscode4,featpool4=word2code(rfeat4,4,256,None,None)
    
    out1,probloss1=mydecode('codeword1',cen1,featscode1,featpool1,1,0.01)
    out11=out1[0]
    out2,probloss2=mydecode('codeword2',cen2,featscode2,featpool2,2,0.01)
    out21,out22=out2
    out3,probloss3=mydecode('codeword3',cen3,featscode3,featpool3,3,0.01)
    out31,out32,out33=out3
    out4,probloss4=mydecode('codeword4',cen4,featscode4,featpool4,4,0.01)
    out41,out42,out43,out44=out4

    loss=[]
    loss1,loss2,loss3,loss4=[],[],[],[]
    trainstep=[]
    print('>>>>>>>>>>>>>>>>>>')
    loss2.append(0.5*tf.expand_dims(detail_multi_k_func(cen2,cen1,out21,128,[16,32],[1,0.5,0.5,0.3],useall=False,use_type='c'),axis=-1))
    loss2.append(0.5*tf.expand_dims(detail_multi_k_func(cen1,datain,out22,512,[32,64],[1,0.5,0.5,0.3,0.2,0.2],useall=False,use_type='c'),axis=-1))

    print('>>>>>>>>>>>>>>>>>>')
    loss3.append(0.33*tf.expand_dims(detail_multi_k_func(cen3,cen2,out31,32,[16,32],[1.0,0.5,0.5,0.3],useall=False,use_type='c'),axis=-1))#//////////////////
    loss3.append(0.33*tf.expand_dims(detail_multi_k_func(cen2,cen1,out32,128,[32,64],[1,0.5,0.5,0.3],useall=False,use_type='c'),axis=-1))
    loss3.append(0.33*tf.expand_dims(detail_multi_k_func(cen1,datain,out33,512,[64,128],[1,0.5,0.5,0.3,0.2,0.2],useall=False,use_type='c'),axis=-1))

    loss4.append(0.25*tf.expand_dims(detail_multi_k_func(cen4,cen3,out41,8,[8,16],[1.0,0.5,0.5,0.3],useall=False,use_type='c'),axis=-1))
    loss4.append(0.25*tf.expand_dims(detail_multi_k_func(cen3,cen2,out42,32,[16,32],[1.0,0.5,0.5,0.3],useall=False,use_type='c'),axis=-1))#//////////////////
    loss4.append(0.25*tf.expand_dims(detail_multi_k_func(cen2,cen1,out43,128,[32,64],[1,0.5,0.5,0.3],useall=False,use_type='c'),axis=-1))
    loss4.append(0.25*tf.expand_dims(detail_multi_k_func(cen1,datain,out44,512,[64,128],[1,0.5,0.5,0.3,0.2,0.2],useall=False,use_type='c'),axis=-1))
    loss=loss2+loss3+loss4
    loss=tf.concat(loss,axis=-1)
    print('>>>>>>>>>>>>>>>>>>')

    exploss=tf.add_n(tf.get_collection('expand_dis'))
    kmindist=tf.add_n(tf.get_collection('kmindist'))

    cenex=tf.add_n(tf.get_collection('cenex'))
    decayloss=alpha0*tf.reduce_sum(tf.get_collection('decays'))
    loss=tf.reduce_sum(loss)+0.01*kmindist+0.01*exploss+decayloss

    trainvars=tf.GraphKeys.TRAINABLE_VARIABLES
    var2=[v for v in tf.get_collection(trainvars) if v.name.split('/')[0]!='flow']
    advar=[v for v in tf.get_collection(trainvars) if v.name.split('/')[0]!='Dinc_3' and v.name.split(':')[0]!='is_training' and v.name.split('/')[0]!='flow']
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    zhengze=tf.add_n([regularizer(v) for v in advar])
    loss=loss+0.001*zhengze
    trainstep=tf.train.AdamOptimizer(learning_rate=0.0002).minimize(loss, global_step=global_step,var_list=var2)
    config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    config.gpu_options.allow_growth=True
    savepath='./'
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(max_to_keep=10)
        sess.run(tf.global_variables_initializer())
        if os.path.exists(savepath+'checkpoint'):
            print('here load')
            saver.restore(sess, tf.train.latest_checkpoint(savepath))
        datalist=[]

        for j in range(FILE_NUM):
            traindata = getdata.load_h5(os.path.join(DATA_DIR, trainfiles[j]))
            datalist.append(traindata)
        for i in range(EPOCH_ITER_TIME):
            for j in range(FILE_NUM):
                traindata=copy.deepcopy(datalist[j])
                ids=list(range(len(traindata)))
                random.shuffle(ids)
                traindata=traindata[ids,:,:]

                allnum=int(len(traindata)/BATCH_SIZE)*BATCH_SIZE
                batch_num=int(allnum/BATCH_SIZE)
                for batch in range(batch_num):
                    start_idx = (batch * BATCH_SIZE) % allnum
                    end_idx=(batch*BATCH_SIZE)%allnum+BATCH_SIZE
                    batch_point = traindata[start_idx:end_idx]

                    batch_point=shuffle_points(batch_point)
                    batch_point=rotate_perturbation_point_cloud(batch_point, angle_sigma=2*pi, angle_clip=2*pi)

                    #import numpy as np
                    kklist=[np.power(2,v) for v in list(range(7,12))]
                    knum=kklist[np.random.choice(len(kklist),1,p=[0.15,0.15,0.15,0.15,0.4])[0]]
                    if knum<2048:
                        didx=list(range(knum))+list(np.random.choice(knum,2048-knum,replace=True))
                        batch_point=batch_point[:,didx]

                    feed_dict = {pointcloud_pl: batch_point}
                    resi = sess.run([trainstep,loss,cenex,kmindist,tf.get_collection('decays'),zhengze], feed_dict=feed_dict)
                    if batch % 16 == 0:
                        print('epoch: %d '%i,'file: %d '%j,'batch: %d' %batch)
                        print('loss: ',resi[1])
                        print('knum: ',knum)
                        print('cenloss: ',resi[2])

                        print('kmindist:',resi[3])
                        print('decays:',resi[4])
                        print('regularization: ', resi[-2])
            if (i+1)%10==0:
                save_path = saver.save(sess, savepath+'model',global_step=i)
if __name__=='__main__':
    train()
