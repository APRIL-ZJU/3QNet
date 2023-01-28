import tensorflow as tf
from numpy import *
import numpy as np
import os
import getdata
import tf_util
import copy
import random
import point_choose

from tf_ops.CD import tf_nndistance
from tf_ops.emd import tf_auctionmatch
#from tf_ops.CD import tf_nndistance
#from tf_ops.sampling import tf_sampling
#from tf_ops.grouping import tf_grouping
from pointnet_util import tf_sampling,tf_grouping, pointnet_sa_module_msg
from provider import shuffle_points,jitter_point_cloud,rotate_perturbation_point_cloud,random_scale_point_cloud
from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
from tf_ops.grouping.tf_grouping import query_ball_point, group_point

DATA_DIR=getdata.getspdir()
filelist=os.listdir(DATA_DIR)

trainfiles=getdata.getfile(os.path.join(DATA_DIR,'train_files.txt'))
#testfiles=getdata.getfile(os.path.join(DATA_DIR,'test_files.txt'))

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
    #print(shape)
    weight = tf.get_variable(name=name,shape=shape,initializer=tf.contrib.layers.xavier_initializer())
    tf.summary.histogram(name+'/weights',weight)
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weight))
    return weight
def get_bias_variable(shape,value,name):
    bias = tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    #biases = _variable_on_cpu('biases', [num_output_channels],
    #                            tf.constant_initializer(0.0))
    #bias=tf.Variable(tf.constant(value, shape=shape, name=name,dtype=tf.float32))
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
def ful_maxpooling(tensor):
    return tf.reduce_max(tensor,axis=1)
def get_voxel(fileNum,voxel_size):
    data=tf_util.load_data(os.path.join(DATA_DIR,'voxel_file'+str(fileNum)+'.npy'))
    return data['voxel'+str(voxel_size)]
def get_covariance(fileNum):
    data = tf_util.load_data(os.path.join(DATA_DIR, 'voxel_file' + str(fileNum) + '.npy'))
    return data['covar_data']
def loss_func(input_voxel,output_voxel):
    model_cross=tf.reduce_sum(input_voxel*tf.log(tf.clip_by_value(output_voxel,1e-10,1.0))+(1.0-input_voxel)*tf.log(tf.clip_by_value(1.0-output_voxel,1e-10,1.0)),axis=-1)
    result=-tf.reduce_mean(model_cross)
    return result
def loss_func_l2(input_voxel,output_voxel):
    return tf.reduce_sum(tf.square(input_voxel-output_voxel))
#def sampling(npoint,xyz,use_type='f'):
#    if use_type=='f':
#        idx=tf_sampling.farthest_point_sample(npoint, xyz)
#        new_xyz=tf_sampling.gather_point(xyz,idx)
#    return idx,new_xyz
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
        #print('idx',idx)
        #assert False
    else:
        _,id0 = tf_grouping.knn_point(1, xyz, new_xyz)
        valdist,idx = tf_grouping.knn_point(nsample, xyz, new_xyz)
        idx=tf.where(tf.greater(valdist,radius),tf.tile(id0,[1,1,nsample]),idx)
        #print(valdist,idx,id0)
        #assert False

        #idx, pts_cnt = tf_grouping.query_ball_point(radius, nsample, xyz, new_xyz)
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

def local_net(scope,sample_dim,xyz,featvec,r_list,k_list,layers_list,use_xyz=True,use_all=False):

    with tf.variable_scope(scope) as sc:
        newfeatvec=[]
        if use_all:
            ptnum=xyz.get_shape()[1].value
            centers_batch=tf.tile(tf.constant([i for i in range(BATCH_SIZE)],shape=[BATCH_SIZE,1,1]),multiples=[1,ptnum,1])
            centers_pt=tf.tile(tf.constant([i for i in range(ptnum)],shape=[1,ptnum,1]),multiples=[BATCH_SIZE,1,1])
            centers_id=tf.concat([centers_batch,centers_pt],axis=-1)
            centers_coor=xyz 
        else:
            centers_id=point_choose.farthest_sampling(sample_dim,xyz,featvec=featvec,batch=BATCH_SIZE)
            #print(xyz,centers_id)
            centers_coor=tf.gather_nd(xyz,centers_id)
        for i in range(len(r_list)):
            r=r_list[i]
            k=k_list[i]
            group_id=point_choose.local_devide(sample_dim,k,r,xyz,centers_id,featvec,batch=BATCH_SIZE)
            group_coor=tf.gather_nd(xyz,group_id)
            group_coor=group_coor-tf.tile(tf.expand_dims(centers_coor,axis=2),multiples=[1,1,k,1])
            if featvec is not None:
                group_points=tf.gather_nd(featvec,group_id)
                if use_xyz:
                    group_points=tf.concat([group_points,group_coor],axis=-1)
            else:
                group_points=group_coor
            for j,out_channel in enumerate(layers_list[i]):
                group_points=conv2d(scope='conv%d_%d'%(i,j),inputs=group_points,num_outchannels=out_channel,kernel_size=[1,1],padding='VALID')
            newfeat=tf.reduce_max(group_points,axis=2)
            #newfeat=tf.squeeze(conv2d(scope='add_conv%d'%(i),inputs=group_points,num_outchannels=out_channel,kernel_size=[1,k],padding='VALID'))
            newfeatvec.append(newfeat)
    new_featvec_tensor=tf.concat(newfeatvec,axis=-1)
    return centers_coor,new_featvec_tensor
def global_mlp(scope,xyz,mlp,use_bnorm=False):
    with tf.variable_scope(scope):
        tensor=xyz
        for i,outchannel in enumerate(mlp):
            tensor=conv2d('ini_layer%d'%i,tensor,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        result=tf.reduce_max(tensor,axis=1,keepdims=True)
    return result
#batch*2048*1*128,batch*2048*1*128
def encode_cell(scope,input_tensor,state_tensor,state_len=128,code_len=128,mlp=[256,128],mlpout=[512,128],reuse=False,use_bnorm=False):
    with tf.variable_scope(scope,reuse=reuse):
        #input_info=conv2d('input_trans',input_tensor,code_len/2,[1,1],padding='VALID',use_bnorm=use_bnorm)#batch*2048*1*64
        #input_info=conv2d('input_trans1',input_info,code_len,[1,1],padding='VALID',use_bnorm=use_bnorm)#batch*2048*1*64
        input_info=input_tensor
        if state_tensor is not None:
            #new_state=conv2d('state_trans',state_tensor,128,[1,1],padding='VALID',use_bnorm=use_bnorm)#batch*2048*1*128
            state_info=tf.tile(state_tensor,multiples=[1,tf.shape(input_tensor)[1],1,1])#batch*2048*1*128
            new_state=tf.concat([input_info,state_info],axis=-1)#batch*2048*1*256
            #new_state=input_info*state_info
        else:
            new_state=input_info
        #new_state=conv2d('state_trans',new_state,128,[1,1],padding='VALID',use_bnorm=use_bnorm)#batch*2048*1*128

        for i,outchannel in enumerate(mlp):
            new_state=conv2d('state%d'%i,new_state,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        new_state=tf.reduce_max(conv2d('state_end',new_state,state_len,[1,1],padding='VALID',use_bnorm=use_bnorm),axis=1,keepdims=True)
        codeout=new_state
        for i,outchannel in enumerate(mlpout):
            codeout=conv2d('codemlp%d'%i,codeout,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
        #codeout=tf.reduce_max(codeout,axis=1,keepdims=True)
    #batch*2048*1*128,batch*2048*1*128
    return codeout,new_state
#batch*2048*3
def re_encoder(pointcloud,rnum=3,type_pl='p'):
    with tf.variable_scope('E') as scope:
        #point=tf.expand_dims(pointcloud,axis=2)
        if type_pl=='r':
            prob=0.4*random.random()
            #prob=0.4
            point=shuffle_points(prob,pointcloud)
            #point=tf.nn.dropout(pointcloud,keep_prob=prob,noise_shape=[1,tf.shape(pointcloud)[1],1])*prob
            point_input=tf.expand_dims(point,axis=2)
        else:
            point=tf.expand_dims(pointcloud,axis=2)
            point_input=point
        statelen=256
        state0=global_mlp('init_mlp',point_input,[64,128,statelen])
        #point_input=recover_cell('recover1',point_input,mlp1=[128,256],mlp2=[512,1024])

        #state1=global_mlp('init_mlp1',point_input,[64,128,statelen])
        #state2=global_mlp('init_mlp2',point_input,[64,128,statelen])

        codeout1,state=encode_cell('cell',point_input,state0,mlp=[256,384],mlpout=[128,128],state_len=statelen,code_len=128,use_bnorm=False)
        #codeout1=recover_cell('recover1',codeout1,point_input,mlp1=[256,256],mlp2=[128,128])
        codeout2,state=encode_cell('cell',point_input,state,mlp=[256,384],mlpout=[128,128],state_len=statelen,code_len=128,reuse=True,use_bnorm=False)
        #codeout2=recover_cell('recover2',codeout2,point_input,mlp1=[256,256],mlp2=[128,128])
        codeout3,state=encode_cell('cell',point_input,state,mlp=[256,384],mlpout=[128,128],state_len=statelen,code_len=128,reuse=True,use_bnorm=False)
        #codeout3=recover_cell('recover3',codeout3,point_input,mlp1=[256,256],mlp2=[128,128])
        #codeout1,codeout2,codeout3=recover_devide('recover_final',point_input,[codeout1,codeout2,codeout3],mlp=[256,256],state_len=128)
        #state=global_mlp('end_mlp',point_input,[64,128,128])
        #codeout4,state=encode_cell('cell',point_input,state,mlp=[256,128],reuse=True,use_bnorm=False)
        tf.add_to_collection('code1', codeout1)
        tf.add_to_collection('code2', codeout2)
        tf.add_to_collection('code3', codeout3)
        #tf.add_to_collection('code4', codeout4)
        #print(codeout1)
    return tf.squeeze(tf.concat([codeout1,codeout2,codeout3],axis=-1),[1,2])
#reso:1 or batch*1
#pts:batch*32*3
#gpts:batch*2048*3
def oct_constrain(reso,pts,gpts):
    uplimit=tf.reduce_max(gpts,axis=1,keepdims=True)
    downlimit=tf.reduce_min(gpts,axis=1,keepdims=True)
    length=uplimit-downlimit
    voxcen=(uplimit+downlimit)/2

    voxnum=tf.pow(2.0,tf.expand_dims(tf.expand_dims(reso,axis=-1),axis=-1))
    voxlen=length/voxnum
    voxid=tf.abs(pts-voxcen)/voxlen#batch*32*3
    #print(voxid)
    fixid=tf.range(tf.minimum(tf.reduce_max(tf.ceil(voxnum/2)),6.0))-0.5 #0.5,1.0,1.5,2.0,2.5...
    fixid=tf.reshape(fixid,[1,1,1,-1])
    fixid=tf.tile(fixid,[1,1,3,1])#1*1*3*num
    voxid2=tf.expand_dims(voxid,axis=-1)#batch*32*3*1
    voxdis=tf.reduce_max(tf.abs(voxid2-tf.tile(fixid,[tf.shape(voxid)[0],tf.shape(voxid)[1],1,1])),axis=2)#batch*32*num
    voxcons=tf.reduce_mean(tf.reduce_min(voxdis,axis=-1))

    ptdis=tf.abs(tf.expand_dims(voxid,axis=1)-tf.expand_dims(voxid,axis=2))#batch*32*32*3
    ptdis=tf.reduce_max(ptdis,axis=-1)#batch*32*32
    ptcons=tf.reduce_mean(tf.reduce_sum(tf.nn.relu(1-ptdis),axis=-1)-1.0)

    resocons=tf.reduce_mean(reso)
    return voxcons,ptcons,resocons

def reso_define(feats=None,mlp=[128,64],define_type='v'):
    if define_type=='v':
        reso=tf.get_variable(name='resolution',shape=(1,),initializer=tf.contrib.layers.xavier_initializer())
        reso=tf.stop_gradient(tf.ceil(reso)-reso)+reso
    elif define_type=='n' and feats is not None:
        reso=reso_network(feats,mlp=mlp)
        reso=tf.stop_gradient(tf.ceil(reso)-reso)+reso
    else:
        reso=4
    return reso

def reso_network(feats,mlp=[128,128]):
    #tensor=feats
    tensor=tf.reduce_max(feats,axis=1,keepdims=True)
    tensor=tf.expand_dims(tensor,axis=2)
    #print(feats,tensor)
    for i,outchannel in enumerate(mlp):
        tensor=conv2d('reso_network%d'%i,tensor,outchannel,[1,1],padding='VALID')
    tensor=conv2d('reso_out',tensor,1,[1,1],padding='VALID',activation_func=None)
    reso=tf.squeeze(tf.square(tensor),[1,2,3])
    #print(reso)
    return reso

def nonlocal_aggregate(scope,cens,feats,gcens,gfeats,featlen=128,mlp=[128,128],mlp1=[128,128]):
    with tf.variable_scope(scope):
        tensor0=tf.expand_dims(tf.concat([cens,feats],axis=-1),axis=2)
        gtensor=tf.expand_dims(tf.concat([gcens,gfeats],axis=-1),axis=1)
        glen=gtensor.get_shape()[-1].value

        tensor=tf.concat([tf.tile(tensor0,[1,1,tf.shape(gtensor)[2],1]),tf.tile(gtensor,[1,tf.shape(tensor0)[1],1,1])],axis=-1)
        #tensor=tf.tile(gtensor,[1,tf.shape(tensor0)[1],1,1])-tf.tile(tensor0,[1,1,tf.shape(gtensor)[2],1])
        for i,outchannel in enumerate(mlp):
            tensor=conv2d('nonlocal_state%d'%i,tensor,outchannel,[1,1],padding='VALID')
        tensor=conv2d('nonlocal_out',tensor,1+glen,[1,1],padding='VALID',activation_func=None)
        alpha=tf.square(tensor[:,:,:,:1])
        mask=tensor[:,:,:,1:]

        #dismat=tf.sqrt(1e-8+tf.reduce_sum(tf.square(tf.expand_dims(cens,axis=2)-tf.expand_dims(gcens,axis=1)),axis=-1,keepdims=True))#batch*32*32*1
        #dismask=tf.exp(-alpha*dismat)/tf.reduce_sum(tf.exp(-alpha*dismat),axis=2,keepdims=True)
        #print(mask,gtensor)
        #mask=tf.exp(mask)/tf.reduce_sum(tf.exp(mask),axis=2,keepdims=True)

        afeat=tf.reduce_sum(mask*gtensor,axis=2,keepdims=True)
        #print('*************',afeat,dismask,gtensor)
        tensor=tf.concat([tensor0,afeat],axis=-1)
        for i,outchannel in enumerate(mlp1):
            tensor=conv2d('nonlocal_final%d'%i,tensor,outchannel,[1,1],padding='VALID')
        tensor=conv2d('nonlocal_finalout',tensor,3+featlen,[1,1],padding='VALID',activation_func=None)
        offset=tensor[:,:,:,:3]
        newfeats=tensor[:,:,:,3:]
        #newfeats=tensor[:,:,:,3:]
        #print('cens',cens,offset)
        tf.add_to_collection(scope+'_offset',offset)
        newcens=cens+tf.squeeze(offset,[2])
        newfeats=tf.squeeze(newfeats,[2])+feats
        #newfeats=afeat[:,:,:,3:]
        #newfeats=tf.squeeze(newfeats,[2])
        #print(newfeats)
    return newcens,newfeats
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
        #tensor=tf.expand_dims(tf.concat([feats,gfeat],axis=-1),axis=2)
        #for i,outchannel in enumerate(mlp1):
        #    tensor=conv2d('global_feat%d'%i,tensor,outchannel,[1,1],padding='VALID')
        #tensor=conv2d('nonlocal_featout',tensor,feats.get_shape()[-1].value,[1,1],padding='VALID',activation_func=None)
    return newcens,newfeats
#def global_fix(scope,cens,gfeat,mlp=[128,128],mlp1=[128,128]):
#    with tf.variable_scope(scope):
#        #tensor0=tf.expand_dims(tf.concat([cens,feats],axis=-1),axis=2)
#        tensor0=tf.expand_dims(tf.concat([cens,tf.tile(gfeat,[1,tf.shape(cens)[1],1])],axis=-1),axis=2)
#        tensor=tensor0
#        for i,outchannel in enumerate(mlp):
#            tensor=conv2d('global_ptstate%d'%i,tensor,outchannel,[1,1],padding='VALID',activation_func=tf.nn.relu)
#        tensorword=tensor
#        tensor=tf.reduce_max(tensor,axis=1,keepdims=True)
#        tensor=tf.concat([tf.expand_dims(cens,axis=2),tf.tile(tensor,[1,tf.shape(cens)[1],1,1])],axis=-1)
#        for i,outchannel in enumerate(mlp1):
#            tensor=conv2d('global_ptstate2%d'%i,tensor,outchannel,[1,1],padding='VALID',activation_func=tf.nn.relu)
#        tensor=conv2d('global_ptout',tensor,3,[1,1],padding='VALID',activation_func=None)
#        newcens=cens+tf.squeeze(tensor[:,:,:,:3],[2])
#        #newfeats=feats+tf.squeeze(tensor[:,:,:,3:],[2])
#        tf.add_to_collection('non1_offset',tensor[:,:,:,:3])
#        #tensor=tf.expand_dims(tf.concat([feats,gfeat],axis=-1),axis=2)
#        #for i,outchannel in enumerate(mlp1):
#        #    tensor=conv2d('global_feat%d'%i,tensor,outchannel,[1,1],padding='VALID')
#        #tensor=conv2d('nonlocal_featout',tensor,feats.get_shape()[-1].value,[1,1],padding='VALID',activation_func=None)
#        return newcens
def global_layer(scope,pts,mlp=[64,64,128],use_bnorm=False):
    with tf.variable_scope(scope):
        tensor=tf.expand_dims(pts,axis=2)
        for i,outchannel in enumerate(mlp):
            tensor=conv2d('global_ptstate%d'%i,tensor,outchannel,[1,1],padding='VALID',use_bnorm=use_bnorm)
    return tf.squeeze(tensor,2)
#def dis_aggregate(newcens,cens,feats,)
def deform_layer(scope,cens,feats,gfeat,pts,ptfeats,mlp1=[128,128],mlp2=[128,128],use_bnorm=False,use_rdis=False,use_featdis=True):
    with tf.variable_scope(scope):
        #feats=tf.expand_dims(feats,axis=2)
        cenlen=cens.get_shape()[-1].value
        featlen=feats.get_shape()[-1].value
        dislen=0
        tensor=tf.expand_dims(tf.concat([cens,feats,tf.tile(gfeat,[1,tf.shape(feats)[1],1])],axis=-1),axis=2)
        centensor=tf.expand_dims(tf.concat([cens,feats],axis=-1),axis=2)
        for i,outchannel in enumerate(mlp1):
            tensor=conv2d('global_info%d'%i,tensor,outchannel,[1,1],padding='VALID')
        if use_rdis:
            #dislen=cenlen+1
            if use_featdis:
                dislen=cenlen+featlen+1
                distensor=tf.concat([pts,ptfeats],axis=-1)
                cenlens=cenlen+featlen
            else:
                dislen=cenlen+1
                distensor=pts
                cenlens=cenlen
        else:
            if use_featdis:
                dislen=cenlen+cenlen+featlen+featlen
                distensor=tf.concat([pts,ptfeats],axis=-1)
                cenlens=cenlen+featlen
            else:
                dislen=cenlen+cenlen+featlen
                distensor=pts
                cenlens=cenlen
        tensor=conv2d('global_ptout_feat',tensor,dislen,[1,1],padding='VALID',activation_func=None)
        move=tensor[:,:,:,:cenlens]
        paras=tf.square(tensor[:,:,:,cenlens:])
        #print(centensor,move,distensor)
        dismat=tf.reduce_sum(tf.square(centensor+move-tf.expand_dims(distensor,axis=1)),axis=-1,keepdims=True)#batch*32*2048*1
        weights=tf.exp(-paras*dismat)
        #weights=weights[:,:,:,cenlen:]
        weights=tf.reduce_prod(weights,axis=-1,keepdims=True)
        weights=tf.exp(weights)
        weights=weights/(tf.reduce_sum(weights,axis=2,keepdims=True))
        newfeats=tf.reduce_sum(weights*tf.expand_dims(ptfeats,axis=1),axis=2,keepdims=True)#batch*32*1*featlen
        #newfeats=tf.concat([centensor,newfeats],axis=-1)
        for i,outchannel in enumerate(mlp2):
            newfeats=conv2d('global_ptstate%d'%i,newfeats,outchannel,[1,1],padding='VALID')

        newcens=cens+tf.squeeze(move[:,:,:,:cenlen],axis=2)
        
        return newcens,tf.squeeze(newfeats,2)
def F1_Net(scope,data,mlp0=[64,128],mlp=[64,64,1]):
    #result=tf.expand_dims(data,axis=2)
    with tf.variable_scope(scope):
        #kpts,_=grouping(data,data, None, knum, None, knn=True, use_xyz=True)
        #result=kpts-tf.expand_dims(data,axis=2)

        result=tf.expand_dims(data,axis=2)
        for i, num_out_channel in enumerate(mlp0):
            result = conv2d('F_feats%d'%i,result,num_out_channel,[1,1],activation_func=tf.nn.relu)
        feat=tf.reduce_max(result,axis=2,keepdims=True)
        maxfeat=tf.reduce_max(feat,axis=1,keepdims=True)
        #result=tf.concat([tf.expand_dims(data,axis=2),feat],axis=-1)
        result=tf.concat([tf.expand_dims(data,axis=2),tf.tile(maxfeat,[1,tf.shape(feat)[1],1,1])],axis=-1)

        for i, num_out_channel in enumerate(mlp[:-1]):
            result = conv2d('F_trans%d'%i,result,num_out_channel,[1,1],activation_func=tf.nn.relu)

        result = conv2d('F_out%d'%i,result,mlp[-1],[1,1],activation_func=None)
    result=tf.squeeze(result,[2])
    return result
def F_Net(scope,data,mlp0=[64,64],mlp=[64,64,1],knum=8,inputfeat=None):
    #result=tf.expand_dims(data,axis=2)
    with tf.variable_scope(scope):
        kpts,_=grouping(data,data, None, knum, None, knn=True, use_xyz=True)
        result=kpts-tf.expand_dims(data,axis=2)
        
        #result=tf.expand_dims(data,axis=2)
        for i, num_out_channel in enumerate(mlp0):
            result = conv2d('F_feats%d'%i,result,num_out_channel,[1,1],activation_func=tf.nn.relu)
        feat=tf.reduce_max(result,axis=2,keepdims=True)
        maxfeat=tf.reduce_max(feat,axis=1,keepdims=True)
        if inputfeat is not None:
            if inputfeat.get_shape()[1].value>1:
                feat=tf.concat([feat,inputfeat],axis=-1)
            else:
                feat=tf.concat([feat,tf.tile(maxfeat,[1,tf.shape(inputfeat)[1],1,1])],axis=-1)
        result=tf.concat([tf.expand_dims(data,axis=2),feat],axis=-1)
        #result=tf.concat([tf.expand_dims(data,axis=2),feat,tf.tile(maxfeat,[1,tf.shape(feat)[1],1,1])],axis=-1)

        for i, num_out_channel in enumerate(mlp[:-1]):
            result = conv2d('F_trans%d'%i,result,num_out_channel,[1,1],activation_func=tf.nn.relu)

        result = conv2d('F_out%d'%i,result,mlp[-1],[1,1],activation_func=None)
    result=tf.squeeze(result,[2])
    return result
def trans_block(scope,data,feat=None):
    with tf.variable_scope(scope):

        xy=data[:,:,:2]#batch*n*2
        z=data[:,:,2:]#batch*n*1
        #xy=xy+F1_Net('module001',z)
        pz=tf.exp(F_Net('module01',xy,mlp=[64,16,1],inputfeat=feat))
        z=pz*z+F_Net('module1',xy,mlp=[64,16,1],inputfeat=feat)
        data=tf.concat([xy,z],axis=-1)

        yz=data[:,:,1:]
        x=data[:,:,:1]
        #yz=yz+F1_Net('module02',x)
        px=tf.exp(F_Net('module02',yz,mlp=[64,16,1],inputfeat=feat))
        x=px*x+F_Net('module2',yz,mlp=[64,16,1],inputfeat=feat)
        data=tf.concat([x,yz],axis=-1)

        xz=tf.concat([data[:,:,:1],data[:,:,2:]],axis=-1)
        y=data[:,:,1:2]
        #xz=xz+F1_Net('module03',y)
        py=tf.exp(F_Net('module03',xz,mlp=[64,16,1],inputfeat=feat))
        y=py*y+F_Net('module3',xz,mlp=[64,16,1],inputfeat=feat)
        data=tf.concat([data[:,:,:1],y,data[:,:,2:]],axis=-1)

        #xy=data[:,:,:2]#batch*n*2
        #z=data[:,:,2:]#batch*n*1
        #z=z+F_Net('module1',xy,mlp=[64,16,1])
        #data=tf.concat([xy,z],axis=-1)
    return data
def reverse_block(scope,data,feat=None):
    with tf.variable_scope(scope):
        #xy=data[:,:,:2]#batch*n*2
        #z=data[:,:,2:]#batch*n*1
        #z=z-F_Net('module1',xy,mlp=[64,16,1])
        #data=tf.concat([xy,z],axis=-1)

        xz=tf.concat([data[:,:,:1],data[:,:,2:]],axis=-1)
        y=data[:,:,1:2]
        y=y-F_Net('module3',xz,mlp=[64,16,1],inputfeat=feat)
        py=tf.exp(F_Net('module03',xz,mlp=[64,16,1],inputfeat=feat))
        y=y/py
        #xz=xz-F1_Net('module03',y)
        data=tf.concat([data[:,:,:1],y,data[:,:,2:]],axis=-1)

        yz=data[:,:,1:]
        x=data[:,:,:1]
        x=x-F_Net('module2',yz,mlp=[64,16,1],inputfeat=feat)
        px=tf.exp(F_Net('module02',yz,mlp=[64,16,1],inputfeat=feat))
        x=x/px
        #yz=yz-F1_Net('module02',x)
        data=tf.concat([x,yz],axis=-1)

        xy=data[:,:,:2]#batch*n*2
        z=data[:,:,2:]#batch*n*1
        z=z-F_Net('module1',xy,mlp=[64,16,1],inputfeat=feat)
        pz=tf.exp(F_Net('module01',xy,mlp=[64,16,1],inputfeat=feat))
        z=z/pz
        #xy=xy-F1_Net('module001',z)
        data=tf.concat([xy,z],axis=-1)
    return data
#data:batch*n*3
def flow_trans(inputdata):
    data=inputdata
    with tf.variable_scope('flow'):
        result=tf.expand_dims(data,axis=2)
        #kpts,_=grouping(data,data, None, 16, None, knn=True, use_xyz=True)
        #result=kpts-tf.expand_dims(data,axis=2)
        for i, num_out_channel in enumerate([64,64,128]):
            result = conv2d('F_feats%d'%i,result,num_out_channel,[1,1],activation_func=tf.nn.relu)
        feat=tf.reduce_max(result,axis=2,keepdims=True)

        #feat=None
        data=trans_block('block1',data,feat)
        #data=trans_block('block2',data)
    return data,feat
def reverse_trans(inputdata,feat):
    data=inputdata
    with tf.variable_scope('flow',reuse=True):
        #data=reverse_block('block2',data)
        data=reverse_block('block1',data,feat)
    return data
def sample_distri(meanval,varval):
    #result=meanval+varval*tf.random_normal(shape=tf.shape(meanval),mean=0.0,stddev=1.0)
    result=meanval+tf.exp(varval/2)*tf.random_normal(shape=tf.shape(meanval),mean=0.0,stddev=1.0)
    #result=meanval+tf.random_normal(shape=tf.shape(meanval),mean=0.0,stddev=0.1)
    return result
def KLfunc(inmeanval,instdval):
    meanval=tf.reduce_mean(inmeanval,axis=-1,keepdims=True)
    result=tf.exp(instdval)-instdval-1#+tf.square(inmeanval-meanval)
    return result/2
def vae_flow(rawdata,flowdata,ncen,nlength,feat=None):
    with tf.variable_scope('vae_flow'):
        data=flowdata
        varval=F_Net('vae_flow',data,mlp=[128,64,3],knum=16)
    result=sample_distri(data,varval)
    result=result*nlength+ncen
    vaedata=reverse_trans(result,feat)
    cen=sampling(512,rawdata,use_type='f')[-1]
    #loss1=multi_chamfer_func(cen,rawdata,vaedata,512,8,r=0.2,theta1=0.5,theta2=1.0,use_frame=False,use_r=False,use_all=False)
    #loss1=chamfer_big(rawdata,vaedata)[0]
    loss1=emd_func(rawdata,vaedata)[0]
    #loss1=tf.reduce_mean(tf.reduce_sum(tf.square(rawdata-vaedata),axis=-1))
    loss2=tf.reduce_mean(tf.reduce_sum(KLfunc(data,varval),axis=-1))
    loss=loss2+loss1
    return loss,[loss1,loss2]
#def vaeloss(flowin,flowvae):

def encoder(tensor):
    with tf.variable_scope('E'):
        l0_xyz=tensor
        l0_points=None
        ptnum=tensor.get_shape()[1].value
        globalfeat=global_layer('init_layer',l0_xyz,mlp=[64,128,256],use_bnorm=False)
        globalfeat=tf.reduce_max(globalfeat,axis=1,keepdims=True)
        gfeat=globalfeat

        cen1,feat1=pointnet_sa_module_msg(l0_xyz, l0_points, 512, [0.2], [8], [[32,256]], is_training=True, bn_decay=None, scope='layer1', use_nchw=False,use_knn=True)
        rcen1,rfeat1=global_fix('global1',cen1,feat1,gfeat,mlp=[256,256],mlp1=[256,256])

        cen2,feat2=pointnet_sa_module_msg(cen1, feat1, 128, [0.4], [8], [[64,256]], is_training=True, bn_decay=None, scope='layer2',use_nchw=False,use_knn=True)
        rcen2,rfeat2=global_fix('global2',cen2,feat2,gfeat,mlp=[256,256],mlp1=[256,256])

        cen3,feat3=pointnet_sa_module_msg(cen2, feat2, 32, [0.6], [8], [[128,256]], is_training=True, bn_decay=None, scope='layer3',use_nchw=False,use_knn=True)
        rcen3,rfeat3=global_fix('global3',cen3,feat3,gfeat,mlp=[256,256],mlp1=[256,256])

        cen4,feat4=pointnet_sa_module_msg(cen3, feat3, 8, [0.6], [8], [[128,256]], is_training=True, bn_decay=None, scope='layer4',use_nchw=False,use_knn=True)
        rcen4,rfeat4=global_fix('global4',cen4,feat4,gfeat,mlp=[256,256],mlp1=[256,256])

        #rfeat1,rfeat2,rfeat3=feat1,feat2,feat3
                 
        reso=reso_define(feats=feat3,mlp=[128,64],define_type='n')
        voxcons,ptcons,resocons=oct_constrain(reso,cen3,cen3)
        tf.add_to_collection('voxel_constraint',0.01*voxcons+0.01*ptcons+0.001*resocons)

        tf.add_to_collection('cen3',cen3)
    return [rcen1,rcen2,rcen3,rcen4],[rfeat1,rfeat2,rfeat3,rfeat4]
def pt_encoder(feat,mlp=[128,128],ptnum=32):
    with tf.variable_scope('S'):
        new_point=tf.reshape(feat,[-1,1,1,feat.get_shape()[-1].value])
        for i, num_out_channel0 in enumerate(mlp):
            new_point = conv2d('samplenet%d'%i,new_point,num_out_channel0,[1,1])

def full_encoder(tensor):
    with tf.variable_scope('E'):
        tf.add_to_collection('encoder', net)
    return net

def normalize(tensor_data):
    tensormean=tf.reduce_mean(tensor_data,axis=0)
    tensorvar=tf.clip_by_value(tf.sqrt(tf.reduce_mean(tf.square(tensor_data-tensormean),axis=0)),1e-10,10)
    tensornorm=(tensor_data-tensormean)/tensorvar
    print(tensornorm)
    return tensornorm
def deconv_decoder(tensor):
    featLen=tensor.get_shape()[-1].value
    batchNum=tf.shape(tensor)[0]
    input_tensor=tf.reshape(tensor,[-1,1,1,featLen])

    net=deconv('deconv_layer1',inputs=input_tensor,output_shape=[batchNum,8,1,256],kernel_size=[8,1],stride=[1,1],padding='VALID')
    net=deconv('deconv_layer2',inputs=net,output_shape=[batchNum,32,1,128],kernel_size=[8,1],stride=[4,1])
    net=deconv('deconv_layer3',inputs=net,output_shape=[batchNum,128,1,128],kernel_size=[8,1],stride=[4,1])
    net=deconv('deconv_layer4',inputs=net,output_shape=[batchNum,512,1,128],kernel_size=[8,1],stride=[4,1])
    net = deconv('deconv_layer5', inputs=net, output_shape=[batchNum, 1024, 1, 64], kernel_size=[8, 1], stride=[2, 1])
    net = deconv('deconv_layer6', inputs=net, output_shape=[batchNum, 2048, 1, 3], kernel_size=[8, 1], stride=[2, 1],activation_func=None)
    tf.add_to_collection('decoder', net)
    return tf.reshape(net,[-1,2048,3])
def gredual_decoder(tensor):
    featLen = tensor.get_shape()[-1].value
    batchNum = tf.shape(tensor)[0]
    input_tensor = tf.reshape(tensor, [-1, 1, 1, featLen])

    net = deconv('deconv_layer1', inputs=input_tensor, output_shape=[batchNum, 8, 1, 256], kernel_size=[8, 1],stride=[1, 1], padding='VALID')
    net = deconv('deconv_layer2', inputs=net, output_shape=[batchNum, 32, 1, 128], kernel_size=[8, 1], stride=[4, 1])
    net = deconv('deconv_layer3', inputs=net, output_shape=[batchNum, 128, 1, 128], kernel_size=[8, 1], stride=[4, 1])

    net128=conv2d('o128', net,3,[1,1],padding='VALID',activation_func=None)
    tf.add_to_collection('o128', net128)

    net=tf.concat([net128,tf.tile(input_tensor,multiples=[1,128,1,1])],axis=-1)
    net = deconv('deconv_layer4', inputs=net, output_shape=[batchNum, 512, 1, 128], kernel_size=[8, 1], stride=[4, 1])
    net = conv2d('layer4_conv2d',net,128,[1,1],padding='VALID')
    net512=conv2d('o512',net,3,[1,1],padding='VALID',activation_func=None)
    tf.add_to_collection('o512',net512)

    net = tf.concat([net512, tf.tile(input_tensor, multiples=[1, 512, 1, 1])], axis=-1)
    net = deconv('deconv_layer5', inputs=net, output_shape=[batchNum, 2048, 1, 128], kernel_size=[8, 1], stride=[4, 1])
    net = conv2d('layer5_conv2d', net, 128, [1, 1], padding='VALID')
    net2048 = conv2d('o2048', net, 3, [1, 1], padding='VALID',activation_func=None)
    tf.add_to_collection('o2048', net2048)

    return tf.reshape(net128, [-1, 128, 3]),tf.reshape(net512, [-1, 512, 3]),tf.reshape(net2048, [-1, 2048, 3])

def gradual_decoder(tensor):
    featLen = tensor.get_shape()[-1].value
    batchNum = tf.shape(tensor)[0]
    input_tensor = tf.reshape(tensor, [-1, 1, 1, featLen])

    net = deconv('deconv_layer1', inputs=input_tensor, output_shape=[batchNum, 8, 1, 256], kernel_size=[8, 1],stride=[1, 1], padding='VALID')
    net = deconv('deconv_layer2', inputs=net, output_shape=[batchNum, 32, 1, 128], kernel_size=[8, 1], stride=[4, 1])
    net = deconv('deconv_layer3', inputs=net, output_shape=[batchNum, 128, 1, 128], kernel_size=[8, 1], stride=[4, 1])

    net128=conv2d('o128', net,3,[1,1],padding='VALID',activation_func=None)
    tf.add_to_collection('o128', net128)

    net=tf.concat([net128,tf.tile(input_tensor,multiples=[1,128,1,1])],axis=-1)
    net = deconv('deconv_layer4', inputs=net, output_shape=[batchNum,256 , 1, 128], kernel_size=[8, 1], stride=[2, 1])
    net = deconv('deconv_layer5',inputs=net,output_shape=[batchNum,512,1,64],kernel_size=[8,1],stride=[2,1])
    net512=conv2d('o512',net,3,[1,1],padding='VALID',activation_func=None)
    tf.add_to_collection('o512',net512)

    net = tf.concat([net512, tf.tile(input_tensor, multiples=[1, 512, 1, 1])], axis=-1)
    net = deconv('deconv_layer6', inputs=net, output_shape=[batchNum, 1024, 1, 128], kernel_size=[8, 1], stride=[2, 1])
    net = deconv('deconv_layer7', inputs=net, output_shape=[batchNum, 2048, 1, 64], kernel_size=[8, 1], stride=[2, 1])
    net2048 = conv2d('o2048', net, 3, [1, 1], padding='VALID',activation_func=None)
    tf.add_to_collection('o2048', net2048)

    return tf.reshape(net128, [BATCH_SIZE, 128, 3]),tf.reshape(net512, [BATCH_SIZE, 512, 3]),tf.reshape(net2048, [-1, 2048, 3])
   # return tf.reshape(net2048, [-1, 2048, 3])

def gradual2_decoder(tensor):
    featLen = tensor.get_shape()[-1].value
    batchNum = tf.shape(tensor)[0]
    input_tensor = tf.reshape(tensor, [-1, 1, 1, featLen])

    net = deconv('deconv_layer1', inputs=input_tensor, output_shape=[batchNum, 8, 1, 256], kernel_size=[8, 1],stride=[1, 1], padding='VALID')
    net = deconv('deconv_layer2', inputs=net, output_shape=[batchNum, 32, 1, 128], kernel_size=[8, 1], stride=[4, 1])
    net = deconv('deconv_layer3', inputs=net, output_shape=[batchNum, 128, 1, 128], kernel_size=[8, 1], stride=[4, 1])

    net128 = conv2d('128', net, 3, [1, 1], padding='VALID', activation_func=None)
    tf.add_to_collection('o128', net128)

    net = tf.concat([net128, tf.tile(input_tensor, multiples=[1, 128, 1, 1])], axis=-1)
    net = deconv('deconv_layer4', inputs=net, output_shape=[batchNum, 512, 1, 128], kernel_size=[32, 1], stride=[4, 1])

    net = conv2d('conv2d_layer5',  net, 64, [1, 1], padding='VALID')

    net512 = conv2d('512', net, 3, [1, 1], padding='VALID', activation_func=None)
    tf.add_to_collection('o512', net512)

    net = tf.concat([net512, tf.tile(input_tensor, multiples=[1, 512, 1, 1])], axis=-1)
    net = deconv('deconv_layer6', inputs=net, output_shape=[batchNum, 2048, 1, 128], kernel_size=[16, 1], stride=[4, 1])
    net = conv2d('conv2d_layer7', net, 64, [1, 1], padding='VALID')
    net2048 = conv2d('2048', net, 3, [1, 1], padding='VALID', activation_func=None)
    tf.add_to_collection('o2048', net2048)

    return tf.reshape(net128, [-1, 128, 3]),tf.reshape(net512, [-1, 512, 3]),tf.reshape(net2048, [-1, 2048, 3])

def gradual22_decoder(tensor):
    featLen = tensor.get_shape()[-1].value
    batchNum = tf.shape(tensor)[0]
    input_tensor = tf.reshape(tensor, [-1, 1, 1, featLen])

    net = deconv('deconv_layer1', inputs=input_tensor, output_shape=[batchNum, 8, 1, 256], kernel_size=[8, 1],stride=[1, 1], padding='VALID')
    net = deconv('deconv_layer2', inputs=net, output_shape=[batchNum, 32, 1, 128], kernel_size=[8, 1], stride=[4, 1])
    netfeat128 = deconv('deconv_layer3', inputs=net, output_shape=[batchNum, 128, 1, 128], kernel_size=[8, 1], stride=[4, 1])

    net128 = conv2d('128', netfeat128, 3, [1, 1], padding='VALID', activation_func=None)
    tf.add_to_collection('o128', net128)

    net = tf.concat([net128,netfeat128,tf.tile(input_tensor, multiples=[1, 128, 1, 1])], axis=-1)
    net = deconv('deconv_layer4', inputs=net, output_shape=[batchNum, 512, 1, 128], kernel_size=[32, 1], stride=[4, 1])

    netfeat512 = conv2d('conv2d_layer5',  net, 128, [1, 1], padding='VALID')

    net512 = conv2d('512', netfeat512, 3, [1, 1], padding='VALID', activation_func=None)
    tf.add_to_collection('o512', net512)

    net = tf.concat([net512,netfeat512, tf.tile(input_tensor, multiples=[1, 512, 1, 1])], axis=-1)
    net = deconv('deconv_layer6', inputs=net, output_shape=[batchNum, 2048, 1, 128], kernel_size=[16, 1], stride=[4, 1])
    net = conv2d('conv2d_layer7', net, 128, [1, 1], padding='VALID')
    net2048 = conv2d('2048', net, 3, [1, 1], padding='VALID', activation_func=None)
    tf.add_to_collection('o2048', net2048)

    return tf.reshape(net128, [-1, 128, 3]),tf.reshape(net512, [-1, 512, 3]),tf.reshape(net2048, [-1, 2048, 3])



def gradual3_decoder(tensor):
    featLen = tensor.get_shape()[-1].value
    batchNum = tf.shape(tensor)[0]
    input_tensor = tf.reshape(tensor, [-1, 1, 1, featLen])
    with tf.variable_scope('D128'):
        net = deconv('deconv_layer1', inputs=input_tensor, output_shape=[batchNum, 8, 1, 256], kernel_size=[8, 1],stride=[1, 1], padding='VALID')
        net = deconv('deconv_layer2', inputs=net, output_shape=[batchNum, 32, 1, 128], kernel_size=[8, 1], stride=[4, 1])
        net = deconv('deconv_layer3', inputs=net, output_shape=[batchNum, 128, 1, 128], kernel_size=[8, 1], stride=[4, 1])
        net128=conv2d('o128', net,3,[1,1],padding='VALID',activation_func=None)
    tf.add_to_collection('o128', net128)
    with tf.variable_scope('D512'):
        net=tf.concat([net128,tf.tile(input_tensor,multiples=[1,128,1,1])],axis=-1)
        net = deconv('deconv_layer4', inputs=net, output_shape=[batchNum, 128, 8, 128], kernel_size=[1, 8], stride=[1, 1],padding='VALID')
        net = deconv('deconv_layer5', inputs=net, output_shape=[batchNum, 512, 8, 128], kernel_size=[32, 1], stride=[4, 1])
        net512 = conv2d('layer5_conv2d',net,3,[1,8],padding='VALID',activation_func=None)
        #net512=conv2d('o512',net,3,[1,1],padding='VALID',activation_func=None)
    tf.add_to_collection('o512',net512)
    with tf.variable_scope('D2048'):
        net = tf.concat([net512, tf.tile(input_tensor, multiples=[1, 512, 1, 1])], axis=-1)
        net = deconv('deconv_layer6', inputs=net, output_shape=[batchNum,512 , 16, 128], kernel_size=[1, 16], stride=[1, 1],padding='VALID')
        net = deconv('deconv_layer7', inputs=net, output_shape=[batchNum,2048 , 16, 128], kernel_size=[32, 1], stride=[4, 1])
        net2048 = conv2d('layer7_conv2d', net, 3, [1, 16], padding='VALID',activation_func=None)
        #net2048 = conv2d('o2048', net, 3, [1, 1], padding='VALID',activation_func=None)
    tf.add_to_collection('o2048', net2048)

    return tf.reshape(net128, [-1, 128, 3]),tf.reshape(net512, [-1, 512, 3]),tf.reshape(net2048, [-1, 2048, 3])


#def gradual33_decoder(tensor):
#    featLen = tensor.get_shape()[-1].value
#    batchNum = tf.shape(tensor)[0]
#    input_tensor = tf.reshape(tensor, [-1, 1, 1, featLen])
#
#    net = deconv('deconv_layer1', inputs=input_tensor, output_shape=[batchNum, 8, 1, 256], kernel_size=[8, 1],stride=[1, 1], padding='VALID')
#    net = deconv('deconv_layer2', inputs=net, output_shape=[batchNum, 32, 1, 128], kernel_size=[8, 1], stride=[4, 1])
#    netfeat128 = deconv('deconv_layer3', inputs=net, output_shape=[batchNum, 128, 1, 128], kernel_size=[8, 1], stride=[4, 1])
#
#    net128=conv2d('o128', netfeat128,3,[1,1],padding='VALID',activation_func=None)
#    tf.add_to_collection('o128', net128)
#
#    net=tf.concat([net128,netfeat128,tf.tile(input_tensor,multiples=[1,128,1,1])],axis=-1)
#    net = deconv('deconv_layer4', inputs=net, output_shape=[batchNum, 128, 16, 128], kernel_size=[1, 16], stride=[1, 1],padding='VALID')
#    netfeat512 = deconv('deconv_layer5', inputs=net, output_shape=[batchNum, 512, 16, 128], kernel_size=[32, 1], stride=[4, 1])
#   # netfeat512 = deconv('deconv_layer5', inputs=net, output_shape=[batchNum, 512, 8, 128], kernel_size=[4, 1], stride=[4, 1],padding='VALID')
#    net512 = conv2d('layer5_conv2d',netfeat512,3,[1,16],padding='VALID',activation_func=None)
#    
#    tf.add_to_collection('o512',net512)
#
#    net = tf.concat([tf.tile(net512,multiples=[1,1,16,1]),netfeat512, tf.tile(input_tensor, multiples=[1, 512, 16, 1])], axis=-1)
#    net = deconv('deconv_layer6', inputs=net, output_shape=[batchNum,512 , 32, 128], kernel_size=[1, 8], stride=[1, 2])
#    net = deconv('deconv_layer7', inputs=net, output_shape=[batchNum,2048 , 32, 128], kernel_size=[32, 1], stride=[4, 1])
#    #net = deconv('deconv_layer7', inputs=net, output_shape=[batchNum,2048 , 16, 128], kernel_size=[4, 1], stride=[4, 1],padding='VALID')
#    net2048 = conv2d('layer7_conv2d', net, 3, [1, 32], padding='VALID',activation_func=None)
#    tf.add_to_collection('o2048', net2048)
#
#    return tf.reshape(net128, [-1, 128, 3]),tf.reshape(net512, [-1, 512, 3]),tf.reshape(net2048, [-1, 2048, 3])

def gradual33_decoder(tensor):
    featLen = tensor.get_shape()[-1].value
    batchNum = tf.shape(tensor)[0]
    input_tensor = tf.reshape(tensor, [-1, 1, 1, featLen])

    net = deconv('deconv_layer1', inputs=input_tensor, output_shape=[batchNum, 8, 1, 256], kernel_size=[8, 1],stride=[1, 1], padding='VALID')
    net = deconv('deconv_layer2', inputs=net, output_shape=[batchNum, 32, 1, 128], kernel_size=[8, 1], stride=[4, 1])
    netfeat128 = deconv('deconv_layer3', inputs=net, output_shape=[batchNum, 128, 1, 128], kernel_size=[8, 1], stride=[4, 1])

    net128=conv2d('o128', netfeat128,3,[1,1],padding='VALID',activation_func=None)
    tf.add_to_collection('o128', net128)

    net=tf.concat([net128,netfeat128,tf.tile(input_tensor,multiples=[1,128,1,1])],axis=-1)
    net = deconv('deconv_layer4', inputs=net, output_shape=[batchNum, 128, 8, 128], kernel_size=[1, 8], stride=[1, 1],padding='VALID')
    netfeat512 = deconv('deconv_layer5', inputs=net, output_shape=[batchNum, 512, 8, 128], kernel_size=[32, 1], stride=[4, 1])
   # netfeat512 = deconv('deconv_layer5', inputs=net, output_shape=[batchNum, 512, 8, 128], kernel_size=[4, 1], stride=[4, 1],padding='VALID')
    net512 = conv2d('layer5_conv2d',netfeat512,3,[1,8],padding='VALID',activation_func=None)
    
    tf.add_to_collection('o512',net512)

    net = tf.concat([tf.tile(net512,multiples=[1,1,8,1]),netfeat512, tf.tile(input_tensor, multiples=[1, 512, 8, 1])], axis=-1)
    net = deconv('deconv_layer6', inputs=net, output_shape=[batchNum,512 , 16, 128], kernel_size=[1, 16], stride=[1, 2])
    net = deconv('deconv_layer7', inputs=net, output_shape=[batchNum,2048 , 16, 128], kernel_size=[32, 1], stride=[4, 1])
    #net = deconv('deconv_layer7', inputs=net, output_shape=[batchNum,2048 , 16, 128], kernel_size=[4, 1], stride=[4, 1],padding='VALID')
    net2048 = conv2d('layer7_conv2d', net, 3, [1, 16], padding='VALID',activation_func=None)
    tf.add_to_collection('o2048', net2048)

    return tf.reshape(net128, [-1, 128, 3]),tf.reshape(net512, [-1, 512, 3]),tf.reshape(net2048, [-1, 2048, 3])


def gradual_final_30(tensor):
    featLen = tensor.get_shape()[-1].value
    batchNum = tf.shape(tensor)[0]
    input_tensor = tf.reshape(tensor, [-1, 1, 1, featLen])

    net = deconv('deconv_layer1', inputs=input_tensor, output_shape=[batchNum, 8, 1, 512], kernel_size=[8, 1],stride=[1, 1], padding='VALID')
    net = deconv('deconv_layer2', inputs=net, output_shape=[batchNum, 32, 1, 128], kernel_size=[8, 1], stride=[4, 1])
    netfeat128 = deconv('deconv_layer3', inputs=net, output_shape=[batchNum, 128, 1, 128], kernel_size=[8, 1], stride=[4, 1])

    net128=conv2d('o128', netfeat128,3,[1,1],padding='VALID',activation_func=None)
    tf.add_to_collection('o128', net128)

    net=tf.concat([net128,netfeat128,tf.tile(input_tensor,multiples=[1,128,1,1])],axis=-1)
    net = deconv('deconv_layer4', inputs=net, output_shape=[batchNum, 128, 8, 512], kernel_size=[1, 8], stride=[1, 1],padding='VALID')
    netfeat512 = deconv('deconv_layer5', inputs=net, output_shape=[batchNum, 512, 8, 128], kernel_size=[32, 1], stride=[4, 1])
    #netfeat512 = conv2d('layer5_conv2d' , net,64,[1,8],padding='VALID')
   #netfeat512 = deconv('deconv_layer5', inputs=net, output_shape=[batchNum, 512, 8, 128], kernel_size=[4, 1], stride=[4, 1],padding='VALID')
    net512 = conv2d('layer6_conv2d',netfeat512,3,[1,8],padding='VALID',activation_func=None)
    
    tf.add_to_collection('o512',net512)

    net = tf.concat([net512,tf.reshape(netfeat512,[-1,512,1,8*128]), tf.tile(input_tensor, multiples=[1, 512, 1, 1])], axis=-1)
    net = deconv('deconv_layer6', inputs=net, output_shape=[batchNum,512 , 16, 512], kernel_size=[1, 16], stride=[1, 1],padding='VALID')
    net = deconv('deconv_layer7', inputs=net, output_shape=[batchNum,2048 , 16, 128], kernel_size=[32, 1], stride=[4, 1])
    #net = deconv('deconv_layer7', inputs=net, output_shape=[batchNum,2048 , 16, 128], kernel_size=[4, 1], stride=[4, 1],padding='VALID')
    net2048 = conv2d('layer7_conv2d', net, 3, [1, 16], padding='VALID',activation_func=None)
    tf.add_to_collection('o2048', net2048)

    return tf.reshape(net128, [-1, 128, 3]),tf.reshape(net512, [-1, 512, 3]),tf.reshape(net2048, [-1, 2048, 3])

def decompress_layer(scope,tensor,mlp,mlp1,mlp2,up_ratio=4):
    with tf.variable_scope(scope):
        new_point=tensor
        new_point_list=[]
        for i in range(up_ratio):
            newpoint=tensor
            for j, num_out_channel0 in enumerate(mlp):
                newpoint = conv2d('convall%d_%d'%(i,j),newpoint,num_out_channel0,[1,1])
            new_point_list.append(newpoint)
        new_point=tf.concat(new_point_list,axis=1)
        new_feat=new_point
        for j, num_out_channel in enumerate(mlp1):
            new_feat = conv2d('convfeat_%d'%j,new_feat, num_out_channel, [1,1]) 
        for k, num_out_channel2 in enumerate(mlp2):
            new_point = conv2d('convpt_%d'%k,new_point,num_out_channel2,[1,1])
        new_point = conv2d('convptout',new_point,3,[1,1],activation_func=None)
        return new_point,new_feat
def decom_ful(scope,tensor,mlp,mlp1,mlp2,up_ratio=4,fea_dim=512):
    with tf.variable_scope(scope):
        rawpt_len=tensor.get_shape()[1].value
        keynum=16
        new_point=tensor
        for i, num_out_channel0 in enumerate(mlp):
            new_point = conv2d('ful_ini%d'%i,new_point,num_out_channel0,[1,1])
        new_feat=new_point
        for j, num_out_channel in enumerate(mlp1):
            new_feat = conv2d('fulfeat_%d'%j,new_feat, num_out_channel, [1,1])
        new_feat = conv2d('fulfeatout',new_feat,keynum*up_ratio,[1,1])
        new_feat = tf.reshape(new_feat,[-1,rawpt_len*up_ratio,1,keynum])
        new_feat=tf.concat([new_feat,tf.tile(tensor,[1,rawpt_len*up_ratio,1,1])],axis=-1)
        for i,num_out_channel in enumerate([128,128]):
            new_feat = conv2d('incfeat_%d'%i,new_feat, num_out_channel, [1,1])
        new_feat = conv2d('incfeatout_%d'%i,new_feat, fea_dim, [1,1])
        for k, num_out_channel2 in enumerate(mlp2):
            new_point = conv2d('fulpt_%d'%k,new_point,num_out_channel2,[1,1])
        new_point = conv2d('convptout',new_point,3*up_ratio,[1,1],activation_func=None)
        new_point=tf.reshape(new_point,[-1,rawpt_len*up_ratio,1,3])
        return new_point,new_feat

def gradual_final1(tensor):
    tensor=tf.cast(tensor,tf.float32)
    featLen = tensor.get_shape()[-1].value
    batchNum = tf.shape(tensor)[0]
    input_tensor = tf.reshape(tensor, [-1, 1, 1, featLen])
    with tf.variable_scope('D128'):
        #net = conv2d('fully_layer1', input_tensor, 512, [1, 1])
        #net = conv2d('fully_layer2', net,512,[1,1])
        #net = conv2d('fully_layer3', net,384,[1,1],activation_func=None)
        #net128 = tf.reshape(net,[BATCH_SIZE,128,1,3])
        net128,netfeat128=decom_ful('decom_layer0',input_tensor,[512,256],[256],[256],128,128)
    tf.add_to_collection('o128', net128)
    with tf.variable_scope('D512'):
        tensor128=tf.concat([net128,netfeat128,tf.tile(input_tensor,multiples=[1,128,1,1])],axis=-1)
        net512,netfeat512=decom_ful('decom_layer1',tensor128,[512,256],[128,128],[128,64])         
    tf.add_to_collection('o512',net512)
    with tf.variable_scope('D2048'):
        tensor512=tf.concat([net512,netfeat512,tf.tile(input_tensor,multiples=[1,512,1,1])],axis=-1)
        net2048,netfeat=decom_ful('decom_layer2',tensor512,[512,256],[128,128],[128,64])
    tf.add_to_collection('o2048', net2048) 
    return tf.reshape(net128, [-1, 128, 3]),tf.reshape(net512, [-1, 512, 3]),tf.reshape(net2048, [-1, 2048, 3])


def gradual_final(tensor,ex_type='d'):
    #tensor=tf.cast(tensor,tf.float32)
    tensor=tf.cast(tensor,tf.float32)
    featLen = tensor.get_shape()[-1].value
    batchNum = tf.shape(tensor)[0]
    input_tensor = tf.reshape(tensor, [-1, 1, 1, featLen])
    with tf.variable_scope('D128'):
        net = deconv('deconv_layer1', inputs=input_tensor, output_shape=[batchNum, 128, 1, 128], kernel_size=[128, 1],stride=[1, 1], padding='VALID')
        #net = deconv('deconv_layer2', inputs=net, output_shape=[batchNum, 32, 8, 128], kernel_size=[16, 1], stride=[4, 1])

        netfeat128 = deconv('deconv_layer3', inputs=net, output_shape=[batchNum, 128, 8, 64], kernel_size=[1,8], stride=[1, 1],padding='VALID')
        net = conv2d('conv2d_layer4', netfeat128,64,[1,8],padding='VALID')

        net=tf.concat([net,tf.tile(input_tensor,multiples=[1,128,1,1])],axis=-1)
        net=conv2d('conv2d_layer5',net,64,[1,1])
        net128=conv2d('o128', net,3,[1,1],activation_func=None)
    tf.add_to_collection('o128', net128)
    with tf.variable_scope('D512'):
        #net=tf.concat([tf.reshape(netfeat128,[BATCH_SIZE,128,1,-1]),tf.tile(input_tensor,multiples=[1,128,1,1])],axis=-1)
        #net=tf.reshape(netfeat128,[BATCH_SIZE,128,1,-1])
        #net=conv2d('conv_layer',net,512,[1,1])
        net=tf.concat([net128,tf.reshape(netfeat128,[BATCH_SIZE,128,1,-1])],axis=-1)

        _,netl=local_net(scope='layer',sample_dim=128,xyz=tf.reshape(net128,[-1,128,3]),featvec=tf.squeeze(net),r_list=[0.2],k_list=[8],layers_list=[[128,128]],use_all=True)
        netl=tf.expand_dims(netl,axis=2)
        net=tf.concat([net,netl],axis=-1)
        
        net = deconv('deconv_layer7', inputs=net, output_shape=[batchNum, 128, 8, 128], kernel_size=[1, 8], stride=[1, 1],padding='VALID')
        netfeat512 = deconv('deconv_layer8', inputs=net, output_shape=[batchNum, 512, 8, 64], kernel_size=[4, 1], stride=[4, 1],padding='VALID')

        #netfeat512 = conv2d('layer5_conv2d' , net,64,[1,8],padding='VALID')
        #netfeat512 = deconv('deconv_layer5', inputs=net, output_shape=[batchNum, 512, 8, 128], kernel_size=[4, 1], stride=[4, 1],padding='VALID')
        net=conv2d('conv2d_layer9',netfeat512,64,[1,8],padding='VALID')
        net=tf.concat([net,tf.tile(input_tensor,multiples=[1,512,1,1])],axis=-1)

        net=conv2d('conv2d_layer10',net,64,[1,1])
        net512 = conv2d('conv2d_layer11',net,3,[1,1],activation_func=None)
    tf.add_to_collection('o512',net512)
    with tf.variable_scope('D2048'):
        #net = tf.concat([tf.reshape(netfeat512,[BATCH_SIZE,512,1,-1]),tf.tile(input_tensor, multiples=[1, 512, 1, 1])], axis=-1)
        #net=conv2d('conv_layer',net,512,[1,1])
        net=tf.concat([net512,tf.reshape(netfeat512,[BATCH_SIZE,512,1,-1])],axis=-1)
        _,netl=local_net(scope='layer',sample_dim=512,xyz=tf.reshape(net512,[-1,512,3]),featvec=tf.squeeze(net),r_list=[0.4],k_list=[32],layers_list=[[128,128]],use_all=True)
        netl=tf.expand_dims(netl,axis=2)
        net=tf.concat([net,netl],axis=-1)
        #net = conv2d('conv2d_layer8',net,128,[1,1])
        net = deconv('deconv_layer12', inputs=net, output_shape=[batchNum,512 , 8, 128], kernel_size=[1, 8], stride=[1, 1],padding='VALID')
        net = deconv('deconv_layer13', inputs=net, output_shape=[batchNum,2048 , 8, 64], kernel_size=[4, 1], stride=[4, 1],padding='VALID')
        #net = deconv('deconv_layer7', inputs=net, output_shape=[batchNum,2048 , 16, 128], kernel_size=[4, 1], stride=[4, 1],padding='VALID')
        net=conv2d('conv2d_layer14',net,64,[1,8],padding='VALID')
        net=tf.concat([net,tf.tile(input_tensor,multiples=[1,2048,1,1])],axis=-1)
        net=conv2d('conv2d_layer15',net,64,[1,1])
        net2048 = conv2d('conv2d_layer16', net, 3, [1, 1],activation_func=None)
    tf.add_to_collection('o2048', net2048) 
    return tf.reshape(net128, [-1, 128, 3]),tf.reshape(net512, [-1, 512, 3]),tf.reshape(net2048, [-1, 2048, 3])


def fn_decode_cell(scope,center,state_tensor,up_ratio=16,mlp=[256,256],mlp_transmat=[64,9],mlp_grid=[128,64,3],\
                   mlp_mask=[128,128],mlpfn0=[64,64],mlpfn1=[64,64],mlpfn2=[64,64],mlp2=[128,128],grid_scale=0.05,state_len=128,reuse=False):
    with tf.variable_scope(scope,reuse=reuse):
        ptnum=state_tensor.get_shape()[1].value
        digit=tf.log(up_ratio*1.0)/tf.log(2.0)
        #center=tf.expand_dims(center,axis=2)
        new_state=tf.concat([center,state_tensor],axis=-1)
        for i,outchannel in enumerate(mlp):
            new_state=conv2d('basic_state%d'%i,new_state,outchannel,[1,1],padding='VALID')

        grid_feat=new_state
        for i,outchannel in enumerate(mlp_grid[:-1]):
            grid_feat=conv2d('basic_grid%d'%i,grid_feat,outchannel,[1,1],padding='VALID')#batch*64*1*11
        raw_grid_feat=conv2d('basic_grid-111',grid_feat,mlp_grid[-1],[1,1],padding='VALID',activation_func=None)#batch*64*1*11
        xylength=tf.square(raw_grid_feat[:,:,:,:2])
        #grid_feat=initpts*tf.tile(xylength,[1,1,up_ratio,1])
        #size_ratio=tf.minimum(tf.nn.relu(tf.expand_dims(raw_grid_feat[:,:,:,2],axis=-1)),1.0)
        size_ratio=tf.tile(tf.constant(0.5,shape=[1,ptnum,1,1]),[tf.shape(grid_feat)[0],1,1,1])
        ##xgrid_digit=tf.floor(tf.slice(grid_feat,[0,0,0,2],[tf.shape(grid_feat)[0],ptnum,1,1]))
        xgrid_digit=digit*size_ratio
        
        xgrid_size=tf.pow(2.0,xgrid_digit)
        #xgrid_size=tf.minimum(tf.maximum(up_ratio*size_ratio,1.0),16.0)
        ygrid_size=up_ratio/xgrid_size#batch*64*1*1
        initlist=tf.constant(list(range(up_ratio)),shape=[1,1,up_ratio,1],dtype=tf.float32)
        grid_sizes=tf.concat([xgrid_size,ygrid_size],axis=-1)
        grid_dis=tf.minimum(grid_sizes-1,1)/tf.maximum(grid_sizes-1,1)#batch*64*1*2,calculate distance between grids
        #grid_feat=grid_dis*rlist#batch*64*up_ratio*2,0~1 grids
        xdis=tf.expand_dims(grid_dis[:,:,:,0],axis=-1) 
        rlist=initlist*xdis
        xrlist=rlist-rlist//(1.0+xdis)
        yrlist=tf.expand_dims(grid_dis[:,:,:,1],axis=-1)*(rlist-xrlist)
        grid_feat=tf.concat([xrlist,yrlist],axis=-1)
        grid_feat=2*grid_feat*xylength-xylength#batch*64*up_ratio*2,-1~1 grids
        
        raw_state=new_state
        new_state=tf.concat([tf.tile(new_state,[1,1,up_ratio,1]),grid_feat],axis=-1)
        points_out=new_state
        points_out=tf.concat([grid_feat,new_state],axis=-1)
        for i,outchannel in enumerate(mlpfn1):
            points_out=conv2d('fn1_points%d'%i,points_out,outchannel,[1,1],padding='VALID')
        points_out=conv2d('fn1_points_out',points_out,3,[1,1],padding='VALID',activation_func=None)

        #points_out=tf.matmul(tf.tile(trans_matrix,[1,1,up_ratio,1,1]),tf.expand_dims(points_out,axis=-1))
        #points_out=tf.squeeze(points_out,axis=-1)
        points_out=tf.concat([points_out,new_state],axis=-1)
        new_state=points_out
        for i,outchannel in enumerate(mlpfn2):
            points_out=conv2d('fn2_points%d'%i,points_out,outchannel,[1,1],padding='VALID')
        points_out=conv2d('fn2_points_out',points_out,3,[1,1],padding='VALID',activation_func=None)
        new_state=tf.concat([new_state,points_out],axis=-1)

        tf.add_to_collection(scope+str(ptnum),points_out)
        points_out=tf.tile(center,[1,1,up_ratio,1])+points_out
        points_out=tf.reshape(points_out,[-1,ptnum*up_ratio,1,3])

        for i,outchannel in enumerate(mlp2):
            new_state=conv2d('state%d'%i,new_state,outchannel,[1,1],padding='VALID')
        new_state=conv2d('state_out%d'%i,new_state,state_len,[1,1],padding='VALID')
        new_state=tf.reshape(new_state,[-1,ptnum*up_ratio,1,state_len])
    return points_out,new_state
def moore_penrose(matrix):
    #print('matrix',matrix)
    d,u,v=tf.linalg.svd(matrix)
    dnum=d.get_shape()[2].value
    d=tf.tile(tf.reshape(tf.eye(dnum),[1,1,dnum,dnum]),[tf.shape(d)[0],tf.shape(d)[1],1,1])*tf.expand_dims(d,axis=-1)
    #print(d,u,v)
    d0=tf.zeros_like(d)
    d1=tf.ones_like(d)
    d2=0.001*d1
    mat1=tf.where(tf.equal(d,d0),d0,d1)
    mat=tf.where(tf.equal(d,d0),d1,d)
    mat=tf.transpose(mat1/mat,[0,1,3,2])
    result=tf.matmul(v,mat)
    result=tf.matmul(result,tf.transpose(u,[0,1,3,2]))
    return result
def rbf_core(dismat,use_type='r'):
    delta=0.1
    if use_type=='r':
        result=1/(tf.sqrt(dismat+delta))
    elif use_type=='e':
        result=tf.exp(-dismat/(2*delta)**2)
    return result
def rbf_solve(inpts,newpts,k=16,points=None,use_xyz=True):
    rbfpts,kfeat=grouping(xyz=inpts,new_xyz=newpts, radius=0.5, nsample=k, points=points, knn=True, use_xyz=True)#batch*128*16*3,batch*128*16*featlen
    #print(rbfpts)
    dismat=tf.reduce_sum(tf.square(tf.expand_dims(rbfpts,axis=-2)-tf.expand_dims(rbfpts,axis=-3)),axis=-1)#batch*128*16*16
    #print(dismat)
    rbfmat=tf.matrix_inverse(rbf_core(dismat,use_type='r'))
    labelmat=tf.ones((tf.shape(dismat)[0],tf.shape(dismat)[1],tf.shape(dismat)[2],1))#batch*128*16*1
    rbfpara=tf.matmul(rbfmat,labelmat)#batch*128*16*1
    rbf=rbf_pred(rbfpts,rbfpara,kfeat)
    return rbf
def rbf_pred(rbfpts,rbfpara,kfeat):
    rbfdis=rbf_core(tf.reduce_sum(tf.square(rbfpts),axis=-1,keepdims=True),use_type='r')#batch*128*16*1
    rbf=tf.reduce_sum(rbfpara*rbfdis*kfeat,axis=2,keepdims=True)#batch*128*1*featlen
    return rbf
#inpts:batch*128*3,infeat:batch*128*featlen
def knn_interpolate(inpts,infeat,rawpts,rawfeats,knum=8,use_xyz=False,use_type='r'):
    kpts,kfeat=grouping(xyz=rawpts,new_xyz=inpts, radius=0.5, nsample=knum, points=rawfeats, knn=True, use_xyz=True)#batch*128*16*3,batch*128*16*featlen
    if use_xyz:
        kpoints=kpts
        inpoint=inpts
    else:
        kpoints=tf.concat([kpts,kfeat],axis=-1)
        inpoint=tf.concat([inpts,infeat],axis=-1)
    dismat=tf.reduce_sum(tf.square(tf.expand_dims(inpoint,axis=2)-kpoints),axis=-1,keepdims=True)

    if use_type=='r':
        delta=0.01
        ratio=1/(delta+dismat)
        ratio=ratio/tf.reduce_sum(ratio,axis=2,keepdims=True)
        #newfeat=tf.reduce_sum(ratio*kfeat,axis=2,keepdims=True)
    else:
        delta=1.0
        ratio=tf.exp(-delta*dismat)
        ratio=ratio/tf.reduce_sum(ratio,axis=2,keepdims=True)
    newfeat=tf.reduce_sum(ratio*kfeat,axis=2,keepdims=True)
    return newfeat
def data_pool(name,ptnum,dimnum):
    data=tf.get_variable(name=name,shape=[ptnum,dimnum],initializer=tf.contrib.layers.xavier_initializer())
    return data
#b*n*k*f
def cosine_sim(feat1,feat2):
    result=tf.reduce_sum(feat1*feat2,axis=-1)
    result=result/(tf.sqrt(tf.reduce_sum(tf.square(feat1),axis=-1))*tf.sqrt(tf.reduce_sum(tf.square(feat2),axis=-1)))
    result=tf.abs(1-result)
    return result
def get_topk(rawcode,codepool,knum):
    #print(rawcode,codepool,'>>>>>>>>>>')
    valdist,ptid = tf_grouping.knn_point(knum, codepool, rawcode)#batch*n*k
    bid=tf.tile(tf.reshape(tf.range(start=0,limit=tf.shape(rawcode)[0],dtype=tf.int32),[-1,1,1,1]),[1,tf.shape(rawcode)[1],knum,1])
    idx=tf.concat([bid,tf.expand_dims(ptid,axis=-1)],axis=-1)
    kcode=tf.gather_nd(codepool,idx)#batch*n*k*c
    #kdist=cosine_sim(tf.expand_dims(rawcode,axis=2),kcode)
    kdist=tf.reduce_mean(tf.square(tf.expand_dims(rawcode,axis=2)-kcode),axis=-1)#/////////////////////
    #kdist=tf.sqrt(kdist)
    return kdist,ptid,kcode
#rawcode:batch*n*c
#codepool:N*c
#decayratio:1
def code_transform(rawcode,codepool,decayratio,knum=16,distratio=0.001):
    codepool=tf.tile(tf.expand_dims(codepool,axis=0),[tf.shape(rawcode)[0],1,1])
    kdist,_,kcode=get_topk(rawcode,codepool,knum)#batch*n*k*c
    #kdist=kdist/rawcode.get_shape()[-1].value

    _,idx,code1=get_topk(rawcode,codepool,1)
    tf.add_to_collection('codeidx',idx)
    kmask=tf.exp(-kdist/(1e-8+decayratio))/(1e-8+tf.reduce_sum(tf.exp(-kdist/(1e-8+decayratio)),axis=-1,keepdims=True))

    tf.add_to_collection('kdist',kdist)
    tf.add_to_collection('kmask',kmask)
    kmindist=tf.reduce_mean(tf.reduce_min(kdist,axis=-1))
    tf.add_to_collection('kmindist',kmindist)
    #kmask=tf.exp(-kdist*decayratio)/(tf.reduce_sum(tf.exp(-kdist*decayratio),axis=-1,keepdims=True))
    #kmask=tf.exp(-kdist)
    kmask=tf.expand_dims(kmask,axis=-1)
    #print('***************',kmask,kcode)
    kcode=tf.reduce_sum(kmask*kcode,axis=-2,keepdims=True)
    code=tf.stop_gradient(code1-kcode)+kcode
    #code=kcode
    code=tf.squeeze(code,[2])
    return code
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
    #print('************',code,codemax)
    code=tf.concat([startcode,tf.tile(codemax,[1,tf.shape(code)[1],1,1])],axis=-1)

    for i, num_out_channel in enumerate(mlp2):
        code = conv2d('recover_%d'%i,code, num_out_channel, [1,1],activation_func=tf.nn.relu)
    code=conv2d('recover_out',code,ptslen+featlen, [1,1], activation_func=None)
    code=tf.squeeze(code,[2])
    ptsmove=code[:,:,:ptslen]
    featmove=code[:,:,ptslen:]
    #newpts,newfeat=ptsmove,tf.abs(featmove)
    
    newpts=ptscode+ptsmove
    newfeat=featcode+featmove
    return newpts,newfeat
def get_pools(featnum,featlen,num):
    #ptspool=data_pool('ptspool',ptnum,3)
    featpool=data_pool('featspool'+str(num),featnum,featlen)
    #tf.add_to_collection('ptspool',ptspool)
    tf.add_to_collection('featspool',featpool)
    #featpool=tf.square(featpool)
    return featpool
def ubloss(pool1,pool2):
    dismat=tf.expand_dims(pool1,axis=0)-tf.expand_dims(pool2,axis=1)#b2*b1*f
    dismat=tf.reduce_mean(tf.reduce_min(tf.reduce_mean(tf.square(dismat),axis=-1),axis=1))
    return dismat
def word2code(featsin,num=1,poolen=256,ptsdecay=None,featsdecay=None):
    #ptnum=ptsin.get_shape()[1].value
    #featnum=featsin.get_shape()[1].value
    featlen=featsin.get_shape()[-1].value
    featpool=get_pools(poolen,featlen,num)
    ubloss1=ubloss(featpool[:128],featpool)
    ubloss2=ubloss(featpool[:64],featpool)
    tf.add_to_collection('ubloss',ubloss1+ubloss2)
    #if ptsdecay is None:
    #    ptsdecay=tf.square(tf.get_variable(name='ptsdecay',shape=[1],initializer=tf.constant_initializer(1.0)))

    if featsdecay is None:
        featsdecay=tf.square(tf.get_variable(name='featsdecay'+str(num),shape=[1],initializer=tf.constant_initializer(5.0)))
    
    #tf.add_to_collection('decays',ptsdecay)
    tf.add_to_collection('decays',featsdecay)
    #ptscode=code_transform(ptsin,ptspool,ptsdecay,knum=128,distratio=1)
    featscode=code_transform(featsin,featpool,featsdecay,knum=poolen//2,distratio=1)
    return featscode,featpool
def code2word(scope,ptscode,featscode,featpool):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        newpts,newfeat=shape_recover(ptscode,featscode)
        #newpts,newfeat=ptscode,featscode
    return newpts,newfeat
def get_repulsion_loss4(pred, nsample=20, radius=0.07):
    # pred: (batch_size, npoint,3)
    #idx, pts_cnt = query_ball_point(radius, nsample, pred, pred)
    #tf.summary.histogram('smooth/unque_index', pts_cnt)

   #grouped_pred = group_point(pred, idx)  # (batch_size, npoint, nsample, 3)
    #grouped_pred -= tf.expand_dims(pred, 2)

    ##get the uniform loss
    h = 0.03
    pred=tf.expand_dims(pred,axis=0)
    dist_square,_,_ = get_topk(pred,pred,5)
    #dist_square = tf.reduce_sum(grouped_pred ** 2, axis=-1)
    #dist_square, idx = tf.nn.top_k(-dist_square, 5)
    dist_square = dist_square[:, :, 1:]  # remove the first one
    #dist_square = tf.maximum(1e-12, dist_square)
    dist = tf.sqrt(dist_square+1e-12)
    uniform_loss = tf.reduce_mean(1/(0.2+dist))
    #weight = tf.exp(-dist_square / h ** 2)
    #uniform_loss = tf.reduce_mean( - dist * weight)
    return uniform_loss
def uniform_loss(pred,knum=16,mytype='n'):
    cens=sampling(512,pred,'f')[-1]
    dist_square,_,_ = get_topk(cens,pred,64)
    dist_square = dist_square[:, :, 1:]#batch*n*4
    dist=tf.sqrt(dist_square)
    #print(dist)
    #assert False
    mdist=tf.reduce_min(dist,axis=[-1,-2],keepdims=True)
    if mytype=='n':
        result=tf.reduce_mean(tf.reduce_max(tf.reduce_mean(dist,axis=-1),axis=-1))
    else:
        result=tf.reduce_mean(tf.reduce_max(tf.reduce_sum(tf.square(dist-mdist),axis=-1),axis=-1))
    return result
#ptsin:b*n*1*3,b*n*1*flen
def fc_block(ptsin,featsin,featlen=128,mlp=[256,256,256],mlp2=[256,256,256],up_ratio=4):
    ptnum=ptsin.get_shape()[1].value
    words=tf.concat([ptsin,featsin],axis=-1)
    for i,outchannel in enumerate(mlp):
        words=conv2d('fcstate%d'%i,words,outchannel,[1,1],padding='VALID')
    words=conv2d('fcpts',words,3*up_ratio,[1,1],padding='VALID',activation_func=None)
    move=tf.reshape(words,[-1,ptnum,up_ratio,3])
    ptsout=ptsin+move

    move_regular=tf.reduce_mean(tf.reduce_max(tf.reduce_sum(tf.square(move),axis=-1),axis=1))
    tf.add_to_collection('expand_dis',move_regular)

    ptsout=tf.reshape(words,[-1,ptnum*up_ratio,1,3])

    for i,outchannel in enumerate(mlp):
        words=conv2d('fcstate2%d'%i,words,outchannel,[1,1],padding='VALID')
    words=conv2d('fcfeat',words,featlen*up_ratio,[1,1],padding='VALID',activation_func=tf.nn.relu)
    feats=tf.reshape(words,[-1,ptnum*up_ratio,1,featlen])
    return ptsout,feats
def fd_block(ptsin,featsin,featlen=128,mlp=[256,256,256],mlp2=[256,256,256],up_ratio=4):
    bnum=tf.shape(ptsin)[0]
    ptnum=ptsin.get_shape()[1].value
    grid_size=int(sqrt(up_ratio))
    words=tf.concat([ptsin,featsin],axis=-1)
    grid_feat=-1+2*tf.tile(tf.reshape(tf.linspace(0.0,1.0,grid_size),[1,1,-1,1]),[bnum,ptnum,1,2])#batch*ptnum*2*2
    #zgrid_feat=-1+2*tf.tile(tf.reshape(tf.linspace(0.0,1.0,layernum),[1,1,-1,1]),[tf.shape(raw_grid_feat)[0],ptnum,1,1])#batch*64*4*1

    xgrid_feat=tf.expand_dims(grid_feat[:,:,:,0],axis=-1)
    ygrid_feat=tf.expand_dims(grid_feat[:,:,:,1],axis=-1)
    grid_feat=tf.concat([tf.tile(xgrid_feat,[1,1,grid_size,1]),tf.reshape(tf.tile(ygrid_feat,[1,1,1,grid_size]),[-1,ptnum,grid_size*grid_size,1])],axis=-1)#batch*ptnum*4*2
    words=tf.concat([tf.tile(words,[1,1,up_ratio,1]),grid_feat],axis=-1)

    for i,outchannel in enumerate(mlp):
        words=conv2d('fdstate%d'%i,words,outchannel,[1,1],padding='VALID')
    words=conv2d('fdpts',words,3,[1,1],padding='VALID',activation_func=None)
    move=tf.reshape(words,[-1,ptnum,up_ratio,3])
    ptsout=ptsin+move
    ptsout=tf.reshape(words,[-1,ptnum*up_ratio,1,3])

    move_regular=tf.reduce_mean(tf.reduce_max(tf.reduce_sum(tf.square(move),axis=-1),axis=1))
    tf.add_to_collection('expand_dis',move_regular)

    for i,outchannel in enumerate(mlp2):
        words=conv2d('fdstate2%d'%i,words,outchannel,[1,1],padding='VALID')
    words=conv2d('fdstate',words,featlen,[1,1],padding='VALID',activation_func=tf.nn.relu)
    feats=tf.reshape(words,[-1,ptnum*up_ratio,1,featlen])

    return ptsout,feats
def exdeconv_block(ptsin,featsin,featlen=128,up_ratio=4,spacelen=4):
    #with tf.variable_scope('Dinc'):
    #featlen=featsin.get_shape()[-1].value
    shortlen=int(featlen/spacelen)
    batchNum=tf.shape(ptsin)[0]
    rawnum=featsin.get_shape()[1].value
    newnum=up_ratio*rawnum
    net=tf.concat([ptsin,featsin],axis=-1)
    #net1=net
    #net=tf.concat([tf.reshape(featsin,[-1,rawnum,1,featlen])],axis=-1)
    net = deconv('deconv_layer1',inputs=net, output_shape=[batchNum, rawnum, spacelen, featlen], kernel_size=[1, spacelen], stride=[1, 1],padding='VALID')
    #net=net+featsin
    netfeat = deconv('deconv_layer2', inputs=net, output_shape=[batchNum, newnum, spacelen, shortlen], kernel_size=[up_ratio, 1], stride=[up_ratio, 1],padding='VALID',activation_func=None)
    #netadd=tf.reshape(tf.tile(featsin,[1,1,up_ratio,1]),[-1,rawnum*up_ratio,1,featlen])
    #netfeat = netadd+netfeat
    net=conv2d('conv2d_layer3',netfeat,64,[1,spacelen],padding='VALID')
    #net1=conv2d('conv2d_zoom',netfeat,64,[1,spacelen],padding='VALID')

    net=conv2d('conv2d_layer4',net,64,[1,1],padding='VALID')
    net=conv2d('conv2d_layer5',net,64,[1,1],padding='VALID')
    
    #for i,outchannel in enumerate([64,64]):
    #    net1=conv2d('zoom_layer%d'%i,net1,outchannel,[1,1],padding='VALID')
    #zoom=tf.square(conv2d('conv2d_layer6',net1,3,[1,1],activation_func=None))
    #zoom=tf.reshape(tf.tile(zoom,[1,1,up_ratio,1]),[-1,rawnum*up_ratio,1,3])
    #print('..............',zoom,net1)
    #print(ptsin)
    lastnet=tf.reshape(tf.tile(ptsin,[1,1,up_ratio,1]),[-1,up_ratio*ptsin.get_shape()[1].value,1,3])
    netfeat=tf.reshape(netfeat,[-1,newnum,1,featlen])+tf.reshape(tf.tile(featsin,[1,1,up_ratio,1]),[-1,up_ratio*featsin.get_shape()[1].value,1,featlen])
    move = conv2d('conv2d_layer6',net,3,[1,1],activation_func=None)
    move_regular=tf.reduce_mean(tf.reduce_max(tf.reduce_sum(tf.square(move),axis=-1),axis=1))
    tf.add_to_collection('expand_dis',move_regular)
    newpts=move+lastnet
    return newpts, netfeat
def dcom_block(ptsin,featsin,featlen=128,up_ratio=4):
    mlp=[256,256]
    #mlp.append(featlen)

    mlp2=[64,64,64]

    shortlen=int(featlen/up_ratio)
    ptnum=featsin.get_shape()[1].value
    newnum=up_ratio*ptnum
    infeats=tf.reshape(featsin,[-1,ptnum,up_ratio,shortlen])

    infeats=tf.concat([infeats,tf.tile(ptsin,[1,1,up_ratio,1])],axis=-1)
    for i,outchannel in enumerate(mlp):
        infeats=conv2d('dcomin%d'%i,infeats,outchannel,[1,1],padding='VALID')
    allfeat=tf.reduce_sum(infeats,axis=[1,2],keepdims=True)/PT_NUM
    #allfeat=tf.reduce_max(infeats,axis=[1,2],keepdims=True)
    infeats=tf.concat([infeats,tf.tile(allfeat,[1,ptnum,up_ratio,1])],axis=-1)


    netfeat=infeats
    for i,outchannel in enumerate(mlp):
        netfeat=conv2d('dcomstate%d'%i,netfeat,outchannel,[1,1],padding='VALID')
    netfeat=conv2d('dcomfeatout',netfeat,featlen,[1,1],padding='VALID',activation_func=tf.nn.leaky_relu)

    netpts=infeats
    #print(netpts,shortlen)
    for i,outchannel in enumerate(mlp2):
        netpts=conv2d('dcomptsfeat%d'%i,netpts,outchannel,[1,1],padding='VALID')
    move=conv2d('dcomptsout',netpts,3,[1,1],padding='VALID',activation_func=None)

    #lastnet=tf.reshape(tf.tile(ptsin,[1,1,up_ratio,1]),[-1,up_ratio*ptsin.get_shape()[1].value,1,3])
    netfeat=tf.reshape(netfeat,[-1,newnum,1,featlen])+tf.reshape(tf.tile(featsin,[1,1,up_ratio,1]),[-1,up_ratio*ptnum,1,featlen])
    move_regular=tf.reduce_mean(tf.reduce_max(tf.reduce_sum(tf.square(move),axis=-1),axis=1))
    tf.add_to_collection('expand_dis',move_regular)
    newpts=move+ptsin
    newpts=tf.reshape(newpts,[-1,up_ratio*ptsin.get_shape()[1].value,1,3])
    #print(newpts,netfeat)
    return newpts, netfeat
def ref_block(pts,featsin,mlp=[128,128],mlp1=[128,128],up_ratio=4):
    fnum=featsin.get_shape()[1].value
    flen=featsin.get_shape()[-1].value
    feats=tf.reshape(tf.tile(featsin,[1,1,up_ratio,1]),[-1,up_ratio*fnum,1,flen])
    infeats=tf.concat([pts,feats],axis=-1)
    for i,outchannel in enumerate(mlp):
        infeats=conv2d('ref_feats%d'%i,infeats,outchannel,[1,1],padding='VALID')
    allfeat=tf.reduce_sum(infeats,axis=[1,2],keepdims=True)/PT_NUM
    #infeats=tf.concat([infeats,tf.tile(allfeat,[1,ptnum,up_ratio,1])],axis=-1)

    netpts=tf.concat([pts,tf.tile(allfeat,[1,pts.get_shape()[1].value,1,1])],axis=-1)
    for i,outchannel in enumerate(mlp1):
        netpts=conv2d('ref_outfeats%d'%i,netpts,outchannel,[1,1],padding='VALID')
    move=conv2d('ref_out',netpts,3,[1,1],padding='VALID',activation_func=None)
    move_regular=0.01*tf.reduce_mean(tf.reduce_max(tf.reduce_sum(tf.square(move),axis=-1),axis=1))
    tf.add_to_collection('expand_dis',move_regular)
    newpts=move+pts
    return newpts

def progress_block(pts0,feats0,cirnum,featlen=256):
    pts=tf.expand_dims(pts0,axis=2)
    feats=tf.expand_dims(feats0,axis=2)
    ptlist=[]
    #ptlist.append(pts0)
    featlist=[]
    for i in range(cirnum):
        with tf.variable_scope('Dinc',reuse=tf.AUTO_REUSE):
            #feats=tf.concat([pts,feats],axis=-1)
            #pts,feats=exdeconv_block(pts,feats,featlen=featlen,up_ratio=4,spacelen=4)
            infeats=feats
            pts,feats=dcom_block(pts,feats,featlen=featlen,up_ratio=4)
            #opts=ref_block(pts,infeats)
            ptlist.append(tf.squeeze(pts,axis=2))
            featlist.append(tf.reshape(feats,[-1,feats.get_shape()[1].value,featlen]))
    return ptlist,featlist
def progress_decode(cen3,feat3,cirnum=3):
    pt3list,feat3list=progress_block(cen3,feat3,cirnum)
    return pt3list
#def progress_decode(cenlist,featlist0):
#    cen3,cen2,cen1=cenlist
#    feat3,feat2,feat1=featlist0
#    pt3list,feat3list=progress_block(cen3,feat3,3)
#    pt2list,feat2list=progress_block(cen2,feat2,2)
#    pt1list,feat1list=progress_block(cen1,feat1,1)
#    pt11,feat11=pt1list[0],feat1list[0]
#    pt21,pt22=pt2list
#    feat21,feat22=feat2list
#    pt31,pt32,pt33=pt3list
#    feat31,feat32,feat33=feat3list
#    ptlist=[pt11,pt21,pt22,pt31,pt32,pt33]
#    featlist=[feat11,feat21,feat22,feat31,feat32,feat33]
#    return ptlist,featlist
def emd_feat(cen2,feat2,cen1,feat1):
    bnum=tf.shape(cen1)[0]
    npoint=tf.shape(cen1)[1]
    ptemd,idxl=emd_id(cen1,cen2)
    
    bid=tf.tile(tf.reshape(tf.range(start=0,limit=bnum,dtype=tf.int32),[-1,1,1]),[1,npoint,1])
    idx=tf.concat([bid,tf.expand_dims(idxl,axis=-1)],axis=-1)
    newfeat=tf.gather_nd(feat1,idx)#batch*npoint*featlen
    featdist=tf.reduce_sum((feat1 - newfeat) ** 2,axis=-1)
    dist = tf.reduce_mean(featdist)
    loss=ptemd#+dist
    return loss
def progress_loss(rawpts,ptlist,featlist,cenlist,featlist0):
    cen3,cen2,cen1=cenlist
    feat3,feat2,feat1=featlist0
    pt11,pt21,pt22,pt31,pt32,pt33=ptlist
    feat11,feat21,feat22,feat31,feat32,feat33=featlist
    print(pt11,pt22,pt33)
    ls11,ls22,ls33=emd_id(pt11,rawpts)[0],emd_id(pt22,rawpts)[0],emd_id(pt33,rawpts)[0]
    print('*************************')
    ls31,ls32,ls21=emd_feat(cen2,feat2,pt31,feat31),emd_feat(cen1,feat1,pt32,feat32),emd_feat(cen1,feat1,pt21,feat21)
    result=ls11+ls22+ls33
    return result
def emd_id(pred,gt):
    batch_size = pred.get_shape()[0].value
    matchl_out, matchr_out = tf_auctionmatch.auction_match(pred, gt)
    matched_out = tf_sampling.gather_point(gt, matchl_out)
    dist = tf.sqrt(tf.reduce_sum((pred - matched_out) ** 2,axis=-1))
    dist = tf.reduce_mean(dist,axis=-1)

    #cens=tf.reduce_mean(pred,axis=1,keep_dims=True)
    #radius=tf.sqrt(tf.reduce_max(tf.reduce_sum((pred - cens) ** 2,axis=-1),axis=-1))
    #dist_norm = dist / radius

    emd_loss = tf.reduce_mean(dist)
    return emd_loss,matchl_out
#def refine_layer(points,tensor,feats,use_knn=False)
#startpts:batch*32*3
#tensor:batch*codelen
def gradual_final0(net32,netfeat32,startpts=None):
    tensor=netfeat32
    net32=tf.expand_dims(net32,axis=2)
    tensor=tf.cast(tensor,tf.float32)
    featlen = tensor.get_shape()[-1].value
    shortlen= int(featlen/8)
    batchNum = tf.shape(tensor)[0]
    #input_tensor = tf.reshape(tensor, [-1, 1, 1, featLen])
    move=0
    #print(input_tensor)
    #if startpts is None:
    #    with tf.variable_scope('D128'):
    #        net32,netfeat32=decom_ful('init_layer',input_tensor,mlp=[256,256],mlp1=[128],mlp2=[128],up_ratio=32,fea_dim=512)
    #        tf.add_to_collection('o32', net32)
    #else:
    #    startpts=tf.expand_dims(startpts,axis=2)
    #    word=tf.concat([startpts,tf.tile(input_tensor,[1,tf.shape(startpts)[1],1,1])],axis=-1)
    #    with tf.variable_scope('D32'):
    #        for i,outchannel in enumerate([128,64]):
    #            word=conv2d('ptstate%d'%i,word,outchannel,[1,1],padding='VALID')
    #        move=conv2d('pt_out%d'%i,word,3,[1,1],padding='VALID',activation_func=None)
    #        net32=startpts+move
    #        word=tf.concat([net32,tf.tile(input_tensor,[1,tf.shape(startpts)[1],1,1])],axis=-1)
    #        wordlen=word.get_shape()[-1].value
    #        for i,outchannel in enumerate([wordlen,wordlen]):
    #            word=conv2d('state%d'%i,word,outchannel,[1,1],padding='VALID')
    #        netfeat32=conv2d('state_out%d'%i,word,512,[1,1],padding='VALID')
        #net32=startpts
    #print(netfeat32)
    input_tensor=tf.reduce_max(tf.expand_dims(netfeat32,axis=2),axis=1,keepdims=True)
    with tf.variable_scope('Dinc'):
        net=tf.concat([tf.reshape(netfeat32,[-1,32,1,featlen])],axis=-1)
        #net=conv2d('conv_layer',net,128,[1,1])
        #knet=knn_interpolate(tf.squeeze(net32,axis=2),tf.squeeze(net,axis=2),tf.squeeze(net32,axis=2),tf.squeeze(net,axis=2),knum=8,use_xyz=False,use_type='r')
        #net=tf.concat([net,knet],axis=-1)
        #net=tf.concat([net32,net],axis=-1)
        net = deconv('deconv_layer1', inputs=net, output_shape=[batchNum, 32, 8, featlen], kernel_size=[1, 8], stride=[1, 1],padding='VALID')
        netfeat128 = deconv('deconv_layer2', inputs=net, output_shape=[batchNum, 128, 8, shortlen], kernel_size=[4, 1], stride=[4, 1],padding='VALID')
        net=conv2d('conv2d_layer3',netfeat128,64,[1,8],padding='VALID')
        #netfeat128=conv2d('conv2d_layer4',netfeat128,128,[1,1])

        #net=conv2d('conv2d_layer4',net,64,[1,1])
        #net=conv2d('conv2d_layer5',net,64,[1,1])
        lastnet=tf.reshape(tf.tile(net32,[1,1,4,1]),[-1,4*net32.get_shape()[1].value,1,3])
        net128 = conv2d('conv2d_layer6',net,3,[1,1],activation_func=None)
        net128=net128+lastnet
        move=net128
        move_regular=tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(move),axis=-1)),axis=1))
        tf.add_to_collection('expand_dis',move_regular)
        #input_tensor=tf.reduce_max(netfeat32,axis=1,keepdims=True)
        word=tf.concat([net128,tf.tile(input_tensor,[1,tf.shape(net128)[1],1,1])],axis=-1)
        #for i,outchannel in enumerate([128,64]):
        #    word=conv2d('ptstate%d'%i,word,outchannel,[1,1],padding='VALID')
        #net128=net128+conv2d('pt_out%d'%i,word,3,[1,1],padding='VALID',activation_func=None)
        #netfeat128=rbf_solve(tf.squeeze(net32,[2]),tf.squeeze(net128,[2]),k=8,points=tf.squeeze(netfeat32,[2]),use_xyz=True)
        #rbfpts=grouping(xyz=net32,new_xyz=net128, radius=0.5, nsample=16, points=None, knn=True, use_xyz=False)#batch*128*16*3
        #dismat=tf.reduce_sum(tf.square(tf.expand_dims(rbfpts,axis=-2)-tf.expand_dims(rbfpts,axis=-3)),axis=-1)#batch*128*16*16
        
        
    tf.add_to_collection('o128',net128)
    with tf.variable_scope('Dinc',reuse=True):
        net=tf.concat([tf.reshape(netfeat128,[-1,128,1,featlen])],axis=-1)
        print('net',net)
        #knet=knn_interpolate(tf.squeeze(net128,axis=2),tf.squeeze(net,axis=2),tf.squeeze(net128,axis=2),tf.squeeze(net,axis=2),knum=8,use_xyz=False,use_type='r')
        #net=tf.concat([net,knet],axis=-1)
        #net=conv2d('conv_layer',net,128,[1,1])
        #net=tf.concat([net128,net],axis=-1)
        net = deconv('deconv_layer1', inputs=net, output_shape=[batchNum, 128, 8, featlen], kernel_size=[1, 8], stride=[1, 1],padding='VALID')
        netfeat512 = deconv('deconv_layer2', inputs=net, output_shape=[batchNum, 512, 8, shortlen], kernel_size=[4, 1], stride=[4, 1],padding='VALID')
        
        #net=conv2d('conv2d_layer4',netfeat512,64,[1,1])
        #net=conv2d('conv2d_layer5',net,64,[1,1])
        net=conv2d('conv2d_layer3',netfeat512,64,[1,8],padding='VALID')
        #netfeat512=conv2d('conv2d_layer4',netfeat512,128,[1,1])
                
        #net=conv2d('conv2d_layer4',net,64,[1,1])
        #net=conv2d('conv2d_layer5',net,64,[1,1])
        lastnet=tf.reshape(tf.tile(net128,[1,1,4,1]),[-1,4*net128.get_shape()[1].value,1,3])
        net512 = conv2d('conv2d_layer6',net,3,[1,1],activation_func=None)
        net512=net512+lastnet
        word=tf.concat([net512,tf.tile(input_tensor,[1,tf.shape(net512)[1],1,1])],axis=-1)
        #for i,outchannel in enumerate([128,64]):
        #    word=conv2d('ptstate%d'%i,word,outchannel,[1,1],padding='VALID')
        #net512=net512+conv2d('pt_out%d'%i,word,3,[1,1],padding='VALID',activation_func=None)
        #netfeat512=rbf_solve(tf.squeeze(net128,[2]),tf.squeeze(net512,[2]),k=8,points=tf.squeeze(netfeat128,[2]),use_xyz=True)
    tf.add_to_collection('o512',net512)
    with tf.variable_scope('Dinc',reuse=True):
        net = tf.concat([tf.reshape(netfeat512,[-1,512,1,featlen])], axis=-1)
        #knet=knn_interpolate(tf.squeeze(net512,axis=2),tf.squeeze(net,axis=2),tf.squeeze(net512,axis=2),tf.squeeze(net,axis=2),knum=8,use_xyz=False,use_type='r')
        #net=tf.concat([net,knet],axis=-1)
        #net=conv2d('conv_layer',net,128,[1,1])
        #net=tf.concat([net512,net],axis=-1)
        net = deconv('deconv_layer1', inputs=net, output_shape=[batchNum,512 , 8, featlen], kernel_size=[1, 8], stride=[1, 1],padding='VALID')
        net = deconv('deconv_layer2', inputs=net, output_shape=[batchNum,2048 , 8, shortlen], kernel_size=[4, 1], stride=[4, 1],padding='VALID')
        #net=conv2d('conv2d_layer4',net,64,[1,1])
        #net=conv2d('conv2d_layer5',net,64,[1,1])
        net=conv2d('conv2d_layer3',net,64,[1,8],padding='VALID')
        
        #net=conv2d('conv2d_layer4',net,64,[1,1])
        #net=conv2d('conv2d_layer5',net,64,[1,1])
        lastnet=tf.reshape(tf.tile(net512,[1,1,4,1]),[-1,4*net512.get_shape()[1].value,1,3])
        net2048 = conv2d('conv2d_layer6', net, 3, [1, 1],activation_func=None)
        net2048=net2048+lastnet
        word=tf.concat([net2048,tf.tile(input_tensor,[1,tf.shape(net2048)[1],1,1])],axis=-1)
        #for i,outchannel in enumerate([128,64]):
        #    word=conv2d('ptstate%d'%i,word,outchannel,[1,1],padding='VALID')
        #net2048=net2048+conv2d('pt_out%d'%i,word,3,[1,1],padding='VALID',activation_func=None)
    tf.add_to_collection('o2048', net2048)

    return tf.reshape(net128, [-1, 128, 3]),tf.reshape(net512, [-1, 512, 3]),tf.reshape(net2048, [-1, 2048, 3])
def gradual_test(tensor):
    tensor=tf.cast(tensor,tf.float32)
    featLen = tensor.get_shape()[-1].value
    batchNum = tf.shape(tensor)[0]
    input_tensor = tf.reshape(tensor, [-1, 1, 1, featLen])
    #print(input_tensor)
    with tf.variable_scope('D128'):
        net32,netfeat32=decom_ful('init_layer',input_tensor,mlp=[256,256],mlp1=[128],mlp2=[128],up_ratio=32,fea_dim=512)
        tf.add_to_collection('o32', net32)
    #print(netfeat128)
    with tf.variable_scope('Dinc'):
        net=tf.concat([tf.reshape(netfeat32,[-1,32,1,512]),tf.tile(input_tensor,multiples=[1,32,1,1])],axis=-1)
        #net=conv2d('conv_layer',net,512,[1,1])
        #net=tf.concat([net32,net],axis=-1)
        net = deconv('deconv_layer1', inputs=net, output_shape=[batchNum, 32, 8, 128], kernel_size=[1, 8], stride=[1, 1],padding='VALID')
        netfeat128 = deconv('deconv_layer2', inputs=net, output_shape=[batchNum, 128, 8, 64], kernel_size=[4, 1], stride=[4, 1],padding='VALID')
        net=conv2d('conv2d_layer3',netfeat128,64,[1,8],padding='VALID')

        #net=conv2d('conv2d_layer4',net,64,[1,1])
        #net=conv2d('conv2d_layer5',net,64,[1,1])
        net128 = conv2d('conv2d_layer6',net,3,[1,1],activation_func=None)
    tf.add_to_collection('o128',net128)
    with tf.variable_scope('Dinc',reuse=True):
        net=tf.concat([tf.reshape(netfeat128,[-1,128,1,512]),tf.tile(input_tensor,multiples=[1,128,1,1])],axis=-1)
        #net=conv2d('conv_layer',net,512,[1,1])
        #net=tf.concat([net128,net],axis=-1)
        net = deconv('deconv_layer1', inputs=net, output_shape=[batchNum, 128, 8, 128], kernel_size=[1, 8], stride=[1, 1],padding='VALID')
        netfeat512 = deconv('deconv_layer2', inputs=net, output_shape=[batchNum, 512, 8, 64], kernel_size=[4, 1], stride=[4, 1],padding='VALID')
        net=conv2d('conv2d_layer3',netfeat512,64,[1,8],padding='VALID')

        #net=conv2d('conv2d_layer4',net,64,[1,1])
        #net=conv2d('conv2d_layer5',net,64,[1,1])
        net512 = conv2d('conv2d_layer6',net,3,[1,1],activation_func=None)
    tf.add_to_collection('o512',net512)
    with tf.variable_scope('Dinc',reuse=True):
        net = tf.concat([tf.reshape(netfeat512,[-1,512,1,512]),tf.tile(input_tensor, multiples=[1, 512, 1, 1])], axis=-1)
        #net=conv2d('conv_layer',net,512,[1,1])
        #net=tf.concat([net512,net],axis=-1)
        net = deconv('deconv_layer1', inputs=net, output_shape=[batchNum,512 , 8, 128], kernel_size=[1, 8], stride=[1, 1],padding='VALID')
        netfeat2048 = deconv('deconv_layer2', inputs=net, output_shape=[batchNum,2048 , 8, 64], kernel_size=[4, 1], stride=[4, 1],padding='VALID')
        net=conv2d('conv2d_layer3',netfeat2048,64,[1,8],padding='VALID')

        #net=conv2d('conv2d_layer4',net,64,[1,1])
        #net=conv2d('conv2d_layer5',net,64,[1,1])
        net2048 = conv2d('conv2d_layer6', net, 3, [1, 1],activation_func=None)
    tf.add_to_collection('o2048', net2048)
    with tf.variable_scope('Dinc',reuse=True):
        net = tf.concat([tf.reshape(netfeat2048,[-1,2048,1,512]),tf.tile(input_tensor, multiples=[1, 2048, 1, 1])], axis=-1)
        #net=conv2d('conv_layer',net,512,[1,1])
        #net=tf.concat([net2048,net],axis=-1)
        net = deconv('deconv_layer1', inputs=net, output_shape=[batchNum,2048, 8, 128], kernel_size=[1, 8], stride=[1, 1],padding='VALID')
        net = deconv('deconv_layer2', inputs=net, output_shape=[batchNum,8192 , 8, 64], kernel_size=[4, 1], stride=[4, 1],padding='VALID')
        #net=conv2d('conv2d_layer3',net,64,[1,8],padding='VALID')

        #net=conv2d('conv2d_layer4',net,64,[1,1])
        #net=conv2d('conv2d_layer5',net,64,[1,1])
        net8192 = conv2d('conv2d_layer6', net, 3, [1, 1],activation_func=None)
    #tf.add_to_collection('o2048', net2048)
    return net32,net128,net512,net8192
#points2,state=fn_decode_cell('decode_cell',None,net32,netfeat32,up_ratio=4,state_len=128,reuse=False)
def gradual_fn_final0(net32,netfeat32):
    tensor=netfeat32
    tensor=tf.cast(tensor,tf.float32)
    featLen = tensor.get_shape()[-1].value
    batchNum = tf.shape(tensor)[0]
    net32=tf.expand_dims(net32,axis=2)
    netfeat32=tf.expand_dims(netfeat32,axis=2)
    #input_tensor = tf.reshape(tensor, [-1, 1, 1, featLen])
    #print(input_tensor)
    #with tf.variable_scope('D128'):
    #    net32,netfeat32=decom_ful('init_layer',input_tensor,mlp=[256,256],mlp1=[128],mlp2=[128],up_ratio=32,fea_dim=512)
    #    tf.add_to_collection('o32', net32)
    
    with tf.variable_scope('Dinc'):
        net128,netfeat128=fn_decode_cell('decode_cell',net32,netfeat32,up_ratio=4,state_len=512,reuse=False)
        tf.add_to_collection('o128',net128)
    with tf.variable_scope('Dinc',reuse=True):
        net512,netfeat512=fn_decode_cell('decode_cell',net128,netfeat128,up_ratio=4,state_len=512,reuse=True)
        tf.add_to_collection('o512',net512)
    with tf.variable_scope('Dinc',reuse=True):
        net2048,netfeat2048=fn_decode_cell('decode_cell',net512,netfeat512,up_ratio=4,state_len=512,reuse=True)
        tf.add_to_collection('o2048', net2048)
    return tf.reshape(net32, [-1, 32, 3]),tf.reshape(net128, [-1, 128, 3]),tf.reshape(net512, [-1, 512, 3]),tf.reshape(net2048, [-1, 2048, 3])
def gradual_fn_test(tensor):
    tensor=tf.cast(tensor,tf.float32)
    featLen = tensor.get_shape()[-1].value
    batchNum = tf.shape(tensor)[0]
    input_tensor = tf.reshape(tensor, [-1, 1, 1, featLen])
    #print(input_tensor)
    with tf.variable_scope('D128'):
        net32,netfeat32=decom_ful('init_layer',input_tensor,mlp=[256,256],mlp1=[128],mlp2=[128],up_ratio=32,fea_dim=512)
        tf.add_to_collection('o32', net32)
    with tf.variable_scope('Dinc'):
        net128,netfeat128=fn_decode_cell('decode_cell',net32,netfeat32,up_ratio=4,state_len=512,reuse=False)
        tf.add_to_collection('o128',net128)
    with tf.variable_scope('Dinc',reuse=True):
        net512,netfeat512=fn_decode_cell('decode_cell',net128,netfeat128,up_ratio=4,state_len=512,reuse=True)
        tf.add_to_collection('o512',net512)
    with tf.variable_scope('Dinc',reuse=True):
        net2048,netfeat2048=fn_decode_cell('decode_cell',net512,netfeat512,up_ratio=4,state_len=512,reuse=True)
        tf.add_to_collection('o2048', net2048)
    with tf.variable_scope('Dinc',reuse=True):
        net8192,netfeat8192=fn_decode_cell('decode_cell',net2048,netfeat2048,up_ratio=4,state_len=512,reuse=True)
        
    return tf.reshape(net32, [-1, 32, 3]),tf.reshape(net128, [-1, 128, 3]),tf.reshape(net512, [-1, 512, 3]),tf.reshape(net8192, [-1, 8192, 3])

def gradual_choose(tensor):
    featLen = tensor.get_shape()[-1].value
    batchNum = tf.shape(tensor)[0]
    input_tensor = tf.reshape(tensor, [-1, 1, 1, featLen])
    with tf.variable_scope('D128'):
        #net = deconv('deconv_layer1', inputs=input_tensor, output_shape=[batchNum, 8, 8, 512], kernel_size=[8, 8],stride=[1, 1], padding='VALID')
        #net = deconv('deconv_layer2', inputs=net, output_shape=[batchNum, 32, 8, 128], kernel_size=[8, 1], stride=[4, 1])
        #netfeat128 = deconv('deconv_layer3', inputs=net, output_shape=[batchNum, 128, 8, 128], kernel_size=[32, 1], stride=[4, 1])
        net128,netfeat128=decom_ful('decom_layer0',input_tensor,[512,256],[256],[256,256],128,128)
        netfeat128 = conv2d('conv2d_layer4', netfeat128,128,[1,1])
        netfeat128 = conv2d('conv2d_layer5',netfeat128,128,[1,1])

        #net=tf.reshape(conv2d('o128', net,3,[1,1],activation_func=None),[-1,128*8,3])
        #net128=tf.reshape(tf.gather_nd(net,point_choose.farthest_sampling(128,net,featvec=None,batch=BATCH_SIZE)),[-1,128,1,3])
    tf.add_to_collection('o128', net128)
    with tf.variable_scope('D512'):
        net=tf.concat([net128,tf.reshape(netfeat128,[BATCH_SIZE,128,1,-1]),tf.tile(input_tensor,multiples=[1,128,1,1])],axis=-1)
        net=conv2d('conv_layer',net,128,[1,1])
        net=tf.concat([net128,net],axis=-1)

        netfeat512 = deconv('deconv_layer7', inputs=net, output_shape=[batchNum, 128, 32, 128], kernel_size=[1, 32], stride=[1, 1],padding='VALID')
        #netfeat512 = deconv('deconv_layer8', inputs=net, output_shape=[batchNum, 512, 8, 128], kernel_size=[32, 1], stride=[4, 1])
        
        #netfeat512 = conv2d('layer5_conv2d' , net,64,[1,8],padding='VALID')
        #netfeat512 = deconv('deconv_layer5', inputs=net, output_shape=[batchNum, 512, 8, 128], kernel_size=[4, 1], stride=[4, 1],padding='VALID')
        net=conv2d('conv2d_layer9',netfeat512,64,[1,1])
        net=conv2d('conv2d_layer10',net,64,[1,1])
        net32_512=conv2d('o512', net,3,[1,1],activation_func=None)#batch*128*32*3

        idx,net=sampling(512,tf.reshape(net32_512,[-1,32*128,3]))
        bb=tf.constant([i for i in range(BATCH_SIZE)],shape=[BATCH_SIZE,1,1])
        batch_index=tf.tile(bb,[1,512,1])
        idx=tf.concat([batch_index,tf.expand_dims(idx,axis=-1)],axis=-1)

        netfeat512=tf.gather_nd(tf.reshape(netfeat512,[-1,32*128,128]),idx)
        netfeat512=tf.expand_dims(netfeat512,axis=2)
        netfeat512=conv2d('conv_layer2',netfeat512,256,[1,1])
        netfeat512=conv2d('conv_layer3',netfeat512,128,[1,1])
 
        net512=tf.reshape(net,[-1,512,1,3])

    tf.add_to_collection('o512',net512)
    with tf.variable_scope('D2048'):
        net = tf.concat([net512,netfeat512,tf.tile(input_tensor, multiples=[1, 512, 1, 1])], axis=-1)
        net=conv2d('conv_layer',net,128,[1,1])
        net=tf.concat([net512,net],axis=-1)

        #net = conv2d('conv2d_layer8',net,128,[1,1])
        net = deconv('deconv_layer12', inputs=net, output_shape=[batchNum,512 , 32, 128], kernel_size=[1, 32], stride=[1, 1],padding='VALID')
        #net = deconv('deconv_layer13', inputs=net, output_shape=[batchNum,2048 , 8, 128], kernel_size=[32, 1], stride=[4, 1])
        #net = deconv('deconv_layer7', inputs=net, output_shape=[batchNum,2048 , 16, 128], kernel_size=[4, 1], stride=[4, 1],padding='VALID')
        net=conv2d('conv2d_layer14',net,64,[1,1])
        net=conv2d('conv2d_layer15',net,64,[1,1])
        net32_2048=conv2d('o2048', net,3,[1,1],activation_func=None)
        idx,net=sampling(2048,tf.reshape(net32_2048,[-1,32*512,3]))

        net2048=tf.reshape(net,[-1,2048,1,3])


    tf.add_to_collection('o2048', net2048)

    return tf.reshape(net128, [-1, 128, 3]),net32_512,tf.reshape(net512, [-1, 512, 3]),net32_2048,tf.reshape(net2048, [-1, 2048, 3])

def rbf_expand(xyz,input_tensor,mlp,k=16,n=4,proj_type='g',up_ratio=4,use_all=True):
    #xyz:batch*128*3
    #input_tensor:batch*128*1*128
    input_len=input_tensor.get_shape()[1].value
    input_tensor=tf.squeeze(input_tensor,[2])#batch*128*128
    dis_xyz=tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(xyz,axis=2)-tf.expand_dims(xyz,axis=1)),axis=-1))#batch*128*128
    knum=k
    if use_all:
        base_xyz=tf.tile(tf.expand_dims(xyz,axis=2),[1,1,input_len,1]) #batch*128*128*3
        base_feat=tf.tile(tf.expand_dims(input_tensor,axis=2),[1,1,input_len,1])#batch*128*128*128
        knum=input_len
    else:
        #dis_xyz=tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(xyz,axis=2)-tf.expand_dims(xyz,axis=1)),axis=-1))
        base_xyz,base_idx=get_kneighbor(xyz,dis_xyz,k)#batch*128*k*3
        base_feat=tf.gather_nd(input_tensor,base_idx)#batch*128*k*128
    theta=n*tf.reduce_min(tf.reduce_min(dis_xyz,axis=-1),axis=-1)
    in_xyz=tf.expand_dims(xyz,axis=2)
    if proj_type=='g':
        guassian=tf.exp(-tf.reduce_sum(tf.square(in_xyz-base_xyz),axis=-1)/tf.reshape(tf.square(theta),[BATCH_SIZE,1,1])) #batch*128*k
    ex_tensor=[]
    affect_feat=tf.expand_dims(guassian,axis=-1)*base_feat
    for i in range(up_ratio):
        kernel=get_weight_variable([1,1,knum,1],1e-3,'weights%d'%i)
        ex_tensor.append(tf.reduce_sum(kernel*affect_feat,axis=2))
    out_tensor=tf.expand_dims(tf.concat(ex_tensor,axis=1),axis=2)#batch*512*1*128
    tensor=out_tensor
    for i,outchannel in enumerate(mlp):
        tensor=conv2d('to_point%d'%i,tensor,outchannel,[1,1],padding='VALID')
    pt = tf.squeeze(conv2d('topt', tensor, 3, [1, 1],activation_func=None),[2])
    return pt,out_tensor

def gradual_rbf(tensor):
    featLen = tensor.get_shape()[-1].value
    batchNum = tf.shape(tensor)[0]
    input_tensor = tf.reshape(tensor, [-1, 1, 1, featLen])
    with tf.variable_scope('D128'):
        net128,netfeat128=decom_ful('decom_layer0',input_tensor,[512,256],[256],[256],128,128)
        net128=tf.squeeze(net128,[2])
    tf.add_to_collection('o128', net128)
    with tf.variable_scope('D512'):
        net512,netfeat512=rbf_expand(net128,netfeat128,[64,64,64],use_all=False)
    tf.add_to_collection('o512',net512)
    with tf.variable_scope('D2048'):
        net2048,netfeat2048=rbf_expand(net512,netfeat512,[64,64,64],use_all=False)
    tf.add_to_collection('o2048', net2048)
    return tf.reshape(net128, [-1, 128, 3]),tf.reshape(net512, [-1, 512, 3]),tf.reshape(net2048, [-1, 2048, 3])


#def get_kneighbor(ptdata,dis_oself,k):#
#    kindex=tf.expand_dims(tf.nn.top_k(-dis_oself,k)[1],axis=-1)#batch*2048*k*1
#    batchindex=tf.reshape(tf.constant([i for i in range(BATCH_SIZE)]),[BATCH_SIZE,1,1,1])
#    index=tf.concat([tf.tile(batchindex,multiples=[1,tf.shape(kindex)[1],k,1]),kindex],axis=-1)#batch*2048*k*2
#    kneighbor=tf.gather_nd(ptdata,index)#batch*2048*k*3
#    return kneighbor,index
#
#def get_cen_kneighbor(cen_index,ptdata,dis_oself,k):#
#    dis_self=tf.gather_nd(dis_oself,cen_index)
#    kindex=tf.expand_dims(tf.nn.top_k(-dis_self,k)[1],axis=-1)#batch*n*k*1
#    
#    batchindex=tf.reshape(tf.constant([i for i in range(BATCH_SIZE)]),[BATCH_SIZE,1,1,1])
#    index=tf.concat([tf.tile(batchindex,multiples=[1,tf.shape(kindex)[1],k,1]),kindex],axis=-1)#batch*n*k*2
#    kneighbor=tf.gather_nd(ptdata,index)#batch*n*k*3
#    return kneighbor,index

def regression_128_func(input,output):
    frame_loss=regression_func(input,output)
    

def regress_with_fardis(input,output):
    inout_loss=regression_func(input,output)
    dis_oself=tf.reduce_sum(tf.square(tf.expand_dims(output, axis=2) - tf.expand_dims(output, axis=1)), axis=-1)
    # dis_oself=dis_oself+tf.matrix_diag(tf.constant(10.0,shape=[BATCH_SIZE,input.get_shape()[1]]))
    # self_loss1=tf.reduce_mean(tf.reciprocal(tf.reduce_max(tf.reduce_min(dis_oself,axis=-1),axis=-1)))
    self_loss2 = tf.reduce_mean(tf.reciprocal(tf.reduce_min(tf.reduce_sum(dis_oself, axis=-1), axis=-1)))
   # dis_oself = dis_oself + tf.matrix_diag(tf.constant(10.0, shape=[BATCH_SIZE, input.get_shape()[1]]))
    #self_loss=-tf.reduce_mean(tf.log(tf.reduce_mean(tf.reduce_min(dis_oself,axis=-1),axis=-1)))
    loss=inout_loss+self_loss2
    return loss


def get_kneighbor(ptdata,dis_oself,k):
    kindex=tf.expand_dims(tf.nn.top_k(-dis_oself,k)[1],axis=-1)#batch*2048*k*1
    batchindex=tf.reshape(tf.constant([i for i in range(BATCH_SIZE)]),[BATCH_SIZE,1,1,1])
    index=tf.concat([tf.tile(batchindex,multiples=[1,tf.shape(kindex)[1],k,1]),kindex],axis=-1)#batch*2048*k*2
    kneighbor=tf.gather_nd(ptdata,index)#batch*2048*k*3
    return kneighbor,index
def get_cen_kneighbor(ptdata,dis_self,k):
    kindex=tf.expand_dims(tf.nn.top_k(-dis_self,k)[1],axis=-1)#batch*2048*k*1
    batchindex=tf.reshape(tf.constant([i for i in range(BATCH_SIZE)]),[BATCH_SIZE,1,1,1])
    
    index=tf.concat([tf.tile(batchindex,multiples=[1,tf.shape(kindex)[1],k,1]),kindex],axis=-1)#batch*2048*k*2
    kneighbor=tf.gather_nd(ptdata,index)#batch*2048*k*3
    return kneighbor,index
#make k points near a point as far as possible from it
def fardis_func(dis_oself,k=6):
    kdis=-tf.nn.top_k(-dis_oself,k)[0]
    kmeandis=tf.reduce_mean(tf.reduce_sum(kdis,axis=-1)/(k-1),axis=-1)#because there is a zeros for itself
    kfarloss=-tf.reduce_mean(tf.log(kmeandis))
    return kfarloss
def equaldis_func(dis_oself,k=6):
    kdis = tf.log(-tf.nn.top_k(-dis_oself, k)[0]+1e-5) #batch*2048*k
    kmeandis=tf.expand_dims(tf.reduce_mean(kdis,axis=-1),axis=-1)
    #kdisstd=tf.reduce_max(tf.log(tf.reduce_max(kdis,axis=-1))-tf.log((tf.reduce_min(kdis,axis=-1)+1e-10)))
    kdisstd=tf.sqrt(tf.reduce_max(tf.reduce_mean(tf.square(kdis-kmeandis),axis=-1),axis=-1))
    #kdisstd=tf.sqrt(kdisvar)
    result=tf.reduce_mean(kdisstd)
    return result
#calculate normal directions of every point by guassian PCA,return batch*2048*3
def getdirection(output,dis_oself,k=16):
    kneighbor,_=get_kneighbor(output,dis_oself,k)#batch*2048*k*3
    kndis=tf.sqrt(tf.reduce_sum(tf.square(kneighbor-tf.expand_dims(output,axis=2)),axis=-1))#batch*2048*k
    radius=tf.reduce_max(kndis,axis=-1)/3 #batch*2048
    A=tf.expand_dims(output,axis=2)-kneighbor #batch*2048*k*3
    guassian=tf.exp(-tf.reduce_sum(tf.square(A),axis=-1)/tf.square(tf.expand_dims(radius,axis=-1))) #batch*2048*k
    B=tf.transpose(A,[0,1,3,2])*tf.expand_dims(guassian,axis=2) #batch*2048*3*k
    neighvar=tf.matmul(B,A) #batch*2048*3*3
    _,vecs=tf.self_adjoint_eig(neighvar)
    normalvec=vecs[:,:,:,0] #batch*2048*3
    return normalvec
#make k points neighbor have similiar directions
def smooth_func(output,dis_oself,k=8):
    normalvecs=getdirection(output,dis_oself)
    _,kindex=get_kneighbor(output,dis_oself,k)
    kneighborvecs=tf.gather_nd(normalvecs,kindex) #batch*2048*k*3
    kvecs=tf.reduce_sum(tf.expand_dims(kneighborvecs,axis=2)*tf.expand_dims(kneighborvecs,axis=3),axis=-1) #batch*2048*k*k
    kvecsmax=tf.reduce_max(tf.reduce_max(tf.sqrt(1-tf.square(kvecs)),axis=-1),axis=-1)#?
    kvecs_loss=tf.reduce_mean(tf.reduce_mean(kvecsmax,axis=-1))
    return kvecs_loss,kvecsmax
def rms_func(input,output):
    dis_i=tf.reduce_sum(tf.square(tf.expand_dims(input,axis=2)-tf.tile(tf.expand_dims(output,axis=1),multiples=[1,tf.shape(input)[1],1,1])),axis=-1)
    dis_o=tf.transpose(dis_i,[0,2,1])
    dis_ii=tf.reduce_mean(tf.reduce_min(dis_i,axis=-1))
    dis_oo=tf.reduce_mean(tf.reduce_min(dis_o,axis=-1))
    dis=tf.sqrt(tf.reduce_mean(tf.maximum(dis_ii,dis_oo)))
    return dis
def chamfer_func(input,output):
    dis_i=tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(input,axis=2)-tf.tile(tf.expand_dims(output,axis=1),multiples=[1,tf.shape(input)[1],1,1])),axis=-1))
    dis_o=tf.transpose(dis_i,[0,2,1])
    dis_ii=tf.reduce_mean(tf.reduce_min(dis_i,axis=-1))
    dis_oo=tf.reduce_mean(tf.reduce_min(dis_o,axis=-1))
    dis=tf.reduce_mean(tf.maximum(dis_ii,dis_oo))
    return dis
def chamfer_big(pcd1, pcd2):
    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1 = tf.reduce_mean(tf.sqrt(dist1),axis=1)#+0.1*tf.reduce_max(dist1,axis=1)
    dist2 = tf.reduce_mean(tf.sqrt(dist2),axis=1)#+0.1*tf.reduce_max(dist2,axis=1)
    #dist1 = tf.reduce_mean(dist1,axis=1)#+0.1*tf.reduce_max(dist1,axis=1)
    #dist2 = tf.reduce_mean(dist2,axis=1)#+0.1*tf.reduce_max(dist2,axis=1)
    dist=tf.reduce_mean((dist1+dist2)/2)
    #dist=tf.reduce_mean(tf.maximum(dist1,dist2))#+0.5*tf.reduce_max(tf.maximum(dist1,dist2))
    return dist,idx1
#pcd:batch*num*3
def normalize(pcd1,pcd):
    upb=tf.reduce_max(pcd,axis=1,keepdims=True)
    downb=tf.reduce_min(pcd,axis=1,keepdims=True)
    cen=(upb+downb)/2
    length=(upb-downb)
    length=tf.reduce_max(length,axis=-1,keepdims=True)
    result=(pcd-cen)/(length)
    result1=(pcd1-cen)/(length)
    return result1,result
#pcd1,pcd2:batch*n*k*3
def chamfer_local(pcda,pcdb):
    ptnum=pcda.get_shape()[1].value
    knum=pcda.get_shape()[2].value
    pcd1=tf.reshape(pcda,[-1,knum,3])
    pcd2=tf.reshape(pcdb,[-1,knum,3])
    #pcd1,pcd2=normalize(pcd1,pcd2)
    #pcd1,pcd2=normalize(pcd1),normalize(pcd2)

    dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(pcd1, pcd2)
    dist1=tf.sqrt(dist1)
    dist2=tf.sqrt(dist2)
    #weights1=tf.exp(tf.reshape(dist1,[-1,ptnum,knum]))
    #weights1=weights1/tf.reduce_sum(weights1,axis=1,keepdims=True)
    #dist1=tf.reduce_sum(weights1*tf.reshape(dist1,[-1,ptnum,knum]),axis=1)
    #
    #weights2=tf.exp(tf.reshape(dist2,[-1,ptnum,knum]))
    #weights2=weights2/tf.reduce_sum(weights2,axis=1,keepdims=True)
    #dist2=tf.reduce_sum(weights2*tf.reshape(dist2,[-1,ptnum,knum]),axis=1)
    #print('--------------',pcd1,pcd2,dist1) 
    #dist1=tf.reduce_mean(tf.reshape(tf.sqrt(dist1),[-1,ptnum,knum]),axis=1)#+0.1*tf.reduce_max(tf.reshape(tf.sqrt(dist1),[-1,ptnum,knum]),axis=1)#batch*k
    #dist2=tf.reduce_mean(tf.reshape(tf.sqrt(dist2),[-1,ptnum,knum]),axis=1)#+0.1*tf.reduce_max(tf.reshape(tf.sqrt(dist2),[-1,ptnum,knum]),axis=1)#batch*k
    dist1=tf.reduce_mean(tf.reshape(dist1,[-1,ptnum,knum]),axis=2)#+0.1*tf.reduce_max(tf.reshape(dist1,[-1,ptnum,knum]),axis=1)#batch*k
    dist2=tf.reduce_mean(tf.reshape(dist2,[-1,ptnum,knum]),axis=2)#+0.1*tf.reduce_max(tf.reshape(dist2,[-1,ptnum,knum]),axis=1)#batch*k
    #dist1,dist2=tf.sqrt(dist1),tf.sqrt(dist2)
    
    #dist=tf.reduce_mean(tf.maximum(tf.reduce_mean(dist1,axis=1)+tf.reduce_max(dist1,axis=1),tf.reduce_mean(dist2,axis=1)+tf.reduce_max(dist2,axis=1)))#(batch,)
    #dist=tf.reduce_mean(tf.maximum(tf.reduce_mean(dist1,axis=1),tf.reduce_mean(dist2,axis=1)))#(batch,)
    #dist=tf.reduce_mean(0.5*(tf.reduce_mean(dist1,axis=1)+tf.reduce_mean(dist2,axis=1)))#(batch,)
    #dist=tf.reduce_mean(tf.maximum(tf.reduce_mean(dist1,axis=1),tf.reduce_mean(dist2,axis=1)))#(batch,)
    #print(pcda,pcdb,dist1,dist2)
    #assert False
    dist=tf.reduce_mean(tf.maximum(dist1,dist2))
    #dist=tf.reduce_mean(0.5*tf.sqrt(dist1+dist2))
    return dist,idx1
def emd_func(pred,gt):
    batch_size = pred.get_shape()[0].value
    matchl_out, matchr_out = tf_auctionmatch.auction_match(pred, gt)
    matched_out = tf_sampling.gather_point(gt, matchl_out)
    dist = tf.sqrt(tf.reduce_sum((pred - matched_out) ** 2,axis=-1))
    #dist = tf.reduce_sum((pred - matched_out) ** 2,axis=-1)
    dist = tf.reduce_mean(dist,axis=-1)

    cens=tf.reduce_mean(pred,axis=1,keep_dims=True)
    radius=tf.sqrt(tf.reduce_max(tf.reduce_sum((pred - cens) ** 2,axis=-1),axis=-1))
    #dist = tf.reshape((pred - matched_out) ** 2, shape=(batch_size, -1))
    #dist = tf.reduce_mean(dist, axis=1, keep_dims=True)
    #print(matched_out,dist)
    dist_norm = dist / radius

    emd_loss = tf.reduce_mean(dist_norm)
    return tf.reduce_mean(dist),radius
def emd_big(pred,gt):
    batch_size = pred.get_shape()[0].value
    matchl_out, matchr_out = tf_auctionmatch.auction_match(pred, gt)
    matched_out = tf_sampling.gather_point(gt, matchl_out)
    dist = tf.sqrt(tf.reduce_sum((pred - matched_out) ** 2,axis=-1))
    #dist = tf.reduce_sum((pred - matched_out) ** 2,axis=-1)
    dist = tf.reduce_max(dist,axis=-1)

    cens=tf.reduce_mean(pred,axis=1,keep_dims=True)
    radius=tf.sqrt(tf.reduce_max(tf.reduce_sum((pred - cens) ** 2,axis=-1),axis=-1))
    #dist = tf.reshape((pred - matched_out) ** 2, shape=(batch_size, -1))
    #dist = tf.reduce_mean(dist, axis=1, keep_dims=True)
    #print(matched_out,dist)
    dist_norm = dist / radius

    emd_loss = tf.reduce_mean(tf.reduce_mean(dist_norm,axis=-1))
    return emd_loss,radius

#make sample points' neighbor satisfy shape constraint,cen shape batch*n*3
def multi_rms_func(cen,input,output,n,k,r=0.2,theta1=0.5,theta2=0.5,use_frame=True,use_r=False,use_all=False):
    #in_index=point_choose.farthest_sampling(n,input) #batch*n*2
    if use_all:
        in_cen=input
    else:
        in_cen=tf.get_collection('i'+str(n))[0]
    #in_cen=tf.gather_nd(input,in_index)
    if use_r:
        out_kneighbor=tf.gather_nd(output,point_choose.local_cen_devide(n,k,r,output,in_cen,batch=BATCH_SIZE))-tf.expand_dims(in_cen,axis=2)
        in_kneighbor=tf.gather_nd(input,point_choose.local_cen_devide(n,k,r,input,in_cen,batch=BATCH_SIZE))-tf.expand_dims(in_cen,axis=2)
    else:
        dis_ci=tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(in_cen,axis=2)-tf.expand_dims(input,axis=1)),axis=-1))#batch*n*2048
        dis_co=tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(in_cen,axis=2)-tf.expand_dims(output,axis=1)),axis=-1))#batch*n*2048
        out_kneighbor=get_cen_kneighbor(output,dis_co,k)[0]-tf.expand_dims(in_cen,axis=2) #batch*n*k*3
        in_kneighbor=get_cen_kneighbor(input,dis_ci,k)[0]-tf.expand_dims(in_cen,axis=2) #batch*n*k*3

    in_mat=tf.reduce_sum(tf.square(tf.expand_dims(in_kneighbor,axis=3)-tf.expand_dims(out_kneighbor,axis=2)),axis=-1)# batch*n*k*k
    out_mat=tf.transpose(in_mat,[0,1,3,2])
    in_loss=tf.reduce_mean(tf.reduce_mean(tf.reduce_min(in_mat,axis=-1),axis=-1),axis=-1) #batch*n*k*k find sum loss of all locals
    out_loss=tf.reduce_mean(tf.reduce_mean(tf.reduce_min(out_mat,axis=-1),axis=-1),axis=-1)
    local_loss=tf.sqrt(tf.reduce_mean(tf.maximum(in_loss,out_loss)))

    if use_frame:
        #frame_loss=regression_func(in_cen,cen)
        frame_loss=rms_func(input,output)
        return theta1*frame_loss+theta2*local_loss
    else:
        return theta2*local_loss
def multi_chamfer_func(cen,inputpts,output,n,k,r=0.2,theta1=0.5,theta2=0.5,use_frame=True,use_r=False,use_all=False):
    if use_all:
        in_cen=inputpts
    else:
        in_cen=cen
        #in_cen=tf.get_collection('i'+str(n))
    #in_cen=tf.gather_nd(input,in_index)
    #ptnum=input.get_shape()[1].value
    if use_r:
        out_kneighbor,_=grouping(xyz=output,new_xyz=in_cen, radius=r, nsample=k, points=None, knn=False, use_xyz=False)
        #print(out_kneighbor)
        in_kneighbor,_=grouping(xyz=inputpts,new_xyz=in_cen, radius=r, nsample=k, points=None, knn=False, use_xyz=False)
        #out_kneighbor=tf.gather_nd(output,point_choose.local_cen_devide(n,k,r,output,in_cen,batch=BATCH_SIZE))-tf.expand_dims(in_cen,axis=2)
        #in_kneighbor=tf.gather_nd(input,point_choose.local_cen_devide(n,k,r,input,in_cen,batch=BATCH_SIZE))-tf.expand_dims(in_cen,axis=2)
    else:
        out_kneighbor,_=grouping(output,new_xyz=in_cen, radius=r, nsample=k, points=None, knn=True, use_xyz=True)
        in_kneighbor,_=grouping(inputpts,new_xyz=in_cen, radius=r, nsample=k, points=None, knn=True, use_xyz=True)

        #dis_ci=tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(in_cen,axis=2)-tf.expand_dims(input,axis=1)),axis=-1))#batch*n*2048
        #dis_co=tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(in_cen,axis=2)-tf.expand_dims(output,axis=1)),axis=-1))#batch*n*2048
        #out_kneighbor=get_cen_kneighbor(output,dis_co,k)[0]-tf.expand_dims(in_cen,axis=2) #batch*n*k*3
        #in_kneighbor=get_cen_kneighbor(input,dis_ci,k)[0]-tf.expand_dims(in_cen,axis=2) #batch*n*k*3

    #in_mat=tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(in_kneighbor,axis=3)-tf.expand_dims(out_kneighbor,axis=2)),axis=-1))# batch*n*k*k
    #out_mat=tf.transpose(in_mat,[0,1,3,2])
    #in_loss=tf.reduce_mean(tf.reduce_mean(tf.reduce_min(in_mat,axis=-1),axis=-1),axis=-1) #batch*n*k*k find sum loss of all locals
    #out_loss=tf.reduce_mean(tf.reduce_mean(tf.reduce_min(out_mat,axis=-1),axis=-1),axis=-1)
    #local_loss=tf.reduce_mean(tf.maximum(in_loss,out_loss))
    #kout=tf.reshape(out_kneighbor,[-1,k,3])
    #kin=tf.reshape(in_kneighbor,[-1,k,3])
    #local_loss=chamfer_big(kout,kin)[0]
    local_loss=chamfer_local(out_kneighbor,in_kneighbor)[0]
    if use_frame:
        #frame_loss=regression_func(in_cen,cen)
        #frame_loss=chamfer_big(inputpts,output)[0]
        frame_loss=chamfer_big(inputpts,output)[0]
        #print(frame_loss,local_loss)
        return theta1*frame_loss+theta2*local_loss
    else:
        return theta2*local_loss
def multi_emd_func(cen,input,output,n,k,r=0.2,theta1=0.5,theta2=0.5,use_frame=True,use_r=False,use_all=False):
    if use_all:
        in_cen=input
    else:
        in_cen=tf.get_collection('i'+str(n))[0]
    #in_cen=tf.gather_nd(input,in_index)
    if use_r:
        out_kneighbor=tf.gather_nd(output,point_choose.local_cen_devide(n,k,r,output,in_cen,batch=batch_size))-tf.expand_dims(in_cen,axis=2)
        in_kneighbor=tf.gather_nd(input,point_choose.local_cen_devide(n,k,r,input,in_cen,batch=batch_size))-tf.expand_dims(in_cen,axis=2)
    else:
        dis_ci=tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(in_cen,axis=2)-tf.expand_dims(input,axis=1)),axis=-1))#batch*n*2048
        dis_co=tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(in_cen,axis=2)-tf.expand_dims(output,axis=1)),axis=-1))#batch*n*2048
        out_kneighbor=get_cen_kneighbor(output,dis_co,k)[0]-tf.expand_dims(in_cen,axis=2) #batch*n*k*3
        in_kneighbor=get_cen_kneighbor(input,dis_ci,k)[0]-tf.expand_dims(in_cen,axis=2) #batch*n*k*3
    
    local_loss=emd_func(tf.reshape(in_kneighbor,[-1,k,3]),tf.reshape(out_kneighbor,[-1,k,3]))
    
    if use_frame:
        #frame_loss=regression_func(in_cen,cen)
        frame_loss=emd_func(input,output)
        return theta1*frame_loss+theta2*local_loss
    else:
        return theta2*local_loss


def detail_multi_r_func(cen,input,output,n,klist,rlist,theta,useall=False,use_type='r'):
    if use_type=='r':
        detail_func=multi_rms_func
    if use_type=='c':
        detail_func=multi_chamfer_func 
    if use_type=='e':
        detail_func=multi_emd_func
    loss=detail_func(cen,input,output,n,klist[0],rlist[0],theta1=theta[0],theta2=theta[1],use_frame=False,use_r=False,use_all=useall)
    for i in range(1,len(klist)):
        loss=loss+detail_func(cen,input,output,n,klist[i],rlist[i],theta2=theta[i+1],use_frame=False,use_r=False,use_all=useall)
    #loss=-(1/tf.log(loss))
    return loss
def detail_multi_k_func(cen,input,output,n,klist,theta,useall=False,use_type='e'):
    if use_type=='r':
        detail_func=multi_rms_func
    if use_type=='c':
        detail_func=multi_chamfer_func
    if use_type=='e':
        detail_func=multi_emd_func
    loss=detail_func(cen,input,output,n,klist[0],theta[0],theta[1],use_r=False,use_all=useall)
    for i in range(1,len(klist)):
        loss=loss+detail_func(cen,input,output,n,klist[i],theta[i+1],use_frame=False,use_r=False,use_all=useall)
    return loss
def detail_choose(cen,outcen,inputdata,outputdata,radius,norm_radius,knn=True):
    nsample=outputdata.get_shape()[-2].value
    
    #print(nsample)
    matchl_out, matchr_out = tf_auctionmatch.auction_match(cen, outcen)
    matchcen = tf_sampling.gather_point(cen, matchr_out)
    
    #matchcen=outcen
    #print(111)
    #print(matchcen,inputdata)
    matchedout=grouping(inputdata,matchcen, radius, nsample, None, knn=knn, use_xyz=True)
    #print(222)
    outdata = outputdata - tf.tile(tf.expand_dims(matchcen, 2), [1,1,nsample,1])
    #normrs = tf.reshape(tf.tile(norm_radius,[1,nsample]),[-1,1])
    result= chamfer_func(tf.reshape(matchedout,[-1,nsample,3]),tf.reshape(outdata,[-1,nsample,3]))
    #result=tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(matchedout-outdata),axis=-1)),axis=-1),axis=-1)/norm_radius)
    #cens=tf.reduce_mean(cen,axis=1,keep_dims=True)
    #radius=tf.sqrt(tf.reduce_max(tf.reduce_sum((cen - cens) ** 2,axis=-1),axis=-1))

    return result 
def choose_loss(cen,outcen,inputdata,output32,outputdata,radius,knn=True):
    basic_loss,norm_radius=emd_func(inputdata,outputdata)
    dloss=detail_choose(cen,outcen,inputdata,output32,radius,norm_radius,knn=knn)
    return 0.2*basic_loss+0.8*dloss
    
#make codeword more sparse
def codelimit_func(codeword):
    #return tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(codeword),axis=-1)))
    codeword=tf.cast(codeword,tf.float32)
    return tf.reduce_mean(tf.reduce_sum(codeword,axis=-1))

##make sample points' neighbor satisfy shape constraint    
#def detail_func(input,output,n,k,theta1=0.5,theta2=0.5):
#    dis_iself=tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(input,axis=2)-tf.expand_dims(input,axis=1)),axis=-1))
#    in_index=point_choose.farthest_sampling(n,input) #batch*n*2
#    #out_index=point_choose.farthest_sampling(n,output)
#    in_cen=tf.gather_nd(input,in_index)
#    #out_cen=tf.gather_nd(output,out_index) #batch*n*3
#    #get k neighbor points in input of centers from output 
#    dis_io=tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(input,axis=2)-tf.expand_dims(output,axis=1)),axis=-1)) #batch*2048*2048
#    out_kneighbor=get_cen_kneighbor(in_index,output,dis_io,k)[0]-tf.expand_dims(in_cen,axis=2) #batch*n*k*3
#    in_kneighbor=get_cen_kneighbor(in_index,input,dis_iself,k)[0]-tf.expand_dims(in_cen,axis=2) #batch*n*k*3
#    
#    in_mat=tf.sqrt(tf.reduce_sum(tf.square(tf.expand_dims(in_kneighbor,axis=3)-tf.expand_dims(out_kneighbor,axis=2)),axis=-1))# batch*n*k*k
#    out_mat=tf.transpose(in_mat,[0,1,3,2])
#    in_loss=tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(tf.reduce_min(in_mat,axis=-1),axis=-1),axis=-1)) #batch*n*k*k find max of all locals
#    out_loss=tf.reduce_mean(tf.reduce_mean(tf.reduce_sum(tf.reduce_min(out_mat,axis=-1),axis=-1),axis=-1))
#    local_loss=tf.maximum(in_loss,out_loss)
#    
#    #frame_loss=regression_func(in_cen,out_cen)
#    return local_loss

def cond1(inmat,_,i):
    return tf.less(tf.reduce_min(inmat),100.0)
   # return tf.cast(tf.minimum(tf.cast(tf.less(i,3),tf.int32),tf.cast(tf.less(tf.reduce_min(inmat),99),tf.int32)),tf.bool)
   # return tf.less(i,3) and tf.less(tf.reduce_min(inmat),99)
def loop_body(inmat,invalue,i):
    batchnum=tf.shape(inmat)[0]
    row_len=inmat.get_shape()[1].value
    col_len=inmat.get_shape()[2].value
    row_compare_mat=tf.tile(tf.constant([i for i in range(row_len)],shape=[1,row_len,1],dtype=tf.int64),multiples=[batchnum,1,col_len])
    col_compare_mat=tf.tile(tf.constant([i for i in range(col_len)],shape=[1,1,col_len],dtype=tf.int64),multiples=[batchnum,row_len,1])

    batch_rodex=tf.tile(tf.constant([i for i in range(BATCH_SIZE)],shape=[BATCH_SIZE,1,1],dtype=tf.int64),multiples=[1,col_len,1])
    row_index=tf.argmin(inmat,axis=1)
    row_index=tf.expand_dims(row_index,axis=1)#batchnum*1*col_len
    row_mat=tf.tile(tf.expand_dims(tf.reduce_all(tf.not_equal(row_index,row_compare_mat),axis=-1),axis=-1),multiples=[1,1,col_len])

    inmat1=tf.gather_nd(inmat,tf.concat([batch_rodex,tf.transpose(row_index,[0,2,1])],axis=-1))
    col_index=tf.expand_dims(tf.argmin(inmat1,axis=-1),axis=-1)#batchnum*row_len*1
    col_mat=tf.tile(tf.expand_dims(tf.reduce_all(tf.not_equal(col_index,col_compare_mat),axis=1),axis=1),multiples=[1,row_len,1])
    where_mat=tf.cast(tf.minimum(tf.cast(row_mat,dtype=tf.int32),tf.cast(col_mat,dtype=tf.int32)),dtype=tf.bool)
    outmat=tf.where(where_mat,inmat,tf.constant(100.0,shape=[BATCH_SIZE,row_len,col_len]))

    count_mat=tf.tile(row_index,[1,col_len,1])
    row_cp=tf.transpose(row_index,[0,2,1])
    count_num=tf.reduce_sum(tf.cast(tf.equal(row_cp,count_mat),dtype=tf.float32),axis=1)
    cal_mat=tf.reduce_min(inmat1,axis=-1)
    cal_mat=tf.where(tf.less(cal_mat,100.0),cal_mat,tf.constant(0.0,shape=[BATCH_SIZE,row_len]))
    result=invalue+tf.reduce_mean(tf.reduce_sum(cal_mat/count_num,axis=-1))

    return outmat,result,i+1

def regress_order_func(input,output):
    dis = tf.reduce_sum(tf.square(tf.expand_dims(input, axis=2) - tf.expand_dims(output, axis=1)), axis=-1)
    row_len = dis.get_shape()[1].value
    col_len = dis.get_shape()[2].value
    dis_v=tf.concat([tf.constant(100.0,shape=[BATCH_SIZE,row_len,1]),dis],axis=-1)
    dis_v=tf.concat([tf.constant(100.0,shape=[BATCH_SIZE,1,col_len+1]),dis_v],axis=1)
    final_mat,loss,_=tf.while_loop(cond1,loop_body,[dis_v,0.0,0])
    final_mat=final_mat[:,1:,1:]
    return loss

def get_gradients(loss,scope):
    with tf.variable_scope(scope,reuse=True):
        wlayer = tf.gradients(loss, tf.get_variable(name='weights'))
    tf.summary.histogram(scope+'/gradients',wlayer)
    return wlayer
#batch*n*3
def normal_trans(data):
    uplimit=tf.reduce_max(data,axis=1,keepdims=True)
    downlimit=tf.reduce_min(data,axis=1,keepdims=True)
    cen=(uplimit+downlimit)/2
    length=(uplimit-downlimit)/2
    result=(data-cen)/length#tf.reduce_max(length,axis=-1,keepdims=True)
    return result,cen,length
    #return data,0,1
def get_distri(dim):
    result=tf.random_uniform(dim, minval=-1,maxval=1, dtype=tf.float32)
    return result
def npnorm(data):
    maxv=np.max(data,axis=1,keepdims=True)
    minv=np.min(data,axis=1,keepdims=True)
    cenv=(maxv+minv)/2
    length=(maxv-minv)
    #length=np.max(length,axis=1,keepdims=True)
    ndata=2*(data-cenv)/length
    return ndata
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

    #datain,_,_=normal_trans(pointcloud_pl)
    datain=pointcloud_pl
    pt8=sampling(8,datain,use_type='f')[-1]
    pt32=sampling(32,datain,use_type='f')[-1]
    pt128=sampling(128,datain,use_type='f')[-1]
    pt512=sampling(512,datain,use_type='f')[-1]
    
    cens,feats=encoder(datain)
    global_step=tf.Variable(0,trainable=False)
    epoch_step=tf.cast(BATCH_SIZE*global_step/(2048*6),tf.int32)
    alpha0=tf.train.piecewise_constant(epoch_step,[50,100],[0.01,0.01,0.01],'alpha_op')
    #alpha1=tf.train.piecewise_constant(global_step,[1000,30000],[0.01,0.01,0.01],'alpha_op1')
    
    cen1,cen2,cen3,cen4=cens
    rfeat1,rfeat2,rfeat3,rfeat4=feats

    featscode1,featpool1=word2code(rfeat1,1,32,None,None)
    featscode2,featpool2=word2code(rfeat2,2,128,None,None)
    featscode3,featpool3=word2code(rfeat3,3,256,None,None)
    featscode4,featpool4=word2code(rfeat4,4,256,None,None)

    #featloss=tf.reduce_mean(tf.reduce_mean(tf.abs(featscode1-rfeat1),axis=-1))+tf.reduce_mean(tf.reduce_mean(tf.abs(featscode2-rfeat2),axis=-1))+tf.reduce_mean(tf.reduce_mean(tf.abs(featscode3-rfeat3),axis=-1))
    
    out1,probloss1=mydecode('codeword1',cen1,featscode1,featpool1,1,0.01)
    out11=out1[0]
    out2,probloss2=mydecode('codeword2',cen2,featscode2,featpool2,2,0.01)
    out21,out22=out2
    out3,probloss3=mydecode('codeword3',cen3,featscode3,featpool3,3,0.01)
    out31,out32,out33=out3
    out4,probloss4=mydecode('codeword4',cen4,featscode4,featpool4,4,0.01)
    out41,out42,out43,out44=out4

    #scaleloss=tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.reduce_max(datain,axis=1)-tf.reduce_max(cen1,axis=1)),axis=-1))\
    #        +tf.sqrt(tf.reduce_sum(tf.square(tf.reduce_min(datain,axis=1)-tf.reduce_min(cen1,axis=1)),axis=-1)))
    #scaleloss+=tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.reduce_max(datain,axis=1)-tf.reduce_max(cen2,axis=1)),axis=-1))\
    #        +tf.sqrt(tf.reduce_sum(tf.square(tf.reduce_min(datain,axis=1)-tf.reduce_min(cen2,axis=1)),axis=-1)))
    scaleloss=tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.reduce_max(datain,axis=1)-tf.reduce_max(cen3,axis=1)),axis=-1))\
            +tf.sqrt(tf.reduce_sum(tf.square(tf.reduce_min(datain,axis=1)-tf.reduce_min(cen3,axis=1)),axis=-1)))

    loss=[]
    loss1,loss2,loss3,loss4=[],[],[],[]
    trainstep=[]

    #print('>>>>>>>>>>>>>>>>>>')
    ##loss1.append(tf.expand_dims(detail_multi_r_func(pt512,datain,out11,512,[64,128],[0.05,0.1],[1,0.5,0.5,0.3,0.2,0.2],useall=False,use_type='c'),axis=-1))
    ##loss1.append(emd_func(datain,out11)[0])
    #loss2.append(0.5*tf.expand_dims(chamfer_big(pt512,out21)[0],axis=-1))
    #loss2.append(0.5*tf.expand_dims(chamfer_big(datain,out22)[0],axis=-1))

    #print('>>>>>>>>>>>>>>>>>>')
    ##loss3.append(0.33*tf.expand_dims(detail_multi_r_func(pt32,pt128,out31,32,[16,32],[0.2,0.4],[1.0,0.5,0.5,0.3],useall=False,use_type='c'),axis=-1))#//////////////////
    ##loss3.append(0.33*tf.expand_dims(detail_multi_r_func(pt128,pt512,out32,128,[32,64],[0.1,0.2],[1,0.5,0.5,0.3],useall=False,use_type='c'),axis=-1))
    ##loss3.append(tf.expand_dims(detail_multi_r_func(pt512,datain,out33,512,[64,128],[0.05,0.1],[1,0.5,0.5,0.3,0.2,0.2],useall=False,use_type='c'),axis=-1))

    #loss3.append(0.33*tf.expand_dims(chamfer_big(pt128,out31)[0],axis=-1))
    #loss3.append(0.33*tf.expand_dims(chamfer_big(pt512,out32)[0],axis=-1))
    #loss3.append(0.33*tf.expand_dims(chamfer_big(datain,out33)[0],axis=-1))

    ##loss4.append(tf.expand_dims(detail_multi_r_func(pt512,datain,out44,512,[64,128],[0.05,0.1],[1,0.5,0.5,0.3,0.2,0.2],useall=False,use_type='c'),axis=-1))
    #loss4.append(0.25*tf.expand_dims(chamfer_big(pt32,out41)[0],axis=-1))
    #loss4.append(0.25*tf.expand_dims(chamfer_big(pt128,out42)[0],axis=-1))
    #loss4.append(0.25*tf.expand_dims(chamfer_big(pt512,out43)[0],axis=-1))
    #loss4.append(0.25*tf.expand_dims(chamfer_big(datain,out44)[0],axis=-1))
    print('>>>>>>>>>>>>>>>>>>')
    loss2.append(0.5*tf.expand_dims(detail_multi_r_func(cen2,cen1,out21,128,[16,32],[0.1,0.2],[1,0.5,0.5,0.3],useall=False,use_type='c'),axis=-1))
    loss2.append(0.5*tf.expand_dims(detail_multi_r_func(cen1,datain,out22,512,[32,64],[0.05,0.1],[1,0.5,0.5,0.3,0.2,0.2],useall=False,use_type='c'),axis=-1))

    print('>>>>>>>>>>>>>>>>>>')
    loss3.append(0.33*tf.expand_dims(detail_multi_r_func(cen3,cen2,out31,32,[16,32],[0.2,0.4],[1.0,0.5,0.5,0.3],useall=False,use_type='c'),axis=-1))#//////////////////
    loss3.append(0.33*tf.expand_dims(detail_multi_r_func(cen2,cen1,out32,128,[32,64],[0.1,0.2],[1,0.5,0.5,0.3],useall=False,use_type='c'),axis=-1))
    loss3.append(0.33*tf.expand_dims(detail_multi_r_func(cen1,datain,out33,512,[64,128],[0.05,0.1],[1,0.5,0.5,0.3,0.2,0.2],useall=False,use_type='c'),axis=-1))

    loss4.append(0.25*tf.expand_dims(detail_multi_r_func(cen4,cen3,out41,8,[8,16],[0.2,0.4],[1.0,0.5,0.5,0.3],useall=False,use_type='c'),axis=-1))
    loss4.append(0.25*tf.expand_dims(detail_multi_r_func(cen3,cen2,out42,32,[16,32],[0.2,0.4],[1.0,0.5,0.5,0.3],useall=False,use_type='c'),axis=-1))#//////////////////
    loss4.append(0.25*tf.expand_dims(detail_multi_r_func(cen2,cen1,out43,128,[32,64],[0.1,0.2],[1,0.5,0.5,0.3],useall=False,use_type='c'),axis=-1))
    loss4.append(0.25*tf.expand_dims(detail_multi_r_func(cen1,datain,out44,512,[64,128],[0.05,0.1],[1,0.5,0.5,0.3,0.2,0.2],useall=False,use_type='c'),axis=-1))
    #loss=tf.concat(loss1,axis=-1)+tf.concat(loss2,axis=-1)+tf.concat(loss3,axis=-1)
    loss=loss2+loss3+loss4
    loss=tf.concat(loss,axis=-1)
    #probloss=probloss1+probloss2+probloss3
    print('>>>>>>>>>>>>>>>>>>')

    exploss=tf.add_n(tf.get_collection('expand_dis'))
    kmindist=tf.add_n(tf.get_collection('kmindist'))

    #featpool=tf.get_collection('featspool')
    #print(featpool1,featpool2,featpool3)
    #ftpooluni=get_repulsion_loss4(featpool3)
    ftpooluni=get_repulsion_loss4(featpool2)+get_repulsion_loss4(featpool3)
    #cenex=tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(cen-cen0),axis=-1)),axis=1))
    #print(tf.get_collection('cenex'))
    cenex=tf.add_n(tf.get_collection('cenex'))
    pool_loss=tf.add_n(tf.get_collection('ubloss'))
    decayloss=alpha0*tf.reduce_sum(tf.get_collection('decays'))
    loss=tf.reduce_sum(loss)+0.01*kmindist+0.01*exploss+decayloss
    #loss=tf.reduce_sum(loss)+0.01*kmindist+0.01*exploss+alpha0*tf.add_n(tf.get_collection('decays'))+0.01*cenex#+0.1*scaleloss#+0.2*probloss#+0.001*tf.add_n(tf.get_collection('losses'))

    trainvars=tf.GraphKeys.TRAINABLE_VARIABLES
    var2=[v for v in tf.get_collection(trainvars) if v.name.split('/')[0]!='flow']
    advar=[v for v in tf.get_collection(trainvars) if v.name.split('/')[0]!='Dinc_3' and v.name.split(':')[0]!='is_training' and v.name.split('/')[0]!='flow']
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    zhengze=tf.add_n([regularizer(v) for v in advar])
    loss=loss+0.001*zhengze
    trainstep=tf.train.AdamOptimizer(learning_rate=0.0002).minimize(loss, global_step=global_step,var_list=var2)
    config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
    config.gpu_options.allow_growth=True
    #kmask=tf.reduce_max(tf.get_collection('kmask'),axis=-1)
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(max_to_keep=10)
        sess.run(tf.global_variables_initializer())
        print('im here')
        if os.path.exists('./modelvv_test/checkpoint'):
            print('here load')
            saver.restore(sess, tf.train.latest_checkpoint('./modelvv_test/'))
            #tf.train.Saver(var_list=advar).restore(sess, tf.train.latest_checkpoint('./modelvv_temp/'))
        #merged = tf.summary.merge_all()
        #writer = tf.summary.FileWriter("logs/", sess.graph)
        print('here,here')
        datalist=[]

        for j in range(FILE_NUM):
            traindata = getdata.load_h5(os.path.join(DATA_DIR, trainfiles[j]))
            datalist.append(traindata)
        for i in range(EPOCH_ITER_TIME):
            for j in range(FILE_NUM):
                #traindata=datalist[j]
                traindata = getdata.load_h5(os.path.join(DATA_DIR, trainfiles[j]))
                #basepath='../../hard_disk/KITTI/odometry/sequences'
                #traindata,_=getdata.load_k_kitti(basepath,'00','000000.bin',bsize=16,knum=2048)#
                
                #traindata=[traindata[7]]
                #traindata=getdata.rotate_point_cloud(traindata)                
                #random.shuffle(traindata)
                #traindata.swapaxes(1,0)
                #random.shuffle(traindata)
                #traindata.swapaxes(1,0)
                ids=list(range(len(traindata)))
                random.shuffle(ids)
                traindata=traindata[ids,:,:]

                #import numpy as np
                #kklist=[np.power(2,v) for v in list(range(9,12))]
                #knum=kklist[np.random.choice(len(kklist),1,p=[0.25,0.25,0.5])[0]]
                ##print(knum)
                #if knum<2048:
                #    didx=list(range(knum))+list(np.random.choice(knum,2048-knum,replace=True))
                #    traindata=traindata[:,didx]

                allnum=int(len(traindata)/BATCH_SIZE)*BATCH_SIZE
                batch_num=int(allnum/BATCH_SIZE)
                for batch in range(batch_num):
                    #traindata,_=getdata.load_k_kitti(basepath,'00','000000.bin',bsize=BATCH_SIZE,knum=2048)

                    start_idx = (batch * BATCH_SIZE) % allnum
                    end_idx=(batch*BATCH_SIZE)%allnum+BATCH_SIZE
                    batch_point = traindata[start_idx:end_idx]

                    batch_point=shuffle_points(batch_point)
                    #batch_point=random_scale_point_cloud(batch_point,0.8,1.25)
                    #batch_point=jitter_point_cloud(batch_point,0.01,0.2)
                    batch_point=rotate_perturbation_point_cloud(batch_point, angle_sigma=2*pi, angle_clip=2*pi)

                    #import numpy as np
                    kklist=[np.power(2,v) for v in list(range(7,12))]
                    knum=kklist[np.random.choice(len(kklist),1,p=[0.15,0.15,0.15,0.15,0.4])[0]]
                    #print(knum)
                    #knum=512
                    if knum<2048:
                        didx=list(range(knum))+list(np.random.choice(knum,2048-knum,replace=True))
                        batch_point=batch_point[:,didx]
                    #if np.random.rand()>0.5:
                    #    batch_point=npnorm(batch_point)

                    feed_dict = {pointcloud_pl: batch_point}
                    resi = sess.run([trainstep,loss,cenex,pool_loss,kmindist,tf.get_collection('decays'),zhengze], feed_dict=feed_dict)
                    #if i<20:
                    #resi=sess.run([flowstep,[flowshow,recloss],tf.get_collection('decays'),exploss,cenex,kmindist,zhengze,word],feed_dict=feed_dict)
                    #else:
                    #resi = sess.run([trainstep,loss,tf.get_collection('decays'),exploss,cenex,kmindist,zhengze,word], feed_dict=feed_dict)
                    #sess.run(trainstep,feed_dict=feed_dict)
                    #save_path = saver.save(sess, './modelvv/model')
                    if batch % 16 == 0:
                        #save_path = saver.save(sess, './modelvv/model',global_step=batch)
                        #resishow = resi[-1]
                        #resi_vec0 = resishow[0]
                        #result = sess.run(merged, feed_dict=feed_dict)
                        #writer.add_summary(result, batch)
                        print('epoch: %d '%i,'file: %d '%j,'batch: %d' %batch)
                        print('loss: ',resi[1])
                        print('knum: ',knum)
                        print('cenloss: ',resi[2])

                        print('exp loss:',resi[3])
                        #print('cens move:',resi[4],isnan(resi[4]).any())
                        print('kmindist:',resi[4])
                        print('decays:',resi[5])
                        #print('max one of first codeword: ',max(resishow[0]))
                        #print('num of first codeword nozeros: ',len(resi_vec0[resi_vec0 != 0]))
                        print('regularization: ', resi[-2])
            if (i+1)%10==0:
                save_path = saver.save(sess, './modelvv_test/model',global_step=i)
if __name__=='__main__':
    train()
