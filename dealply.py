from plyfile import PlyData
import numpy as np
import sys
import h5py
import os
import open3d as o3d
def load_h5(h5_filename):
    f=h5py.File(h5_filename)
    data=f['data'][:]
    return data
def load_ply_data(filename):
  '''
  load data from ply file.
  '''

  f = open(filename)
  #1.read all points
  points = []
  for line in f:
    #only x,y,z
    wordslist = line.split(' ')
    try:
      x, y, z = float(wordslist[0]),float(wordslist[1]),float(wordslist[2])
    except ValueError:
      continue
    points.append([x,y,z])
  points = np.array(points)
  points = points.astype(np.float32)#np.uint8
  # print(filename,'\n','length:',points.shape)
  f.close()

  return points
def write_ply_data(filename, points):
  '''
  write data to ply file.
  '''
  if os.path.exists(filename):
      os.system('rm '+filename)
  f = open(filename,'a+')
  #print('data.shape:',data.shape)
  f.writelines(['ply\n','format ascii 1.0\n'])
  f.write('element vertex '+str(points.shape[0])+'\n')
  f.writelines(['property float x\n','property float y\n','property float z\n'])
  f.write('end_header\n')
  for _, point in enumerate(points):
    f.writelines([str(point[0]), ' ', str(point[1]), ' ',str(point[2]), '\n'])
  f.close()

  return
def write_ply_alpha(filename, points):
    if os.path.exists(filename):
        os.system('rm '+filename)
    f = open(filename,'a+')
    #print('data.shape:',data.shape)
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(points.shape[0])+'\n')
    f.writelines(['property float x\n','property float y\n','property float z\n', 'property float nx\n', 'property float ny\n','property float nz\n'])
    #f.writelines(['property float x\n','property float y\n','property float z\n', 'property uchar red\n', 'property uchar green\n', 'property uchar blue\n'])
    f.write('end_header\n')
    for _, point in enumerate(points):
      #point.astype(np.float16)
      f.writelines([str(point[0]), ' ', str(point[1]), ' ',str(point[2]), ' ', str(point[3]), ' ','0', ' ', '0', '\n'])
      #a=point[3]

      #b=int((a+1)/2)
      #a=a-b
          
      #f.writelines([str(point[0]), ' ', str(point[1]), ' ',str(point[2]), ' ', str(np.uint8(a)), ' ', str(np.uint8(b)), ' ', '0', '\n'])
    f.close()

    return
def load_ply_alpha(filename):
    '''
  load data from ply file.
  '''

    f = open(filename)
    #1.read all points
    points = []
    for line in f:
        #only x,y,z
        wordslist = line.split(' ')
        try:
          x, y, z, al = float(wordslist[0]),float(wordslist[1]),float(wordslist[2]),float(wordslist[3])#,float(wordslist[4])
        except ValueError:
          continue
        points.append([x,y,z,al])
    points = np.array(points)
    points = points.astype(np.float32)#np.uint8
    # print(filename,'\n','length:',points.shape)
    f.close()

    return points
def trans_ply(filename):
    plydata = PlyData.read(filename)
    data=plydata.elements[0].data
    os.system('rm -r '+filename)
    #os.wait()
    #write_ply_data(filename,data)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(outpath, pcd, write_ascii=True)
def trans_alpha(filename):
    plydata = PlyData.read(filename)
    data=plydata.elements[0].data
    os.system('rm -r '+filename)
    #os.wait()
    #print(np.max(data,axis=1))
    write_ply_alpha(filename,data)
if __name__=='__main__':
    #basepath='/home/xk/codetest/data/KITTI/odometry/sequences'
    #path=basepath+'/'+'00'+'/'+'velodyne/'+'000000.bin'
    #data=np.fromfile(path,dtype=np.float32,count=-1)
    #data=np.reshape(data,[-1,4])[:,:3]

    data=load_ply_alpha('/home/xk/codetest/data/plymodels/8iVFB/redandblack_vox10_1550.ply')
    #write_ply_data('kitti0.ply',data)
    write_ply_data('test.ply',data)
    os.system('./draco_encoder -point_cloud -i test.ply -o test.drc -cl 10 -qp 10')
    os.system('./draco_decoder -point_cloud -i test.drc -o testout.ply')
    trans_ply('testout.ply')
    trans_ply('outputs.ply')
    #plydata = PlyData.read('outputs.ply')
    #print(plydata.elements[0].data)
