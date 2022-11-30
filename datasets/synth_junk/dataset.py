import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import json
import codecs
import numpy as np
import sys
import torchvision.transforms as transforms
import argparse
import json
import time
import random
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
import yaml
import cv2
from omegaconf import OmegaConf
import csv
import open3d as o3d
import imageio.v2 as imageio

CONFIG_PATH = '/data/config.yaml'

def load_image(
  path: str,
  gray16: bool = False,
  normalize: bool = False
) -> np.ndarray:
  """Load image file

  Args:
    path (str): path to image
    gray16 (bool, optional): whether it's a 16bit gray-scale image.
      Defaults to False.
    normalize (bool, optional): whether to normalize the image.
      Defaults to False.

  Returns:
    np.ndarray: loaded image
  """
  # load image file and convert it to numpy array
  img = np.asarray(imageio.imread(path))
  if gray16:
    # load 16 bit gray-scale image and normalize
    img = img.astype(np.float64) / 65535.0
  elif normalize:
    # normalize rgb image
    img = img.astype(np.float64) / 255.
  return img

def load_camera_intrinsics_3x3(path: str) -> np.ndarray:
  with open(path, 'r') as f:
    lines = csv.reader(f, delimiter='\t')
    P3x3 = np.array([
      [float(n) for n in line]
      for line in lines if len(line) >= 3
    ], dtype=np.float32)
  return P3x3

class PoseDataset(data.Dataset):
  def __init__(self, mode, num, add_noise, root, noise_trans, refine):
    self.objlist = [1, 2, 3 ,4]
    self.mode = mode

    self.list_rgb = []
    self.list_depth = []
    self.list_label = []
    self.list_obj = []
    self.list_rank = []
    self.meta = {}
    self.pt = {}
    self.root = root
    self.noise_trans = noise_trans
    self.refine = refine

    conf = OmegaConf.load(CONFIG_PATH)
    self.conf = conf

    if self.mode == 'train':
      annotation_path = conf.pose_dataset.train.annotation_path
      image_path = conf.pose_dataset.train.image_path
    else:
      annotation_path = conf.pose_dataset.test.annotation_path
      image_path = conf.pose_dataset.test.image_path

    self.annotation_path = annotation_path
    self.image_path = image_path

    # load dataset, annotations
    with open(annotation_path, 'r') as f:
      j = json.load(f)

    annotation_list = j['annotations']
    image_list = j['images']
    image_info_lookup = {}
    for image_info in image_list:
      image_info_lookup[image_info['id']] = image_info

    self.annotation_list = annotation_list
    self.image_info_lookup = image_info_lookup

    # load camera intrinsics
    P3x3 = load_camera_intrinsics_3x3(conf.camera.intrinsics)
    self.camera_intrinsics = P3x3

    self.cam_cx = P3x3[0, 2]
    self.cam_cy = P3x3[1, 2]
    self.cam_fx = P3x3[0, 0]
    self.cam_fy = P3x3[1, 1]

    # load 3d models
    pt = {}
    for model_name, model_conf in conf.models.items():
      pcd = o3d.io.read_point_cloud(model_conf.path)
      pt[model_conf.category_id] = np.asarray(pcd.points)

    self.pt = pt

    self.length = len(self.annotation_list)

    # ========

    # item_count = 0
    # for item in self.objlist:
    #   if self.mode == 'train':
    #     input_file = open('{0}/data/{1}/train.txt'.format(self.root, '%02d' % item))
    #   else:
    #     input_file = open('{0}/data/{1}/test.txt'.format(self.root, '%02d' % item))
    #   while 1:
    #     item_count += 1
    #     input_line = input_file.readline()
    #     if self.mode == 'test' and item_count % 10 != 0:
    #       continue
    #     if not input_line:
    #       break
    #     if input_line[-1:] == '\n':
    #       input_line = input_line[:-1]
    #     self.list_rgb.append('{0}/data/{1}/rgb/{2}.png'.format(self.root, '%02d' % item, input_line))
    #     self.list_depth.append('{0}/data/{1}/depth/{2}.png'.format(self.root, '%02d' % item, input_line))
    #     if self.mode == 'eval':
    #       self.list_label.append('{0}/segnet_results/{1}_label/{2}_label.png'.format(self.root, '%02d' % item, input_line))
    #     else:
    #       self.list_label.append('{0}/data/{1}/mask/{2}.png'.format(self.root, '%02d' % item, input_line))
        
    #     self.list_obj.append(item)
    #     self.list_rank.append(int(input_line))

    #   meta_file = open('{0}/data/{1}/gt.yml'.format(self.root, '%02d' % item), 'r')
    #   self.meta[item] = yaml.load(meta_file)
    #   self.pt[item] = ply_vtx('{0}/models/obj_{1}.ply'.format(self.root, '%02d' % item))
      
    #   print("Object {0} buffer loaded".format(item))

    # self.length = len(self.list_rgb)

    # self.cam_cx = 325.26110
    # self.cam_cy = 242.04899
    # self.cam_fx = 572.41140
    # self.cam_fy = 573.57043

    self.xmap = np.array([[j for i in range(640)] for j in range(480)])
    self.ymap = np.array([[i for i in range(640)] for j in range(480)])
    
    self.num = num
    self.add_noise = add_noise
    self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
    self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    self.border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
    self.num_pt_mesh_large = 500
    self.num_pt_mesh_small = 500
    self.symmetry_obj_idx = []

  def __getitem__(self, index):
    anno_info = self.annotation_list[index]
    image_id = anno_info['image_id']
    category_id = anno_info['category_id']
    image_info = self.image_info_lookup[image_id]

    file_name = image_info['file_name']
    height = image_info['height']
    width = image_info['width']
    image_number = os.path.splitext(file_name)[0].split('_')[-1] #{:06d}

    # load color
    color_path = os.path.join(self.image_path, f'color_{image_number}.jpg')
    color_image = Image.open(color_path) # PIL.Image

    # load depth
    depth_path = os.path.join(self.image_path, f'depth_{image_number}.png')
    depth_image = load_image(depth_path, gray16=True) # (h, w)
    # load segmentation mask
    seg_path = os.path.join(self.image_path, f'seg_{image_number}.png')
    seg_image = load_image(seg_path) # (h, w)
    mask_label = np.zeros_like(seg_image, dtype=np.uint8)
    mask_label[seg_image == category_id] = 255 # (h, w) uint8

    mask_depth = ma.getmaskarray(ma.masked_not_equal(depth_image, 0))
    mask_label = ma.getmaskarray(ma.masked_equal(mask_label, 255))


    mask = mask_label * mask_depth

    # color augmentation
    if self.add_noise:
      color_image = self.trancolor(color_image)

    color_image = np.array(color_image)[...,:3]
    color_image = np.transpose(color_image, (2, 0, 1)) # (c, h, w)
    img_masked = color_image

    # crop image region
    rmin, rmax, cmin, cmax = get_bbox(anno_info['bbox'])
    img_masked = img_masked[:, rmin:rmax, cmin:cmax]

    # get ground truth pose
    Rt = np.asarray(anno_info['pose'], dtype=np.float32).copy()
    target_r = Rt[:3, :3]
    target_t = Rt[:3, -1]
    add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])


    choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
    if len(choose) == 0:
      cc = torch.LongTensor([0])
      return(cc, cc, cc, cc, cc, cc)
    
    if len(choose) > self.num:
      c_mask = np.zeros(len(choose), dtype=int)
      c_mask[:self.num] = 1
      np.random.shuffle(c_mask)
      choose = choose[c_mask.nonzero()]
    else:
      choose = np.pad(choose, (0, self.num - len(choose)), 'wrap')


    depth_masked = depth_image[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    choose = np.array([choose])

    cam_scale = 1.0
    pt2 = depth_masked / cam_scale
    pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
    pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
    cloud = np.concatenate((pt0, pt1, pt2), axis=1)

    # if self.add_noise:
    #   cloud = np.add(cloud, add_t)


    model_points = self.pt[category_id].copy()
    dellist = [j for j in range(0, len(model_points))]
    dellist = random.sample(dellist, len(model_points) - self.num_pt_mesh_small)
    model_points = np.delete(model_points, dellist, axis=0)

    target = np.dot(model_points, target_r.T)
    # if self.add_noise:
    #   target = np.add(target, target_t + add_t)
    #   out_t = target_t + add_t
    # else:
    target = np.add(target, target_t)
    out_t = target_t

    return torch.from_numpy(cloud.astype(np.float32)), \
         torch.LongTensor(choose.astype(np.int32)), \
         self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
         torch.from_numpy(target.astype(np.float32)), \
         torch.from_numpy(model_points.astype(np.float32)), \
         torch.LongTensor([self.objlist.index(category_id)])

  # def __getitem__(self, index):
    # img = Image.open(self.list_rgb[index])
    # ori_img = np.array(img)
    # depth = np.array(Image.open(self.list_depth[index]))
    # label = np.array(Image.open(self.list_label[index]))
    # obj = self.list_obj[index]
    # rank = self.list_rank[index]        

    # if obj == 2:
    #   for i in range(0, len(self.meta[obj][rank])):
    #     if self.meta[obj][rank][i]['obj_id'] == 2:
    #       meta = self.meta[obj][rank][i]
    #       break
    # else:
    #   meta = self.meta[obj][rank][0]

    # mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
    # if self.mode == 'eval':
    #   mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))
    # else:
    #   mask_label = ma.getmaskarray(ma.masked_equal(label, np.array([255, 255, 255])))[:, :, 0]
    
    # mask = mask_label * mask_depth

    # if self.add_noise:
    #   img = self.trancolor(img)

    # img = np.array(img)[:, :, :3]
    # img = np.transpose(img, (2, 0, 1))
    # img_masked = img

    # if self.mode == 'eval':
    #   rmin, rmax, cmin, cmax = get_bbox(mask_to_bbox(mask_label))
    # else:
    #   rmin, rmax, cmin, cmax = get_bbox(meta['obj_bb'])

    # img_masked = img_masked[:, rmin:rmax, cmin:cmax]
    
    # target_r = np.resize(np.array(meta['cam_R_m2c']), (3, 3))
    # target_t = np.array(meta['cam_t_m2c'])
    # add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

    # choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
    # if len(choose) == 0:
    #   cc = torch.LongTensor([0])
    #   return(cc, cc, cc, cc, cc, cc)

    # if len(choose) > self.num:
    #   c_mask = np.zeros(len(choose), dtype=int)
    #   c_mask[:self.num] = 1
    #   np.random.shuffle(c_mask)
    #   choose = choose[c_mask.nonzero()]
    # else:
    #   choose = np.pad(choose, (0, self.num - len(choose)), 'wrap')
    
    # depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    # xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    # ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
    # choose = np.array([choose])

    # cam_scale = 1.0
    # pt2 = depth_masked / cam_scale
    # pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
    # pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
    # cloud = np.concatenate((pt0, pt1, pt2), axis=1)
    # cloud = cloud / 1000.0

    # if self.add_noise:
    #   cloud = np.add(cloud, add_t)

    #fw = open('evaluation_result/{0}_cld.xyz'.format(index), 'w')
    #for it in cloud:
    #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
    #fw.close()

    # model_points = self.pt[obj] / 1000.0
    # dellist = [j for j in range(0, len(model_points))]
    # dellist = random.sample(dellist, len(model_points) - self.num_pt_mesh_small)
    # model_points = np.delete(model_points, dellist, axis=0)

    #fw = open('evaluation_result/{0}_model_points.xyz'.format(index), 'w')
    #for it in model_points:
    #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
    #fw.close()

    # target = np.dot(model_points, target_r.T)
    # if self.add_noise:
    #   target = np.add(target, target_t / 1000.0 + add_t)
    #   out_t = target_t / 1000.0 + add_t
    # else:
    #   target = np.add(target, target_t / 1000.0)
    #   out_t = target_t / 1000.0

    #fw = open('evaluation_result/{0}_tar.xyz'.format(index), 'w')
    #for it in target:
    #    fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
    #fw.close()

    # return torch.from_numpy(cloud.astype(np.float32)), \
    #      torch.LongTensor(choose.astype(np.int32)), \
    #      self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
    #      torch.from_numpy(target.astype(np.float32)), \
    #      torch.from_numpy(model_points.astype(np.float32)), \
    #      torch.LongTensor([self.objlist.index(obj)])

  def __len__(self):
    return self.length

  def get_sym_list(self):
    return self.symmetry_obj_idx

  def get_num_points_mesh(self):
    if self.refine:
      return self.num_pt_mesh_large
    else:
      return self.num_pt_mesh_small



border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640


def mask_to_bbox(mask):
  mask = mask.astype(np.uint8)
  contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


  x = 0
  y = 0
  w = 0
  h = 0
  for contour in contours:
    tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(contour)
    if tmp_w * tmp_h > w * h:
      x = tmp_x
      y = tmp_y
      w = tmp_w
      h = tmp_h
  return [x, y, w, h]


def get_bbox(bbox):
  bbx = [bbox[1], bbox[1] + bbox[3], bbox[0], bbox[0] + bbox[2]]
  if bbx[0] < 0:
    bbx[0] = 0
  if bbx[1] >= 480:
    bbx[1] = 479
  if bbx[2] < 0:
    bbx[2] = 0
  if bbx[3] >= 640:
    bbx[3] = 639                
  rmin, rmax, cmin, cmax = bbx[0], bbx[1], bbx[2], bbx[3]
  r_b = rmax - rmin
  for tt in range(len(border_list)):
    if r_b > border_list[tt] and r_b < border_list[tt + 1]:
      r_b = border_list[tt + 1]
      break
  c_b = cmax - cmin
  for tt in range(len(border_list)):
    if c_b > border_list[tt] and c_b < border_list[tt + 1]:
      c_b = border_list[tt + 1]
      break
  center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
  rmin = center[0] - int(r_b / 2)
  rmax = center[0] + int(r_b / 2)
  cmin = center[1] - int(c_b / 2)
  cmax = center[1] + int(c_b / 2)
  if rmin < 0:
    delt = -rmin
    rmin = 0
    rmax += delt
  if cmin < 0:
    delt = -cmin
    cmin = 0
    cmax += delt
  if rmax > 480:
    delt = rmax - 480
    rmax = 480
    rmin -= delt
  if cmax > 640:
    delt = cmax - 640
    cmax = 640
    cmin -= delt
  return rmin, rmax, cmin, cmax


def ply_vtx(path):
  f = open(path)
  assert f.readline().strip() == "ply"
  f.readline()
  f.readline()
  N = int(f.readline().split()[-1])
  while f.readline().strip() != "end_header":
    continue
  pts = []
  for _ in range(N):
    pts.append(np.float32(f.readline().split()[:3]))
  return np.array(pts)
