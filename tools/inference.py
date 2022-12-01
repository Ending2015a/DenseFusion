import os
import cv2
import copy
import numpy as np
import imageio.v2 as imageio
from omegaconf import OmegaConf

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

import torch
import torchvision.transforms as transforms
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import quaternion_matrix, quaternion_from_matrix
import skimage.morphology
import numpy.ma as ma
import csv
import matplotlib.pyplot as plt
import open3d as o3d
import einops

from wp_pose import utils


conf = OmegaConf.load('/data/config.yaml')

MODEL_PATH = '/output/maskrcnn-v2/model_0000999.pth'
THRESHOLD = 0.5
OUTPUT_DIR = '/output/inference-densefusion-v2/0003/'
CONFIG_PATH = '/output/maskrcnn-v2/config.yaml'
DEPTH_SCALE = 7
POSE_NET_PATH = '/tools/DenseFusion/trained_models/synth_junk_v2/pose_model_3_0.024766791697746638.pth'
POSE_REFINE_NET_PATH = '/tools/DenseFusion/trained_models/synth_junk_v2/pose_refine_model_7_0.018782259248803848.pth'
NUM_POINTS = 500
NUM_OBJECTS = 4
INTRINSICS = '/data/intrinsics.txt'
REFINE_ITERATION = 4

def load_camera_intrinsics_3x3(path: str) -> np.ndarray:
  with open(path, 'r') as f:
    lines = csv.reader(f, delimiter='\t')
    P3x3 = np.array([
      [float(n) for n in line]
      for line in lines if len(line) >= 3
    ], dtype=np.float32)
  return P3x3

cam = utils.FancyCamera.load(INTRINSICS, width=640, height=480)

CAM_CX = cam.cx
CAM_CY = cam.cy
CAM_FX = cam.fx
CAM_FY = cam.fy

cfg = get_cfg()
cfg.merge_from_file(CONFIG_PATH)
cfg.MODEL.WEIGHTS = MODEL_PATH
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = THRESHOLD

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

NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640
XMAP = np.array([[j for i in range(640)] for j in range(480)])
YMAP = np.array([[i for i in range(640)] for j in range(480)])


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

def preprocess_batch(color, depth, instances, cat_id):
  # instances: detectron2.structures.Instances
  # preprocess mask
  label = instances.pred_masks.numpy()[0]
  brush = skimage.morphology.disk(conf.maskrcnn.dilate_size)
  label = skimage.morphology.binary_dilation(label, brush)
  label = label.astype(np.uint8) * 255 # [0, 255] mask
  # preprocess bbox
  bboxfp = instances.pred_boxes.tensor.numpy()[0]
  bbox_min = np.floor(bboxfp[0:2])
  bbox_max = np.ceil(bboxfp[2:4])
  x, y = bbox_min.tolist()
  w, h = (bbox_max - bbox_min).tolist()
  bbox = np.asarray([x, y, w, h]).astype(np.int64).tolist()
  rmin, rmax, cmin, cmax = get_bbox(bbox)
  # preprocess image
  img = np.array(color)[:, :, :3]
  img = np.transpose(img, (2, 0, 1))
  img_masked = img
  img_masked = img_masked[:, rmin:rmax, cmin:cmax]
  # preprocess depth
  depth_image = depth
  mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
  mask_label = ma.getmaskarray(ma.masked_equal(label, np.array(255)))

  mask = mask_label * mask_depth
  choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
  if len(choose) > NUM_POINTS:
    c_mask = np.zeros(len(choose), dtype=int)
    c_mask[:NUM_POINTS] = 1
    np.random.shuffle(c_mask)
    choose = choose[c_mask.nonzero()]
  else:
    choose = np.pad(choose, (0, NUM_POINTS - len(choose)), 'wrap')
  
  depth_masked = depth_image[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
  xmap_masked = XMAP[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
  ymap_masked = YMAP[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
  choose = np.array([choose])

  cam_scale = 1.0
  pt2 = depth_masked / cam_scale
  pt0 = (ymap_masked - CAM_CX) * pt2 / CAM_FX
  pt1 = (xmap_masked - CAM_CY) * pt2 / CAM_FY
  cloud = np.concatenate((pt0, pt1, pt2), axis=1)

  img = NORM(torch.from_numpy(img_masked.astype(np.float32)))
  points = torch.from_numpy(cloud.astype(np.float32))
  choose = torch.LongTensor(choose.astype(np.int32))
  idx = torch.LongTensor([cat_id])
  
  # cuda
  img = img.unsqueeze(axis=0).cuda()
  points = points.unsqueeze(axis=0).cuda()
  choose = choose.unsqueeze(axis=0).cuda()
  idx = idx.unsqueeze(axis=0).cuda()

  return (img, points, choose, idx)

# setup mask-rcnn predictor
predictor = DefaultPredictor(cfg)

# setup densefusion predictor
pose_net = PoseNet(num_points=NUM_POINTS, num_obj=NUM_OBJECTS)
pose_refine_net = PoseRefineNet(num_points=NUM_POINTS, num_obj=NUM_OBJECTS)
pose_net.load_state_dict(torch.load(POSE_NET_PATH))
pose_refine_net.load_state_dict(torch.load(POSE_REFINE_NET_PATH))
pose_net.eval()
pose_refine_net.eval()
pose_net.cuda()
pose_refine_net.cuda()

os.makedirs(OUTPUT_DIR, exist_ok=True)

@torch.no_grad()
def detect_pose(color, depth, cat_instances, cat_id):
  img, points, choose, idx = preprocess_batch(color, depth, cat_instances, cat_id)

  quaternion_th, translation_th, confidence_th, embedding_th = pose_net(img, points, choose, idx)
  quaternion_th = quaternion_th / torch.norm(quaternion_th, dim=2, keepdim=True)
  confidence_th = confidence_th.reshape(1, NUM_POINTS)
  how_max, which_max = torch.max(confidence_th, 1)
  translation_th = translation_th.reshape(NUM_POINTS, 1, 3)

  Q = quaternion_th[0][which_max[0]].view(-1).cpu().numpy()
  t = translation_th[which_max[0]].view(-1).cpu().numpy()
  Pt = (points.view(NUM_POINTS, 1, 3) + translation_th)[which_max[0]].view(-1).cpu().numpy()

  for it in range(REFINE_ITERATION):

    T = torch.from_numpy(Pt.astype(np.float32)).cuda().view(1, 3).repeat(NUM_POINTS, 1).contiguous().view(1, NUM_POINTS, 3)
    Rt = quaternion_matrix(Q)
    R = torch.from_numpy(Rt[:3, :3].astype(np.float32)).cuda().view(1, 3, 3)
    Rt[:3, -1] = Pt

    new_points = torch.bmm((points - T), R).contiguous()
    quaternion_th, translation_th = pose_refine_net(new_points, embedding_th, idx)
    quaternion_th = quaternion_th.view(1, 1, -1)
    quaternion_th = quaternion_th / torch.norm(quaternion_th, dim=2, keepdim=True)
    Q_2 = quaternion_th.view(-1).cpu().numpy()
    t_2 = translation_th.view(-1).cpu().numpy()
    Rt_2 = quaternion_matrix(Q_2)
    Rt_2[:3, -1] = t_2

    #Rt_final = np.einsum('ij,kj->ki', Rt_2, Rt)
    Rt_final = np.matmul(Rt, Rt_2)
    R_final = copy.deepcopy(Rt_final)
    R_final[:3, 3] = 0
    Q_final = quaternion_from_matrix(R_final, True)
    t_final = np.array([Rt_final[0, 3], Rt_final[1, 3], Rt_final[2, 3]])

    Q = Q_final
    Pt = t_final

  Rt = quaternion_matrix(Q)
  Rt[:3, -1] = Pt

  return Rt


total_images = 26
for i in range(0, total_images):
  print(f'Inferencing {i:2d}/{total_images}')
  color = load_image(f"/data/color_0000{i:02d}.png")
  depth = load_image(f"/data/depth_0000{i:02d}.png", gray16=True)
  assert color is not None
  assert depth is not None

  depth = depth * DEPTH_SCALE

  print(' - Forward MaskRCNN')

  outputs = predictor(color[...,::-1])
  instances = outputs['instances'].to('cpu')

  fig = plt.figure()
  height, width = color.shape[:2]
  plt.imshow(color)

  print(' - Forward PoseNet')

  for model_name, cat_id in conf.maskrcnn.labels.items():
    cat_instances = instances[instances.pred_classes == cat_id]
    if len(cat_instances) == 0:
      continue

    Rt = detect_pose(color, depth, cat_instances, cat_id)

    model_conf = conf.models[model_name]
    model = o3d.io.read_point_cloud(model_conf.path)

    axis_points = np.asarray([
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1]
    ], dtype=np.float64) * 0.05

    axis_points = cam.project(axis_points, Rt)
    utils.draw_axis(axis_points)

    aabb = model.get_axis_aligned_bounding_box()
    aabb_points = np.asarray(aabb.get_box_points())
    aabb_points = cam.project(aabb_points, Rt)

    utils.draw_bbox(aabb_points, model_conf.color)
  
  output_path = os.path.join(OUTPUT_DIR, f'pose_0000{i:02d}.png')
  plt.xlim((0, width))
  plt.ylim((0, height))
  plt.axis('off')
  plt.gca().invert_yaxis()
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  fig.savefig(
    output_path,
    dpi=200,
    facecolor='white',
    bbox_inches='tight',
    pad_inches=0.0
  )
  print(f'Visualization saved to: {output_path}')

  plt.close('all')
  
  
