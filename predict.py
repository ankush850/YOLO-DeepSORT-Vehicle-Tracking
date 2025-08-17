import hydra
import torch
import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque, defaultdict
import numpy as np
import math
import threading

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
data_deque = {}
deepsort = None
object_counter = {}
object_counter1 = {}
line = [(100, 500), (1050, 500)]
_lock = threading.Lock()

class EMA:
    def __init__(self, beta=0.9, init=0.0):
        self.beta = beta
        self.v = init
        self.ready = False
    def update(self, x):
        if not self.ready:
            self.v = x
            self.ready = True
        else:
            self.v = self.beta * self.v + (1 - self.beta) * x
        return self.v
    def value(self):
        return self.v

class Stopwatch:
    def __init__(self):
        self.t = None
    def start(self):
        self.t = time.time()
    def stop(self):
        if self.t is None:
            return 0.0
        d = time.time() - self.t
        self.t = None
        return d

class DynThreshold:
    def __init__(self, base=0.25, min_v=0.1, max_v=0.7):
        self.base = base
        self.min_v = min_v
        self.max_v = max_v
        self.ema_motion = EMA(0.85, 0.0)
        self.ema_size = EMA(0.85, 0.0)
    def update(self, motion_mag, mean_area):
        m = self.ema_motion.update(motion_mag)
        s = self.ema_size.update(mean_area)
        v = self.base + 0.15 * (1 - np.tanh(m)) + 0.1 * (1 - np.tanh(s))
        return float(max(self.min_v, min(self.max_v, v)))

class FrameSkipper:
    def __init__(self, min_stride=1, max_stride=5):
        self.min_stride = min_stride
        self.max_stride = max_stride
        self.ema_fps = EMA(0.9, 30.0)
        self.ema_load = EMA(0.9, 0.5)
        self.cnt = 0
    def update_and_should_process(self, fps, load_hint=0.5):
        f = self.ema_fps.update(fps)
        l = self.ema_load.update(load_hint)
        target = 1 if f < 25 else 2 if f < 35 else 3 if f < 45 else 4
        if l > 0.75:
            target = min(self.max_stride, target + 1)
        target = max(self.min_stride, min(self.max_stride, target))
        self.cnt = (self.cnt + 1) % target
        return self.cnt == 0

def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    use_cuda = torch.cuda.is_available()
    deepsort = DeepSort(
        cfg_deep.DEEPSORT.REID_CKPT,
        max_dist=cfg_deep.DEEPSORT.MAX_DIST,
        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg_deep.DEEPSORT.MAX_AGE,
        n_init=cfg_deep.DEEPSORT.N_INIT,
        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
        use_cuda=use_cuda,
    )

def xyxy_to_xywh(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):
    if label == 0:
        color = (85,45,255)
    elif label == 2:
        color = (222,82,175)
    elif label == 3:
        color = (0, 204, 255)
    elif label == 5:
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def get_direction(point1, point2):
    s = ""
    if point1[1] > point2[1]:
        s += "South"
    elif point1[1] < point2[1]:
        s += "North"
    else:
        s += ""
    if point1[0] > point2[0]:
        s += "East"
    elif point1[0] < point2[0]:
        s += "West"
    else:
        s += ""
    return s

def motion_magnitude(q):
    if len(q) < 2:
        return 0.0
    s = 0.0
    n = 0
    for i in range(1, len(q)):
        a = q[i-1]
        b = q[i]
        if a is None or b is None:
            continue
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])
        s += math.sqrt(dx*dx + dy*dy)
        n += 1
    return s / max(1, n)

def area_mean(bbox):
    if len(bbox) == 0:
        return 0.0
    a = 0.0
    for x1,y1,x2,y2 in bbox:
        w = max(0.0, float(x2 - x1))
        h = max(0.0, float(y2 - y1))
        a += w*h
    return a / max(1, len(bbox))

def aspect_valid(x1,y1,x2,y2, min_ar=0.2, max_ar=6.0):
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    ar = max(w,h) / min(w,h)
    return (ar >= 1.0 and ar <= max_ar) or (ar < 1.0 and (1.0/ar) <= max_ar)

def filter_det(det, img_shape, min_area=50*50, max_area_ratio=0.7, min_h=12, min_w=12):
    if len(det) == 0:
        return det
    H, W = img_shape[:2]
    max_area = max_area_ratio * W * H
    out = []
    for *xyxy, conf, cls in det:
        x1,y1,x2,y2 = [float(v) for v in xyxy]
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        a = w*h
        if a < min_area:
            continue
        if a > max_area:
            continue
        if w < min_w or h < min_h:
            continue
        if not aspect_valid(x1,y1,x2,y2):
            continue
        out.append([x1,y1,x2,y2, float(conf), float(cls)])
    if len(out) == 0:
        return det[:0]
    t = torch.tensor(out, device=det.device if hasattr(det, "device") else None)
    return t

def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1)
    ih = max(0.0, y2 - y1)
    inter = iw * ih
    aa = max(0.0, (a[2]-a[0])*(a[3]-a[1]))
    bb = max(0.0, (b[2]-b[0])*(b[3]-b[1]))
    u = aa + bb - inter + 1e-9
    return inter / u

def dedup_boxes(det, iou_thr=0.95):
    if len(det) <= 1:
        return det
    arr = det.cpu().numpy()
    keep = []
    used = np.zeros(len(arr), dtype=bool)
    for i in range(len(arr)):
        if used[i]:
            continue
        used[i] = True
        keep.append(i)
        for j in range(i+1, len(arr)):
            if used[j]:
                continue
            if int(arr[i,5]) != int(arr[j,5]):
                continue
            if iou(arr[i,:4], arr[j,:4]) > iou_thr:
                if arr[i,4] >= arr[j,4]:
                    used[j] = True
                else:
                    used[i] = True
                    keep[-1] = j
    idx = np.array(keep, dtype=int)
    t = torch.from_numpy(arr[idx]).to(det.device if hasattr(det,"device") else None)
    return t

def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0)):
    cv2.line(img, line[0], line[1], (46,162,112), 3)
    height, width, _ = img.shape
    lost = []
    for key in list(data_deque):
        if identities is None or key not in identities:
            lost.append(key)
    for k in lost:
        data_deque.pop(k, None)
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        center = (int((x2+x1)/ 2), int((y2+y2)/2))
        id = int(identities[i]) if identities is not None else 0
        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = '{}{:d}'.format("", id) + ":"+ '%s' % (obj_name)
        data_deque[id].appendleft(center)
        if len(data_deque[id]) >= 2:
            direction = get_direction(data_deque[id][0], data_deque[id][1])
            if intersect(data_deque[id][0], data_deque[id][1], line[0], line[1]):
                cv2.line(img, line[0], line[1], (255, 255, 255), 3)
                if "South" in direction:
                    if obj_name not in object_counter:
                        object_counter[obj_name] = 1
                    else:
                        object_counter[obj_name] += 1
                if "North" in direction:
                    if obj_name not in object_counter1:
                        object_counter1[obj_name] = 1
                    else:
                        object_counter1[obj_name] += 1
        UI_box(box, img, label=label, color=color, line_thickness=2)
        for i in range(1, len(data_deque[id])):
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
        for idx, (key, value) in enumerate(object_counter1.items()):
            cnt_str = str(key) + ":" +str(value)
            cv2.line(img, (width - 500,25), (width,25), [85,45,255], 40)
            cv2.putText(img, f'Number of Vehicles Entering', (width - 500, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
            cv2.line(img, (width - 150, 65 + (idx*40)), (width, 65 + (idx*40)), [85, 45, 255], 30)
            cv2.putText(img, cnt_str, (width - 150, 75 + (idx*40)), 0, 1, [255, 255, 255], thickness = 2, lineType = cv2.LINE_AA)
        for idx, (key, value) in enumerate(object_counter.items()):
            cnt_str1 = str(key) + ":" +str(value)
            cv2.line(img, (20,25), (500,25), [85,45,255], 40)
            cv2.putText(img, f'Numbers of Vehicles Leaving', (11, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
            cv2.line(img, (20,65+ (idx*40)), (127,65+ (idx*40)), [85,45,255], 30)
            cv2.putText(img, cnt_str1, (11, 75+ (idx*40)), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
    return img

def to_tensor(img, device, fp16):
    t = torch.from_numpy(img).to(device, non_blocking=True)
    t = t.half() if fp16 else t.float()
    t /= 255.0
    return t

def estimate_scene_motion():
    s = 0.0
    n = 0
    for k,v in data_deque.items():
        s += motion_magnitude(v)
        n += 1
    if n == 0:
        return 0.0
    return s / n

def safe_update_tracker(xywhs, confss, oids, im0):
    global deepsort
    try:
        return deepsort.update(xywhs, confss, oids, im0)
    except Exception:
        try:
            deepsort.use_cuda = False
            return deepsort.update(xywhs, confss, oids, im0)
        except Exception:
            return np.zeros((0, 6), dtype=np.float32)

def pack_xywh_conf_cls(det):
    xywh_bboxs = []
    confs = []
    oids = []
    for *xyxy, conf, cls in det:
        x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
        xywh_obj = [x_c, y_c, bbox_w, bbox_h]
        xywh_bboxs.append(xywh_obj)
        confs.append([float(conf)])
        oids.append(int(cls))
    if len(xywh_bboxs) == 0:
        return None, None, None
    xywhs = torch.tensor(xywh_bboxs)
    confss = torch.tensor(confs)
    return xywhs, confss, oids

def boost_vehicle_conf(det, names, boost=0.05):
    if len(det) == 0:
        return det
    arr = det.clone()
    for i in range(arr.shape[0]):
        c = int(arr[i,5].item())
        if c in (2,3,5,7) if len(names) > 7 else c in (2,3,5):
            arr[i,4] = float(min(0.999, arr[i,4].item() + boost))
    return arr

class DetectionPredictor(BasePredictor):
    def __init__(self, cfg, overrides=None):
        super().__init__(cfg, overrides)
        self._timer = Stopwatch()
        self._fps_ema = EMA(0.9, 30.0)
        self._dyn = DynThreshold(base=float(getattr(cfg, "conf", 0.25)))
        self._skipper = FrameSkipper()
        self._last_det = None
        self._last_out = None
        self._last_im0 = None
        self._frame_i = 0
        self.all_outputs = []
        cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        t = to_tensor(img, self.model.device, self.model.fp16)
        return t

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds, self.args.conf, self.args.iou, agnostic=self.args.agnostic_nms, max_det=self.args.max_det)
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        save_path = str(self.save_dir / p.name)
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]
        self.annotator = self.get_annotator(im0)
        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            self._last_det = None
            self._last_out = None
            self._last_im0 = im0
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        det = filter_det(det, im0.shape, min_area=24*24, max_area_ratio=0.8, min_h=14, min_w=14)
        det = dedup_boxes(det, iou_thr=0.97)
        det = boost_vehicle_conf(det, self.model.names, 0.03)
        motion = estimate_scene_motion()
        scene_mean_area = area_mean(det[:, :4].cpu().numpy()) if len(det) else 0.0
        dyn_conf = self._dyn.update(motion, scene_mean_area/(im0.shape[0]*im0.shape[1]+1e-9))
        self.args.conf = float(max(0.05, min(0.9, dyn_conf)))
        self._timer.start()
        xywh_bboxs = []
        confs = []
        oids = []
        xywhs = None
        confss = None
        outputs = []
        for *xyxy, conf, cls in reversed(det):
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append([conf.item()])
            oids.append(int(cls))
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)
        outputs = safe_update_tracker(xywhs, confss, oids, im0)
        dt = self._timer.stop()
        fps = 1.0 / max(1e-6, dt)
        s_fps = self._fps_ema.update(fps)
        _ = self._skipper.update_and_should_process(s_fps, load_hint=min(1.0, len(det)/50.0))
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            draw_boxes(im0, bbox_xyxy, self.model.names, object_id,identities)
        self._last_det = det
        self._last_out = outputs
        self._last_im0 = im0
        return log_string

@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    init_tracker()
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()

if __name__ == "__main__":
    predict()
