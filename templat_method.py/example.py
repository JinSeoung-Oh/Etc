##  I want to Add def box_coor 
##  At first, define box_coor function in templte
##    def box_coor(self, results, batch):
##        #print('boxbox')
##        return results

## And then, have to define box_coor function in this script very deteail.

import torch
from ultralytics.yolo.engine.results import Results
from predict import BasePredictor 
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import argparse
import cv2

class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=1, example='yolov8n-face.pt')


    #def preprocess(self, img):
    #    img = img[0]
    #    img = np.array(img)
    #    img = img.transpose((2, 0, 1))[::-1]
    #    img = img[np.newaxis, :,:,:]
    #    print(img.shape)
    #    img = torch.from_numpy(img.copy()).to("cuda")
    #    img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
    #    img /= 255  # 0 - 255 to 0.0 - 1.0
    #    return img

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img, classes=None):
        preds = ops.non_max_suppression(preds,
                                        0.25,
                                        0.7,
                                        agnostic=False,
                                        max_det=1000,
                                        classes=None)

        #print('!!!', preds)
        results = []
        for i, pred in enumerate(preds):
            #print('???',pred)
            shape = orig_img[i].shape if isinstance(orig_img, list) else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
            #print('?!?!?!?', pred)
            results.append(Results(boxes=pred, orig_shape=shape[:2]))
        return results

    def write_results(self, idx, results, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam or self.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = results[idx].boxes  
        if len(det) == 0:
            return log_string
        for c in det.cls.unique():
            n = (det.cls == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        # write
        for d in reversed(det):
            cls, conf = d.cls.squeeze(), d.conf.squeeze()
            if self.args.save_txt:  # Write to file
                line = (cls, *(d.xywhn.view(-1).tolist()), conf) \
                    if self.args.save_conf else (cls, *(d.xywhn.view(-1).tolist()))  # label format
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                self.annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))
            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(d.xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string


   # add box_coor def from BasePredictor
    def box_coor(self, idx, results):
        #print('????')
        det = results[idx].boxes
        return det


def predict(cfg=DEFAULT_CFG):
    cfg.model = "yolov8-face/yolov8n-face.pt"
    origin_path = 'yolov8-face/examples/original'
    part_path = 'yolov8-face/examples/after'
    ori_path = [(os.path.join(origin_path, f)) for f in os.listdir(origin_path)]
    after_path = [(os.path.join(part_path, f)) for f in os.listdir(part_path)]

    box_info = {}
   
    cfg.source_ori = ori_path[0]
    #'yolov8-face/examples/test_de.png'
   
    predictor = DetectionPredictor(cfg)
    box_ori = predictor.stream_inference(source = cfg.source_ori, model = cfg.model, verbose=True)

    for data in box_ori:
        ori_coor = data.boxes
        box_info['ori'] = ori_coor

    boxes = []
    for i in range(len(after_path)):
        path = after_path[i]
        cfg.source_part = path
        box_part = predictor.stream_inference(source = cfg.source_part, model = cfg.model, verbose=True)
        for data in box_part:
            part = data.boxes
            #print('???',part[0])
            boxes.append(part)
           
    box_info['after'] = boxes    
       

    return box_info
   

box = predict()
   

ori_direct = 'yolov8_face/examples/original'
after_direct = 'yolov8_face/examples/after'

ori_path = [(os.path.join(ori_direct, f)) for f in os.listdir(ori_direct)]
after_path = [(os.path.join(after_direct, f)) for f in os.listdir(after_direct)]


for i in range(len(ori_path)):
    ori_img = cv2.imread(str(ori_path[i]))
    out_path = './yolov8_face/face_result/' + 'face_deidentity_' + str(i) +'.png'
   
    for k in ragne(len(after_path)):

        after_img = cv2.imread(str(after_path[i]))

        ori_box = box['ori'].xyxy
        after_box = box['after'][0].xyxy

        for i in range(len(ori_box)):
            ori = ori_box[i].detach().cpu().numpy()
            print(ori)
    #af = after_box[i].detach().cpu().numpy()

            x_min = int(ori[0])
            y_min = int(ori[1])
            x_max = int(ori[2])
            y_max = int(ori[3])

            af_face = after_img[y_min:y_max, x_min:x_max]
            ori_img[y_min:y_max, x_min:x_max] = af_face
   
   
         cv2.imwrite(out_path, ori_img)
