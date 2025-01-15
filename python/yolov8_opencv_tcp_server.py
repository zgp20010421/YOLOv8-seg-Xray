#===----------------------------------------------------------------------===#
#
# Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
import os
import json
import time
import cv2
import socket
from datetime import datetime
import argparse
import numpy as np
import sophon.sail as sail
import logging
from postprocess_numpy import PostProcess
# from utils import class_names
from utils import *
logging.basicConfig(level=logging.INFO)

class YOLOv8:
    def __init__(self, args):
        # load bmodel
        self.net = sail.Engine(args.bmodel, args.dev_id, sail.IOMode.SYSIO)
        logging.info("load {} success!".format(args.bmodel))
        self.graph_name = self.net.get_graph_names()[0]
        self.input_name = self.net.get_input_names(self.graph_name)[0]
        self.output_names = self.net.get_output_names(self.graph_name)
        self.input_shape = self.net.get_input_shape(self.graph_name, self.input_name)

        self.batch_size = self.input_shape[0]
        self.net_h = self.input_shape[2]
        self.net_w = self.input_shape[3]
        
        self.conf_thresh = args.conf_thresh
        self.nms_thresh = args.nms_thresh
        self.agnostic = False
        self.multi_label = False
        self.max_det = 300

        self.postprocess = PostProcess(
            conf_thres=self.conf_thresh,
            iou_thres=self.nms_thresh
        )

        # add tcp_server
        self.server_ip = args.server_ip
        self.server_port = args.server_port
        self.now = None
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.server_ip, self.server_port))
        self.server_socket.listen(10)

        # add class_names
        self.class_names = class_names
        
        # Related to TPU post-processing
        if 'use_tpu_opt' in getattr(args, '__dict__', {}):
            self.use_tpu_opt = args.use_tpu_opt
        else:
            self.use_tpu_opt = False
        
        self.tpu_opt_process = None
        self.handle = sail.Handle(args.dev_id)

        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0      
       
    def init(self):
        self.preprocess_time = 0.0
        self.inference_time = 0.0
        self.postprocess_time = 0.0

    def preprocess(self, ori_img):
        """
        pre-processing
        Args:
            img: numpy.ndarray -- (h,w,3)

        Returns: (3,h,w) numpy.ndarray after pre-processing

        """
        letterbox_img, ratio, (tx1, ty1) = self.letterbox(
            ori_img,
            new_shape=(self.net_h, self.net_w),
            color=(114, 114, 114),
            auto=False,
            scaleFill=False,
            scaleup=True,
            stride=32
        )

        img = letterbox_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = img.astype(np.float32)
        img = np.ascontiguousarray(img / 255.0)
        return img, ratio, (tx1, ty1) 

    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

        return im, ratio, (dw, dh)

    def predict(self, input_img, img_num):
        input_data = {self.input_name: input_img}
        outputs = self.net.process(self.graph_name, input_data)

        # resort
        out_keys = list(outputs.keys())
        ord = []
        for n in self.output_names:
            for i, k in enumerate(out_keys):
                if n == k:
                    ord.append(i)
                    break
        out = [outputs[out_keys[i]][:img_num] for i in ord]

        return out

    def __call__(self, img_list):
        img_num = len(img_list)
        ori_size_list = []
        preprocessed_img_list = []
        ratio_list = []
        txy_list = []
        for ori_img in img_list:
            ori_h, ori_w = ori_img.shape[:2]
            ori_size_list.append((ori_h, ori_w))
            start_time = time.time()
            preprocessed_img, ratio, (tx1, ty1) = self.preprocess(ori_img)
            self.preprocess_time += time.time() - start_time
            preprocessed_img_list.append(preprocessed_img)
            ratio_list.append(ratio)
            txy_list.append([tx1, ty1])
        
        if img_num == self.batch_size:
            input_img = np.stack(preprocessed_img_list)
        else:
            input_img = np.zeros(self.input_shape, dtype='float32')
            input_img[:img_num] = np.stack(preprocessed_img_list)
        
        start_time = time.time()
        outputs = self.predict(input_img, img_num)
        self.inference_time += time.time() - start_time

        # TPU post processing
        if self.use_tpu_opt:
            getmask_bmodel_path = args.getmask_bmodel
            
            if self.tpu_opt_process is None:
                detection_shape = list(outputs[0].shape)  # 4 116 8400
                segmentation_shape = list(outputs[1].shape) # 4 32 160 160  
                self.tpu_opt_process = sail.algo_yolov8_seg_post_tpu_opt(getmask_bmodel_path, args.dev_id, detection_shape, segmentation_shape, self.net_h, self.net_w)
            
            results = []
            for i in range(img_num):
                
                detection_input = dict(detection_input = sail.Tensor(self.handle, outputs[0][i:i+1, :, :], True))
                segmentation_input = dict(segmentation_input = sail.Tensor(self.handle, outputs[1][i:i+1, :, :, :], True))
                
                start_time = time.time()
                results_sail = self.tpu_opt_process.process(detection_input, segmentation_input, ori_size_list[i][0], ori_size_list[i][1], self.conf_thresh, self.nms_thresh, True, False)
                self.postprocess_time += time.time() - start_time
                
                boxes = []
                contours = []
                masks = []
                for item in results_sail:
                    boxes.append(list(item[:6]))
                    contours.append([item[6]])
                    masks.append(np.array(item[7]))
                
                result_tuple = (boxes, contours, masks)
                results.append(result_tuple)
              
        else:
            start_time = time.time()
            results = self.postprocess(outputs, ori_size_list, ratio_list, txy_list)
            self.postprocess_time += time.time() - start_time
        return results

def main(args):
    # check params
    if not os.path.exists(args.bmodel):
        raise FileNotFoundError('{} is not existed.'.format(args.bmodel))
    if args.use_tpu_opt:
        if not os.path.exists(args.getmask_bmodel):
            raise FileNotFoundError('{} is not existed.'.format(args.getmask_bmodel))
    # creat save path
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_img_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir) 

    yolov8 = YOLOv8(args)
    batch_size = yolov8.batch_size
    
    # warm up 
    # for i in range(10):
    #     results = yolov8([np.zeros((640, 640, 3))])
    yolov8.init()

    decode_time = 0.0
    cn = 0
    # test images
    while True:
        client_socket, client_address = yolov8.server_socket.accept()
        logging.info(f"客户端 {client_address} 已连接")
        try:
            # 接收图像数据长度
            length_data = receive_data(client_socket, 4)
            if not length_data:
                logging.error("未接收到图像数据长度")
                client_socket.close()
                continue
            image_length = int.from_bytes(length_data, 'big')
            logging.info(f"image_length: {image_length}")
            # 接收图像数据
            image_data = receive_data(client_socket, image_length)
            if not image_data:
                logging.error("未接收到图像数据")
                return
            # 处理图像（这里简单地打印图像形状）
            yolov8.now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            logging.info(f"接收到图像长度: {len(image_data)},时间: {yolov8.now}")

            # 将接收到的字节数据转换为图像 decode
            img_list = []
            results_list = []
            start_time = time.time()
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                    logging.error("images imdecode is None.")
                    continue
            if len(image.shape) != 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            decode_time += time.time() - start_time
            img_list.append(image)
            if (len(img_list) == batch_size and len(img_list)):
                # predict
                results = yolov8(img_list) 
                for i in range(len(results)): 
                    cn += 1
                    boxes, segments, masks = results[i]
                    current_time = time.time()
                    # 将时间戳转换为本地时间的时间元组
                    local_time = time.localtime(current_time)
                    formatted_time = time.strftime('%Y_%m_%d_%H_%M_%S', local_time)
                    save_basename = 'res_opencv_tcp_{}_{}'.format('x_ray', formatted_time)
                    save_name = os.path.join(output_img_dir, save_basename)

                    pred_im = yolov8.postprocess.draw_and_visualize(save_name, img_list[i], boxes, segments, vis=False, save=True)
                    
                    res_dict = dict()
                    res_dict['image_name'] = save_basename
                    res_dict['class_num'] = len(yolov8.class_names)
                    res_dict['bboxes'] = []
                    res_dict['segs'] = []
                    for idx in range(len(boxes)):
                        rles = single_encode(masks[idx])
                        bbox_dict = dict()
                        x1, y1, x2, y2, score, category_id = boxes[idx]
                        bbox_dict['bbox'] = [float(round(x1, 3)), float(round(y1, 3)), float(round(x2 - x1,3)), float(round(y2 -y1, 3))]
                        bbox_dict['bbox'] = [int(float(round(x1, 3))), int(float(round(y1, 3))), int(float(round(x2 - x1,3))), int(float(round(y2 -y1, 3)))]
                        bbox_dict['category_id'] = int(category_id)
                        bbox_dict['score'] = float(round(score,5))
                        bbox_dict['class_name'] = yolov8.class_names[int(category_id)]
                        bbox_dict['time:'] = yolov8.now
                        res_dict['bboxes'].append(bbox_dict)
                        res_dict['segs'].append(rles)
                        
                    results_list.append(res_dict)
                img_list.clear()
            img_list.clear()
        
            json_name = os.path.split(args.bmodel)[-1] + "_opencv" + "_python_result_{}.json".format(formatted_time)
            with open(os.path.join(output_dir, json_name), 'w') as jf:
                # json.dump(results_list, jf)
                json.dump(results_list, jf, indent=4, ensure_ascii=False)
                results_list.clear()
            logging.info("result saved in {}".format(os.path.join(output_dir, json_name)))

            # 发送图像数据回客户端
            _, image_encode = cv2.imencode('.jpg', pred_im) # 图像编码
            image_bytes = image_encode.tobytes() # 图像字节流
            send_data(client_socket, image_bytes)
            yolov8.now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            logging.info(f"发送客户端的图像长度: {len(pred_im)},时间: {yolov8.now}")

            # 发送JSON文件
            with open(os.path.join(output_dir, json_name), 'r') as file:
                json_data = json.load(file)
            # 将JSON数据转换为字节流
            json_string = json.dumps(json_data)
            data_to_send = json_string.encode('utf-8')
            # 发送数据
            send_data(client_socket, data_to_send)
            yolov8.now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            logging.info(f"发送客户端JSON文件: {json_name},时间: {yolov8.now}")

        except socket.error as e:
            logging.error(f"处理客户端连接时出现错误: {e}")
            client_socket.close()
        finally:
            client_socket.close()

        # calculate speed  
        logging.info("------------------ Predict Time Info ----------------------")
        decode_time = decode_time / cn
        preprocess_time = yolov8.preprocess_time / cn
        inference_time = yolov8.inference_time / cn
        postprocess_time = yolov8.postprocess_time / cn
        logging.info("decode_time(ms): {:.2f}".format(decode_time * 1000))
        logging.info("preprocess_time(ms): {:.2f}".format(preprocess_time * 1000))
        logging.info("inference_time(ms): {:.2f}".format(inference_time * 1000))
        logging.info("postprocess_time(ms): {:.2f}".format(postprocess_time * 1000))
        logging.info("------------------ Finish Predict result ----------------------\n")

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--output_dir', type=str, default='./results', help='save of path')
    parser.add_argument('--server_ip', type=str, default='192.168.10.1', help='server of ip')
    parser.add_argument('--server_port', type=int, default=6666, help='server of port')
    parser.add_argument('--bmodel', type=str, default='../models/bm1684/yolov8m_Xray_F32_1b.bmodel', help='path of bmodel')
    parser.add_argument('--dev_id', type=int, default=0, help='dev id')
    parser.add_argument('--conf_thresh', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.7, help='nms threshold')
    parser.add_argument('--use_tpu_opt', action="store_true", default=False, help='use TPU to accelerate postprocessing')
    parser.add_argument('--getmask_bmodel', type=str, default='../models/yolov8s_getmask_32_fp32.bmodel', help='path of getmask bmodel')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = argsparser()
    main(args)
    print('all done.')