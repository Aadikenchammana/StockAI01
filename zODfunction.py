import time
from pathlib import Path
import os
import subprocess
import sys
import pytz
from datetime import datetime
import calendar
import json
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import matplotlib.pyplot as plt

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import glob



#subprocess.run(['pip', 'install', '-qr', 'requirements.txt'])
#------------------------------------------------------------------------
# A I   P R E P 
#------------------------------------------------------------------------
def instancePrint(lst):
    string = ""
    for item in lst:
        string+= " "+str(item)
    string = string[1:]
    print("zOD:",string)
def clear_jpg_files(directory,typ):
    # Get all jpg files in the directory
    if typ == "jpg":
        jpg_files = glob.glob(os.path.join(directory, '*.jpg'))
    else:
        jpg_files = glob.glob(os.path.join(directory, '*.txt'))
    
    # Iterate over each jpg file
    for jpg_file in jpg_files:
        if (os.path.exists(jpg_file)):
            os.remove(jpg_file)
    instancePrint(["DIRECTORY CLEARED:",directory])
def read_json_file(file_path,prices):
    max_attempts = 10
    current_attempt = 1

    while current_attempt <= max_attempts:
        try:
            with open(file_path) as file:
                data = json.load(file)
            return data
        except json.JSONDecodeError:
            instancePrint([f"Attempt {current_attempt}: Failed to read JSON file. Retrying..."])
            current_attempt += 1
            time.sleep(0.05)  # Wait for 1 second before retrying

    instancePrint(["Max attempts reached. Unable to read JSON file."])
    return prices

def ma(lst,interval):
    if interval == 0:
        return lst
    output = []
    ma_lst = []
    for item in lst:
        ma_lst.append(item)
        if len(ma_lst) > interval:
            ma_lst.pop(0)
        result = 0
        for val in ma_lst:
            result+= float(val)
        result = result/len(ma_lst)
        output.append(result)
    return output

def current_time(timezone):
    time = pytz.timezone(timezone) 
    time = datetime.now(time)
    
    day = calendar.day_name[time.weekday()]
    dt = time.strftime("%Y-%m-%d %H:%M:%S")
    time = time.strftime("%H:%M:%S")

    return dt, time, day

def extract_hs(current_symbols,current_prices,dt_list,name,dimension):
    with open('zODworkspace//results//'+name+'.txt', 'r') as f:
        bbox = f.read().strip()
    bboxes = bbox.splitlines()
    output = {}
    for bbox in bboxes:
        bbox = bbox.split()
        x_center = float(bbox[1])
        y_center = float(bbox[2])
        width = float(bbox[3])
        height = float(bbox[4])
        if 0/dimension < y_center < 1/dimension:
            if 0/dimension < x_center < 1/dimension:
                typ = 1
                x_dimension = 0
                y_dimension = 0
            elif 1/dimension < x_center < 2/dimension:
                typ = 2
                x_dimension = 1
                y_dimension = 0
            elif 2/dimension < x_center < 3/dimension:
                typ = 3
                x_dimension = 2
                y_dimension = 0
            elif 3/dimension < x_center < 4/dimension:
                typ = 4
                x_dimension = 3
                y_dimension = 0
        elif 1/dimension < y_center < 2/dimension:
            if 0/dimension < x_center < 1/dimension:
                typ = 5
                x_dimension = 0
                y_dimension = 1
            elif 1/dimension < x_center < 2/dimension:
                typ = 6
                x_dimension = 1
                y_dimension = 1
            elif 2/dimension < x_center < 3/dimension:
                typ = 7
                x_dimension = 2
                y_dimension = 1
            elif 3/dimension < x_center < 4/dimension:
                typ = 8
                x_dimension = 3
                y_dimension = 1
        elif 2/dimension < y_center < 3/dimension:
            if 0/dimension < x_center < 1/dimension:
                typ = 9
                x_dimension = 0
                y_dimension = 2
            elif 1/dimension < x_center < 2/dimension:
                typ = 10
                x_dimension = 1
                y_dimension = 2
            elif 2/dimension < x_center < 3/dimension:
                typ = 11
                x_dimension = 2
                y_dimension = 2
            elif 3/dimension < x_center < 4/dimension:
                typ = 12
                x_dimension = 3
                y_dimension = 2
        elif 3/dimension < y_center < 4/dimension:
            if 0/dimension < x_center < 1/dimension:
                typ = 13
                x_dimension = 0
                y_dimension = 3
            elif 1/dimension < x_center < 2/dimension:
                typ = 14
                x_dimension = 1
                y_dimension = 3
            elif 2/dimension < x_center < 3/dimension:
                typ = 15
                x_dimension = 2
                y_dimension = 3
            elif 3/dimension < x_center < 4/dimension:
                typ = 16
                x_dimension = 3
                y_dimension = 3
        instancePrint ([typ])
        lst = current_prices[typ-1]
        symb = current_symbols[typ-1]

        size = 1262
        individual = 275
        outer = 25
        inner = 37

        width = width*size/individual
        height = height*size/individual

        x_center = (x_center*size - (x_dimension)*individual-x_dimension*inner-outer)/individual
        y_center = (y_center*size - (y_dimension)*individual-y_dimension*inner-outer)/individual

        x_ln = len(lst)
        y_ln = max(lst) - min(lst)

        #"""
        x_max = int((x_center+width/2)*x_ln)
        x_min = int((x_center-width/2)*x_ln)
        y_max = int((y_center+height/2)*y_ln)
        y_min = int((y_center-height/2)*y_ln)

        diff = x_ln - x_max
        x_max += diff
        x_min += diff
        flag = True
        if "1" in dt_list:
            flag = False
            if dt_list.index("1") < 130:
                flag = True
        flag = True
        instancePrint([symb])
        instancePrint([x_center,y_center,height,width])
        if x_center > 0.7 and len(dt_list) - x_min < 200:
            output[symb] = dt_list[x_min-1]
    instancePrint([output])
    return output

def predicting(dataset,source, weights, view_img, save_txt, imgsz, trace,device,half,model,classify,webcam,save_dir,names,save_img,colors,conf_thres, iou_thres, save_conf, nosave, classes, agnostic_nms, update, project, name, exist_ok,old_img_b,old_img_w,old_img_h,augment):
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        if webcam:  # batch_size >= 1
            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
        else:
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # img.jpg
        txt_path = str(save_dir / p.stem)  # img.txt
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        with open(txt_path + '.txt', 'w') as f:
            f.write('')
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / p.stem)  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    #instancePrint("XYXY",xyxy)
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        string = str(int(xyxy[0]))+" "+str(int(xyxy[1]))+" "+str(int(xyxy[2]))+" "+str(int(xyxy[3]))
                        with open(txt_path + '.txt', 'a') as f:
                              #f.write(string+'\n')
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            instancePrint([f'{s}{(1E3 * (t2 - t1)):.1f}'])

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    #instancePrint(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)








def OD():

    instancePrint(["YURRRR"])

    #------------------------------------------------------------------------

    with torch.no_grad():
        instancePrint(["1"])
        source, weights, view_img, save_txt, imgsz, trace, device = 'zODworkspace//save','zODweights_new//weights4_1_1.pt', False, False, 640, True,''
        conf_thres, iou_thres, save_conf, nosave, classes, agnostic_nms, update, project, name, exist_ok,augment = 0.25, 0.45, False, False, None, False, False, 'runs/detect', 'exp', False,False
        print(source, weights, view_img, save_txt, imgsz, trace)
        print(type(source),type(weights))
        save_img = False#True
        save_txt = True
        webcam = False
        save_dir = Path("zODworkspace//results")
        set_logging()
        total_device = select_device("")
        device = total_device#select_device(device)#opt.device)
        print(device)
        half = device.type != 'cpu'
        model = attempt_load(weights, map_location=device)
        stride = int(model.stride.max())
        imgsz = check_img_size(imgsz, s=stride)
        instancePrint(["2"])
        if trace:
            model = TracedModel(model, device, imgsz)#opt.img_size)
        if half:
            model.half()
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        instancePrint(["3"])
        #-----------------------------------
        #WORKSPACE
        #-----------------------------------
        status = "start"
        plot_flag = True
        AI_flag = True
        dimension = 4
        x_size = 1262 #230
        y_size = 1262 #170
        my_dpi = 121
        prices = {}
        max_ln = 180
        with open('zODworkspace//watchlist.txt', 'w') as f:
            json.dump({}, f)
        with open('zODworkspace//total_watchlist.txt', 'w') as f:
            json.dump({}, f)
        clear_jpg_files("zODworkspace//save","jpg")
        clear_jpg_files("zODworkspace//results","jpg") 
        clear_jpg_files("zODworkspace//results","txt")
        start_time = time.time()
        while True:
            instancePrint(["--------------------"])
            instancePrint(["STARTING NEW ITERATION"])
            instancePrint(["TIME",time.time()-start_time,time.time(),start_time])
            instancePrint(["--------------------"])
            start_time = time.time()
            dt,now,day = current_time("America/New_York")
            if plot_flag:
                prices = read_json_file('zODworkspace//prices.txt',prices)
                dt_list = prices["dt_list"]
                dt_list = dt_list[-max_ln:]
                del prices["dt_list"]
                symbols = list(prices.keys())
                iteration = 0
                current_symbols = []
                current_prices = []
                plt.figure(figsize=(x_size/my_dpi, y_size/my_dpi), dpi=my_dpi)
                plt.subplots_adjust(left=0, right=1, bottom=0, top=1,hspace=0.086, wspace=0.086)
                if len(symbols)%(dimension**2) != 0:
                    for i in range(dimension**2 - len(symbols)%(dimension**2)):
                        symbols.append('temp_symbol'+str(i))
                        prices['temp_symbol'+str(i)] = [0 for item in dt_list]
                for symbol in symbols:
                    ttemp = time.time()
                    iteration+=1
                    y = prices[symbol]
                    y = y[-max_ln:]
                    y = ma(y,10)
                    current_symbols.append(symbol)
                    current_prices.append(y)
                    plt.subplot(dimension,dimension,iteration)
                    plt.plot(y,label=str(symbol),antialiased=False)
                    plt.gca().spines['top'].set_visible(False)
                    plt.gca().spines['bottom'].set_visible(False) 
                    plt.gca().spines['left'].set_visible(False)
                    plt.gca().spines['right'].set_visible(False)
                    plt.gca().set_xlabel('')
                    plt.gca().set_ylabel('')
                    plt.gca().set_xticks([])
                    plt.gca().set_yticks([])
                    print("PLOT TIME",time.time()-ttemp)
                    if iteration == dimension**2:
                        ttemp = time.time()
                        name = ""
                        for symb in current_symbols:
                            if name == "":
                                name += symb
                            else:
                                name += ","+symb
                        
                        file_name = "zODworkspace//save//"+name+".jpg"
                        instancePrint([name])
                        plt.savefig(file_name, format='jpeg')
                        print("SAVE TIME",time.time()-ttemp)
                        ttemp = time.time()
                        if AI_flag:
                            #data
                            dataset = LoadImages(source, img_size=imgsz, stride=stride)
                            # prediction 
                            if device.type != 'cpu':
                                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                            old_img_w = old_img_h = imgsz
                            old_img_b = 1
                            t0 = time.time()
        
                            predicting(dataset,source, weights, view_img, save_txt, imgsz, trace,device,half,model,classify,webcam,save_dir,names,save_img,colors,conf_thres, iou_thres, save_conf, nosave, classes, agnostic_nms, update, project, name, exist_ok,old_img_b,old_img_w,old_img_h,augment)
                            ttemp = time.time()
                            print("PREDICTION TIME",ttemp-t0)
                            results = extract_hs(current_symbols,current_prices,dt_list,name,dimension)
                            print("EXTRACTION TIME",time.time()-ttemp)
                            ttemp = time.time()
                            with open('zODworkspace//watchlist.txt', 'r') as f:
                                watchlist = json.loads(f.read())
                            with open('zODworkspace//total_watchlist.txt', 'r') as f:
                                total_watchlist = json.loads(f.read())
                            PD_intervals = [5,20]
                            for key in results.keys():
                                for interval in PD_intervals:
                                    watchlist[key+"_"+str(interval)] = results[key]
                                    total_watchlist[key+"_"+str(interval)] = results[key]

                            sorted_keys = sorted(list(watchlist.keys()))
                            sorted_watchlist = {}
                            for key in sorted_keys:
                                sorted_watchlist[key] = watchlist[key]
                            watchlist = sorted_watchlist
                            with open('zODworkspace//watchlist.txt', 'w') as f:
                                json.dump(watchlist, f)
                            with open('zODworkspace//total_watchlist.txt', 'w') as f:
                                json.dump(total_watchlist, f)
                            print("FINAL TIME",time.time()-ttemp)
                            ttemp = time.time()
                        os.remove(file_name)
                        plt.close('all')
                        plt.figure(figsize=(x_size/my_dpi, y_size/my_dpi), dpi=my_dpi)
                        plt.subplots_adjust(left=0, right=1, bottom=0, top=1,hspace=0.086, wspace=0.086)
                        iteration = 0
                        current_symbols = []
                        current_prices = []
                        ttemp = time.time()
            if time.time()-start_time < 30:
                instancePrint(["TIME TAKEN:",(time.time()-start_time)])
                time.sleep(int(30-(time.time()-start_time)))

