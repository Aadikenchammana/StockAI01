import argparse
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
import shutil
import glob


subprocess.run(['pip', 'install', '-qr', 'requirements.txt'])
#------------------------------------------------------------------------
# A I   P R E P 
#------------------------------------------------------------------------
def clear_jpg_files(directory,typ):
    # Get all jpg files in the directory
    if typ == "jpg":
        jpg_files = glob.glob(os.path.join(directory, '*.jpg'))
    else:
        jpg_files = glob.glob(os.path.join(directory, '*.txt'))
    
    # Iterate over each jpg file
    for jpg_file in jpg_files:
        os.remove(jpg_file)
    print("DIRECTORY CLEARED:",directory)
def determine_valid(lstx,lsty,end_x,end_y):
    mn = min(lsty)
    flag = True
    print(lstx)
    lst = [item-mn for item in lsty]
    print(lsty)
    left_shoulder_x,t1_x,head_x,t2_x,right_shoulder_x = lstx
    left_shoulder,t1,head,t2,right_shoulder = lsty
    diff = max(lsty) - min(lsty)
    if not(left_shoulder > t1 and head > t1 and right_shoulder > t1):
        print("FALSE: HEIGHT 1")
        return False
    if not(left_shoulder > t2 and head > t2 and right_shoulder > t2):
        print("FALSE: HEIGHT 2")
        return False
    if abs(t1-t2)/diff > 0.4:
        print("FALSE: DIFF")
        return False
    if not(2 > left_shoulder/right_shoulder > 0.5):
        print("FALSE: SHOULDERS 1")
        return False
    if not(head/right_shoulder > 1 and head/left_shoulder > 1):
        print("FALSE: HEAD 1")
        return False
    if right_shoulder_x  - left_shoulder_x < 30:
        print("FALSE: LENGTH 1")
        return False
    
    head -= (t1+t2)/2
    left_shoulder -= t1
    right_shoulder -= t2
    if not(3 > left_shoulder/right_shoulder > 0.33):
        print("FALSE: SHOULDERS 2")
        return False
    
    d1 = t1_x - left_shoulder_x
    d2 = head_x - t1_x
    d3 = t2_x - head_x
    d4 = right_shoulder_x - t2_x
    d5 = right_shoulder_x - t2_x

    coef = 4

    if max([d1,d2,d3,d4,d5])/min([d1,d2,d3,d4,d5]) > coef and False:
        print("FALSE: D1")
        return False
    
    return True
def read_json_file(file_path,prices):
    max_attempts = 10
    current_attempt = 1

    while current_attempt <= max_attempts:
        try:
            with open(file_path) as file:
                data = json.load(file)
            return data
        except json.JSONDecodeError:
            print(f"Attempt {current_attempt}: Failed to read JSON file. Retrying...")
            current_attempt += 1
            time.sleep(0.05)  # Wait for 1 second before retrying

    print("Max attempts reached. Unable to read JSON file.")
    return prices
def max_index_find(lst):
    #print("max_finding")
    num = lst[0]
    index = 0
    for i in range(len(lst)):
        if lst[i] >= num:
            num = lst[i]
            index = i
    return num,index
def min_index_find(lst):
    #print("min_finding")
    num = lst[0]
    index = 0
    for i in range(len(lst)):
        if lst[i] <= num:
            num = lst[i]
            index = i
    return num,index
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

def extract_hs(symbol,ma_interval,x,y,ogx,ogy,img_width,img_height):
    #print("--------")
    #print(symbol,ma_interval)
    classes = ["peak","trough"]
    #img_width += 38
    #img_height += 38
    with open('zworkspace//results//'+symbol+"_"+str(ma_interval)+'.txt', 'r') as f:
        bbox = f.read().strip()
    bboxes = bbox.splitlines()
    x_output = []
    y_output = []
    class_output = []
    temp_x = []
    prev = 0
    counter = 0
    xs = []
    class_names = {}
    
    for bbox in bboxes:
        bbox = bbox.split(" ")
        class_num = int(bbox[0])
        x_center = float(bbox[1])
        y_center = float(bbox[2])
        width = float(bbox[3])
        height = float(bbox[4])

        border = 30#86
        x_ln = len(x)
        y_ln = max(y) - min(y)
        
        og_center = x_center
        x_center = (x_center*img_width-border)/(img_width-border*2)
        y_center = (y_center*img_height-border)/(img_height-border*2)
        width = (width*img_width)/(img_width-border*2)
        height = (height*img_height)/(img_height-border*2)
        if int(og_center*100) != 95 and int(x_center*x_ln)!= 0:
            x_center = int(x_center*x_ln)
            class_name = classes[class_num]
            xs.append(x_center)
            class_names [x_center] = class_name
    xs.sort()
    lst = []
    for x_val in xs:
        lst.append(class_names[x_val])
    class_names = lst

    for i in range(len(xs)):
        x_center = xs[i]
        class_name = class_names[i]
        min_index =  x_center - int(ma_interval)*2
        max_index =  x_center+1
        if min_index < 0:
            min_index = 0
        if max_index > x_ln -1:
            max_index = x_ln-1
        if min_index < prev:
                min_index = prev
        if not(i == len(xs)-1):
            if max_index > xs[i+1]:
                max_index = xs[i+1]
        if max_index - min_index > 1:
            y_list = ogy[min_index:max_index]
            if class_name == "peak":
                mx, indx = max_index_find(y_list)
            else:
                mn, indx = min_index_find(y_list)
        else:
            indx = 0
        
        index = min_index+indx
        prev = index
        x_output.append(index)
        y_output.append(ogy[index])
        class_output.append(class_name)
        counter+=1
    new_x = []
    new_y = []
    new_class = []
    for i in range(1,len(x_output)):
        if class_output[i-1] != class_output[i]:
            new_x.append(x_output[i])
            new_y.append(y_output[i])
            new_class.append(class_output[i])
        elif (y_output[i-1] < y_output[i] and class_output[i] == "peak") or (y_output[i-1] > y_output[i] and class_output[i] == "trough"):
            if len(new_x) < 1:
                new_x.append(x_output[i])
                new_y.append(y_output[i])
                new_class.append(class_output[i])
            else:
                nln = len(new_x)-1
                new_x[nln] = x_output[i]
                new_y[nln] = y_output[i]
                new_class[nln] = class_output[i]
    if len(x_output) > 0:
        x_output = [x_output[0]]+new_x
        y_output = [y_output[0]]+new_y
        class_output = [class_output[0]]+new_class
    #x_output,y_output,class_output = new_x,new_y,new_class
    return x_output,y_output,class_output
def add_to_dct_lst(dct,ky,val):
    if ky not in dct.keys():
        dct[ky] = [val]
    else:
        dct[ky] = dct[ky] + [val]
    return dct

def predicting(dataset,source, weights, view_img, save_txt, imgsz, trace,device,half,model,classify,webcam,save_dir,names,save_img,colors):
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
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
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
                    #print("XYXY",xyxy)
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        string = str(int(xyxy[0]))+" "+str(int(xyxy[1]))+" "+str(int(xyxy[2]))+" "+str(int(xyxy[3]))
                        with open(txt_path + '.txt', 'a') as f:
                            #f.write(string+'\n')
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            # Print time (inference + NMS)
            #print(f'{s}{(1E3 * (t2 - t1)):.1f}')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    #print(f" The image with the result is saved in: {save_path}")
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


print("YURRRR")
#"""
parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='PD_weights//weights1_5_3.pt', help='model.pt path(s)')
parser.add_argument('--source', type=str, default='zworkspace//save', help='source')  # file/folder, 0 for webcam
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
opt = parser.parse_args()
print(opt)

#------------------------------------------------------------------------

with torch.no_grad():
    print("-----------------------")
    print("1")
    print("-----------------------")
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = True#not opt.nosave and not source.endswith('.txt')
    save_txt = True
    webcam = False
    save_dir = Path("zworkspace//results")#Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    print(save_dir,type(save_dir))
    #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    if trace:
        model = TracedModel(model, device, opt.img_size)
    if half:
        model.half()
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    print("-----------------------")
    print("2")
    print("-----------------------")
    #-----------------------------------
    #WORKSPACE
    #-----------------------------------
    status = "start"
    x_size = 638 #230
    y_size = 638 #170
    my_dpi = 192
    intervals = [5,20]
    AI_flag = True
    no_fly_list = []
    prices = {}
    watchlist = {}
    compression = 160
    compressions = {}
    shift = 10
    if input("clear files") == "True":
        clear_jpg_files("zworkspace//current","jpg")
        clear_jpg_files("zworkspace//false","jpg")
        clear_jpg_files("zworkspace//found//small","jpg")
        clear_jpg_files("zworkspace//found//small","txt")
        clear_jpg_files("zworkspace//found//big","jpg")
        clear_jpg_files("zworkspace//found//big","txt")
        clear_jpg_files("zworkspace//results","jpg")
        clear_jpg_files("zworkspace//results","txt")
        clear_jpg_files("zworkspace//save","jpg")
        clear_jpg_files("zworkspace//trends","jpg")
    start_time = time.time()
    timers = {}
    timer_intervals = {}
    watchlist = {}
    max_trend_len = 540
    eliminate = []
    max_detection_len = 180
    prices = read_json_file('MS yolo//zworkspace//prices.txt',prices)
    for key in prices.keys():
        for ma_interval in intervals:
            timers[key+"_"+str(ma_interval)] = 0
            timer_intervals[key+"_"+str(ma_interval)] = 1
    while True:
        print("--------------------")
        print("STARTING NEW ITERATION")
        print("TIME",time.time()-start_time)
        print("--------------------")
        start_time = time.time()
        dt,now,day = current_time("America/New_York")
        if (day != "Sunday" or day != "Saturday") and '09:30:00' < now < '16:00:00' or True:
            if status != "open":
                print("MARKET OPEN:",dt)
                status = "open"
            prev_watchlist = watchlist
            watchlist = read_json_file('MS yolo//zworkspace//watchlist.txt',watchlist)
            symbols = list(watchlist.keys())
            remove = []
            prices = read_json_file('MS yolo//zworkspace//prices.txt',prices)
            dt_list = prices["dt_list"]
            print("--------------------")
            print("CURRENT DT:",dt_list[len(dt_list)-1])
            print("--------------------")


            for symbol in prev_watchlist.keys():
                if symbol in no_fly_list:
                    if prev_watchlist[symbol] == watchlist[symbol]:
                        no_fly_list.remove(symbol)
            temp = symbols
            symbols = []
            for symbol in temp:
                start = watchlist[symbol]
                if str(start) in dt_list and symbol not in no_fly_list:
                    lst = prices[symbol.split("_")[0]]
                    start_index = dt_list.index(start) - shift
                    lst = lst[start_index:]
                    x = []
                    for i in range(len(lst)):
                        x.append(i)
                    if len(x) > 180:
                        remove.append(symbol)
                    else:
                        symbols.append(symbol)
                elif symbol in no_fly_list:
                    if os.path.exists("zworkspace//current//"+symbol+".jpg"):
                        os.remove("zworkspace//results//"+symbol+".jpg")
                        os.remove("zworkspace//results//"+symbol+".txt")
                        os.remove("zworkspace//current//"+symbol+".jpg")
                else:
                    remove.append(symbol)
            for symbol in remove:
                print(symbol,"STOPPING")
                
                if symbol not in no_fly_list:
                    if os.path.exists("zworkspace//current//"+symbol+".jpg"):
                        source_path = "zworkspace//current//"+symbol+".jpg"
                        destination_path = "zworkspace//false//"+symbol+"-"+dt+".jpg"
                        shutil.copyfile(source_path, destination_path)
                        os.remove("zworkspace//results//"+symbol+".jpg")
                        os.remove("zworkspace//results//"+symbol+".txt")
                        os.remove("zworkspace//current//"+symbol+".jpg")
                else:
                    no_fly_list.remove(symbol)
            
            

            flag = False
            if watchlist != {}:
                flag = True
                og_dt_list = prices["dt_list"]
                del prices["dt_list"]
                eliminate = []
                for symbol in symbols:
                    current_timer = timers[symbol]
                    current_timer += 1
                    current_timer_interval = timer_intervals[symbol]
                    if current_timer == current_timer_interval:
                        current_timer = 0
                        results= {}
                        symbol_name = symbol.split("_")[0]
                        ma_interval = int(symbol.split("_")[1])
                        with open('zworkspace//results//'+symbol+'.txt', 'w') as f:
                            f.write('')
                        plt.figure(figsize=(x_size/my_dpi, y_size/my_dpi), dpi=my_dpi)
                        y = prices[symbol_name]
                        y = y[-max_detection_len:]
                        dt_list = og_dt_list[-max_detection_len:]
                        start = watchlist[symbol]
                        start_index = dt_list.index(start) - shift
                        
                        compression_flag = False

                        if compression_flag:
                            hs_ln = len(y) - start_index -1
                            compression = hs_ln*3
                            
                            temp_x = []
                            temp_y = []
                            for i in range(compression):
                                temp_x.append("compression"+str(i))
                                temp_y.append(y[0])
                            compressions[symbol] = compression
                            dt_list = temp_x+dt_list
                            y = temp_y+y
                            tmp = []
                            for i in range(len(y)):
                                if i < start_index+compression:
                                    tmp.append(y[start_index+compression])
                                else:
                                    tmp.append(y[i])
                            y = tmp
                        else:
                            compressions[symbol] = 0
                            dt_list = dt_list
                            y = y


                        ogx = dt_list
                        ogy = y
                        y = ma(y,ma_interval)
                        x = []
                        for i in range(len(y)):
                            x.append(i)
                        plt.plot(x,y,label=str(symbol))
                        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                        plt.gca().spines['top'].set_visible(False)
                        plt.gca().spines['bottom'].set_visible(False) 
                        plt.gca().spines['left'].set_visible(False)
                        plt.gca().spines['right'].set_visible(False)
                        plt.gca().set_xlabel('')
                        plt.gca().set_ylabel('')
                        plt.gca().set_xticks([])
                        plt.gca().set_yticks([])
                        file_name = "zworkspace//save//"+symbol+".jpg"
                        plt.savefig(file_name, format='jpeg')
                        if AI_flag:
                            #data
                            dataset = LoadImages(source, img_size=imgsz, stride=stride)
                            # prediction 
                            if device.type != 'cpu':
                                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                            old_img_w = old_img_h = imgsz
                            old_img_b = 1
                            t0 = time.time()
                            predicting(dataset,source, weights, view_img, save_txt, imgsz, trace,device,half,model,classify,webcam,save_dir,names,save_img,colors)
                            #quit()
                        os.remove(file_name)
                        plt.close('all')
                        result_x,result_y,result_class = extract_hs(symbol_name,ma_interval,dt_list,y,x,ogy,x_size,y_size)
                        

                        print("----------------------------")
                        print(symbol)

                        current_compression = compressions[symbol]
                        x = [item - current_compression for item in x]
                        print(len(ogx),len(ogy))
                        x_list = x[current_compression:]
                        y_list = ogy[current_compression:]
                        plt.plot(x_list,y_list)
                        

                        ln = len(result_x)
                        x = result_x
                        y = result_y
                        classes = result_class
                        x = [item - current_compression for item in x]
                        plt.scatter(x,y)

                        
                        plt.savefig("zworkspace//current//"+symbol+".jpg", format='jpeg')
                        plt.close('all')

                        
                        

                        def check_trend(points_x,points_y,x,y,symbol,trend_ln,detection_ln,usx,usy):
                            global x_size
                            global y_size
                            global my_dpi
                            points_x = [item + trend_ln - detection_ln for item in points_x]
                            print(usx)
                            usx = [item + trend_ln - detection_ln for item in usx]
                            print(usx)
                            
                            left_shoulder_x,t1_x,head_x,t2_x,right_shoulder_x = points_x
                            left_shoulder,t1,head,t2,right_shoulder = points_y
                            ogx = x
                            ogy = y
                            y = ma(y,40)
                            plt.figure(figsize=(x_size/my_dpi, y_size/my_dpi), dpi=my_dpi)
                            plt.plot(x,y,label=str(symbol))
                            plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
                            plt.gca().spines['top'].set_visible(False)
                            plt.gca().spines['bottom'].set_visible(False) 
                            plt.gca().spines['left'].set_visible(False)
                            plt.gca().spines['right'].set_visible(False)
                            plt.gca().set_xlabel('')
                            plt.gca().set_ylabel('')
                            plt.gca().set_xticks([])
                            plt.gca().set_yticks([])
                            file_name = "zworkspace//save//CHECKING TREND:"+symbol+"_"+"40"+".jpg"
                            symbol_name = "CHECKING TREND:"+symbol
                            ma_interval = "40"
                            plt.savefig(file_name, format='jpeg')
                            dataset = LoadImages(source, img_size=imgsz, stride=stride)
                            if device.type != 'cpu':
                                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                            old_img_w = old_img_h = imgsz
                            old_img_b = 1
                            t0 = time.time()
                            predicting(dataset,source, weights, view_img, save_txt, imgsz, trace,device,half,model,classify,webcam,save_dir,names,save_img,colors)
                            os.remove(file_name)
                            plt.close('all')
                            result_x,result_y,result_class = extract_hs(symbol_name,ma_interval,x,y,x,ogy,x_size,y_size)
                            trend_x_list = []
                            trend_y_list = []
                            trend_class_list = []
                            for i in range(len(result_x)):
                                if result_x[i] < left_shoulder_x:
                                    trend_x_list.append(result_x[i])
                                    trend_y_list.append(result_y[i])
                                    trend_class_list.append(result_class[i])
                            if len(trend_x_list) < 1:
                                return False
                            trend_x = trend_x_list[len(trend_x_list)-1]
                            trend_y = trend_y_list[len(trend_y_list)-1]
                            if left_shoulder_x - trend_x < 4:
                                return False
                            lst_x = x[trend_x:left_shoulder_x-3]
                            lst_y = ogy[trend_x:left_shoulder_x-3]
                            print(trend_x,left_shoulder_x,lst_x,lst_y)
                            trend_class = trend_class_list[len(trend_class_list)-1]
                            len_trend = left_shoulder_x - trend_x
                            height_trend = t1 - trend_y
                            plt.clf()
                            plt.scatter(trend_x_list,trend_y_list,color = 'red')
                            plt.scatter(usx,usy,color="blue")
                            plt.plot(ogx,ogy)
                            plt.plot(x,y)
                            plt.savefig("zworkspace//trends//"+symbol+"TREND.jpg", format='jpeg')
                            print(max(lst_y), left_shoulder,ogy[left_shoulder_x])
                            print(max(lst_y) < left_shoulder)
                            print(height_trend > 1*(head - t1))
                            print(len_trend >= 0.5*(right_shoulder_x - left_shoulder_x))
                            if len_trend >= 0.3*(right_shoulder_x - left_shoulder_x) and height_trend > 1*(head - t1) and max(ma(lst_y,5)) < left_shoulder:
                                return True
                            
                            return False
                            

                        points = []
                        for item in x:
                            if item > start_index + compressions[symbol]:
                                points.append(item)
                        unshortened_x = x
                        unshortened_y = y
                        x = x[-5:]
                        y = y[-5:]
                        if len(y) > 5:
                            x = x[-5:]
                            y = y[-5:]
                            classes = classes[-5:]
                        if len(points) > 8:
                            no_fly_list.append(symbol)
                            eliminate.append(symbol)
                            source_path = "zworkspace//current//"+symbol+".jpg"
                            destination_path = "zworkspace//false//"+symbol+"-"+dt+".jpg"
                            shutil.copyfile(source_path, destination_path)
                        else:
                            if len(y) == 5:
                                right_shoulder = y[0]
                                t1 = y[1]
                                head = y[2]
                                t2 = y[3]
                                left_shoulder = y[4]
                                end = ogy[len(ogy)-1]
                                end_x = len(ogy) - 1
                                #print(right_shoulder,t1,head,t2,left_shoulder,end)
                                if determine_valid(x,[left_shoulder,t1,head,t2,right_shoulder],end_x,end):
                                    if end < min(t1,t2):
                                        print("developed")
                                        y_trend_lst = prices[symbol_name]
                                        y_trend_lst = y_trend_lst
                                        dt_list = og_dt_list[-1*max_trend_len:]
                                        x_trend_lst = []
                                        for i in range(len(y_trend_lst)):
                                            x_trend_lst.append(i)
                                        if check_trend(x,y,x_trend_lst,y_trend_lst,symbol,max_trend_len,max_detection_len,unshortened_x,unshortened_y):
                                            print("developed")
                                            source_path = "zworkspace//current//"+symbol+".jpg"
                                            if (head - (t1+t2)/2)/((t1+t2)/2) > 0.0045:
                                                destination_path = "zworkspace//found//big//"+symbol+".jpg"
                                                txt_path = "zworkspace//found//big//"+symbol+".txt"
                                            else:
                                                destination_path = "zworkspace//found//small//"+symbol+".jpg"
                                                txt_path = "zworkspace//found//small//"+symbol+".txt"
                                            if not(os.path.exists(destination_path)):
                                                shutil.copyfile(source_path, destination_path)
                                                dt_list = dt_list[compressions[symbol]:]
                                                with open(txt_path, 'w') as f:
                                                    print(x)
                                                    print(len(dt_list))
                                                    print(max_trend_len - max_detection_len)
                                                    json.dump({"start":dt_list[x[0]+max_trend_len - max_detection_len],"end":dt_list[x[4]+max_trend_len - max_detection_len]}, f)
                                            no_fly_list.append(symbol)
                                    else:
                                        print("waiting")
                                        current_timer_interval = 1
                                else:
                                    print("False")
                                    current_timer_interval = 5
                            else:
                                current_timer_interval = 15
                                print("Not enough points")
                            
                    timers[symbol] = current_timer
                    timer_intervals[symbol] = current_timer_interval
                for symbol in eliminate:
                    if symbol in watchlist.keys():
                        del watchlist[symbol]
                with open('MS yolo//zworkspace//watchlist.txt', 'w') as f:
                    json.dump(watchlist, f)
                    

            
                    
            if flag:
                print("TIME TAKEN:",(time.time()-start_time))
                if time.time()-start_time < 10:
                    time.sleep(int(10-(time.time()-start_time)))
            else:
                time.sleep(20)
                    
        else:
            if status != "closed":
                print("MARKET CLOSED:",dt)
                status = "closed"
    #-----------------------------------
    #-----------------------------------

    print(f'Done. ({time.time() - t0:.3f}s)')

