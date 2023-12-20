

#subprocess.run(['pip', 'install', '-qr', 'requirements.txt'])
#------------------------------------------------------------------------
# A I   P R E P 
#------------------------------------------------------------------------
def printDir():
    import os
    current_directory = os.getcwd()
    output = ""
    for item in os.listdir(current_directory):
        output+=item+","
    instancePrint([output])
def findRange(lst):
    return max(lst) - min(lst)
def instancePrint(lst):
    string = ""
    for item in lst:
        string+= " "+str(item)
    string = string[1:]
    print("zPD:",string)
def read_json_file(file_path,prices):
    import time
    import json
    max_attempts = 10
    current_attempt = 1

    while current_attempt <= max_attempts:
        try:
            with open(file_path, "r") as f:
                data = json.loads(f.read())
            return data
        except json.JSONDecodeError as e:
            instancePrint([file_path])
            instancePrint([f"Attempt {current_attempt}: Failed to read JSON file. Error: {e}"])
            instancePrint([f"Attempt {current_attempt}: Failed to read JSON file. Retrying..."])
            current_attempt += 1
            time.sleep(0.05)  # Wait for 1 second before retrying

    instancePrint(["Max attempts reached. Unable to read JSON file."])
    return prices
def determine_valid(lstx,lsty,end_x,end_y):
    import json
    
    mn = min(lsty)
    flag = True
    lsty = [item-mn for item in lsty]
    left_shoulder_x,t1_x,head_x,t2_x,right_shoulder_x = lstx
    left_shoulder,t1,head,t2,right_shoulder = lsty
    diff = max(lsty) - min(lsty)
    for i in range(1,len(lstx)):
        if lstx[i] - lstx[i-1] < 6:
            h = 1
            #instancePrint(["FALSE: LENGTH 1"])
            #return False
    if not(left_shoulder > t1 and head > t1 and right_shoulder > t1):
        #instancePrint(["FALSE: HEIGHT 1"])
        return False
    if not(left_shoulder > t2 and head > t2 and right_shoulder > t2):
        #instancePrint(["FALSE: HEIGHT 2"])
        return False
    if abs(t1-t2)/diff > 0.4:
        #instancePrint(["FALSE: DIFF"])
        return False
    if not(3 > left_shoulder/right_shoulder > 0.33):
        #instancePrint(["FALSE: SHOULDERS 1"])
        return False
    if not(head/right_shoulder > 1 and head/left_shoulder > 1):
        #instancePrint(["FALSE: HEAD 1"])
        return False
    if right_shoulder_x  - left_shoulder_x < 30:
        #instancePrint(["FALSE: LENGTH 1"])
        return False
    
    head -= (t1+t2)/2
    left_shoulder -= t1
    right_shoulder -= t2
    if not(5 > left_shoulder/right_shoulder > 0.2):
        #instancePrint(["FALSE: SHOULDERS 2"])
        return False
    
    d1 = t1_x - left_shoulder_x
    d2 = head_x - t1_x
    d3 = t2_x - head_x
    d4 = right_shoulder_x - t2_x
    d5 = right_shoulder_x - t2_x

    coef = 4
    
    if max([d1,d2,d3,d4,d5])/min([d1,d2,d3,d4,d5]) > coef and False:
        #instancePrint(["FALSE: D1"])
        return False
    return True
def read_json_file(file_path,prices):
    import json
    import time
    import os
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
def max_index_find(lst):
    num = lst[0]
    index = 0
    for i in range(len(lst)):
        if lst[i] >= num:
            num = lst[i]
            index = i
    return num,index
def min_index_find(lst):
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
    import pytz
    from datetime import datetime
    import calendar
    time = pytz.timezone(timezone) 
    time = datetime.now(time)
    
    day = calendar.day_name[time.weekday()]
    dt = time.strftime("%Y-%m-%d %H:%M:%S")
    time = time.strftime("%H:%M:%S")

    return dt, time, day
def extract_hs(name,current_symbols,current_prices,current_ma_prices,dt_list,dimension):
    with open('zPDworkspace/results/'+name+'.txt', 'r') as f:
        bbox = f.read().strip()
    bboxes = bbox.splitlines()
    x_dict = {}
    y_dict = {}
    class_dict = {}
    uc_x_dict = {}
    uc_y_dict = {}
    uc_class_dict = {}
    prices_dict = {}
    ma_prices_dict = {}
    calibration = {}
    sizex = 800
    sizey = 800
    indx = 136
    indy = 109
    outerx = 80
    outery = 105
    outery2 = 86
    innerx = 32
    innery = 58
    for i in range(len(current_symbols)):
        prices_dict[current_symbols[i]] = current_prices[i]
        ma_prices_dict[current_symbols[i]] = current_ma_prices[i]

    classes = ["peak","trough"]

    initial_results = {}
    for i in range(16):
        initial_results[current_symbols[i]] = [[],{}]
    for bbox in bboxes:
        bbox = bbox.split()
        class_num = int(bbox[0])
        x_center = float(bbox[1])
        y_center = float(bbox[2])
        width = float(bbox[3])
        height = float(bbox[4])
        if (outery+0*innery+0*indy - innery/2)/sizey < y_center < (outery+0*innery+1*indy + innery/2)/sizey:
            if (outerx+0*innerx+0*indx - innerx/2)/sizex < x_center < (outerx+0*innerx+1*indx + innerx/2)/sizex:
                typ = 1
                x_dimension = 0
                y_dimension = 0
            elif (outerx+1*innerx+1*indx - innerx/2)/sizex < x_center < (outerx+1*innerx+2*indx + innerx/2)/sizex:
                typ = 2
                x_dimension = 1
                y_dimension = 0
            elif (outerx+2*innerx+2*indx - innerx/2)/sizex < x_center < (outerx+2*innerx+3*indx + innerx/2)/sizex:
                typ = 3
                x_dimension = 2
                y_dimension = 0
            elif (outerx+3*innerx+3*indx - innerx/2)/sizex < x_center < (outerx+3*innerx+4*indx + innerx/2)/sizex:
                typ = 4
                x_dimension = 3
                y_dimension = 0
        elif (outery+1*innery+1*indy - innery/2)/sizey < y_center < (outery+1*innery+2*indy + innery/2)/sizey:
            if (outerx+0*innerx+0*indx - innerx/2)/sizex < x_center < (outerx+0*innerx+1*indx + innerx/2)/sizex:
                typ = 5
                x_dimension = 0
                y_dimension = 1
            elif (outerx+1*innerx+1*indx - innerx/2)/sizex < x_center < (outerx+1*innerx+2*indx + innerx/2)/sizex:
                typ = 6
                x_dimension = 1
                y_dimension = 1
            elif (outerx+2*innerx+2*indx - innerx/2)/sizex < x_center < (outerx+2*innerx+3*indx + innerx/2)/sizex:
                typ = 7
                x_dimension = 2
                y_dimension = 1
            elif (outerx+3*innerx+3*indx - innerx/2)/sizex < x_center < (outerx+3*innerx+4*indx + innerx/2)/sizex:
                typ = 8
                x_dimension = 3
                y_dimension = 1
        elif (outery+2*innery+2*indy - innery/2)/sizey < y_center < (outery+2*innery+3*indy + innery/2)/sizey:
            if (outerx+0*innerx+0*indx - innerx/2)/sizex < x_center < (outerx+0*innerx+1*indx + innerx/2)/sizex:
                typ = 9
                x_dimension = 0
                y_dimension = 2
            elif (outerx+1*innerx+1*indx - innerx/2)/sizex < x_center < (outerx+1*innerx+2*indx + innerx/2)/sizex:
                typ = 10
                x_dimension = 1
                y_dimension = 2
            elif (outerx+2*innerx+2*indx - innerx/2)/sizex < x_center < (outerx+2*innerx+3*indx + innerx/2)/sizex:
                typ = 11
                x_dimension = 2
                y_dimension = 2
            elif (outerx+3*innerx+3*indx - innerx/2)/sizex < x_center < (outerx+3*innerx+4*indx + innerx/2)/sizex:
                typ = 12
                x_dimension = 3
                y_dimension = 2
        elif (outery+3*innery+3*indy - innery/2)/sizey < y_center < (outery+3*innery+4*indy + innery/2)/sizey:
            if (outerx+0*innerx+0*indx - innerx/2)/sizex < x_center < (outerx+0*innerx+1*indx + innerx/2)/sizex:
                typ = 13
                x_dimension = 0
                y_dimension = 3
            elif (outerx+1*innerx+1*indx - innerx/2)/sizex < x_center < (outerx+1*innerx+2*indx + innerx/2)/sizex:
                typ = 14
                x_dimension = 1
                y_dimension = 3
            elif (outerx+2*innerx+2*indx - innerx/2)/sizex < x_center < (outerx+2*innerx+3*indx + innerx/2)/sizex:
                typ = 15
                x_dimension = 2
                y_dimension = 3
            elif (outerx+3*innerx+3*indx - innerx/2)/sizex < x_center < (outerx+3*innerx+4*indx + innerx/2)/sizex:
                typ = 16
                x_dimension = 3
                y_dimension = 3

        lst = current_prices[typ-1]
        symb = current_symbols[typ-1]

        size = 1262
        individual = 270
        outer = 13
        inner = 52

        width = width*sizex/indx
        height = height*sizey/indy
        x_center = (x_center*sizex - (x_dimension)*indx-x_dimension*innerx-outerx)/indx
        y_center = (y_center*sizey - (y_dimension)*indy-y_dimension*innery-outery)/indy

        x_ln = len(lst)
        y_ln = max(lst) - min(lst)
        if (x_center*100) < 99 and int(x_center*x_ln)!= 0:
            x_center = int(x_center*x_ln)
            class_name = classes[class_num]
            initial_results[symb][0] = initial_results[symb][0] + [x_center]
            initial_results[symb][1][x_center] = class_name
    keys = list(initial_results.keys())
    cleaned_keys = []
    for key in keys:
        if not(key[:5] == "tsymb" or key[:10] == "tempsymbol"):
            cleaned_keys.append(key)
        else:
            x_dict[key] = []
            y_dict[key] = []
            class_dict[key] = []
            uc_x_dict[key] = []
            uc_y_dict[key] = []
            uc_class_dict[key] = []
            
    keys = cleaned_keys
    for key in keys:
        ma_interval = int(key.split("_")[1])
        x_output = []
        y_output = []
        ogy = prices_dict[key]
        ma_y = ma_prices_dict[key]
        class_output = []
        uc_x = []
        uc_y = []
        uc_class = []
        temp_x = []
        prev = 0
        counter = 0
        xs = initial_results[key][0]
        class_names = initial_results[key][1]
        xs.sort()
        lst = []
        for x_val in xs:
            lst.append(class_names[x_val])
        class_names = lst

        for i in range(len(xs)):
            x_center = xs[i]
            class_name = class_names[i]
            min_index =  x_center - int(ma_interval)*3
            max_index =  x_center+1
            uc_min_index = x_center
            uc_max_index = x_center+1
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
            uc_index = uc_min_index
            if uc_index > len(ma_y) - 1:
                uc_index = len(ma_y) - 1
            if uc_index < 0:
                uc_index = 0

            prev = index
            x_output.append(index)
            y_output.append(ogy[index])
            class_output.append(class_name)
            uc_x.append(uc_index)
            uc_y.append(ma_y[uc_index])
            uc_class.append(class_name)
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
        x_dict[key] = x_output
        y_dict[key] = y_output
        class_dict[key] = class_output
        uc_x_dict[key] = uc_x
        uc_y_dict[key] = uc_y
        uc_class_dict[key] = uc_class
    #return uc_x_dict,uc_y_dict,uc_class_dict

    return x_dict,y_dict,class_dict
def add_to_dct_lst(dct,ky,val):
    if ky not in dct.keys():
        dct[ky] = [val]
    else:
        dct[ky] = dct[ky] + [val]
    return dct

def predicting(dataset,source, weights, view_img, save_txt, imgsz, trace,device,half,model,classify,webcam,save_dir,names,save_img,colors,conf_thres, iou_thres, save_conf, nosave, optClasses, agnostic_nms, update, project, name, exist_ok,old_img_b,old_img_w,old_img_h,augment):
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
    from pyqtgraph.Qt import QtGui, QtCore
    import pyqtgraph as pg
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from models.experimental import attempt_load
    from utils.datasets import LoadStreams, LoadImages
    from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
        scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
    from utils.plots import plot_one_box
    from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
    import glob
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
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=optClasses, agnostic=agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        if webcam:  # batch_size >= 1
            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
        else:
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
        p = Path(p)  # to Path
        save_path = str(save_dir / p.name)  # img.png
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
            save_path = str(save_dir / p.name)  # img.png
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
def combine(lst):
    output = 0
    for item in lst:
        output += item
    return output
def check_trend(symbol,x,prices,dt_list,points_x,points_y,usx,usy,detection_ln,trend_ln,source, weights, view_img, save_txt, imgsz, trace,device,half,model,classify,webcam,save_dir,names,save_img,colors,x_size,y_size,my_dpi,stride,dimension,fig,conf_thres, iou_thres, save_conf, nosave, optClasses, agnostic_nms, update, project,exist_ok,augment):
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
    from pyqtgraph.Qt import QtGui, QtCore
    import pyqtgraph as pg
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from models.experimental import attempt_load
    from utils.datasets import LoadStreams, LoadImages
    from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
        scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
    from utils.plots import plot_one_box
    from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
    import glob
    points_x = [item + trend_ln - detection_ln for item in points_x]
    usx = [item + trend_ln - detection_ln for item in usx]
    
    left_shoulder_x,t1_x,head_x,t2_x,right_shoulder_x = points_x
    left_shoulder,t1,head,t2,right_shoulder = points_y
    symbol = symbol+"_40"
    ogx = x
    ogy = prices
    y = ma(prices,40)
    current_symbols = [symbol]
    current_prices = [prices]
    for i in range(15):
        current_symbols.append("tsymb"+str(i)+"_40")
        current_prices.append([0 for item in prices])
    name = ""
    for symb in current_symbols:
        
        if name == "":
            name += symb
        else:
            name += ","+symb
    file_name = "zPDworkspace/save/"+name+".jpg"
    txt_name = "zPDworkspace/results/"+name+".txt"
    result_file_name = "zPDworkspace/results/"+name+".jpg"
    
    with open(txt_name, 'w') as f:
        f.write('')
    y_dict = {}
    for i in range(len(current_prices)):
        y_dict[str(i+1)] = current_prices[i]
    t = time.time()
    fig.for_each_trace(lambda trace: trace.update(y=y_dict[trace.name]))
    instancePrint([time.time() - t])
    t = time.time()
    fig.write_image(file_name)
    instancePrint(["sub", time.time() - t])
    ttemp = time.time()
        #data
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    # prediction 
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    t0 = time.time()
    predicting(dataset,source, weights, view_img, save_txt, imgsz, trace,device,half,model,classify,webcam,save_dir,names,save_img,colors,conf_thres, iou_thres, save_conf, nosave, optClasses, agnostic_nms, update, project, name, exist_ok,old_img_b,old_img_w,old_img_h,augment)
    #quit()
    name = ""
    for tsymbol in current_symbols:
        name += tsymbol+","
    name = name[:-1]
    x_dict,y_dict,class_dict = extract_hs(name,current_symbols,current_prices,current_prices,dt_list,dimension)
    os.remove(file_name)
    #os.remove(txt_name)

    result_x = x_dict[symbol]
    result_y = y_dict[symbol]
    result_class = class_dict[symbol]
    trend_x_list = []
    trend_y_list = []
    trend_class_list = []
    unfiltered_x = result_x
    unfiltered_y = result_y

    peak_x = []
    peak_y = []
    trough_x = []
    trough_y = []
    for i in range(len(result_x)):
        if result_class[i] == 'trough':
            trough_x.append(result_x[i])
            trough_y.append(result_y[i])
        elif result_class[i] == 'peak':
            peak_x.append(result_x[i])
            peak_y.append(result_y[i])

    for i in range(len(result_x)):
        if result_x[i] < left_shoulder_x:# and result_class[i] == 'trough':
            trend_x_list.append(result_x[i])
            trend_y_list.append(result_y[i])
            trend_class_list.append(result_class[i])
    if len(trend_x_list) < 1:
        return 0,0,False
    trend_x = trend_x_list[len(trend_x_list)-1]
    trend_y = trend_y_list[len(trend_y_list)-1]
    if left_shoulder_x - trend_x < 4:
        return 0,0,False
    lst_x = x[trend_x:left_shoulder_x-3]
    lst_y = ogy[trend_x:left_shoulder_x-3]
    trend_class = trend_class_list[len(trend_class_list)-1]
    len_trend = left_shoulder_x - trend_x
    height_trend = min(t1,t2) - trend_y
    plt.close("all")
    plt.clf()
    plt.scatter(peak_x,peak_y,color = 'red')
    plt.scatter(trough_x,trough_y,color = 'green')
    plt.scatter(usx,usy,color="blue")
    plt.scatter(trend_x_list,trend_y_list,color="yellow")
    plt.plot(ogx,ogy)
    plt.plot(x,y)
    plt.scatter([trend_x],[trend_y], color="purple")
    plt.savefig("zPDworkspace/trends/"+symbol+"TREND.jpg", format='jpeg')
    changes = []
    prev = lst_y[0]
    for i in range(len(lst_y)):
        changes.append(lst_y[i] - prev)
        prev = lst_y[i]
    new_changes = []
    for change in changes:
        if abs(change) > 1*height_trend:
            new_changes.append(change)
    vol_adjusted_height_flag = True#height_trend - combine(new_changes) > 1.5*(head - min(t1,t2))
    if len_trend >= 0.3*(right_shoulder_x - left_shoulder_x) and height_trend > 1.5*(head - min(t1,t2)) and max(ma(lst_y,5)) < left_shoulder and vol_adjusted_height_flag:
        return height_trend, trend_ln,True
    
    return height_trend, trend_ln,False



def PD():
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
    from pyqtgraph.Qt import QtGui, QtCore
    import pyqtgraph as pg
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from models.experimental import attempt_load
    from utils.datasets import LoadStreams, LoadImages
    from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
        scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
    from utils.plots import plot_one_box
    from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
    import glob
    import shutil
    instancePrint(["YURRRR"])
    with torch.no_grad():
        instancePrint(["1"])
        #source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
        source, weights, view_img, save_txt, imgsz, trace, device = 'zPDworkspace//save','zPD_weights_new/weights4_1_2.pt', False, False, 640, True,''
        conf_thres, iou_thres, save_conf, nosave, optClasses, agnostic_nms, update, project, name, exist_ok,augment = 0.25, 0.45, False, False, None, False, False, 'runs/detect', 'exp', False,False
        save_img = True#not opt.nosave and not source.endswith('.txt')
        save_txt = True
        webcam = False
        save_dir = Path("zPDworkspace//results")#Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
        #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
        set_logging()
        device = select_device(device)#opt.device)
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

        x = []
        y = []
        for i in range(180):
            x.append(i)
            y.append(i)
        l1 = go.Scatter(x=x, y=y, name="1",marker_color='rgb(26, 123, 184)')
        l2 = go.Scatter(x=x, y=y, name="2",marker_color='rgb(26, 123, 184)')
        l3 = go.Scatter(x=x, y=y, name="3",marker_color='rgb(26, 123, 184)')
        l4 = go.Scatter(x=x, y=y, name="4",marker_color='rgb(26, 123, 184)')
        l5 = go.Scatter(x=x, y=y, name="5",marker_color='rgb(26, 123, 184)')
        l6 = go.Scatter(x=x, y=y, name="6",marker_color='rgb(26, 123, 184)')
        l7 = go.Scatter(x=x, y=y, name="7",marker_color='rgb(26, 123, 184)')
        l8 = go.Scatter(x=x, y=y, name="8",marker_color='rgb(26, 123, 184)')
        l9 = go.Scatter(x=x, y=y, name="9",marker_color='rgb(26, 123, 184)')
        l10 = go.Scatter(x=x, y=y, name="10",marker_color='rgb(26, 123, 184)')
        l11 = go.Scatter(x=x, y=y, name="11",marker_color='rgb(26, 123, 184)')
        l12 = go.Scatter(x=x, y=y, name="12",marker_color='rgb(26, 123, 184)')
        l13 = go.Scatter(x=x, y=y, name="13",marker_color='rgb(26, 123, 184)')
        l14 = go.Scatter(x=x, y=y, name="14",marker_color='rgb(26, 123, 184)')
        l15 = go.Scatter(x=x, y=y, name="15",marker_color='rgb(26, 123, 184)')
        l16 = go.Scatter(x=x, y=y, name="16",marker_color='rgb(26, 123, 184)')


        fig = make_subplots(rows=4, cols=4, shared_yaxes=False)
        fig.update_layout(height=800,width=800,showlegend=False)
        fig.update_layout(
            plot_bgcolor='white'
        )
        fig.update_xaxes(showticklabels=False,gridcolor="white")
        fig.update_yaxes(showticklabels=False,gridcolor="white")
        fig.add_trace(l1, row=1, col=1)
        fig.add_trace(l2, row=1, col=2)
        fig.add_trace(l3, row=1, col=3)
        fig.add_trace(l4, row=1, col=4)

        fig.add_trace(l5, row=2, col=1)
        fig.add_trace(l6, row=2, col=2)
        fig.add_trace(l7, row=2, col=3)
        fig.add_trace(l8, row=2, col=4)

        fig.add_trace(l9, row=3, col=1)
        fig.add_trace(l10, row=3, col=2)
        fig.add_trace(l11, row=3, col=3)
        fig.add_trace(l12, row=3, col=4)

        fig.add_trace(l13, row=4, col=1)
        fig.add_trace(l14, row=4, col=2)
        fig.add_trace(l15, row=4, col=3)
        fig.add_trace(l16, row=4, col=4)
        #-----------------------------------
        #WORKSPACE
        #-----------------------------------
        status = "start"
        x_size = 1262 #230
        y_size = 1262 #170
        my_dpi = 192
        intervals = [5,20]
        AI_flag = True
        prices = {}
        watchlist = {}
        compression = 160
        compressions = {}
        ep_det_lst = []
        ep_det_stor = {}
        shift = 10
        currentFlag = False
        clear_files("zPDworkspace/current","jpg")
        clear_files("zPDworkspace/false","jpg")
        clear_files("zPDworkspace/found/small","jpg")
        clear_files("zPDworkspace/found/big","jpg")
        clear_files("zPDworkspace/found/zregression","jpg")
        clear_files("zPDworkspace/results","jpg")
        clear_files("zPDworkspace/save","jpg")
        clear_files("zPDworkspace/trends","jpg")

        clear_files("zPDworkspace/current","png")
        clear_files("zPDworkspace/false","png")
        clear_files("zPDworkspace/found/small","png")
        clear_files("zPDworkspace/found/big","png")
        clear_files("zPDworkspace/found/zregression","png")
        clear_files("zPDworkspace/results","png")
        clear_files("zPDworkspace/save","png")
        clear_files("zPDworkspace/trends","png")

        clear_files("zPDworkspace/found/small","txt")
        clear_files("zPDworkspace/found/big","txt")
        clear_files("zPDworkspace/found/zregression","txt")
        clear_files("zPDworkspace/results","txt")
        clear_files("zPDworkspace/save","jpg")
        clear_files("zPDworkspace/save","txt")
        start_time = time.time()
        watchlist = {}
        max_trend_len = 540
        max_detection_len = 180
        dimension = 4
        prices = read_json_file('zODworkspace/prices.txt',prices)
        while True:
            instancePrint(["--------------------"])
            instancePrint(["STARTING NEW ITERATION"])
            instancePrint(["TIME",time.time()-start_time])
            start_time = time.time()
            dt,now,day = current_time("America/New_York")
            prev_watchlist = watchlist
            watchlist = read_json_file('zODworkspace/watchlist.txt',watchlist)
            og_watchlist = read_json_file('zODworkspace/watchlist.txt',watchlist)
            symbols = list(watchlist.keys())
            remove = []
            prices = read_json_file('zODworkspace/prices.txt',prices)
            state = prices["state"]
            if state["continue"] == "False":
                instancePrint(["PROCESS BROKEN"])
                break
            del prices["state"]
            dt_list = prices["dt_list"]
            instancePrint(["CURRENT DT:",dt_list[len(dt_list)-1]])
            currDt = dt_list[len(dt_list)-1]
            instancePrint(["--------------------"])
            flag = False
            if watchlist != {}:
                flag = True
                og_dt_list = prices["dt_list"]
                del prices["dt_list"]
                iteration = 0
                current_symbols = []
                current_prices = []
                current_symbols = []
                current_prices = []
                current_ma_prices = []
                prices_dict = {}

                cleaned_symbols = []
                for symbol in symbols:
                    start = watchlist[symbol]
                    valid_flag = False
                    dt_list = og_dt_list[-max_detection_len:]
                    if str(start) in dt_list:
                        valid_flag = True
                        start_index = dt_list.index(start)
                        lst = dt_list[start_index:]
                    #instancePrint([dt_list])
                    if valid_flag and str(start) in dt_list and len(lst) < 150:
                        cleaned_symbols.append(symbol)
                    else:
                        del watchlist[symbol]
                symbols = cleaned_symbols
                if len(symbols)%(dimension**2) != 0:
                    for i in range(dimension**2 - len(symbols)%(dimension**2)):
                        symbols.append('tempsymbol:'+str(i)+'_1')
                        prices['tempsymbol:'+str(i)] = [0 for item in dt_list]
                new_false_list = []
                new_true_big_list = []
                new_true_small_list = []
                result_jpg_list = []
                result_txt_list = []
                for symbol in symbols:
                    if True:
                        iteration+=1
                        results= {}
                        symbol_name = symbol.split("_")[0]
                        ma_interval = int(symbol.split("_")[1])
                        y = prices[symbol_name]
                        y = y[-max_detection_len:]

                        if symbol.split(":")[0] != "tempsymbol":
                            start = watchlist[symbol]
                            start_index = dt_list.index(start) - shift
                        else:
                            start_index = 30
                        
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
                        prices_dict[symbol] = [ogx,x,ogy,y]
                        current_symbols.append(symbol)
                        current_prices.append(ogy)
                        current_ma_prices.append(y)
                        if iteration == dimension**2:
                            name = ""
                            for symb in current_symbols:
                                if name == "":
                                    name += symb
                                else:
                                    name += ","+symb
                            
                            file_name = "zPDworkspace/save/"+name+".jpg"
                            txt_name = "zPDworkspace/results/"+name+".txt"
                            result_file_name = "zPDworkspace/results/"+name+".jpg"
                            result_jpg_list.append(result_file_name)
                            result_txt_list.append(txt_name)
                            with open(txt_name, 'w') as f:
                                f.write('')
                            y_dict = {}
                            for i in range(len(current_ma_prices)):
                                y_dict[str(i+1)] = current_ma_prices[i]
                            t = time.time()
                            fig.for_each_trace(lambda trace: trace.update(y=y_dict[trace.name]))
                            instancePrint([time.time() - t])
                            t = time.time()
                            fig.write_image(file_name)
                            instancePrint(["sub", time.time() - t])
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
                                predicting(dataset,source, weights, view_img, save_txt, imgsz, trace,device,half,model,classify,webcam,save_dir,names,save_img,colors,conf_thres, iou_thres, save_conf, nosave, optClasses, agnostic_nms, update, project, name, exist_ok,old_img_b,old_img_w,old_img_h,augment)
                                ttemp = time.time()
                                instancePrint(["PREDICTION:",ttemp - t0])
                            if True:
                                ttemp = time.time()
                                x_dict,y_dict,class_dict = extract_hs(name,current_symbols,current_prices,current_ma_prices,dt_list,dimension)
                                instancePrint(["EXTRACT",time.time() - ttemp])
                                ttemp = time.time()
                                current_keys = []
                                for key in x_dict.keys():
                                    if key.split(":")[0] != "tempsymbol":
                                        current_keys.append(key)
                                for key in current_keys:
                                    #instancePrint(["--------"])
                                    #instancePrint([key])
                                    result_x = x_dict[key]
                                    result_y = y_dict[key]
                                    result_class = class_dict[key]
                                    symbol = key
                                    symbol_name = symbol.split("_")[0]
                                    ma_interval = int(symbol.split("_")[1])
                                    current_compression = compressions[symbol]
                                    ogx,x,ogy,y = prices_dict[symbol]
                                    x_list = x[current_compression:]
                                    y_list = ogy[current_compression:]
                                    ln = len(result_x)
                                    x = result_x
                                    y = result_y
                                    classes = result_class
                                    x = [item - current_compression for item in x]
                                    if currentFlag:
                                        plt.close('all')
                                        plt.plot(x_list,y_list)
                                        plt.scatter(x,y)
                                        plt.savefig("zPDworkspace/current/"+symbol+".jpg", format='jpeg')
                                        plt.close('all')

                                    points = []
                                    for item in x:
                                        if item > start_index + compressions[symbol]:
                                            points.append(item)
                                    unshortened_x = x
                                    unshortened_y = y

                                    checking_ln = 5
                                    if len(y) > checking_ln:
                                        x = x[-checking_ln:]
                                        y = y[-checking_ln:]
                                        classes = classes[-checking_ln:]
                                    if (ma_interval == 5 and len(points) > 9) or (ma_interval == 20 and len(points) > 7):
                                        new_false_list.append(symbol)
                                    else:
                                        if len(y) > 4:
                                            if len(y) == 5:
                                                repeats = 1
                                            if len(y) == 6:
                                                repeats = 2
                                            if len(y) == 7:
                                                repeats = 3
                                            unrepeated_y = y
                                            unrepeated_x = x
                                            unrepeated_classes = classes
                                            for repeat_num in range(repeats):
                                                left_shoulder = unrepeated_y[repeat_num]
                                                t1 = unrepeated_y[repeat_num+1]
                                                head = unrepeated_y[repeat_num+2]
                                                t2 = unrepeated_y[repeat_num+3]
                                                right_shoulder = unrepeated_y[repeat_num+4]
                                                end = ogy[len(ogy)-1]
                                                end_x = len(ogy) - 1
                                                y = unrepeated_y[repeat_num:repeat_num+5]
                                                x = unrepeated_x[repeat_num:repeat_num+5]
                                                classes = unrepeated_classes[repeat_num:repeat_num+5]
                                                after_y_list = prices[symbol_name]
                                                after_y_list = after_y_list[-max_detection_len:]
                                                after_y_list = ogy[x[len(x)-1]+3:]
                                                if determine_valid(x,[left_shoulder,t1,head,t2,right_shoulder],end_x,end):
                                                    if len(after_y_list) > 0:
                                                        after_bool = max(after_y_list) <  right_shoulder
                                                    else:
                                                        after_bool = True
                                                    after_bool = True
                                                    if (end < min(t1,t2)) and after_bool:
                                                        #instancePrint(["developed"])
                                                        y_trend_lst = prices[symbol_name]
                                                        y_trend_lst = y_trend_lst
                                                        dt_list = og_dt_list[-1*max_trend_len:]
                                                        x_trend_lst = []
                                                        for i in range(len(y_trend_lst)):
                                                            x_trend_lst.append(i)
                                                        print("checking trend")
                                                        trend_height, trend_ln, trend_flag = check_trend(symbol_name,x_trend_lst,y_trend_lst,dt_list,x,y,unshortened_x,unshortened_y,max_detection_len,max_trend_len,source, weights, view_img, save_txt, imgsz, trace,device,half,model,classify,webcam,save_dir,names,save_img,colors,x_size,y_size,my_dpi,stride,dimension,fig, conf_thres, iou_thres, save_conf, nosave, optClasses, agnostic_nms, update, project,exist_ok, augment)
                                                        if trend_flag:
                                                            #instancePrint(["trend verified"])
                                                            source_path = "zPDworkspace/current/"+symbol+".jpg"
                                                            if (t1+t2)/2 != 0 and (head - (t1+t2)/2)/((t1+t2)/2) > 0.004:
                                                                destination_path = "zPDworkspace/found/zregression/"+symbol+"_"+currDt+".jpg"#"zPDworkspace/found/big/"+symbol+"_"+currDt+".jpg"
                                                                txt_path = "zPDworkspace/found/zregression/"+symbol+"_"+currDt+".txt"# "zPDworkspace/found/big/"+symbol+"_"+currDt+".txt"
                                                                new_true_big_list.append(symbol)
                                                            else:
                                                                destination_path = "zPDworkspace/found/zregression/"+symbol+"_"+currDt+".jpg"#"zPDworkspace/found/small/"+symbol+"_"+currDt+".jpg"
                                                                txt_path = "zPDworkspace/found/zregression/"+symbol+"_"+currDt+".txt"#"zPDworkspace/found/small/"+symbol+"_"+currDt+".txt"
                                                                new_true_small_list.append(symbol)
                                                            if not(os.path.exists(txt_path)):
                                                                if currentFlag:
                                                                    shutil.copyfile(source_path, destination_path)
                                                                dt_list = dt_list[compressions[symbol]:]
                                                                with open(txt_path, 'w') as f:
                                                                    current_dt = dt_list[len(dt_list)-1]
                                                                    temp_dict = {}
                                                                    temp_dict["symbol"] = symbol
                                                                    temp_dict["time"] = current_dt
                                                                    temp = [og_dt_list[-max_detection_len:][item] for item in x]
                                                                    temp_dict["x"] = temp #[item+max_trend_len - max_detection_len for item in x]
                                                                    temp_dict["y"] = y
                                                                    temp_dict["class"] = classes
                                                                    temp_dict["trend"] = [trend_ln,trend_height]
                                                                    json.dump(temp_dict, f)
                                                        else:
                                                            new_false_list.append(symbol)
                                                    else:
                                                        hi = 1
                                                        #instancePrint(["waiting"])
                                                else:
                                                    hi = 1
                                                    #instancePrint(["False"])
                                        else:
                                            hi1 = 2
                                            #instancePrint(["Not enough points"])
                                instancePrint(["CALCULATION",time.time() - ttemp,(time.time() - ttemp)/len(current_keys)])
                                ttemp = time.time()
                            os.remove(file_name)
                            plt.close('all')
                            iteration = 0
                            current_symbols = []
                            current_prices = []
                            current_ma_prices = []
                            prices_dict = {}

                for tmp in new_true_small_list+new_true_big_list:
                    if tmp in watchlist.keys():
                        del watchlist[tmp]
                for tmp in new_false_list:
                    source_path = "zPDworkspace/current/"+tmp+".jpg"
                    destination_path = "zPDworkspace/false/"+tmp+".jpg"
                    if not(os.path.exists(destination_path)):
                        if currentFlag:
                            shutil.copyfile(source_path, destination_path)
                            
                    if tmp in watchlist.keys():
                        del watchlist[tmp]
                current_watchlist = read_json_file('zODworkspace/watchlist.txt',watchlist)
                temp_remove = []
                for key in current_watchlist.keys():
                    if key not in list(watchlist.keys()) and key in list(og_watchlist.keys()):
                        if og_watchlist[key]==current_watchlist[key]:
                            temp_remove.append(key)
                for key in temp_remove:
                    del current_watchlist[key]

                with open('zODworkspace/watchlist.txt', 'w') as f:
                    json.dump(current_watchlist, f)
                    
            
            jpg_files = glob.glob(os.path.join("zPDworkspace/results", '*.jpg'))
            txt_files = glob.glob(os.path.join("zPDworkspace/results", '*.txt'))
            
            # Iterate over each jpg file
            for txt_file in txt_files:
                if txt_file not in result_txt_list:
                        os.remove(txt_file)
            for jpg_file in jpg_files:
                
                if jpg_file not in result_jpg_list:
                        os.remove(jpg_file)

            
            if flag:
                instancePrint(["TIME TAKEN:",(time.time()-start_time)])
                if time.time()-start_time < 10:
                    time.sleep(int(10-(time.time()-start_time)))
            else:
                time.sleep(20)
