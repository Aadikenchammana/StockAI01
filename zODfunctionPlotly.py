


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
    print(output)
def findRange(lst):
    return max(lst) - min(lst)
def instancePrint(lst):
    string = ""
    for item in lst:
        string+= " "+str(item)
    string = string[1:]
    print("zOD:",string)
def clear_png_files(directory,typ):
    import glob
    import os
    # Get all png files in the directory
    print("CLEARING:")
    if typ == "png":
        png_files = glob.glob(os.path.join(directory, '*.png'))
    elif typ == "jpg":
        png_files = glob.glob(os.path.join(directory, '*.jpg'))
    else:
        png_files = glob.glob(os.path.join(directory, '*.txt'))
    
    # Iterate over each png file
    for png_file in png_files:
        if (os.path.exists(png_file)):
            os.remove(png_file)
    instancePrint(["DIRECTORY CLEARED:",directory])
def read_json_file(file_path,prices):
    import time
    import json
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
    import pytz
    from datetime import datetime
    import calendar
    time = pytz.timezone(timezone) 
    time = datetime.now(time)
    
    day = calendar.day_name[time.weekday()]
    dt = time.strftime("%Y-%m-%d %H:%M:%S")
    time = time.strftime("%H:%M:%S")

    return dt, time, day

def extract_hs(current_symbols,current_prices,dt_list,name,dimension):
    import os
    import json
    with open('zODworkspace//results//'+name+'.txt', 'r') as f:
        bbox = f.read().strip()
    bboxes = bbox.splitlines()
    output = {}
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
    for bbox in bboxes:
        bbox = bbox.split()
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
        #instancePrint ([typ])
        lst = current_prices[typ-1]
        symb = current_symbols[typ-1]


        width = width*sizex/indx
        height = height*sizey/indy
        x_center = (x_center*sizex - (x_dimension)*indx-x_dimension*innerx-outerx)/indx
        y_center = (y_center*sizey - (y_dimension)*indy-y_dimension*innery-outery)/indy

        x_ln = len(lst)
        y_ln = max(lst) - min(lst)
        x_max = int((x_center+width/2)*x_ln)
        x_min = int((x_center-width/2)*x_ln)
        y_max = int((y_center+height/2)*y_ln)
        y_min = int((y_center-height/2)*y_ln)

        diff = x_ln - x_max
        x_max = 180
        if x_min < 1:
            x_min = 1
        flag = True
        if "1" in dt_list:
            flag = False
            if dt_list.index("1") < 130:
                flag = True
        flag = True
        #instancePrint([symb])
        #instancePrint([x_center,y_center,height,width])
        if x_center > 0.7 and len(dt_list) - x_min < 200:
            output[symb] = dt_list[x_min-1]
            calibration[symb] = [x_min,x_max,y_min,y_max]
    #instancePrint([output])
    return output,calibration

def predicting(dataset,source, weights, view_img, save_txt, imgsz, trace,device,half,model,classify,webcam,save_dir,names,save_img,colors,conf_thres, iou_thres, save_conf, nosave, classes, agnostic_nms, update, project, name, exist_ok,old_img_b,old_img_w,old_img_h,augment):
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
            #instancePrint([f'{s}{(1E3 * (t2 - t1)):.1f}'])

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







def OD():
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
    instancePrint(["YURRRR"])

    #------------------------------------------------------------------------

    with torch.no_grad():
        instancePrint(["1"])
        #source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
        source, weights, view_img, save_txt, imgsz, trace, device = 'zODworkspace//save','zODweights_new//weights4_1_1.pt', False, False, 640, True,''
        conf_thres, iou_thres, save_conf, nosave, classes, agnostic_nms, update, project, name, exist_ok,augment = 0.25, 0.45, False, False, None, False, False, 'runs/detect', 'exp', False,False
        save_img = True#not opt.nosave and not source.endswith('.txt')
        save_txt = True
        webcam = False
        save_dir = Path("zODworkspace//results")#Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
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
        plot_flag = True
        AI_flag = True
        dimension = 4
        x_size = 1262 #230
        y_size = 1262 #170
        my_dpi = 121
        prices = {}
        calibration = {}
        max_ln = 180
        with open('zODworkspace//watchlist.txt', 'w') as f:
            json.dump({}, f)
        with open('zODworkspace//total_watchlist.txt', 'w') as f:
            json.dump({}, f)
        clear_png_files("zODworkspace//save","png")
        clear_png_files("zODworkspace//results","png") 
        clear_png_files("zODworkspace//save","jpg")
        clear_png_files("zODworkspace//results","jpg") 
        clear_png_files("zODworkspace//results","txt")
        start_time = time.time()
        while True:
            instancePrint(["--------------------"])
            instancePrint(["STARTING NEW ITERATION"])
            instancePrint(["TIME",time.time()-start_time,time.time(),start_time])
            instancePrint(["--------------------"])
            start_time = time.time()
            dt,now,day = current_time("America/New_York")
            print(1,plot_flag)
            if plot_flag:
                prices = read_json_file('zODworkspace//prices.txt',prices)
                state = prices["state"]
                if state["continue"] == "False":
                    instancePrint(["PROCESS BROKEN"])
                    break
                dt_list = prices["dt_list"]
                dt_list = dt_list[-max_ln:]
                del prices["dt_list"]
                del prices["state"]
                symbols = list(prices.keys())
                iteration = 0
                current_symbols = []
                current_prices = []
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
                    x = []
                    for i in range(len(y)):
                        x.append(i)
                    current_symbols.append(symbol)
                    current_prices.append(y)
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
                            instancePrint(["PREDICTION:",ttemp - t0])
                            results, points = extract_hs(current_symbols,current_prices,dt_list,name,dimension)
                            ttemp = time.time()
                            for key in points.keys():
                                calibration[key] = points[key]
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
                            ttemp = time.time()
                            
                        os.remove(file_name)
                        iteration = 0
                        current_symbols = []
                        current_prices = []
                        ttemp = time.time()
            if time.time()-start_time < 30:
                instancePrint(["TIME TAKEN:",(time.time()-start_time)])
                time.sleep(int(30-(time.time()-start_time)))
