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
from PyQt5 import QtWidgets
import pyqtgraph as pg
import pyqtgraph.exporters
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph import GraphicsLayoutWidget


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
def findRange(lst):
    return max(lst) - min(lst)
def instancePrint(lst):
    string = ""
    for item in lst:
        string+= " "+str(item)
    string = string[1:]
    print("zOD:",string)
def clear_png_files(directory,typ):
    # Get all png files in the directory
    if typ == "png":
        png_files = glob.glob(os.path.join(directory, '*.png'))
    else:
        png_files = glob.glob(os.path.join(directory, '*.txt'))
    
    # Iterate over each png file
    for png_file in png_files:
        if (os.path.exists(png_file)):
            os.remove(png_file)
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
    calibration = {}
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

        sizex = 800
        sizey = 600
        indx = 154
        indy = 109
        outerx = 30
        outery = 27
        innerx = 41
        innery = 37

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
        x_max = x_ln - 1
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
            calibration[symb] = [x_min,x_max,y_min,y_max]
    instancePrint([output])
    return output,calibration

def predicting(dataset,source, weights, view_img, save_txt, imgsz, trace,device,half,model,classify,webcam,save_dir,names,save_img,colors,conf_thres, iou_thres, save_conf, nosave, classes, agnostic_nms, update, project, name, exist_ok,old_img_b,old_img_w,augment):
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
def saveImage(current_prices, name, win,trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11, trace12, trace13, trace14, trace15, trace16):
    exporter = pg.exporters.ImageExporter(win.ci)
    exporter.parameters()['width'] = 800
    #exporter.parameters()['height'] = 800
    currPrices = current_prices
    t = time.time()
    trace1.setData(currPrices[0])
    trace2.setData(currPrices[1])
    trace3.setData(currPrices[2])
    trace4.setData(currPrices[3])
    trace5.setData(currPrices[4])
    trace6.setData(currPrices[5])
    trace7.setData(currPrices[6])
    trace8.setData(currPrices[7])
    trace9.setData(currPrices[8])
    trace10.setData(currPrices[9])
    trace11.setData(currPrices[10])
    trace12.setData(currPrices[11])
    trace13.setData(currPrices[12])
    trace14.setData(currPrices[13])
    trace15.setData(currPrices[14])
    trace16.setData(currPrices[15])
    print(time.time() - t)
    t = time.time()
    exporter.export(name)
    print("sub", time.time() - t)
    QtCore.QCoreApplication.quit()







def OD():

    instancePrint(["YURRRR"])

    #------------------------------------------------------------------------

    with torch.no_grad():
        instancePrint(["1"])
        #source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
        source, weights, view_img, save_txt, imgsz, trace, device = 'zODworkspace//save','zODweights_new//weights4_1_1.pt', False, False, 640, True,''
        conf_thres, iou_thres, save_conf, nosave, classes, agnostic_nms, update, project, name, exist_ok,augment = 0.25, 0.45, False, False, None, False, False, 'runs/detect', 'exp', False,False
        print(source, weights, view_img, save_txt, imgsz, trace)
        print(type(source),type(weights))
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

        app = QtWidgets.QApplication([])
        win = pg.GraphicsLayoutWidget()
        win.setWindowTitle('Test pyqtgraph export')
        win.setBackground('w') 
        pen = pg.mkPen(color=(26, 123, 184),width = 3)
        win.resize(600, 600)
        ln = 180
        # PLOT 1
        plot1 = win.addPlot(row=0, col=0)
        trace1 = plot1.plot(np.random.random(ln),pen=pen)
        plot1.enableAutoRange(pg.ViewBox.XYAxes)
        plot1.getAxis('bottom').setVisible(False)
        plot1.getAxis('left').setVisible(False)
        # PLOT 2
        plot2 = win.addPlot(row=0, col=1)
        trace2 = plot2.plot(np.random.random(ln),pen=pen)
        plot2.enableAutoRange(pg.ViewBox.XYAxes)
        plot2.getAxis('bottom').setVisible(False)
        plot2.getAxis('left').setVisible(False)
        # PLOT 3
        plot3 = win.addPlot(row=0, col=2)
        trace3 = plot3.plot(np.random.random(ln),pen=pen)
        plot3.enableAutoRange(pg.ViewBox.XYAxes)
        plot3.getAxis('bottom').setVisible(False)
        plot3.getAxis('left').setVisible(False)
        # PLOT 4
        plot4 = win.addPlot(row=0, col=3)
        trace4 = plot4.plot(np.random.random(ln),pen=pen)
        plot4.enableAutoRange(pg.ViewBox.XYAxes)
        plot4.getAxis('bottom').setVisible(False)
        plot4.getAxis('left').setVisible(False)
        # PLOT 5
        plot5 = win.addPlot(row=1, col=0)
        trace5 = plot5.plot(np.random.random(ln),pen=pen)
        plot5.enableAutoRange(pg.ViewBox.XYAxes)
        plot5.getAxis('bottom').setVisible(False)
        plot5.getAxis('left').setVisible(False)
        # PLOT 6
        plot6 = win.addPlot(row=1, col=1)
        trace6 = plot6.plot(np.random.random(ln),pen=pen)
        plot6.enableAutoRange(pg.ViewBox.XYAxes)
        plot6.getAxis('bottom').setVisible(False)
        plot6.getAxis('left').setVisible(False)
        # PLOT 7
        plot7 = win.addPlot(row=1, col=2)
        trace7 = plot7.plot(np.random.random(ln),pen=pen)
        plot7.enableAutoRange(pg.ViewBox.XYAxes)
        plot7.getAxis('bottom').setVisible(False)
        plot7.getAxis('left').setVisible(False)
        # PLOT 8
        plot8 = win.addPlot(row=1, col=3)
        trace8 = plot8.plot(np.random.random(ln),pen=pen)
        plot8.enableAutoRange(pg.ViewBox.XYAxes)
        plot8.getAxis('bottom').setVisible(False)
        plot8.getAxis('left').setVisible(False)
        # PLOT 9
        plot9 = win.addPlot(row=2, col=0)
        trace9 = plot9.plot(np.random.random(ln),pen=pen)
        plot9.enableAutoRange(pg.ViewBox.XYAxes)
        plot9.getAxis('bottom').setVisible(False)
        plot9.getAxis('left').setVisible(False)
        # PLOT 10
        plot10 = win.addPlot(row=2, col=1)
        trace10 = plot10.plot(np.random.random(ln),pen=pen)
        plot10.enableAutoRange(pg.ViewBox.XYAxes)
        plot10.getAxis('bottom').setVisible(False)
        plot10.getAxis('left').setVisible(False)
        # PLOT 11
        plot11 = win.addPlot(row=2, col=2)
        trace11 = plot11.plot(np.random.random(ln),pen=pen)
        plot11.enableAutoRange(pg.ViewBox.XYAxes)
        plot11.getAxis('bottom').setVisible(False)
        plot11.getAxis('left').setVisible(False)
        # PLOT 12
        plot12 = win.addPlot(row=2, col=3)
        trace12 = plot12.plot(np.random.random(ln),pen=pen)
        plot12.enableAutoRange(pg.ViewBox.XYAxes)
        plot12.getAxis('bottom').setVisible(False)
        plot12.getAxis('left').setVisible(False)
        # PLOT 13
        plot13 = win.addPlot(row=3, col=0)
        trace13 = plot13.plot(np.random.random(ln),pen=pen)
        plot13.enableAutoRange(pg.ViewBox.XYAxes)
        plot13.getAxis('bottom').setVisible(False)
        plot13.getAxis('left').setVisible(False)
        # PLOT 14
        plot14 = win.addPlot(row=3, col=1)
        trace14 = plot14.plot(np.random.random(ln),pen=pen)
        plot14.enableAutoRange(pg.ViewBox.XYAxes)
        plot14.getAxis('bottom').setVisible(False)
        plot14.getAxis('left').setVisible(False)
        # PLOT 15
        plot15 = win.addPlot(row=3, col=2)
        trace15 = plot15.plot(np.random.random(ln),pen=pen)
        plot15.enableAutoRange(pg.ViewBox.XYAxes)
        plot15.getAxis('bottom').setVisible(False)
        plot15.getAxis('left').setVisible(False)
        # PLOT 16
        plot16 = win.addPlot(row=3, col=3)
        trace16 = plot16.plot(np.random.random(ln),pen=pen)
        plot16.enableAutoRange(pg.ViewBox.XYAxes)
        plot16.getAxis('bottom').setVisible(False)
        plot16.getAxis('left').setVisible(False)


        exporter = pg.exporters.ImageExporter(win.ci)
        exporter.parameters()['width'] = 800
        timer = QtCore.QTimer()
        traceDict = {"1":trace1,"2":trace2,"3":trace3,"4":trace4,"5":trace5,"6":trace6,"7":trace7,"8":trace8,"9":trace9,"10":trace10,"11":trace11,"12":trace12,"13":trace13,"14":trace14,"15":trace15,"16":trace16}

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
        clear_png_files("zODworkspace//results","txt")
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
                    print(iteration,len(y),findRange(y))
                    if iteration == dimension**2:
                        ttemp = time.time()
                        name = ""
                        for symb in current_symbols:
                            if name == "":
                                name += symb
                            else:
                                name += ","+symb
                        
                        file_name = "zODworkspace//save//"+name+".png"
                        instancePrint([name])
                        timer.singleShot(1, lambda: saveImage(current_prices, file_name, win,trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11, trace12, trace13, trace14, trace15, trace16))
                        app.exec_()
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
                            predicting(dataset,source, weights, view_img, save_txt, imgsz, trace,device,half,model,classify,webcam,save_dir,names,save_img,colors,conf_thres, iou_thres, save_conf, nosave, classes, agnostic_nms, update, project, name, exist_ok,old_img_b,old_img_w,augment)
                            ttemp = time.time()
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
OD()

