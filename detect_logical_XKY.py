# -*- coding: utf-8 -*-
# @Modified by: Yufeng and Hongxu
# @ProjectName:yolov5-pyqt5

import sys
import cv2
import time
import argparse
import random
import torch
import numpy as np
import pandas as pd
import subprocess
import os
import torch.backends.cudnn as cudnn
from moviepy.video.io.VideoFileClip import VideoFileClip
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QInputDialog, QFileDialog

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QProgressDialog, QMessageBox
from PyQt5.QtCore import QTimer
from utils.general import xyxy2xywh


from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.plots import plot_one_box2, plot_one_point

from ui.ori_ui.new_ui import Ui_MainWindow # 导入detect_ui的界面

from PyQt5.QtCore import QThread, pyqtSignal

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    coords[:, :8] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    return coords

class VideoTrimThread(QThread):
    """
    Thread class for trimming video asynchronously.
    """
    finished = pyqtSignal(str)  # Signal emitted when the task is finished
    error = pyqtSignal(str)  # Signal emitted when an error occurs

    def __init__(self, video_path, save_path, start_time, end_time, parent=None):
        super(VideoTrimThread, self).__init__(parent)
        self.video_path = video_path
        self.save_path = save_path
        self.start_time = start_time
        self.end_time = end_time

    def run(self):
        try:
            # Load video
            clip = VideoFileClip(self.video_path)

            # Trim video
            trimmed_video = clip.subclip(self.start_time, self.end_time)
            trimmed_video.write_videofile(self.save_path, codec="libx264", audio_codec="aac")            # Emit finished signal with the save path
            self.finished.emit(self.save_path)
        except Exception as e:
            # Emit error signal with the error message
            self.error.emit(str(e))


def read_second_last_column(file_path):
    """
    Reads the second last column from a given .csv file.

    :param file_path: Path to the .sv file
    :return: List of values from the second last column
    """
    try:
        # Load the .csv file using pandas
        df = pd.read_csv(file_path, sep=',')

        # Check if the file has enough columns
        if df.shape[1] < 2:
            raise ValueError("The file does not have enough columns to extract the second last column.")

        # Extract the second last column
        second_last_column = df.iloc[:, -2]

        # Return the values as a list
        return second_last_column.tolist()

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


class UI_Logic_Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):

        # To save the detection information
        self.detection_info = []

        super(UI_Logic_Window, self).__init__(parent)
        self.timer_video = QtCore.QTimer()  # Create a timer
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.init_slots()
        self.cap = cv2.VideoCapture()
        self.num_stop = 1  # Control for pause/play signals
        self.output_folder = 'output/'
        self.vid_writer = None
        self.last_frame = None

        self.points = []  # Store all bottom-right corner points globally
        self.status = []  # Store all status globally
        # Initial model file name
        self.openfile_name_model = None

    # 控件绑定相关操作
    def init_slots(self):
        # Left Area
        self.ui.pushButton_classify.clicked.connect(self.classify)
        self.ui.pushButton_weights.clicked.connect(self.open_model)
        self.ui.pushButton_video.clicked.connect(self.button_video_open)
        # Bottom Area
        self.ui.pushButton_stop.clicked.connect(self.button_video_stop)
        self.ui.pushButton_finish.clicked.connect(self.finish_detect)
        # Right Area
        self.ui.pushButton_saveCoordinates.clicked.connect(self.save_coordinates)
        self.ui.pushButton_trimVideo.clicked.connect(self.crop_video)
        self.ui.pushButton_saveLastFrame.clicked.connect(self.save_last_frame)
        self.ui.pushButton_timestamp.clicked.connect(self.timestamp_action)

        self.timer_video.timeout.connect(self.show_video_frame) # 定时器超时，将槽绑定至show_video_frame

    # ---------------------- Left Area ----------------------#

    # 打开权重文件
    def open_model(self):
        self.openfile_name_model, _ = QFileDialog.getOpenFileName(self.ui.pushButton_weights, 'Select weights file',
                                                             'weights/')
        if not self.openfile_name_model:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"File not opened.", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            print('Weight Path: ' + str(self.openfile_name_model))
            self.model_init()

    # 加载相关参数，并初始化模型
    def model_init(self):
        # 模型相关参数配置
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, default=[0], help='filter by class: --class 0')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        self.opt = parser.parse_args()
        print(self.opt)
        # 默认使用opt中的设置（权重等）来对模型进行初始化
        source, weights, view_img, save_txt, imgsz = self.opt.source, self.opt.weights, self.opt.view_img, self.opt.save_txt, self.opt.img_size

        # 若openfile_name_model不为空，则使用此权重进行初始化
        if self.openfile_name_model:
            weights = self.openfile_name_model
            print("Using button choose model")

        self.device = select_device(self.opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        cudnn.benchmark = True

        # Load model
        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if self.half:
          self.model.half()  # to FP16

        # Get names and colors
        print("------------ Name ------------")
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = (0, 255, 0)
        print(self.names)
        print("Model initialized.")
        # 设置提示框
        QtWidgets.QMessageBox.information(self, u"Notice", u"Model Initialized.", buttons=QtWidgets.QMessageBox.Ok,
                                      defaultButton=QtWidgets.QMessageBox.Ok)

    def classify(self):
        """
        Function to execute the XGB_LDA_Classification_Code.py script with a loading spinner.
        """
        # Specify the path to the classification script
        script_path = os.path.join(os.getcwd(), "XGB_LDA_Classification_Code.py")

        # Check if the script exists
        if not os.path.exists(script_path):
            QtWidgets.QMessageBox.warning(self, "Error", f"Script not found: {script_path}")
            return

        # Create a loading spinner
        loading_spinner = QtWidgets.QLabel(self)
        loading_spinner.setText("Classification in progress...")
        loading_spinner.setAlignment(QtCore.Qt.AlignCenter)
        movie = QtGui.QMovie("loading_spinner.gif")  # Use a .gif file for the spinner
        loading_spinner.setMovie(movie)
        movie.start()

        # Show the spinner as a modal dialog
        spinner_dialog = QDialog(self)
        spinner_dialog.setWindowTitle("Please Wait")
        spinner_dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        layout = QVBoxLayout()
        layout.addWidget(loading_spinner)
        spinner_dialog.setLayout(layout)
        spinner_dialog.resize(200, 150)
        spinner_dialog.show()

        # Define a function to run the script
        def run_script():
            try:
                # Execute the Python script as a subprocess
                process = subprocess.run(
                    ["python", script_path],
                    check=True,  # Raise an error if the process fails
                    stdout=subprocess.PIPE,  # Capture standard output
                    stderr=subprocess.PIPE  # Capture standard error
                )

                # Close the spinner dialog
                spinner_dialog.close()

                # Display the output of the script in a message box
                output = process.stdout.decode("utf-8")
                QMessageBox.information(self, "Classification Result", output)

            except subprocess.CalledProcessError as e:
                # If the script fails, capture and display the error
                spinner_dialog.close()
                error_message = e.stderr.decode("utf-8")
                QMessageBox.critical(self, "Error", f"An error occurred:\n{error_message}")

        # Use a QTimer to execute the script asynchronously (to keep UI responsive)
        QTimer.singleShot(100, run_script)


# new ---------------------- Video Detection ----------------------#
    def show_results_precision(img, xywh, conf, landmarks, class_num, point, points):
        h, w, c = img.shape
        tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
        x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
        y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
        x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
        y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)

        clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 125), (0, 0, 50)]

        point_avg_x, point_avg_y = 0, 0
        for i in range(4):
            point_x = int(landmarks[2 * i] * w)
            point_y = int(landmarks[2 * i + 1] * h)
            point_avg_x += point_x
            point_avg_y += point_y

        cv2.circle(img, (point_avg_x // 4, point_avg_y // 4), tl + 3, clors[0], -1)
        return img, point_avg_x // 4, point_avg_y // 4

    # 目标检测
    def detect(self, name_list, img):
        """
        :param name_list: list that stores the detection results
        :param img: image to be detected
        :return: info_show: text information to be displayed
        """
        showimg = img
        with torch.no_grad():
            img = letterbox(img, new_shape=self.opt.img_size)[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Inference
            pred = self.model(img, augment=self.opt.augment)[0]
            # Apply NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes,
                                       agnostic=self.opt.agnostic_nms)

            info_show = ""

            status = read_second_last_column('predictions.csv')

            # Process detections
            for i, det in enumerate(pred):
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], showimg.shape).round()
                    det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13], showimg.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        # status = random.randint(0,1)
                        # 绘制检测框
                        plot_one_box2(xyxy, showimg, label=f"{cls} {conf:.2f}", color=(0, 255, 0), line_thickness=2)
                        # 绘制关键点
                        for j in range(0, len(landmarks), 2):
                        cv2.circle(showimg, (int(landmarks[j]), int(landmarks[j + 1])),
                                   radius=3, color=(0, 0, 255), thickness=-1)

                        cur_status = status[i]

                        # draw box
                        single_info = plot_one_box2(xyxy, showimg, cur_status)

                        c2 = (int(xyxy[2]), int(xyxy[3]))
                        self.points.append(c2)
                        self.status.append(cur_status)
                        info_show += single_info + "\n"
                    # Draw all previous points
                    for ind in range(len(self.points)):
                        point = self.points[ind]
                        cur_status = self.status[ind]
                        cur_color = (0, 255, 0) if cur_status == 0 else (0, 0, 255)
                        cv2.circle(showimg, point, radius=3, color=cur_color, thickness=-1)  # Draw all points
        return  info_show

    def set_video_name_and_path(self):
        # 获取当前系统时间，作为img和video的文件名
        now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        # if vid_cap:  # video
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 视频检测结果存储位置
        save_path = self.output_folder + 'video_output/' + now + '.mp4'
        return fps, w, h, save_path # fps是帧率，w是宽度，h是高度

    # 打开视频并检测
    def button_video_open(self):
        video_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select the video", "data/", "*.mp4;;*.avi;;All Files(*)")
        flag = self.cap.open(video_name)
        if not flag:
            QtWidgets.QMessageBox.warning(self, u"Warning", u"Fail to open the video.", buttons=QtWidgets.QMessageBox.Ok,defaultButton=QtWidgets.QMessageBox.Ok)
        else:
            #-------------------------写入视频----------------------------------#
            fps, w, h, save_path = self.set_video_name_and_path()
            self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

            self.timer_video.start(30) # 以30ms为间隔，启动或重启定时器
            # 进行视频识别时，关闭其他按键点击功能
            self.ui.pushButton_video.setDisabled(True)


    # 定义视频帧显示操作
    def show_video_frame(self):
        name_list = []
        flag, img = self.cap.read()
        if img is not None:
            info_show = self.detect(name_list, img) # 检测结果写入到原始img上
            self.last_frame = img.copy()  # 保存当前帧作为最后一帧
            self.detection_info.append(info_show)
            self.vid_writer.write(img) # 检测结果写入视频
            print(info_show)

            self.ui.textBrowser.setText(info_show) # display the detection result in the textBrowser

            show = cv2.resize(img, (640, 480)) # 直接将原始img上的检测结果进行显示
            self.result = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(self.result.data, self.result.shape[1], self.result.shape[0],QtGui.QImage.Format_RGB888)
            self.ui.label.setPixmap(QtGui.QPixmap.fromImage(showImage))
            self.ui.label.setScaledContents(True)  # 设置图像自适应界面大小

        else:
            self.timer_video.stop()
            # 读写结束，释放资源
            # 保存视频的最后一帧为图片
            self.cap.release() # 释放video_capture资源
            self.vid_writer.release() # 释放video_writer资源
            self.ui.label.clear()
            # 视频帧显示期间，禁用其他检测按键功能
            self.ui.pushButton_video.setDisabled(False)

    # ---------------------- Right Area ----------------------#
    # Save the last frame as an image
    def save_last_frame(self):
        if self.last_frame is not None:
            now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
            file_path = self.output_folder + 'img_output/' + now + '_last_frame.jpg'
            cv2.imwrite(file_path, self.last_frame)
            QtWidgets.QMessageBox.information(self, u"Notice", u"Last frame saved.", buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)

    def save_coordinates(self):
        df = pd.DataFrame(self.detection_info)
        df.to_csv(self.output_folder + 'coordinates_output/coordinates.csv', index=False)
        QtWidgets.QMessageBox.information(self, u"Notice", u"Coordinates Saved.", buttons=QtWidgets.QMessageBox.Ok,
                                          defaultButton=QtWidgets.QMessageBox.Ok)


    def crop_video(self):
        # select video file
        video_path, _ = QFileDialog.getOpenFileName(self, "select video", "", "Video file (*.mp4 *.avi *.mkv)")
        if not video_path:
            return

        # get start time
        start_time, ok1 = QInputDialog.getText(self, "Start time", "Please enter the start time (format: seconds or mm:ss):")
        if not ok1 or not start_time.strip():
            return
        start_time = self.convert_to_seconds(start_time.strip())

        # get end time
        end_time, ok2 = QInputDialog.getText(self, "End time", "Please enter the end time (format: seconds or mm:ss):")
        if not ok2 or not end_time.strip():
            return
        end_time = self.convert_to_seconds(end_time.strip())

        # select save path
        save_path, _ = QFileDialog.getSaveFileName(self, "Trim Video", "", "Video file (*.mp4)")
        if not save_path:
            return

        # trim video
        try:
            self.trim_video(video_path, save_path, start_time, end_time)
        except Exception as e:
            print(f"Trim failed. Error: {e}")
            QtWidgets.QMessageBox.warning(self, u"Trim Failed", f"Trim Failed. \nError: {e}", buttons=QtWidgets.QMessageBox.Ok,)
        else:
            print(f"Video trimmed Successfully! \n Video saved to: {save_path}")
            QtWidgets.QMessageBox.information(self, u"Trim Successfully", f"Video trimmed successfully!\n Saved to: {save_path}", buttons=QtWidgets.QMessageBox.Ok,
                                              defaultButton=QtWidgets.QMessageBox.Ok)

    def trim_video_noProgressBar(self, video_path, save_path, start_time, end_time):
        # load video
        clip = VideoFileClip(video_path)

        # trim video
        trimmed_video = clip.subclip(start_time, end_time)
        trimmed_video = trimmed_video.fl_time(lambda t: t, apply_to=['video'], keep_duration=True)

        # save trimmed video
        trimmed_video.write_videofile(save_path, codec="libx264", audio_codec="aac")

    def trim_video(self, video_path, save_path, start_time, end_time):
        """
        Trim video with a loading spinner using a background thread.
        """
        # Create a loading spinner
        loading_spinner = QtWidgets.QLabel(self)
        loading_spinner.setAlignment(QtCore.Qt.AlignCenter)
        movie = QtGui.QMovie("loading_spinner.gif")
        loading_spinner.setMovie(movie)
        movie.start()

        # Show the spinner as a modal dialog
        self.spinner_dialog = QDialog(self)
        self.spinner_dialog.setWindowTitle("Processing")
        self.spinner_dialog.setWindowModality(QtCore.Qt.ApplicationModal)
        layout = QVBoxLayout()
        layout.addWidget(loading_spinner)
        self.spinner_dialog.setLayout(layout)
        # self.spinner_dialog.resize(100, 75)
        self.spinner_dialog.show()

        # Create and start the background thread
        self.trim_thread = VideoTrimThread(video_path, save_path, start_time, end_time)
        self.trim_thread.finished.connect(self.on_trim_finished)
        self.trim_thread.error.connect(self.on_trim_error)
        self.trim_thread.start()

    def on_trim_finished(self, save_path):
        """
        Slot called when video trimming is finished successfully.
        """
        # Close spinner dialog
        if self.spinner_dialog:
            self.spinner_dialog.close()

        self.spinner_dialog.close()
        QtWidgets.QMessageBox.information(self, "Success", f"Video trimmed successfully!\n> Saved to: {save_path}")

    def on_trim_error(self, error_message):
        """
        Slot called when an error occurs during video trimming.
        """
        # Close spinner dialog
        if self.spinner_dialog:
            self.spinner_dialog.close()
        self.spinner_dialog.close()
        QtWidgets.QMessageBox.critical(self, "Error", f"An error occurred: {error_message}")


    def convert_to_seconds(self, time_str):
        # convert time to seconds
        if ":" in time_str:
            minutes, seconds = map(int, time_str.split(":"))
            return minutes * 60 + seconds
        return int(time_str)

    def timestamp_action(self):
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num, ok = QInputDialog.getInt(self, "Input the frame number.", f"Please enter the frame number（1 - {total_frames}）:",
                                            min=1, max=total_frames)
        if ok:
            self.show_frame(frame_num)

    def show_frame(self, frame_num):
        # 定位到指定帧
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)  # 帧数从 0 开始
        ret, frame = self.cap.read()
        if not ret:
            print(f"Unable to read Frame {frame_num} !")
            return

        # 计算时间戳
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        time_seconds = frame_num / fps
        time_formatted = f"{int(time_seconds // 60)}:{int(time_seconds % 60):02d}"

        # 显示到新窗口
        self.display_frame(frame, time_formatted)

    def display_frame(self, frame, timestamp):
        # 创建新窗口
        new_window = QWidget()
        new_window.setWindowTitle("Frame Display")
        new_window.setGeometry(200, 200, 640, 480)

        # 转换帧为 QPixmap 格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # 设置布局
        layout = QVBoxLayout()

        # 显示帧
        label_image = QLabel()
        label_image.setPixmap(pixmap)
        label_image.setAlignment(Qt.AlignCenter)
        layout.addWidget(label_image)

        # 显示时间戳
        label_time = QLabel(f"Timestamp: {timestamp}")
        label_time.setAlignment(Qt.AlignCenter)
        layout.addWidget(label_time)

        new_window.setLayout(layout)
        new_window.show()

        # 保持窗口对象，防止被垃圾回收
        self.new_window = new_window


    # ---------------------- Bottom Area ----------------------#

    # Pause/Continue the video detection
    def button_video_stop(self):
        self.timer_video.blockSignals(False)
        # Pause the video detection
        # if QTimer is activated and triggered
        if self.timer_video.isActive() == True and self.num_stop%2 == 1:
            self.ui.pushButton_stop.setText(u'Continue Detection')
            self.num_stop = self.num_stop + 1 # Set the signal to even
            self.timer_video.blockSignals(True)
        # Continue the video detection
        else:
            self.num_stop = self.num_stop + 1
            self.ui.pushButton_stop.setText(u'Pause Detection')

    # 结束视频检测
    def finish_detect(self):
        # self.timer_video.stop()
        self.cap.release()  # Release video_capture resources
        self.vid_writer.release()  # Release video_writer resources
        self.ui.label.clear() # Clear the label
        # During the video frame display, disable other detection button functions
        self.ui.pushButton_video.setDisabled(False)

        # After the detection is completed, check whether the pause function is reset,
        # and restore the pause function to the initial state
        # Note: clicking pause, num_stop is in an even state
        if self.num_stop%2 == 0:
            print("Reset stop/begin!")
            self.ui.pushButton_stop.setText(u'Pause Detection')
            self.num_stop = self.num_stop + 1
            self.timer_video.blockSignals(False)




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    current_ui = UI_Logic_Window()
    current_ui.show()
    sys.exit(app.exec_())