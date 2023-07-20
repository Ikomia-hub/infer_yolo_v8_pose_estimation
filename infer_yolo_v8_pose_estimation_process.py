# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import copy
from ikomia import core, dataprocess, utils
from ultralytics import YOLO
import torch
import os
from ultralytics import download

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------


class InferYoloV8PoseEstimationParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.model_name = "yolov8m-pose"
        self.cuda = torch.cuda.is_available()
        self.input_size = 640
        self.conf_thres = 0.5
        self.iou_thres = 0.7
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        self.model_name = str(param_map["model_name"])
        self.cuda = utils.strtobool(param_map["cuda"])
        self.input_size = int(param_map["input_size"])
        self.conf_thres = float(param_map["conf_thres"])
        self.iou_thres = float(param_map["iou_thres"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["model_name"] = str(self.model_name)
        param_map["cuda"] = str(self.cuda)
        param_map["input_size"] = str(self.input_size)
        param_map["conf_thres"] = str(self.conf_thres)
        param_map["iou_thres"] = str(self.iou_thres)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferYoloV8PoseEstimation(dataprocess.CKeypointDetectionTask):

    def __init__(self, name, param):
        dataprocess.CKeypointDetectionTask.__init__(self, name)
        # Add input/output of the process here
        # Create parameters class
        if param is None:
            self.set_param_object(InferYoloV8PoseEstimationParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.device = torch.device("cpu")
        self.model = None
        self.half = False
        self.model_name = None
        self.classes = ["person"]
        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                         [7, 13], [6, 7], [6, 8], [7, 9], [
                             8, 10], [9, 11], [2, 3],
                         [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

        self.palette = [[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]]
        self.repo = 'ultralytics/assets'
        self.version = 'v0.0.0'

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call begin_task_run() for initialization
        self.begin_task_run()
        # Get parameters :
        param = self.get_param_object()

        # Get input :
        input = self.get_input(0)

        # Get image from input/output (numpy array):
        src_image = input.get_image()

        # Load model
        if param.update or self.model is None:
            self.device = torch.device(
                "cuda") if param.cuda else torch.device("cpu")
            self.half = True if param.cuda else False
            # Set path
            model_folder = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), "weights")
            model_weights = os.path.join(
                str(model_folder), f'{param.model_name}.pt')
            # Download model if not exist
            if not os.path.isfile(model_weights):
                url = f'https://github.com/{self.repo}/releases/download/{self.version}/{param.model_name}.pt'
                download(url=url, dir=model_folder, unzip=True)
            self.model = YOLO(model_weights)

            # Set Keypoints links
            keypoint_links = []
            for (start_pt_idx, end_pt_idx), color in zip(self.skeleton, self.palette):
                link = dataprocess.CKeypointLink()
                link.start_point_index = start_pt_idx
                link.end_point_index = end_pt_idx
                link.color = color
                keypoint_links.append(link)
            self.set_keypoint_links(keypoint_links)
            self.set_object_names(self.classes)
            param.update = False

        # Run detection
        results = self.model.predict(
            src_image,
            save=False,
            imgsz=param.input_size,
            conf=param.conf_thres,
            iou=param.iou_thres,
            half=self.half,
            device=self.device
        )

        # Get output data
        boxes = results[0].boxes.xyxy
        confidences = results[0].boxes.conf
        keypoints_lists = results[0].keypoints.xy

        for i, (box, conf, keypoints) in enumerate(zip(boxes, confidences, keypoints_lists)):
            box = box.detach().cpu().numpy()
            keypoints = keypoints.detach().cpu().numpy()
            box_x1, box_y1, box_x2, box_y2 = box[0], box[1], box[2], box[3]
            widht = box_x2 - box_x1
            height = box_y2 - box_y1
            kpts_data = keypoints

            # Set Keypoints links
            keypts = []
            kept_kp_id = []
            for link in self.get_keypoint_links():
                kp1, kp2 = kpts_data[link.start_point_index -
                                     1], kpts_data[link.end_point_index-1]
                x1, y1 = kp1
                x2, y2 = kp2
                if link.start_point_index not in kept_kp_id:
                    kept_kp_id.append(link.start_point_index)
                    keypts.append(
                        (link.start_point_index, dataprocess.CPointF(float(x1), float(y1))))
                if link.end_point_index not in kept_kp_id:
                    kept_kp_id.append(link.end_point_index)
                    keypts.append(
                        (link.end_point_index, dataprocess.CPointF(float(x2), float(y2))))

            # Add object to display
            self.add_object(
                i,
                0,
                float(conf),
                float(box_x1),
                float(box_y1),
                float(widht),
                float(height),
                keypts
            )

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferYoloV8PoseEstimationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_yolo_v8_pose_estimation"
        self.info.short_description = "Inference with YOLOv8 pose estimation models"
        self.info.description = "This algorithm proposes inference for pose estimation " \
                                "with YOLOv8 models. " \
            # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Pose"
        self.info.version = "1.0.1"
        self.info.icon_path = "icons/icon.png"
        self.info.authors = "Jocher, G., Chaurasia, A., & Qiu, J"
        self.info.article = "YOLO by Ultralytics"
        self.info.journal = ""
        self.info.year = 2023
        self.info.license = "AGPL-3.0"
        # URL of documentation
        self.info.documentation_link = "https://docs.ultralytics.com/"
        # Code source repository
        self.info.repository = "https://github.com/ultralytics/ultralytics"
        # Keywords used for search
        self.info.keywords = "YOLO, pose, estimation, keypoints, ultralytics, coco"

    def create(self, param=None):
        # Create process object
        return InferYoloV8PoseEstimation(self.info.name, param)
