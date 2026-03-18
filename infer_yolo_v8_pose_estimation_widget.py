# PyQt GUI framework
from PyQt6.QtWidgets import *

from torch.cuda import is_available

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion

from infer_yolo_v8_pose_estimation.infer_yolo_v8_pose_estimation_process import InferYoloV8PoseEstimationParam

# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferYoloV8PoseEstimationWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferYoloV8PoseEstimationParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        # Cuda
        self.check_cuda = pyqtutils.append_check(
            self.grid_layout, "Cuda", self.parameters.cuda and is_available())
        self.check_cuda.setEnabled(is_available())

        # Model name
        self.combo_model = pyqtutils.append_combo(
            self.grid_layout, "Model name")
        self.combo_model.addItem("yolov8n-pose")
        self.combo_model.addItem("yolov8s-pose")
        self.combo_model.addItem("yolov8m-pose")
        self.combo_model.addItem("yolov8l-pose")
        self.combo_model.addItem("yolov8x-pose")
        self.combo_model.addItem("YOLOv8x-pose-p6")

        self.combo_model.setCurrentText(self.parameters.model_name)


        # Input size
        self.spin_input_size = pyqtutils.append_spin(
                                            self.grid_layout,
                                            "Input size",
                                            self.parameters.input_size
        )

        # Confidence threshold
        self.spin_conf_thres = pyqtutils.append_double_spin(
            self.grid_layout,
            "Confidence threshold",
            self.parameters.conf_thres,
            min=0.,
            max=1.,
            step=0.01,
            decimals=2
        )

        # Confidence IOU
        self.spin_iou_thres = pyqtutils.append_double_spin(
                                            self.grid_layout,
                                            "Confidence IoU",
                                            self.parameters.iou_thres,
                                            min=0.,
                                            max=1.,
                                            step=0.01,
                                            decimals=2
        )

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)


    def on_apply(self):
        # Apply button clicked slot
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.cuda = self.check_cuda.isChecked()
        self.parameters.input_size = self.spin_input_size.value()
        self.parameters.conf_thres = self.spin_conf_thres.value()
        self.parameters.iou_thres = self.spin_iou_thres.value()
        self.parameters.update = True

        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferYoloV8PoseEstimationWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_yolo_v8_pose_estimation"

    def create(self, param):
        # Create widget object
        return InferYoloV8PoseEstimationWidget(param, None)
