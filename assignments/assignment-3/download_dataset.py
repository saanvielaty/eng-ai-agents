
from roboflow import Roboflow

rf = Roboflow(api_key="@")

project = rf.workspace("drone-detection-g4d3g").project("drone-detection-a1tsf")

version = project.version(6)

dataset = version.download("yolov8")

print("Done! Dataset at:", dataset.location)

