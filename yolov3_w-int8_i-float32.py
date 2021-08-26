"""
Deploy YOLOv3 in DarkNet
========================

Please install CFFI and CV2 before running this script
pip3 install cffi
pip3 install opencv-python
"""
import numpy as np

import os
import sys
import shutil
import time

import tvm
from tvm import relay

from core.yolov4 import filter_boxes
import tensorflow as tf
from core.config import cfg

model_path = './yolov3-416-int8_tf2_3_0.tflite'
img_path = './VOCdevkit/VOC2007/JPEGImages/'
detection_path = './Object-Detection-Metrics/detections/'
mAP_path = './Object-Detection-Metrics/'
input_name = 'input_1'
input_size = 416
data_type = 'float32'
data_shape = (1, 416, 416, 3)
resulting_file_directory = './tvm_generated_files/'

######################################################################
# Set target
# ----------

target = 'llvm'
target_host = 'llvm'
ctx = tvm.cpu(0)

######################################################################
# Load a TFLite model
# -------------------

import os
tflite_model_file = os.path.join(model_path)
tflite_model_buf = open(tflite_model_file, "rb").read()

# Get TFLite model from buffer
try:
    import tflite
    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

######################################################################
# Convert the TFLite model into Relay IR
# --------------------------------------

import tvm.relay as relay
dtype_dict = {input_name: data_type}
shape_dict = {input_name: data_shape}

mod, params = relay.frontend.from_tflite(tflite_model,
                                         shape_dict=shape_dict,
                                         dtype_dict=dtype_dict)

print("Printing relay module to relay_module.txt...")
with open('relay_module.txt', 'w') as f:
    print(mod.astext(show_meta_data=False), file=f)

######################################################################
# Compile the Relay module
# ------------------------

with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize":True}):
    graph, lib, params = relay.build(mod, target=target, target_host=target_host, params=params)

######################################################################
# Generate resulting files
# ------------------------
'''
print("Printing host code to host_code.cc...")
with open('host_code.cc', 'w') as f:
    print(lib.get_source(), file=f)

print("Printing device code to device_code.cl...")
with open('device_code.cl', 'w') as f:
    print(lib.imported_modules[0].get_source(), file=f)

print("Printing meta json to device_code.tvm_meta.json...")
lib.imported_modules[0].save("device_code", "cl")
os.remove("device_code")

print("Printing binary parameters to binary_params.bin...")
with open('binary_params.bin', 'wb') as writer:
    writer.write(relay.save_param_dict(params))
    writer.close()

print("Printing graph to graph.json...")
with open('graph.json', 'w') as f:
    print(graph, file=f)
'''
######################################################################
# Move all resulting files to a directory
# ---------------------------------------
'''
import shutil

try:
    shutil.rmtree(resulting_file_directory)
except OSError as e:
    print("Preparing a directory for resulting files")

os.mkdir(resulting_file_directory)

shutil.move('host_code.cc', resulting_file_directory)
shutil.move('device_code.cl', resulting_file_directory)
shutil.move('device_code.tvm_meta.json', resulting_file_directory)
shutil.move('binary_params.bin', resulting_file_directory)
shutil.move('graph.json', resulting_file_directory)
'''
######################################################################
# Utilities
# ---------

def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

######################################################################
# Output class name and bounding box
# ----------------------------------

def get_tvm_result(image_path, img_name, detection_file_name, detection_path):
    from tvm.contrib import graph_runtime
    m = graph_runtime.create(graph, lib, ctx)
    import cv2
    original_image = cv2.imread(image_path+img_name)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.0
    images_data = []
    for i in range(1):
        images_data.append(image_data)
    images_data = np.asarray(images_data).astype(np.float32)
    m.set_input(input_name, tvm.nd.array(images_data.astype(data_type)))
    m.set_input(**params)
    timeStart = time.time()
    m.run()
    timeEnd = time.time()
    print("Inference time: %f" % (timeEnd - timeStart))

    boxes, pred_conf = filter_boxes(m.get_output(0).asnumpy(), 
                        m.get_output(1).asnumpy(), score_threshold=0.5,
                        input_shape=tf.constant([input_size, input_size]))
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.5)

    image_h, image_w, _ = original_image.shape
    out_boxes, out_scores, out_classes, num_boxes = boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()
    classes = read_class_names(cfg.YOLO.CLASSES)
    num_classes = len(classes)
    print("Number of detected objects:", num_boxes)
    for i in range(num_boxes[0]):
        if int(out_classes[0][i]) < 0 or int(out_classes[0][i]) > num_classes: continue

        coor = out_boxes[0][i]
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)
        left, top, right, bottom = coor[1], coor[0], coor[3], coor[2]

        class_ind = int(out_classes[0][i])
        class_name = classes[class_ind]

        score = out_scores[0][i]
        #print("<class_name> <score> <left> <top> <right> <bottom>:", class_name, score, left, top, right, bottom)
        with open(detection_path + detection_file_name, 'a') as f:
            print(class_name, score, int(left), int(top), int(right), int(bottom), file=f)

######################################################################
# Detect objects with  YOLOv3
# ---------------------------

print("\nStart to detect objects with  YOLOv3...")
file_list = os.listdir(img_path)
print('Number of images:', len(file_list))
for file_name in file_list:
    img_id = os.path.splitext(file_name)[0]
    detection_file_name = img_id + '.txt'
    if not os.path.isfile(detection_path + detection_file_name):
        os.mknod(detection_path + detection_file_name)
    #with open(detection_path + detection_file_name, 'w') as f:
        #print('', file=f)
    print('\nDetecting objects in', file_name)
    get_tvm_result(img_path, file_name, detection_file_name, detection_path)
    print('Saving detection to', detection_file_name)

######################################################################
# Calculate mAP
# -------------

os.chdir(mAP_path)
os.system('python3 pascalvoc.py -np')
