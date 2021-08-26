"""
Deploy SSD
==========

Please install CFFI and CV2 before running this script
pip3 install cffi
pip3 install opencv-python
"""
import os
import shutil
import time

import tvm
from tvm import relay
from tvm.contrib import graph_runtime

import cv2
import numpy as np
import tensorflow as tf

model_path = './ssd_w-int8_i-int8.tflite'
img_path = './VOCdevkit/VOC2007/JPEGImages/'
detection_path = './Object-Detection-Metrics/detections/'
mAP_path = './Object-Detection-Metrics/'
input_name = 'normalized_input_image_tensor'
input_size = 300
data_type = 'int8'
data_shape = (1, 300, 300, 3)
resulting_file_directory = './tvm_generated_files/'
score_threshold = 0.5

label2string = \
{
	0:   "person",
	1:   "bicycle",
	2:   "car",
	3:   "motorcycle",
	4:   "airplane",
	5:   "bus",
	6:   "train",
	7:   "truck",
	8:   "boat",
	9:   "traffic_light",
	10:  "fire_hydrant",
	12:  "stop_sign",
	13:  "parking_meter",
	14:  "bench",
	15:  "bird",
	16:  "cat",
	17:  "dog",
	18:  "horse",
	19:  "sheep",
	20:  "cow",
	21:  "elephant",
	22:  "bear",
	23:  "zebra",
	24:  "giraffe",
	26:  "backpack",
	27:  "umbrella",
	30:  "handbag",
	31:  "tie",
	32:  "suitcase",
	33:  "frisbee",
	34:  "skis",
	35:  "snowboard",
	36:  "sports_ball",
	37:  "kite",
	38:  "baseball_bat",
	39:  "baseball_glove",
	40:  "skateboard",
	41:  "surfboard",
	42:  "tennis_racket",
	43:  "bottle",
	45:  "wine_glass",
	46:  "cup",
	47:  "fork",
	48:  "knife",
	49:  "spoon",
	50:  "bowl",
	51:  "banana",
	52:  "apple",
	53:  "sandwich",
	54:  "orange",
	55:  "broccoli",
	56:  "carrot",
	57:  "hot_dog",
	58:  "pizza",
	59:  "donut",
	60:  "cake",
	61:  "chair",
	62:  "couch",
	63:  "potted_plant",
	64:  "bed",
	66:  "dining_table",
	69:  "toilet",
	71:  "tv",
	72:  "laptop",
	73:  "mouse",
	74:  "remote",
	75:  "keyboard",
	76:  "cell_phone",
	77:  "microwave",
	78:  "oven",
	79:  "toaster",
	80:  "sink",
	81:  "refrigerator",
	83:  "book",
	84:  "clock",
	85:  "vase",
	86:  "scissors",
	87:  "teddy_bear",
	88:  "hair_drier",
	89:  "toothbrush",
}

######################################################################
# Set target
# ----------

target = 'llvm'
target_host = 'llvm'
ctx = tvm.cpu(0)

######################################################################
# Load the TFLite model
# ---------------------

import os
tflite_model_file = os.path.join(model_path)
tflite_model_buf = open(tflite_model_file, "rb").read()

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
# Output class name and bounding box
# ----------------------------------

def get_tvm_result(image_path, img_name, detection_file_name, detection_path):
    m = graph_runtime.create(graph, lib, ctx)

    img_org = cv2.imread(image_path+img_name)
    img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) # (1, 300, 300, 3)
    #flatten_image_data = img.flatten()
    #np.savetxt('input.txt', flatten_image_data, delimiter='\n')

    m.set_input(input_name, tvm.nd.array(img.astype(data_type)))
    m.set_input(**params)

    timeStart = time.time()
    m.run()
    timeEnd = time.time()
    print("Inference time: %f" % (timeEnd - timeStart))

    bounding_boxs, class_IDs, scores, number = m.get_output(0), m.get_output(1), m.get_output(2), m.get_output(3)
    bounding_boxs = bounding_boxs.asnumpy()
    class_IDs = class_IDs.asnumpy()
    scores = scores.asnumpy()
    number = number.asnumpy()[0]

    num_effective_boxes = 0
    for i in range(number):
        if scores[0, i] > score_threshold:
            num_effective_boxes += 1
            box = bounding_boxs[0, i, :]
            x0 = int(box[1] * img_org.shape[1])
            y0 = int(box[0] * img_org.shape[0])
            x1 = int(box[3] * img_org.shape[1])
            y1 = int(box[2] * img_org.shape[0])
            with open(detection_path + detection_file_name, 'a') as f:
                print(str(label2string[int(class_IDs[0, i])]),
                    scores[0, i], int(x0), int(y1), int(x1), int(y0), file=f)
    print("Number of detected objects:", num_effective_boxes)

######################################################################
# Detect objects with SSD
# -----------------------

print("\nStart to detect objects with SSD...")
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
# ----------------------------------

os.chdir(mAP_path)
os.system('python3 pascalvoc.py -np')

