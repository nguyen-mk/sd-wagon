import cv2
import logging
import datetime
import time
import numpy as np
#import edgetpu.detection.engine
import pycoral.adapters.classify
import pycoral.adapters.detect
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter
# from tflite_runtime.interpreter import load_delegate
from pycoral.utils.dataset import read_label_file
from PIL import Image
from traffic_objects import *

_SHOW_IMAGE = False


class ObjectsOnRoadProcessor(object):
    """
    This class 1) detects what objects (namely traffic signs and people) are on the road
    and 2) controls the car navigation (speed/steering) accordingly
    """

    def __init__(self,
                 car=None,
                 speed_limit=80,
                 model='/home/pi/SDCar/models/road_signs_quantized_edgetpu.tflite',
                 label='/home/pi/SDCar/models/road_sign_labels.txt',
                 width=640,
                 height=480):
        # model: This MUST be a tflite model that was specifically compiled for Edge TPU.
        # https://coral.withgoogle.com/web-compiler/
        logging.info('Creating a ObjectsOnRoadProcessor...')
        self.width = width
        self.height = height

        # initialize car
        self.car = car
        self.speed_limit = speed_limit
        self.speed = speed_limit

        # initialize TensorFlow models
        with open(label, 'r') as f:
            pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
            self.labels = dict((int(k), v) for k, v in pairs)
#         self.labels = read_label_file(label)
        # initial edge TPU engine
        logging.info('Initialize Edge TPU with model %s...' % model)
        #self.engine = edgetpu.detection.engine.DetectionEngine(model)
        self.engine = make_interpreter(model)
        self.engine.allocate_tensors()
        self.min_confidence = 0.3
        self.num_of_objects = 3
        logging.info('Initialize Edge TPU with model done.')

        # initialize open cv for drawing boxes
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bottomLeftCornerOfText = (10, height - 10)
        self.fontScale = 1
        self.fontColor = (255, 255, 255)  # white
        self.boxColor = (0, 0, 255)  # RED
        self.boxLineWidth = 1
        self.lineType = 2
        self.annotate_text = ""
        self.annotate_text_time = time.time()
        self.time_to_show_prediction = 1.0  # ms

        #
        self.traffic_objects = {0:GreenTrafficLight(),
                                1:Person(),
                                2:RedTrafficLight(),
                                3:SpeedLimit(25),
                                4:SpeedLimit(40),
                                5:StopSign()}

    def process_objects_on_road(self, frame):
        # Main entry point of the Road Object Handler
        logging.debug('Processing objects.................................')
        objects, final_frame = self.detect_objects(frame)
        self.control_car(objects)
        logging.debug('Processing objects END..............................')

        return final_frame

    def control_car(self, objects):
        logging.debug('Control car...')
        car_state = {"speed": self.speed_limit, "speed_limit": self.speed_limit}

        if len(objects) == 0:
            logging.debug('No objects detected, drive at speed limit of %s.' % self.speed_limit)

        contain_stop_sign = False
        for obj in objects:
            #obj_label = self.labels[obj.label_id]
            #processor = self.traffic_objects[obj.label_id]
            obj_label = self.labels[obj.id]
            processor = self.traffic_objects[obj.id]
            if processor.is_close_by(obj, self.height):
                processor.set_car_state(car_state)
            else:
                logging.debug("[%s] object detected, but it is too far, ignoring. " % obj_label)
            if obj_label == 'Stop':
                contain_stop_sign = True

        if not contain_stop_sign:
            self.traffic_objects[5].clear()

        self.resume_driving(car_state)

    def resume_driving(self, car_state):
        old_speed = self.speed
        self.speed_limit = car_state['speed_limit']
        self.speed = car_state['speed']

        if self.speed == 0:
            self.set_speed(0)
        else:
            self.set_speed(self.speed_limit)
        logging.debug('Current Speed = %d, New Speed = %d' % (old_speed, self.speed))

        if self.speed == 0:
            logging.debug('full stop for 1 seconds')
            time.sleep(1)

    def set_speed(self, speed):
        # Use this setter, so we can test this class without a car attached
        self.speed = speed
        if self.car is not None:
            logging.debug("Actually setting car speed to %d" % speed)
            self.car.back_wheels.speed = speed



    ############################
    # Frame processing steps
    ############################
    def detect_objects(self, frame):
        logging.debug('Detecting objects...')

        # call tpu for inference
        
        start_ms = time.time()
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_RGB)
        _, scale = common.set_resized_input(
                 self.engine, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
#         pycoral.adapters.common.set_input(self.engine, img_pil)
#         input_details = self.engine.get_input_details()
#         output_details = self.engine.get_output_details()
#         input_index = self.engine.get_input_details()[0]["index"]
#         output_index = self.engine.get_output_details()[0]["index"]
# 
#         sizes = self.engine.get_input_details()[0]["shape"]
#         print('--------------------------')
#         print(sizes)
#         input_data = np.ones(sizes)
#         scale, zero_point = input_details[0]['quantization']
#         input_data = input_data / scale + zero_point
#         input_data = input_data.astype(np.uint8)
# 
#         common.set_input(self.engine, input_data)
#         self.engine.set_tensor(input_index, input_data)
              
        
        self.engine.invoke()
#         objects =self.engine.get_tensor(output_index)
        objects = pycoral.adapters.detect.get_objects(self.engine,
                                    score_threshold=self.min_confidence,image_scale=scale)
        
        
#        objects = self.engine.DetectWithImage(img_pil, threshold=self.min_confidence, keep_aspect_ratio=True, relative_coord=False, top_k=self.num_of_objects)
        
        if objects:
            for obj in objects:
                #height = obj.bounding_box[1][1]-obj.bounding_box[0][1]
                #width = obj.bounding_box[1][0]-obj.bounding_box[0][0]
                height = obj.bbox.ymax - obj.bbox.ymin
                width = obj.bbox.xmax - obj.bbox.xmin
                logging.debug("%s, %.0f%% w=%.0f h=%.0f" % (self.labels[obj.id], obj.score * 100, width, height))
                #box = obj.bounding_box
                box = obj.bbox
                coord_top_left = (int(box.xmin), int(box.ymin))
                coord_bottom_right = (int(box.xmax), int(box.ymax))
                cv2.rectangle(frame, coord_top_left, coord_bottom_right, self.boxColor, self.boxLineWidth)
                annotate_text = "%s %.0f%%" % (self.labels[obj.id], obj.score * 100)
                print(self.labels[obj.id])
                coord_top_left = (coord_top_left[0], coord_top_left[1] + 15)
                cv2.putText(frame, annotate_text, coord_top_left, self.font, self.fontScale, self.boxColor, self.lineType)
        else:
            logging.debug('No object detected')

        elapsed_ms = time.time() - start_ms

        annotate_summary = "%.1f FPS" % (1.0/elapsed_ms)
        logging.debug(annotate_summary)
        cv2.putText(frame, annotate_summary, self.bottomLeftCornerOfText, self.font, self.fontScale, self.fontColor, self.lineType)
#         cv2.imshow('Detected Objects', frame)

        return objects, frame


############################
# Utility Functions
############################
def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)


############################
# Test Functions
############################
def test_photo(file):
    object_processor = ObjectsOnRoadProcessor()
    frame = cv2.imread(file)
    combo_image = object_processor.process_objects_on_road(frame)
#     frame = np.asarray(combo_image)
    show_image('Detected Objects', frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_stop_sign():
    # this simulates a car at stop sign
    object_processor = ObjectsOnRoadProcessor()
    frame = cv2.imread('/home/pi/DeepPiCar/driver/data/objects/stop_sign.jpg')
    combo_image = object_processor.process_objects_on_road(frame)
    show_image('Stop 1', combo_image)
    time.sleep(1)
    frame = cv2.imread('/home/pi/DeepPiCar/driver/data/objects/stop_sign.jpg')
    combo_image = object_processor.process_objects_on_road(frame)
    show_image('Stop 2', combo_image)
    time.sleep(2)
    frame = cv2.imread('/home/pi/DeepPiCar/driver/data/objects/stop_sign.jpg')
    combo_image = object_processor.process_objects_on_road(frame)
    show_image('Stop 3', combo_image)
    time.sleep(1)
    frame = cv2.imread('/home/pi/DeepPiCar/driver/data/objects/green_light.jpg')
    combo_image = object_processor.process_objects_on_road(frame)
    show_image('Stop 4', combo_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_video(video_file):
    object_processor = ObjectsOnRoadProcessor()
    cap = cv2.VideoCapture(video_file + '.avi')

    # skip first second of video.
    for i in range(3):
        _, frame = cap.read()

    video_type = cv2.VideoWriter_fourcc(*'XVID')
    date_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    video_overlay = cv2.VideoWriter("%s_overlay_%s.avi" % (video_file, date_str), video_type, 20.0, (320, 240))
    try:
        i = 0
        while cap.isOpened():
            _, frame = cap.read()
            cv2.imwrite("%s_%03d.png" % (video_file, i), frame)

            combo_image = object_processor.process_objects_on_road(frame)
            cv2.imwrite("%s_overlay_%03d.png" % (video_file, i), combo_image)
            video_overlay.write(combo_image)

            cv2.imshow("Detected Objects", combo_image)

            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        video_overlay.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)-5s:%(asctime)s: %(message)s')

    # These processors contains no state
    test_photo('/home/pi/DeepPiCar/driver/data/objects/red_light.jpg')
    test_photo('/home/pi/DeepPiCar/driver/data/objects/person.jpg')
    test_photo('/home/pi/DeepPiCar/driver/data/objects/limit_40.jpg')
    test_photo('/home/pi/DeepPiCar/driver/data/objects/limit_25.jpg')
    test_photo('/home/pi/DeepPiCar/driver/data/objects/green_light.jpg')
    test_photo('/home/pi/DeepPiCar/driver/data/objects/no_obj.jpg')

    # test stop sign, which carries state
    test_stop_sign()