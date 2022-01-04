import logging
from SDCar import SDCar
import cv2
import datetime
from hand_coded_lane_follower import HandCodedLaneFollower
from objects_on_road_processor import ObjectsOnRoadProcessor
from past.utils import old_div


_SHOW_IMAGE = True


def process_objects_on_road(image):
    image = traffic_sign_processor.process_objects_on_road(image)
    return image

def follow_lane(image):
    image = self.lane_follower.follow_lane(image)
    return image

def cleanup():
    """ Reset the hardware"""
    logging.info('Stopping the car, resetting hardware.')
    
    car.stop_all()
    camera.release()
    video_orig.release()
    video_lane.release()
    video_objs.release()
    cv2.destroyAllWindows()
    
def create_video_recorder(path):
    return cv2.VideoWriter(path, fourcc, 20.0, (SCREEN_WIDTH, SCREEN_HEIGHT))

def show_image(title, frame, show=_SHOW_IMAGE):
    if show:
        cv2.imshow(title, frame)

def map_degree_2_steercmd(x, in_min=45, in_max=135, out_min = 0, out_max = 255):
    return old_div((x - in_min) * (out_max - out_min), (in_max - in_min)) + out_min

INITIAL_SPEED = 0
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

#Create all the objects
car = SDCar()
lane_follower = HandCodedLaneFollower()
traffic_sign_processor = ObjectsOnRoadProcessor()


fourcc = cv2.VideoWriter_fourcc(*'XVID')
datestr = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
video_orig = create_video_recorder('/home/jetson/Videos/car_video%s.avi' % datestr)
video_lane = create_video_recorder('/home/jetson/Videos/car_video_lane%s.avi' % datestr)
video_objs = create_video_recorder('/home/jetson/Videos/car_video_objs%s.avi' % datestr)
camera = cv2.VideoCapture(-1)
camera.set(3, SCREEN_WIDTH)
camera.set(4, SCREEN_HEIGHT)


i=0
while (camera.isOpened()):
    _, image_lane = camera.read()
#     print(image_lane)
    image_objs = image_lane.copy()
#     print(image_objs)
    i += 1
    video_orig.write(image_lane)

    
    image_objs = traffic_sign_processor.process_objects_on_road(image_objs)
    video_objs.write(image_objs)
    show_image('Detected Objects', image_objs)

    image_lane = lane_follower.follow_lane(image_lane)
    video_lane.write(image_lane)
    show_image('Lane Lines', image_lane)
    
 
#     car.accel(20)
    steer_cmd = map_degree_2_steercmd(lane_follower.curr_steering_angle)
#     steer_cmd = lane_follower.curr_steering_angle
    print('Steer cmd :' + str(steer_cmd))
    car.steer(steer_cmd)
    steer_spd = traffic_sign_processor.speed
    car.accel(steer_spd)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cleanup()
        break
    

