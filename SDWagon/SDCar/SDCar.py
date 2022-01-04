import cv2
import numpy as np
import logging
import math
import datetime
import sys
import car_dir as steer_motor
import motor as drive_motor

class SDCar(object):
    
    def __init__(self):
        print('debug')
        busnum = 0
        logging.info('Creating an SDC object')
        steer_motor.setup()
        print('debug2')
        drive_motor.setup()
        drive_motor.ctrl(1)
        self.spd_cmd = 0
#         self.speed = 0
#         self.spd_cur = self.spd_cmd
        self.str_cmd = 0
#         self.str_cur = self.str_cmd
        steer_motor.home()
#         drive_motor.stop()
        

    def steer(self, str_cmd):
        MAX_STEER_DIFF = 30
        steer_motor.turn(str_cmd)

    def accel(self, spd_cmd):
        drive_motor.setSpeed(spd_cmd)

    def stop_all(self):
        steer_motor.home()
        drive_motor.stop()


