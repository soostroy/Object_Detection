from sense_hat import SenseHat
from red_heart import *
import time

sense = SenseHat()


def red():
    sense.clear(255, 0, 0)


def blue():
    sense.clear(0, 0, 255)


def green():
    sense.clear(0, 255, 0)


def yellow():
    sense.clear(255, 255, 0)


def heart():
    for i in range(5):
        sense.set_pixels(images[count % len(images)]())
        time.sleep(.25)
        sense.clear()
        time.sleep(.25)


sense.stick.direction_up = heart
sense.stick.direction_down = blue
sense.stick.direction_left = green
sense.stick.direction_right = yellow
sense.stick.direction_middle = sense.clear
while True:
    # do nothing
    pass
