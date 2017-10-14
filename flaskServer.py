from flask import Flask
import wiringpi
from multiprocessing import Value

#import RPi.GPIO as GPIO

#GPIO.setmode(GPIO.BCM)
#GPIO.setup(18, GPIO.OUT)
#pwm = GPIO.PWM(18,100)
wiringpi.wiringPiSetupPhys()
wiringpi.pinMode(12, 2)

wiringpi.pwmSetMode(0)
wiringpi.pwmSetRange(1024)
wiringpi.pwmSetClock(375)

wiringpi.pwmWrite(32,77)

angle = Value('i', 77)
#while True:
#    wiringpi.pwmWrite(12, 51)
#    wiringpi.delay(1000)
#    wiringpi.pwmWrite(12, 77)
#    wiringpi.delay(1000)
#    wiringpi.pwmWrite(12, 77)
#    wiringpi.delay(1000)

app = Flask(__name__)

@app.route('/left')
def move_left():
    print("Moving to the left")
    with angle.get_lock():
        angle.value -= 5
    wiringpi.pwmWrite(12, angle.value)
    return 'ok'

@app.route('/right')
def move_right():
    print("Moving to the right")
    with angle.get_lock():
        angle.value += 5
    wiringpi.pwmWrite(12, angle.value)
    return 'ok'

if __name__ == '__main__':
    curr = 77
    app.run(debug=True, host='0.0.0.0')
