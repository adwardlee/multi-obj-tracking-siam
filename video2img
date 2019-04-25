import cv2
import argparse

parser = argparse.ArgumentParser(description='input video')
parser.add_argument('-v',type = str, default = 'remove01.avi')
args = parser.parse_args()

cap = cv2.VideoCapture(args.v)
dirs = 'remove1/'
i = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        cv2.imwrite(dirs + str(i) + '.jpg',frame)
        i += 1
    else:
        break
print('read end ')
cap.release()
