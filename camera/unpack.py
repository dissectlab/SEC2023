import cv2
import os

from threading import Thread
import threading

import time


def unpack(root_dir):

    #root_dir = './3'
    start_time = time.time()

    file_name = root_dir + '/' + 'Recording.h264'
    directory = root_dir + '/' + 'Images'

    if not os.path.exists(directory):
        os.mkdir(directory)

    cap = cv2.VideoCapture(file_name, cv2.CAP_FFMPEG)
    i = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imwrite('./' + directory + '/' + str(i) + '.jpg', frame)
        i += 1

    cap.release()

    print(root_dir + "--- %0.2f seconds ---" % (time.time() - start_time))


t0 = Thread(target=unpack, args=('./0',))
t1 = Thread(target=unpack, args=('./1',))
t2 = Thread(target=unpack, args=('./2',))
t3 = Thread(target=unpack, args=('./3',))
t4 = Thread(target=unpack, args=('./4',))

t0.start()
t1.start()
t2.start()
t3.start()
t4.start()

t0.join()
t1.join()
t2.join()
t3.join()
t4.join()

print('Done')
