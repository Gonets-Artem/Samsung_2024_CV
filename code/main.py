import cv2
import numpy as np
import time
import torch
import random

cap = cv2.VideoCapture(0)

width, height = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), #640
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #480
        )
   
# result = cv2.VideoWriter('filename.avi',  
#                          cv2.VideoWriter_fourcc(*'MJPG'), 
#                          10, (width, height)) 
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
# out = cv2.VideoWriter()
# output_file_name = "output_single.mp4"
# out.open(output_file_name, fourcc, fps, (width, height), True)
# i = 0

while True:
    ret, frame = cap.read()

    if (not ret) or (cv2.waitKey(1) & 0xFF == ord('q')):
        break

    # Изменение размера изображения до 324 на 324
    frame_re = torch.as_tensor(np.stack(cv2.resize(frame, (324, 324))))

    # Проверка зоны в рамках эксперимента
    # i += 1
    # if i % 2 == 0:
    #     rec = cv2.rectangle(im, (10, 200), (50, 300), (0, 255, 0), 3)
    # else:
    #     rec = cv2.rectangle(im, (10, 300), (100, 300), (255, 0, 0), 3)
    # cv2.imshow('gray feed', rec)   
    # time.sleep(0.1)  

    # Здесь будет результат модели
    y = [[random.randint(0,1) for _ in range(324)] for _ in range(324)]
    z = [y for _ in range(3)]
    ten_mask = torch.permute(torch.tensor(z), (2,1,0))

    final = torch.mul(ten_mask, frame_re).numpy().astype(np.uint8)
    cv2.imshow('video', final)
    # out.write(final)
    # result.write(final)


cap.release()
# result.release()
cv2.destroyAllWindows()