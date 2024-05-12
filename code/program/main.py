import cv2
import torch
import time
import numpy as np
from model_segmentation import ModelSegmentation
from model_classification import ModelClassification

class VideoEditor:
    def __init__(self, path_segm_people, path_segm_glasses, path_class_glasses):
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_segm_people = ModelSegmentation(model_path=path_segm_people, device=self.torch_device)
        self.model_segm_glasses = ModelSegmentation(model_path=path_segm_glasses, device=self.torch_device)
        self.model_class_glasses = ModelClassification(model_path=path_class_glasses, device=self.torch_device)
        self.is_g = 0
        self.no_g = 0
        self.state = False
        self.cap = cv2.VideoCapture(0)
        self.width = int(self.cap.get(3))
        self.height = int(self.cap.get(4))

    def run(self):
        self.model_class_glasses.set_size(self.width, self.height)
        while True:
            ret, frame = self.cap.read()

            if (not ret) or (cv2.waitKey(1) & 0xFF == ord('q')):
                break

            pred_people = self.model_segm_people.predict(frame)
            pred_glasses = self.model_segm_glasses.predict(frame)
            pred_black = self.model_class_glasses.predict(frame)

            count_glasses = np.count_nonzero(pred_glasses==0)
            count_general = np.count_nonzero(np.multiply(pred_people==0, pred_glasses==0)==1)
            iou = 0 if count_glasses == 0 else count_general/count_glasses

            print("\niou: {:.2f}".format(iou), "pred: {:.3f}".format(pred_black), sep='\t', end='')
            line = torch.tensor(pred_people).unsqueeze(0)
            if pred_black <= 0.5 and iou >= 0.85:
                if self.state == False and self.is_g < 2:
                    self.is_g += 1
                elif self.state == False and self.is_g >= 2:
                    self.is_g, self.state = 0, True
                    print('\tOn', end='')
                    frame = torch.mul(torch.tensor(frame).permute(2,0,1), 
                                      torch.cat([line, line, line], dim=0)
                                    ).permute(1,2,0).numpy().astype(np.uint8)
                else:
                    self.no_g = 0
                    frame = torch.mul(torch.tensor(frame).permute(2,0,1), 
                                      torch.cat([line, line, line], dim=0)
                                    ).permute(1,2,0).numpy().astype(np.uint8)
            else:
                if self.state == True and self.no_g < 3:
                    self.no_g += 1
                    frame = torch.mul(torch.tensor(frame).permute(2,0,1), 
                                      torch.cat([line, line, line], dim=0)
                                    ).permute(1,2,0).numpy().astype(np.uint8)
                elif self.state == True and self.no_g >= 3:
                    self.no_g, self.state = 0, False
                    print('\tOff', end='')
                else:
                    self.is_g = 0
            cv2.imshow('video', frame)

            #time.sleep(0.1) 


if __name__ == '__main__':
    root_path = 'models/'
    file_segm_people = f'{root_path}segmentation_people.pth'
    file_segm_glasses = f'{root_path}segmentation_glasses.pth'
    file_class_glasses = f'{root_path}classification_glasses'
    obj = VideoEditor(file_segm_people, file_segm_glasses, file_class_glasses)
    obj.run()
