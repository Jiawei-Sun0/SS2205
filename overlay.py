import numpy as np
import cv2
import imageio

def create_color(img,r,g,b):
    img = np.tile(img, (3,1,1))
    img = np.transpose(img, (1,2,0))
    # img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    # r0,g0,b0 = cv2.split(img)
    img[:,:,0] = img[:,:,0]*r
    img[:,:,1] = img[:,:,1]*g
    img[:,:,2] = img[:,:,2]*b
    return img
def overlay(pre,mask,ori):
    # img = ((pre > 0) & (mask > 0)).astype(np.uint8)
    # img = create_color(img,0,0,255)
    b_mask = np.tile(mask, (3,1,1))
    b_mask = np.transpose(b_mask, (1,2,0))
    mask = create_color(mask,255,255,255)

    b_pre = np.tile(pre, (3,1,1))
    b_pre = np.transpose(b_pre, (1,2,0))
    pre = create_color(pre,255,0,255)
    
    ori = ori.cpu().numpy()
    ori = ori.transpose(0,2,3,1)
    ori = ori[0,:,:,:]*255
    ori = ori.astype(np.uint8)
    
    # output = cv2.addWeighted(ori,1,pre,0.8,0)
    # output = cv2.addWeighted(output,1,mask,0.5,0)
    output = (ori - (ori * b_mask * 0.5) + (mask * 0.5)).astype(np.uint8)
    output = (output - (output * b_pre * 0.5) + (pre * 0.5)).astype(np.uint8)
    return output