import sys
import cv2
import numpy as np
#import SFEGO_PyCUDA as SFEGO_Backend
import SFEGO_PyOpenCL as SFEGO_Backend

def SFEGO_MultiChannel(input_img, resize_ratio, execute_radius):
    input_height = input_img.shape[0]
    input_width = input_img.shape[1]
    target_height=int(input_height/resize_ratio)
    target_width=int(input_width/resize_ratio)
    resized_color=cv2.resize(input_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    results_gray=[]
    SpatialFrame_results=[]
    for gray in cv2.split(resized_color):
        result_gray=SFEGO_Backend.SFEGO(gray, execute_radius)
        SpatialFrame_result=cv2.resize(result_gray, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
        
        #Store F32 result
        SpatialFrame_results.append(SpatialFrame_result)
        
        #Calculate min, max
        result_gray=SpatialFrame_result
        result_min=np.min(result_gray)
        result_max=np.max(result_gray)
        
        #Real Amplitude
        output_gray=(result_gray-result_min).astype(np.uint8)
        
        #Normalize to uint8 0~255 (remove these two line to return Real Amplitude in UINT8)
        output_gray=255*(result_gray-result_min)/(result_max-result_min)
        output_gray=output_gray.astype(np.uint8)
        
        #Store UINT8 result
        results_gray.append(output_gray.astype(np.uint8))
        
    return cv2.merge(SpatialFrame_results), cv2.merge(results_gray) #float32, uint8

#Read Image
filename = sys.argv[1]
img=cv2.imread(filename)

#Read Radius List
file = open('default_radius')
lines = file.readlines()

Dim=len(lines)
RESULTS=[]
RESULTS.append(img)
for X in range(Dim):
    fields = lines[X].strip().split()
    resize_ratio=float(fields[0])
    execute_radius=int(fields[1])
    # N-channels (As Input Channels) with Float32 (downstream task) and UINT8 (visualize)
    ColorSpatialFrameF32, ColorSpatialFrameU8=SFEGO_MultiChannel(img, resize_ratio, execute_radius)
    RESULTS.append(ColorSpatialFrameU8) #Change to F32 if use in scienefic research
    cv2.imshow('Result', ColorSpatialFrameU8)
    out_filename=filename+"_SFEGO_Color_R"+str(resize_ratio*execute_radius)+"("+str(resize_ratio)+"x"+str(execute_radius)+").png"
    FinalResult=cv2.hconcat([img, ColorSpatialFrameU8])
    cv2.imwrite(out_filename, FinalResult)
    cv2.waitKey(1)
    
FinalResult=cv2.vconcat(RESULTS)
cv2.imwrite(filename+"_SFEGO_Color.png", FinalResult)
