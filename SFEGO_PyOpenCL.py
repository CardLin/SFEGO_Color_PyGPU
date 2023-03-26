import numpy as np
import pyopencl as cl
import math
import cv2
import sys

M_PI=3.14159265358979323846

# Variable for OpenCL device number
Platfrom_ID=0
Device_ID=0

def build_list(radius):
    ar_len=0
    x_list=[]
    y_list=[]
    deg_list=[]
    radius_list=[]
    for i in range(-radius,radius+1):
        for j in range(-radius,radius+1):
            if ((np.sqrt(i*i+j*j) < radius+1.0) and not (i==0 and j==0)):
                x_list.append(j)
                y_list.append(i)
                deg=math.atan2(j,i)
                if deg<0.0:
                    deg+=M_PI*2
                deg_list.append(deg)
                radius_list.append(np.sqrt(i*i+j*j))
    zipped=zip(x_list, y_list, deg_list, radius_list)
    zipped=sorted(zipped, key = lambda x: (x[2], x[3]))
    return zipped

#Initial PyOpenCL
#ctx = cl.create_some_context()
ctx = cl.Context([cl.get_platforms()[Platfrom_ID].get_devices()[Device_ID]])
queue = cl.CommandQueue(ctx)
prg = cl.Program(ctx, open('kernel.cl').read()).build()
knl_gradient = prg.GMEMD_gradient
knl_integral = prg.GMEMD_integral
mf = cl.mem_flags


def SFEGO(input_data, radius):
    #Setup Radius
    ar_list=build_list(radius)
    x_list, y_list, deg_list, radius_list=zip(*ar_list)
    list_len=len(ar_list)
    
    target_height = input_data.shape[0]
    target_width = input_data.shape[1]

    #Convert to Numpy
    np_data = np.asarray(input_data).flatten().astype(np.float32)
    np_x_list = np.asarray(x_list).astype(np.int32)
    np_y_list = np.asarray(y_list).astype(np.int32)
    np_deg_list = np.asarray(deg_list).astype(np.float32)

    #OpenCL Buffer 
    list_x = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_x_list)
    list_y = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_y_list)
    list_deg = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_deg_list)
    data = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np_data)
    diff = cl.Buffer(ctx, mf.READ_WRITE, np_data.nbytes)
    direct = cl.Buffer(ctx, mf.READ_WRITE, np_data.nbytes)
    result = cl.Buffer(ctx, mf.READ_WRITE, np_data.nbytes)

    #OpenCL Execution
    knl_gradient(queue, (target_width, target_height), None, data, diff, direct, list_x, list_y, list_deg, np.int32(list_len), np.int32(target_width), np.int32(target_height))
    knl_integral(queue, (target_width, target_height), None, result, diff, direct, list_x, list_y, list_deg, np.int32(list_len), np.int32(target_width), np.int32(target_height))

    #Get OpenCL Result
    np_result = np.empty_like(np_data)
    cl.enqueue_copy(queue, np_result, result)
    np_result = np_result / list_len

    list_x.release()
    list_y.release()
    list_deg.release()
    data.release()
    diff.release()
    direct.release()
    result.release()

    # Reshape to correct width, height
    result_gray=np_result.reshape((target_height, target_width))
    
    return result_gray

'''
#Read Image
filename = sys.argv[1]
img=cv2.imread(filename)
height = img.shape[0]
width = img.shape[1]
channels = img.shape[2]
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

file = open('default_radius')
for line in file:
    fields = line.strip().split()
    resize_ratio=float(fields[0])
    execute_radius=int(fields[1])
    effective_radius=resize_ratio*execute_radius
    target_height=int(height/resize_ratio)
    target_width=int(width/resize_ratio)
    print(resize_ratio, execute_radius, target_height, target_width)
    resized_gray=cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    # Run SFEGO OpenCL Kernel Code
    result_gray=SFEGO(resized_gray, execute_radius)

    # Use this SpatialFrame_result for down stream task
    SpatialFrame_result=cv2.resize(result_gray, (width, height), interpolation=cv2.INTER_LINEAR)

    #Calculate min, max
    result_gray=SpatialFrame_result
    result_min=np.min(result_gray)
    result_max=np.max(result_gray)

    #Real Amplitude
    output_gray=(result_gray-result_min).astype(np.uint8)
    cv2.imshow('Result ', output_gray)
    cv2.waitKey(1)

    #Normalize to 0~255
    output_gray=255*(result_gray-result_min)/(result_max-result_min)
    output_filename=filename+"_GMEMD_SpatialFrame}"+str(round(effective_radius, 2))+"("+str(resize_ratio)+"x"+str(execute_radius)+").png"
    cv2.imwrite(output_filename, output_gray)
'''



