#coding: utf-8
import json
import os
import requests
import base64
import cv2
import numpy as np
import glob
import re
import time
import threading
import multiprocessing

def get_video(video_root):
    video_dir = []
    for root, _, files in os.walk(video_root):
        if len(files) != 0:
            video_dir.append(root)
    return video_dir

        
def request(url, data):
    try:
        response = requests.post(url=url, data=data)
        return response.text
    except Exception as e:
        print(e)
     
def preprocess(frames):
    '''
        对图像进行解码处理
    :return:
    '''
    img_sequence = []
    for i, frame in enumerate(frames):
        frame = cv2.imread(frame)
        frame = cv2.resize(frame, dsize=(224, 224))
        frames_en = cv2.imencode('.jpg', frame)[1]
        frame_str = base64.b64encode(frames_en)
        img_sequence.append(frame_str)

    return img_sequence

     
def data_process(frames, seq_length):
    assert len(frames) >= seq_length, "frame number {} is less than {}".format(len(frames), seq_length)
    if len(frames) > seq_length:
        tick = len(frames) / float(seq_length)
        offsets = [int(tick / 2.0 + tick * x) for x in range(seq_length)]
        frames = [frames[i] for i in offsets]
    after_frames = preprocess(frames)

    return after_frames

def save_frames(image_dir, output_dir):
    os.system('cp -r {} {}'.format(image_dir, output_dir))

def put_in_order(images_list):
    info_dict = {}
    for single_image in images_list:
        number = re.findall(r'(\d+).jpg', single_image)
        info_dict[single_image] = int(number[0])
    info_dict = sorted(info_dict.items(), key=lambda x: x[1])
    images_list = [single_image[0] for single_image in info_dict]
    return images_list
    
def task(images_dir, i):
    time_sum = 0.0
    #fin = open('result_{}.txt'.format(i), 'w')
    correct_num = 0
    while True:
        for image_dir in images_dir:
            try:
                images_list = glob.glob(image_dir + '/*.jpg')
                images_list = sorted(images_list)
                #assert len(images_list) >= seq_length, 'sequence must larger than 8!'
                #images_list = put_in_order(images_list)

                #对图像b64编码，使用字符串传输
                data = data_process(images_list, seq_length)

                # frames_seq = cv2.imencode('.jpg', mat)[1] 
                # frames_str = base64.b64encode(frames_seq)

                tick_time = time.time()
                req_data = {'seqArray':[{'imgArray':data}]}
                #发送八个图像数据编码手的数据
                res = request(url, repr(req_data))
                time_sum += (time.time() - tick_time)

                res = json.loads(res)
                # max_val = max(res['data'])
                # index = res['data'].index(max_val)
                # fin.write(image_dir + ' ' + str(index) + ' ' + str(max_val) + '\n')
                #print(image_dir)
                #print(res)
                if res['data'][0] == 0:
                     print(images_dir)
                     print(res)
                     #save_frames(images_dir,output_dir)
                     correct_num +=1
            except Exception as e:
                print('Error: {}'.format(e))
   # fin.close()
    print('Correct num: {}'.format(correct_num))
    print('Inference time is: {}'.format(time_sum))
    
if __name__ == '__main__':

    start_time = time.time()
    url = "http://192.168.1.110:8081/ailab/imageprocess"
    # url = "http://47.103.85.135:10010/ailab/imageprocess"
    # set output_dir
    output_dir = './output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #set test dir
    video_root = 'images'
    seq_length = 8
    images_dir = sorted(get_video(video_root))
    # images_dir = [dir for dir in images_dir if 'Pos' in dir]
    # images_dir = [os.path.join(video_dir, file) for file in os.listdir(video_dir)]

    # split image into 4 thread
    num_cpus = 1
    length = len(images_dir)
    split_boundary = length // num_cpus
    threads = []
    for i in range(num_cpus):
        sub_images_dir = images_dir[i * split_boundary: (i+1) * split_boundary]
        #sub_images_dir = images_dir[0: 2]
        # t = threading.Thread(target=task, args=(sub_images_dir, i))
        #把测试目录下的图片送去处理
        t = multiprocessing.Process(target=task, args=(sub_images_dir, i))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    
    # merge all txt files into one file
    os.system('cat {}* > {}'.format('result_', 'result.txt'))
    os.system('rm -rf {}*'.format('result_'))
    
    end_time = time.time()
    print("Consume time is {}".format(end_time - start_time))

