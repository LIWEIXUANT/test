#-*- coding:utf-8 -*-
import os, argparse
import tensorflow as tf
import tensorflow.contrib
import cv2
import numpy as np
from tensorflow.python.framework import graph_util
from resnet_v2 import resnet_v2_50, resnet_arg_scope
slim = tf.contrib.slim
 
dir = os.path.dirname(os.path.realpath(__file__))
 
def freeze_graph(model_folder):
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    # We precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/resnet_v2.pb"
 
    # Before exporting our graph, we need to precise what is our output node
    # this variables is plural, because you can have multiple output nodes
    #freeze之前必须明确哪个是输出结点,也就是我们要得到推论结果的结点
    #输出结点可以看我们模型的定义
    #只有定义了输出结点,freeze才会把得到输出结点所必要的结点都保存下来,或者哪些结点可以丢弃
    #所以,output_node_names必须根据不同的网络进行修改
    output_node_names = "resnet_v2_50/pool5"
 
    # We clear the devices, to allow TensorFlow to control on the loading where it wants operations to be calculated
    clear_devices = True
    
    # We import the meta graph and retrive a Saver
    #saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
    x1 = tf.placeholder(tf.float32, [None, 224, 224, 3])
    with slim.arg_scope(resnet_arg_scope()):
            logits, end_points = resnet_v2_50(x1, num_classes = 2, is_training = False)
    variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
 
    #We start a session and restore the graph weights
    #这边已经将训练好的参数加载进来,也即最后保存的模型是有图,并且图里面已经有参数了,所以才叫做是frozen
    #相当于将参数已经固化在了图当中 
    #read an image
    img = cv2.imread("1.jpg")
    img = cv2.resize(img, dsize=(224,224))
    #img = img[10:309, 10:309]
    img = (img - np.mean(img)) / np.std(img)
    img = np.array([img])
    
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        
        saver.restore(sess, input_checkpoint)

        print('-------------------------------------------')
        for op in graph.get_operations():
          #print(op.name, op.values())
          if 'predictions' in op.name:
            print(op.name, op.values())
            #break
        print('-------------------------------------------')
        y_out = sess.run("resnet_v2_50/predictions/Reshape_1:0", feed_dict={"Placeholder:0": img})
        print(y_out[0].max()) # [[ 0.]] Yay!
        print(y_out[0].argmax())
 
        # We use a built-in TF helper to export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(
            sess, 
            input_graph_def, 
            output_node_names.split(",") # We split on comma for convenience
        ) 
 
        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, help="Model folder to export")
    args = parser.parse_args()
 
    freeze_graph(args.model_folder)
