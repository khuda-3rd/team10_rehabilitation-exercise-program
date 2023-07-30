#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from scipy.signal import find_peaks

class KeyPointClassifier(object):
    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier_0725_lstm.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.landmark_lists = [[0 for j in range(42)] for i in range(56)]
        self.result_list = []

    def __call__(
        self,
        landmark_list,
        label
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.list_update(landmark_list)


        arr = np.array([self.landmark_lists], dtype=np.float32).reshape(1, 56, 42)
        # arr = tf.expand_dims(arr, axis=0)

        self.interpreter.set_tensor(
            input_details_tensor_index,
            arr)
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))
        peak, r_list = self.result_update(result[0][label])

        return peak, r_list
        # return peak, r_list

    
    def result_update(self, result):
        # self.result_list[1:] = self.result_list[0:99] 
        # self.result_list[0] = result

        self.result_list.append(result)
        peak, _ = find_peaks(self.result_list)
        if peak != []:
            temp = self.result_list
            self.result_list = []
            return peak, temp
        else:
            return [], []
        
    def list_update(self, landmark_list):
        for i in range(55):
            self.landmark_lists[i] = self.landmark_lists[i+1]
        self.landmark_lists[55] = landmark_list