import os, time, sys, numpy as np, zipfile, tarfile, shutil, cv2
import six.moves.urllib as urllib
import tensorflow as tf

import multiprocessing as mp
from multiprocessing import Process
from collections import defaultdict
from io import StringIO
import matplotlib
#%matplotlib inline
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from PIL import Image

sys.path.append('./object_detection/')
sys.path.append('..')
#sys.path.append('.')
from utils import ops as utils_ops, label_map_util, visualization_utils as vis_util


class TFOD_predict():

	def preparation(self, PATH_TO_CKPT = 'saved_model/frozen_inference_graph.pb', PATH_TO_LABELS = 'label_map.pbtxt'):
		self.category_index = {}
		self.detection_graph = tf.Graph()

		f = open(PATH_TO_LABELS, 'r').readlines()

		NUM_CLASSES = int(len(f)/4)

		with self.detection_graph.as_default():
			od_graph_def = tf.GraphDef()
			with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
				serialized_graph = fid.read()
				od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(od_graph_def, name='')

		label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
		# print(label_map)
		categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
		# print(categories)
		self.category_index = label_map_util.create_category_index(categories)
		# print(self.category_index)


	def image_to_np(self, image_path):
		image = Image.open(image_path)
		(im_width, im_height) = image.size
		return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

	def all_images_to_np(self, image_paths):
		result = []
		for image_path in image_paths:
			result.append(self.image_to_np(image_path))
		return result

	def predict_image(self, image, all_output_dicts, SPEC, task_id):
		spec = tf.train.ClusterSpec(SPEC)
		server = tf.train.Server(SPEC, job_name='worker', task_index=task_id)			
		graph=self.detection_graph
		with graph.as_default():
			with tf.Session(server.target) as sess:
				# Get handles to input and output tensors
				ops = tf.get_default_graph().get_operations()
				all_tensor_names = {output.name for op in ops for output in op.outputs}
				tensor_dict = {}
				for key in ['num_detections', 'detection_boxes', 'detection_scores',
						'detection_classes', 'detection_masks']:
					tensor_name = key + ':0'
					if tensor_name in all_tensor_names:
						tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
				
				image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

				# Run inference
				output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

				# all outputs are float32 numpy arrays, so convert types as appropriate
				output_dict['num_detections'] = int(output_dict['num_detections'][0])
				output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
				output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
				output_dict['detection_scores'] = output_dict['detection_scores'][0]
		all_output_dicts.put(output_dict)
		return output_dict


	def predict_multiple(self, images):
		all_output_dicts = []
		output = mp.Queue()
		graph=self.detection_graph
		with graph.as_default():
			with tf.Session() as sess:
				# Get handles to input and output tensors
				ops = tf.get_default_graph().get_operations()
				all_tensor_names = {output.name for op in ops for output in op.outputs}
				tensor_dict = {}
				for key in ['num_detections', 'detection_boxes', 'detection_scores',
						'detection_classes', 'detection_masks']:
					tensor_name = key + ':0'
					if tensor_name in all_tensor_names:
						tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
				
				image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

				input_ndarray_list = []
				processes = []

				# Setup a list of processes that we want to run
				# for i, image in enumerate(images):
				# 	print(i)
				# 	input_ndarray = np.expand_dims(image, 0)
				# 	input_ndarray_list.append(input_ndarray)
				# 	processes.append(mp.Process(target=sess.run, args=(tensor_dict, {image_tensor: input_ndarray})))

				# # Run processes
				# for p in processes:
				#     p.start()

				# # Exit the completed processes
				# for p in processes:
				# 	p.join()

    # 			# Get process results from the output queue
				# all_output_dicts = [output.get() for p in processes]
				# 	# Run inference
				# 	output_dict = sess.run(tensor_dict, feed_dict={image_tensor: input_ndarray})					
				#processes = [mp.Process(target=sess.run, args=(tensor_dict, feed_dict={image_tensor: input_ndarray})) for x in range(len(input_ndarray_list))]
				
				for output_dict in all_output_dicts:
					output_dict['num_detections'] = int(output_dict['num_detections'][0])
					output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
					output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
					output_dict['detection_scores'] = output_dict['detection_scores'][0]
					
					all_output_dicts.append(output_dict)


		return all_output_dicts

	def save_predictions_as_image(self, image_np, output_dict, image_path, output_directory):
		category=self.category_index
		# Visualization of the results of a detection.
		vis_util.visualize_boxes_and_labels_on_image_array(
			image_np,
			output_dict['detection_boxes'],
			output_dict['detection_classes'],
			output_dict['detection_scores'],
			category,
			use_normalized_coordinates=True,
			instance_masks=output_dict.get('detection_masks'),
			line_thickness=8)
		try:
			os.makedirs(output_directory)
		except:
			pass
		#IMAGE_SIZE =(12, 8)

		image_file_name = os.path.split(image_path)[1]
		image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
		save_path = os.path.join(output_directory, image_file_name)
		cv2.imwrite(save_path, image_np)

	def save_all_predictions(self, images, outputs, image_paths, output_directory):
		for i in range(len(images)):
			self.save_predictions_as_image(images[i], outputs[i], image_paths[i], output_directory)

	def get_1perclass(self, output_dict):
		output_dict_1perclass = {}
		for i in range(len(output_dict['detection_boxes'])):
			if output_dict['detection_scores'][i] >= 0.5:
				if output_dict['detection_classes'][i] not in output_dict_1perclass:
					for j in output_dict['detection_boxes']:
 						j[0], j[1], j[2], j[3] = j[1] * width, j[3] * width, j[0] * height, j[2] * height
					output_dict_1perclass.update({output_dict['detection_classes'][i]: output_dict['detection_boxes'][i]})
		return output_dict_1perclass



def main():
	predictor = TFOD_predict()
	predictor.preparation()
	input_directory = 'test_images'
	output_directory = 'test_results'
	image_paths = []

	for file in os.listdir(input_directory):
		image_paths.append(os.path.join(input_directory, file))

	print(image_paths)

	# Size, in inches, of the output images.
	# IMAGE_SIZE = (12, 8)

	all_images = predictor.all_images_to_np(image_paths)

	# t1 = time.time()
	# all_output_dicts = predictor.predict_multiple(all_images)
	# predictor.save_all_predictions(all_images, all_output_dicts, image_paths, output_directory)

	# t2 = time.time()
	# print(t2-t1)

	n_workers = 2
	SPEC = {'worker': ['127.0.0.1:12824', '127.0.0.1:12825']}

	all_output_dicts_temp = mp.Queue()

	t1 = time.time()
	job_list = []
	for i, image_np in enumerate(all_images):
		# Actual detection.
		image_path = image_paths[i]
		p = Process(target = predictor.predict_image , args = (image_np, all_output_dicts_temp, SPEC, i))
		p.start()
		job_list.append(p)
		#output_dict = predictor.predict_image(image_np)
		#all_output_dicts.append(output_dict)

	for job in job_list:
		job.join()

	all_output_dicts = [all_output_dicts_temp.get() for job in job_list]

	t2 = time.time()
	print(t2-t1)	
	
	predictor.save_all_predictions(all_images, all_output_dicts, image_paths, output_directory)
		

if __name__ == '__main__':
   main()
