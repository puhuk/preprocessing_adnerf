import os
import cv2
from pathlib import Path
from mtcnn.mtcnn import MTCNN
import numpy as np
from math import log10, sqrt

# draw an image with detected objects
def draw_image_with_boxes(img, result_list):
	# load the image

	# plot each box
	for result in result_list:
		# get coordinates
		x, y, width, height = result['box']
		# create the shape
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		# draw the box
		ax.add_patch(rect)
		# draw the dots
		for key, value in result['keypoints'].items():
			# create and draw dot
			dot = Circle(value, radius=2, color='red')
			ax.add_patch(dot)
	# show the plot
	pyplot.show()

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

movie_list = os.listdir('movies')
print(movie_list)

detector = MTCNN()
for movie in movie_list:

    # extract_wav_cmd = 'ffmpeg -i ' + movie + ' -f wav -ar 16000 ' + wav_file
    # os.system(extract_wav_cmd)

    print(movie)
    name = movie.split('.')[0]

    Path(os.path.join('parsed', name)).mkdir(parents=True, exist_ok=True)
    Path(os.path.join('dataset', name, 'parsing')).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(os.path.join('movies', movie))
    frame_num = 0
    while(True):
    # while(frame_num<20):
        _, frame = cap.read()
        if frame is None:
            break
        cv2.imwrite(os.path.join('parsed', name, str(frame_num) + '.jpg'), frame)
        frame_num = frame_num + 1
    cap.release()

    print("frame num :", frame_num)
    
    psnr_list = []
    mse_list = []
    print("Name", name)
    for i in range(frame_num):
        psnr = 0
        
        orig_img = cv2.imread(os.path.join('../adnerf_new/adnerf/data', 'Obama', 'ori_imgs', str(i) + '.jpg'))
        img = cv2.imread(os.path.join('dataset', name, 'parsing', str(i)+'.jpg'))
        head_img = cv2.imread(os.path.join('../adnerf_new/adnerf/data', 'Obama', 'head_imgs', str(i) + '.jpg'))
        faces = detector.detect_faces(img)
        gen_head = cv2.imread(os.path.join('parsed', name, str(i) + '.jpg'))
        # print(gen_head.shape)

        psnr_list.append(PSNR(gen_head, head_img))


        rect1 = img[faces[0]['keypoints']['left_eye'][1]-5:faces[0]['keypoints']['left_eye'][1]+5, faces[0]['keypoints']['left_eye'][0]-5:faces[0]['keypoints']['left_eye'][0]+5]
        rect2 = img[faces[0]['keypoints']['right_eye'][1]-5:faces[0]['keypoints']['right_eye'][1]+5, faces[0]['keypoints']['right_eye'][0]-5:faces[0]['keypoints']['right_eye'][0]+5]
        rect3 = img[min(faces[0]['keypoints']['mouth_left'][1], faces[0]['keypoints']['mouth_right'][1])-5:min(faces[0]['keypoints']['mouth_left'][1], faces[0]['keypoints']['mouth_right'][1])+5, faces[0]['keypoints']['mouth_left'][0]:faces[0]['keypoints']['mouth_right'][0]]
        
        orig_rect1 = orig_img[faces[0]['keypoints']['left_eye'][1]-5:faces[0]['keypoints']['left_eye'][1]+5, faces[0]['keypoints']['left_eye'][0]-5:faces[0]['keypoints']['left_eye'][0]+5]
        orig_rect2 = orig_img[faces[0]['keypoints']['right_eye'][1]-5:faces[0]['keypoints']['right_eye'][1]+5, faces[0]['keypoints']['right_eye'][0]-5:faces[0]['keypoints']['right_eye'][0]+5]
        orig_rect3 = orig_img[min(faces[0]['keypoints']['mouth_left'][1], faces[0]['keypoints']['mouth_right'][1])-5:min(faces[0]['keypoints']['mouth_left'][1], faces[0]['keypoints']['mouth_right'][1])+5, faces[0]['keypoints']['mouth_left'][0]:faces[0]['keypoints']['mouth_right'][0]]
        
        # psnr += PSNR(orig_rect1, rect1)
        # psnr += PSNR(orig_rect2, rect2)
        # psnr += PSNR(orig_rect3, rect3)
        # psnr = psnr/3

        # psnr_list.append(psnr)

        # mse = np.mean((orig_img - img) ** 2)
        # mse_list.append(mse)

    print(name, np.average(psnr_list))
