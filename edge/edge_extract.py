import cv2
import numpy as np
from datetime import  datetime
from matplotlib import pyplot
import os

empty_img = np.zeros([512, 512])

def return_value(event, x, y, flags, param):
    # inintial number
    value = 0
    ix = 0
    iy = 0
    if event == cv2.EVENT_LBUTTONDOWN:
        ix = x
        iy = y
        value = empty_img[y, x]
        print(value)

# this function to adjust the determine mechanism
def canny_determine_class(val, freq, instrument_first=False):
    idx = 0
    if len(freq) == 1:
        return idx
    if not instrument_first:
        val = val[-2:]
        if val[0] * val[1] == 0 and val[0] + val[1] == 1:
            idx = 1
        elif val[0] * val[1] == 0 and val[0] + val[1] == 2:
            idx = 2
        elif val[0] * val[1] == 0 and val[0] + val[1] == 3:
            idx = 3
        elif val[0] * val[1] == 2:
            idx = 4
        elif val[0] * val[1] == 3:
            idx = 5
        else:
            idx = 6
    else:
        # firstly we get rid of 0
        new_freq = np.delete(freq, 0)
        new_val = np.delete(val, np.where(val == 0))

        # then we find the valid parts id number
        nonzero_num = len(new_val) - (new_freq == 0).astype(np.int).sum()
        if nonzero_num == 1:
            idx = new_val[-1:]
        else:
            val = new_val[-2:]
            if val[0] * val[1] == 2:
                idx = 4
            elif val[0] * val[1] == 3:
                idx = 5
            else:
                idx = 6

    return idx

# point to point operation and need to be improved
def assign_image(src, edge, size=3, instrument_first=False):
    height, width = edge.shape
    src_pad = np.pad(src, ((size, size), (size, size)), 'constant')
    # print(np.max(src_pad))
    classified_image = np.zeros(edge.shape)
    for h in range(height):
        for w in range(width):
            if not edge[h, w] == 0:
                area = np.reshape(src_pad[h:h+2*size, w:w+2*size], ([1, 4*size*size])).astype(np.int)
                freq = np.bincount(area[0])
                val = np.argsort(freq)
                # print(freq)
                # print(val)
                classified_image[h, w] = canny_determine_class(val, freq, instrument_first)

    return classified_image


def canny_edge(image_name, instrument_first=False):
    image = cv2.imread(image_name)

    ## firstly we try canny edge detection
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_image = cv2.GaussianBlur(image, (3, 3), 0)
    edge = cv2.Canny(blur_image, 0, 255)
    # cv2.imshow('res1', blur_image)

    # kernel parameter
    kernel1 = np.ones([3, 3])
    kernel2 = np.ones([5, 5])

    # close to elimilate some isolated point
    closed_edge = cv2.morphologyEx(edge, op=cv2.MORPH_CLOSE, kernel=kernel2, iterations=1)

    # assign the classes by a window
    classified_closed_edge = assign_image(image / 85, closed_edge, size=5, instrument_first=instrument_first)
    # cv2.imshow('res2', closed_edge)
    dilated_edge = cv2.dilate(classified_closed_edge, kernel1, iterations=1)
    # cv2.imshow('res3', dilated_edge)

    return dilated_edge

def dilate_edge(image_name, kernel=np.ones([3, 3])):

    # then we make it by the classes
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    classified_image = image / 85
    step = 1/6
    contour = np.zeros(image.shape)
    for i in [0, 1, 2]:
        for j in range(i+1, 4):

            image_i = np.array(classified_image == i, dtype=np.uint8)
            image_j = np.array(classified_image == j, dtype=np.uint8)

            dilated_image_i = cv2.dilate(image_i, kernel)
            dilated_image_j = cv2.dilate(image_j, kernel)
            intersection = dilated_image_i * dilated_image_j
            # cv2.imshow('intersection', intersection*255)

            # To avoid duplicated add
            latest_contour = np.array(contour > 0, dtype=np.uint8)
            contour[np.where(intersection * latest_contour)] = 0
            contour += step * intersection

            # # the solution below would generate more edge relation with background
            # contour += step * (intersection - intersection * latest_contour)
            step += 1/6
            # print(np.max(contour))
    return contour
    # cv2.imshow('res', contour)
    # cv2.waitKey()


if __name__ == '__main__':

    dataset_path = '/opt/dataset/instrument_segmentation/endovis2017/data/cropped_train/'
    for i in range(1, 9):
        origin_folder = dataset_path + 'instrument_dataset_{}'.format(str(i)) + '/parts_masks/'
        dest_folder = dataset_path + 'instrument_dataset_{}'.format(str(i)) + '/parts_contours/'
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        images = os.listdir(origin_folder)
        for j, image_name in enumerate(images):
            print('current_image: {}/255'.format(j))
            image_path = origin_folder + image_name
            res = canny_edge(image_path, instrument_first=True)
            save_path = dest_folder + image_name
            cv2.imwrite(save_path, res)
        print("vedio_{} complete!".format(i))