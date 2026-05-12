
from __future__ import print_function
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import argparse
import os

'''
directory = r'E:/Lennon_Camera_Project/re/test_homography/re_0.788'
HSI_FILE_PATH = os.path.join(directory ,"HSI_2_Cube_Reconstruction_then_Stitched.png")

RGB_FILE_PATH = os.path.join(directory ,"RGB_2_screenshots_then_Stitched.png")

TEST_OBJ =    r'backup_images/IMG_2428.jpeg'
TEST_SCENE =  r'backup_images/IMG_2429.jpeg'
'''
directory = r'E:/Lennon_Camera_Project/re/test_homography/re_0.788/re_2_by_1'

HSI_FILE_PATH = os.path.join(directory, r'mosaic_fast_grayscale_suffix.png')
RGB_FILE_PATH  = os.path.join(directory, r'RGB_CAMERA_reconstructed.png')

assert os.path.exists(RGB_FILE_PATH)
#general info: https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html
def trial_homography_with_SIFT_algie():
    # code from #https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html info on

    MIN_MATCH_COUNT = 10

    img1 = cv.imread('box.png', cv.IMREAD_GRAYSCALE)          # queryImage
    img2 = cv.imread('box_in_scene.png', cv.IMREAD_GRAYSCALE) # trainImage

    
    # Initiate SIFT detector
    sift = cv.SIFT.create()

    # find the keypoints and descriptors with SIFT

    kp1, des1 = sift.detectAndCompute(image = img1,mask = None)
    kp2, des2 = sift.detectAndCompute(image = img2,mask = None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    return

def trial_homography_other():
    
     #https://pypi.org/project/opencv-contrib-python/ <-- pip install is not enough SURF algorithm patented
     #code modified from https://docs.opencv.org/4.x/d7/dff/tutorial_feature_homography.html
     
    parser = argparse.ArgumentParser(description='Code for Feature Matching with FLANN tutorial.')
    parser.add_argument('--input1', help='Path to input image 1.', default='backup_images/IMG_2428.jpeg')
    parser.add_argument('--input2', help='Path to input image 2.', default='backup_images/IMG_2429.jpeg')
    args = parser.parse_args()

    img_object = cv.imread(cv.samples.findFile(args.input1), cv.IMREAD_GRAYSCALE)
    img_scene = cv.imread(cv.samples.findFile(args.input2), cv.IMREAD_GRAYSCALE)
    if img_object is None or img_scene is None:
        print('Could not open or find the images!')
        exit(0)

    #-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    minHessian = 400
    detector = cv.xfeatures2d.SURF.create(hessianThreshold=minHessian)

    keypoints_obj, descriptors_obj = detector.detectAndCompute(img_object, None)
    keypoints_scene, descriptors_scene = detector.detectAndCompute(img_scene, None)

    #-- Step 2: Matching descriptor vectors with a FLANN based matcher
    # Since SURF is a floating-point descriptor NORM_L2 is used
    matcher = cv.DescriptorMatcher.create(cv.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors_obj, descriptors_scene, 2)

    #-- Filter matches using the Lowe's ratio test
    ratio_thresh = 0.75
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    #-- Draw matches
    img_matches = np.empty((max(img_object.shape[0], img_scene.shape[0]), img_object.shape[1]+img_scene.shape[1], 3), dtype=np.uint8)
    cv.drawMatches(img_object, keypoints_obj, img_scene, keypoints_scene, good_matches, img_matches, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    #-- Localize the object
    obj = np.empty((len(good_matches),2), dtype=np.float32)
    scene = np.empty((len(good_matches),2), dtype=np.float32)
    for i in range(len(good_matches)):
        #-- Get the keypoints from the good matches
        obj[i,0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
        obj[i,1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
        scene[i,0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
        scene[i,1] = keypoints_scene[good_matches[i].trainIdx].pt[1]

    H, _ =  cv.findHomography(obj, scene, cv.RANSAC)

    #-- Get the corners from the image_1 ( the object to be "detected" )
    obj_corners = np.empty((4,1,2), dtype=np.float32)
    obj_corners[0,0,0] = 0
    obj_corners[0,0,1] = 0
    obj_corners[1,0,0] = img_object.shape[1]
    obj_corners[1,0,1] = 0
    obj_corners[2,0,0] = img_object.shape[1]
    obj_corners[2,0,1] = img_object.shape[0]
    obj_corners[3,0,0] = 0
    obj_corners[3,0,1] = img_object.shape[0]

    scene_corners = cv.perspectiveTransform(obj_corners, H)

    #-- Draw lines between the corners (the mapped object in the scene - image_2 )
    cv.line(img_matches, (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])),\
        (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])), (0,255,0), 4)
    cv.line(img_matches, (int(scene_corners[1,0,0] + img_object.shape[1]), int(scene_corners[1,0,1])),\
        (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])), (0,255,0), 4)
    cv.line(img_matches, (int(scene_corners[2,0,0] + img_object.shape[1]), int(scene_corners[2,0,1])),\
        (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])), (0,255,0), 4)
    cv.line(img_matches, (int(scene_corners[3,0,0] + img_object.shape[1]), int(scene_corners[3,0,1])),\
        (int(scene_corners[0,0,0] + img_object.shape[1]), int(scene_corners[0,0,1])), (0,255,0), 4)

    #-- Show detected matches
    cv.imshow('Good Matches & Object detection', img_matches)

    cv.waitKey()


def trial_chatgpt_basic_homography_code():

    # --- Load images ---
    #img1 = cv.imread(TEST_OBJ, cv.IMREAD_COLOR)      # object image
    #img2 = cv.imread(TEST_SCENE, cv.IMREAD_COLOR)    # scene image
    img2 = cv.imread(HSI_FILE_PATH, cv.IMREAD_GRAYSCALE)      # object image
    img1 = cv.imread(RGB_FILE_PATH, cv.IMREAD_COLOR)    # scene image
    
    scale = 1
    #assert img1 and img2 is not None
    print(f' image 1 shape: {img1.shape} ||image 2 shape: {img2.shape}')
    if img1 is None or img2 is None:
        raise ValueError("Could not load images. Check file paths.")

    # --- Step 1: ORB detector ---
    orb = cv.ORB.create(nfeatures=2000)

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        print("No descriptors found.")
        return

    # --- Step 2: Matching ---
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # --- Step 3: Lowe's ratio test ---
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    print(f"Good matches: {len(good)}")

    # --- Step 4: Homography ---
    if len(good) > 10:

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        print("Homography matrix:\n", H)

        # --- Step 5: Project corners of object image ---
        h, w = img1.shape[:2]

        corners = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ]).reshape(-1, 1, 2)

        projected = cv.perspectiveTransform(corners, H)

        # --- Step 6: Draw result on scene ---
        img2_draw = img2.copy()
        cv.polylines(img2_draw, [np.int32(projected)], True, (0, 255, 0), 3)

        # --- Step 7: SHRINK BY 2× (safe scaling) ---

        #img2_color = cv.resize(img2_color, dsize= [int(coordinate/3) for coordinate in img2_color.shape[0:2]]) 
            # ^ doesnt work because of it just clips/cuts doesnt interpolate down
        img2_small = cv.resize(
            img2_draw,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv.INTER_AREA
        )

        projected_small = projected * scale

        # redraw outline (keeps it perfectly aligned)
        cv.polylines(
            img2_small,
            [np.int32(projected_small)],
            True,
            (0, 255, 0),
            3
        )

        # --- Show ---
        cv.imshow("Detected Object (Scaled)", img2_small)
        cv.waitKey(0)
        cv.destroyAllWindows()

    else:
        print("Not enough matches found.")
        
if __name__ == "__main__":
    
    trial_chatgpt_basic_homography_code()
    
    #trial_homography_other()
    