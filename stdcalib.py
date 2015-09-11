# Project Eagle Eye
# Group 15 - UniSA 2015
# Kin Kuen, Liu
ver = '1.3.23'
# Last Updated: 2015-09-11
# 
# Camera Calibration and Image Undistortion using OpenCV standard pinhole camera functions.
# Rational model flag is enabled to compensate radial and tangential effect present in fish-eye lens
# using 8 distortion coefficients (k4, k5, k6).
# 
# This script is in place until further investigation of OpenCV fisheye function is done to calibrate RICOH THETA M15 more accurately.
# See Usage() below for usage
#
# Please read README for more information.
#
# Tested on Python 2.7.8, OpenCV 2.4.11
# API Reference: http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
# 

import sys, glob, time, os, cv2, numpy as np, ast
from datetime import datetime
from elementtree.SimpleXMLWriter import XMLWriter
from eagleeye import EasyArgs, EasyConfig

'''
Camera Calibration (Standard Pinhole Camera)

Reads and loads images using OpenCV image processing function
converts each image to grey scale and detects chessboard pattern

If pattern is found, position of each corner in image plane will be added to the image points array,
along with object points which indicate the individual position of chessboard corners in real world.
Each detection is rendered and output to the original image's location with a postfix of '_pattern'.

Calibration occurs once detected corners are added to object points and image poitns arrays.
k4, k5, k6 coefficients are enabled to decrease the error induced by fish-eye lens,
while OpenCV fisheye functions are being investigated to replace this standard function.

Intrinsic parameters: Camera matrix, distortion coefficents.
Extrinsic parameters: Rotation & Translation vectors.
and root mean square (RMS), which indicates,on average of all images with pattern found,
the difference between position of each reprojected point and their actual point pre-undistortion in pixels.
Furthermore, the sum of error and arithmetical mean are also calculated to measure error in individual picture.

Returns:
rms - Root Mean Square in pixels
camera_matrix - Camera Matrix
dist_coefs - 8 parameters Distortion Coefficients (k4, k5, k6 enabled)
extr - Each image's extrinsic value, e.g. [image path+name, rvecs, tvecs]
intr_error - a dictionary containing rms, total error and arithmetical mean in pixels

API Reference:
http://docs.opencv.org/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
http://docs.opencv.org/modules/highgui/doc/reading_and_writing_images_and_video.html

Coding Reference:
http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration
'''

def stdCalib(imagesArr, patternSize=(9,6), squareSize=1.0, preview_path=False):
    print '\n------- Camera Calibration (Standard - Pinhole Camera) --------'
    
    # termination criteria  used in detecting pattern
    t_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare corner object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    p_size = patternSize
    p_pts = np.zeros((np.prod(p_size), 3), np.float32)
    p_pts[:, :2] = np.indices(p_size).T.reshape(-1, 2)
    p_pts *= squareSize

    # Arrays to store object points and image points of all images
    objpts = [] # 3d points of chessboard corners in object plane
    imgpts = [] # 2d points of chessboard corners in image plane

    img_found = [] # array of images with pattern found
    num_found = 0 # number of pattern found
    total_time = 0 # Total time to process images
    
    for fname in imagesArr:
        print 'Processing', os.path.basename(fname), 
        start = time.clock() # Measure time needed to process an image

        # Load image and check if it exists
        img = cv2.imread(fname)
        if img is None:
          print ' -  Failed to load.'
          continue

        # Convert original image to grey scale
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect chessboard pattern
        found, corners = cv2.findChessboardCorners(grey, p_size, cv2.CALIB_CB_ADAPTIVE_THRESH)

        if found == True:
            print ' *** Pattern found *** ',
            # Get image dimension -> height and width
            h, w = img.shape[:2]

            # Optimise and refine corner finding performance
            cv2.cornerSubPix(grey, corners, (11,11), (-1,-1), t_criteria)

            imgpts.append(corners.reshape(-1, 2))
            objpts.append(p_pts)

            img_found.append(fname)
            num_found += 1
            
            # Draw and display corners detected
            # Then render the pattern on original image with a postfix of '_pattern'
            if preview_path:
                cv2.drawChessboardCorners(img, p_size, corners, found)
                f = os.path.basename(fname).split('.')
                pattern_path = os.path.join(preview_path, ".".join(f[:-1]) + '_pattern' + '.' + f[-1])
                cv2.imwrite(pattern_path, img)
            
        else:
            print ' - No Pattern found. - ', 
            
        t = time.clock() - start # Seconds needed to process an image
        print 'took', "{:.2f}".format(t), 'seconds'
        total_time += t
        # end pattern detection
    # end of chessboard images

    print '\n--------------------------------------------\n'

    print 'Time taken to detect pattern from', len(imagesArr), 'images:', "{:.2f}".format(total_time), 'seconds.'
    
    if num_found == 0:
        print 'Camera Calibration failed : No chessboard pattern size of', p_size, 'has been found.'
        print 'Please check if size of chessboard pattern is correct.'
        raise Exception("No chessboards found")
    elif num_found < 10:
        print 'OpenCV requires at least 10 patterns found for an accurate calibration, please consider taking more images.'
        print 'Chessboard pattern found in %d out of %d images.' % (num_found, len(imagesArr)), ' ({:.2%})'.format(num_found / float(len(imagesArr)))
    else: # >= 10
        print 'Chessboard pattern found in %d out of %d images.' % (num_found, len(imagesArr)), ' ({:.2%})'.format(num_found / float(len(imagesArr)))
        

    print'\n---------------------------------------------\n'

    print 'Calculating intrinsic values ...........'

    # Camera Calibration flags
    cali_flags = cv2.CALIB_RATIONAL_MODEL # enables k4, k5, k6 to better suit lens with large distortion
    #| cv2.cv.CV_CALIB_FIX_K1 | cv2.cv.CV_CALIB_FIX_K2

    # Perform camera calibration (standard function for pinhole camera)
    rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, (w, h), None, None, None, None, cali_flags)

    # Calculate sum of error of all images specified
    imgs_error = [] # Array of images with error assigned
    total_error = 0 # Sum of error in pixels
    for i in xrange(len(objpts)):
        imgpts2, _ = cv2.projectPoints(objpts[i], rvecs[i], tvecs[i], camera_matrix, dist_coefs)
        i_error = cv2.norm(imgpts[i], imgpts2.reshape(-1,2)) / len(imgpts2)

        imgs_error.append([img_found[i], i_error]) # Error per Image
        total_error += i_error

    arth_error = total_error/len(objpts) # Arithmetical mean error
    intr_error = {'rms': rms, 'tot_err': total_error, 'arth_err': arth_error}

    print 'Camera Matrix:\n', camera_matrix, '\n'
    print 'Distortion Coefficients:\n', dist_coefs.ravel(), '\n'
        
    print 'Root Mean Square:', rms, 'pixels'
    print 'Total error:', total_error, 'pixels'
    print 'Arithmetical mean:', arth_error, 'pixels'

    # Format Extrinsics
    #extr = extFormatter(img_found, rvecs, tvecs)
    
    print '------- End of Camera Calibration (Standard - Pinhole Camera) ---------\n'
    return camera_matrix, dist_coefs, intr_error

# .. end of Standard Camera Calibration

'''
Performs undistortion to correct warping effect of each image
based on camera matrix and distortion coefficients calculated and passed in.

The camera matrix is recaculated to avoid losing pixels in the edges,
which happens in default cv2.undistort function.
However, shape of the undistorted image will be changed and 'squished',
additional black pixels will be present surrounding outer edges.

Undistorted image is output to the same location as original image is loaded from
with a postfix of '_undistorted' at the end.

API Reference: http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html
'''
def undistort(path, imagesArr, K, d):

    print '\n-------- Undistort Images ---------'

    for fname in imagesArr:
        print 'Undistorting', os.path.basename(fname),
        
        img = cv2.imread(fname)
        if img is None:
          print ' -  Failed to load.'
          continue
        
        h, w = img.shape[:2]
        
        # Calculate new optimal matrix to avoid losing pixels in the edges after undistortion
        new_matrix, roi = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 1)

        # Generate undistorted image
        #newimg = cv2.undistort(img, K, d, None, new_matrix)
        
        
        # Alternative undistortion via remapping
        mapx, mapy = cv2.initUndistortRectifyMap(K, d, None, new_matrix, (w, h), cv2.CV_32FC1)
        newimg = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        # Output undistorted image to the same location with postfix '_undistorted'
        f = os.path.basename(fname).split('.')
        newimg_path = os.path.join(path, ".".join(f[:-1]) + '_undistorted.' + f[-1])
        cv2.imwrite(newimg_path, newimg)
        print '----->', newimg_path

    print 'Undistorted', len(imagesArr), 'images.'
    print '-------- End of Undistortion ---------\n'

# .. end of Undistortion   


'''
Outputs Intrinsic parameters to xmldata folder as a XML file
'''

def intWriter(path, camMat, distCoe, calibError={}):
    try:
        print 'Generating Intrinsic Parameters to:', path, '...'
        with open(path, 'w') as int_xml:
            w = XMLWriter(int_xml)
            w.declaration()
            
            # Camera Intrinsic (Root)
            root = w.start('StdIntrinsic')
            
            # Camera Matrix
            w.element('CamMat',
                     fx=str(camMat[0][0]), fy=str(camMat[1][1]),
                     cx=str(camMat[0][2]), cy=str(camMat[1][2]))
            
            # Distortion Coefficients
            if (len(distCoe[0]) == 8): # 8 coefficients Rational Model, k4 k5 k6 enabled
                w.element('DistCoe', k1=str(distCoe[0][0]), k2=str(distCoe[0][1]),
                          p1=str(distCoe[0][2]), p2=str(distCoe[0][3]),
                          k3=str(distCoe[0][4]), k4=str(distCoe[0][5]),
                          k5=str(distCoe[0][6]), k6=str(distCoe[0][7]))
            elif (len(distCoe[0]) == 12): # 12 coefficients Prism Model, c1, c2, c3, c4 enabled, new in OpenCV 3.0.0
                w.element('DistCoe', k1=str(distCoe[0][0]), k2=str(distCoe[0][1]),
                               p1=str(distCoe[0][2]), p2=str(distCoe[0][3]),
                               k3=str(distCoe[0][4]), k4=str(distCoe[0][5]),
                               k5=str(distCoe[0][6]), k6=str(distCoe[0][7]),
                               c1=str(distCoe[0][8]), c2=str(distCoe[0][9]),
                                c3=str(distCoe[0][10]),c4=str(distCoe[0][11]))
            else:
                w.element('DistCoe', k1=str(distCoe[0][0]), k2=str(distCoe[0][1]),
                          p1=str(distCoe[0][2]), p2=str(distCoe[0][3]),
                          k3=str(distCoe[0][4]))
            
            # Error values
            if len(calibError) > 0:
                w.element('Error', rms=str(calibError['rms']), total=str(calibError['tot_err']), arth=str(calibError['arth_err']))
            
            w.close(root)
        print 'Intrinsic calibration has been generated successfully.'
        
    except Exception as e:
        print 'ERROR: occurred in writing intrinsic XML file.'
        raise e # keep it bubbling up, to catch in main()

'''
Prints versions of Python and OpenCV.
'''
def version():
    print 'Camera Calibration Script Version: ', ver
    print 'Script is tested using Python 2.7.8 & OpenCV 3.0.0 & NumPy 1.9.0 \n'
    
    print 'Running versions:'
    print 'Python:', sys.version
    print 'OpenCV Version:', cv2.__version__
    print 'NumPy Version:', np.__version__

'''
Prints usage of stdcalib.py
'''
def usage():
    print 'usage: stdcalib.py -output <file path> <multiple jpg files> {-chess_size <pattern: def. 9,6> | -square_size <in mm: def. 1.0> | -preview <preview file folder> | -config <file>}'

def main(sysargs):
    args = EasyArgs(sysargs)
    cfg = EasyConfig(args.config, group="calib")
    
    # argument sanity checks
    if args.usage:
        usage()
        return 0
    elif args.version:
        version()
        return 0
    elif len(args) < 10:
        print "Requires a minimum 10 chessboard images."
        usage()
        return 0
    elif 'output' not in args:
        print "Requires an output path."
        usage()
        return 0
    
    # default args
    s_size = args.square_size or cfg.default_squares
    if args.chess_size:
        p_size = ast.literal_eval("({})".format(args.chess_size))
    else:
        p_size = cfg.default_chess
    
    # create preview folder if specified
    if args.preview and not os.path.exists(args.preview):
        os.makedirs(args.preview)
    
    try: # calibration
        cam_mat, dist, err = stdCalib(args[1:], p_size, s_size, args.preview)
        # XML Output
        intWriter(args.output, cam_mat, dist, err)
    except:
        print "Aborting."
        return 1
    
    # Rectify
    if args.preview:
        undistort(args.preview, args[1:], cam_mat, dist)
    
    print '--------------------------------------'
    print 'Intrinsic Calibration has completed successfully!'
    return 0

if __name__ == '__main__':
    exit(main(sys.argv))

