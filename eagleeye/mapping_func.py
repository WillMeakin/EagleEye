#!/usr/bin/env python2
#
# Project Eagle Eye
# Group 3 - UniSA 2015
# Gwilyn Saunders & Kin Kuen Liu
# version 0.3.17

import cv2, xml.etree.ElementTree as ET, numpy as np
from theta_sides import Theta
from xml_trainer import Xmltrainer

magnitude = lambda x: np.sqrt(np.vdot(x, x))
unit = lambda x: x / magnitude(x)

class Mapper:
    def __init__(self, intrinsic, trainer, cfg, mode=Theta.NonDual):
        # variables
        self.rv = np.asarray([], dtype=np.float32)  # rotation
        self.tv = np.asarray([], dtype=np.float32)  # translation
        self.mode = mode
        
        # load some configs
        self.half_fov = np.radians(cfg.camera_fov) / 2
        pnp = cfg.pnp_flags
        
        # add SOLVEPNP_ prefix if not found
        if not pnp.startswith("SOLVEPNP_"):
            pnp = "SOLVEPNP_" + pnp
        
        # solvepnp flag
        if pnp in cv2.__dict__:
            self.pnp_flags = cv2.__dict__[pnp]
        else:
            print f, "is not found in cv2, reverting to Levenberg-Marquardt"
            self.pnp_flags = cv2.SOLVEPNP_ITERATIVE
        
        
        # open intrinsic, trainer files
        #self.cam, self.distort = self.parseCamIntr(intrinsic)
        self.cam = np.asarray([], dtype=np.float32) #cam matrix
        self.distort = np.asarray([], dtype=np.float32) #dist coeffs
        self.img_pts, self.obj_pts = self.parseTrainer(trainer)
        
        #print "Camera Matrix\n", self.cam
        
        print "\nside:", Theta.name(mode)
        print "img_pts {}".format(len(self.img_pts))
        print "obj_pts {}".format(len(self.obj_pts))
        
        #calculate pose
        self.cam, self.distort, self.rv, self.tv = self.calPose()
        self.R = cv2.Rodrigues(self.rv)[0]
    
    # opens/parses the Intrinsic calib xml file.
    def parseCamIntr(self, xmlpath):
        
        cm, dc = [], []
        cm_dict = {'fx': None, 'fy': None, 'cx': None, 'cy': None}
        dc_dict = {'k1': 0.0, 'k2': 0.0,
                   'k3': 0.0, 'k4': 0.0,
                   'k5': 0.0, 'k6': 0.0,
                   'p1': 0.0, 'p2': 0.0,
                   'c1': 0.0, 'c2': 0.0,
                   'c3': 0.0, 'c4': 0.0
                   }
        
        if xmlpath is None:
            raise IOError('Invalid file path to XML file.')
        
        tree = ET.parse(xmlpath)
        root = tree.getroot()
        
        if len(root) == 0:
            raise IOError('XML file is empty.')
        
        if root.tag != "StdIntrinsic" and self.mode == Theta.NonDual:
            raise IOError("Wrong input file, needs a StdIntrinsic xml file.")
        elif root.tag != "dual_intrinsic" and self.mode != Theta.NonDual:
            raise IOError("Wrong input file, needs a dual_intrinsic xml file.")
        
        if self.mode == Theta.Buttonside:
            root = root.find("Buttonside")
        elif self.mode == Theta.Backside:
            root = root.find("Backside")
        
        for elem in root:
            if elem.tag == 'CamMat':
                cm_dict.update(elem.attrib)
            if elem.tag =='DistCoe':
                dc_dict.update(elem.attrib)
            
        ### TODO: CHECK None Values !!!!
        
        if cm_dict['fx'] and cm_dict['fy'] and cm_dict['cx'] and cm_dict['cy'] is not None:
            # build a 3x3 camera matrix
            cm = np.matrix([[float(cm_dict['fx']), 0, float(cm_dict['cx'])],
                            [0, float(cm_dict['fy']), float(cm_dict['cy'])],
                            [0, 0, 1]
                            ])
            
        if cv2.__version__ >= '3.0.0':
            dc = np.asarray([float(dc_dict['k1']), float(dc_dict['k2']),
                                float(dc_dict['p1']), float(dc_dict['p2']),
                                float(dc_dict['k3']), float(dc_dict['k4']),
                                float(dc_dict['k5']), float(dc_dict['k6']),
                                float(dc_dict['c1']), float(dc_dict['c2']),
                                float(dc_dict['c3']), float(dc_dict['c4'])
                            ])
        else:
            dc = np.asarray([float(dc_dict['k1']), float(dc_dict['k2']),
                                float(dc_dict['p1']), float(dc_dict['p2']),
                                float(dc_dict['k3']), float(dc_dict['k4']),
                                float(dc_dict['k5']), float(dc_dict['k6'])
                            ])
        return cm, dc

    def parseTrainer(self, xmlpath):
        trainer = Xmltrainer(xmlpath, self.mode)
        self.num_training = trainer.total
        
        img_pts = np.asarray(trainer.img_pts(), np.float32)
        obj_pts = np.asarray(trainer.obj_pts(), np.float32)
        
        return img_pts, obj_pts
    
    def calPose(self):
        if len(self.img_pts) < 4 or len(self.obj_pts) < 4:
            raise Exception("Must have at least 4 training points.")
            
        if len(self.img_pts) != len(self.obj_pts):
            raise Exception("Training image points and object points must be equal in size. "
                            "image pts {}, obj pts {}".format(len(self.img_pts), len(self.obj_pts)))

        print "Calculating Camera Pose using the following flags:"

        # solvePnP flags
        # MUST see flag desc: http://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#solvepnp




        print 'obj_pts:\n', self.obj_pts
        print 'img_pts:\n', self.img_pts




        print 'running pnp...'
        w=960
        h=1080
        rms, cam, distort, rv, tv = cv2.calibrateCamera(self.obj_pts, self.img_pts, (w, h), None, None, None, None, cv2.CALIB_RATIONAL_MODEL)

        # levenberg-marquardt iterative method
        retval, cam, distort, rv, tv = cv2.calibrateCamera(
                            self.obj_pts, self.img_pts, 
                            self.cam, None,
                            None, None, self.pnp_flags)
        print 'done'


        print 'cam:\n', cam
        print 'distort:\n', distort
        print 'rv:\n', rv
        print 'tv:\n', tv





        #'''
        #NOT RUNNING
        #http://stackoverflow.com/questions/30271556/opencv-error-through-calibration-tutorial-solvepnpransac
        #iterations = 100
        #reproj_error = 20.0
        #min_inliers = max(4, int(len(self.obj_pts) * 0.9))
        #print self.cam
        #retval, rv, tv, inliers = cv2.solvePnPRansac(
                                      #self.obj_pts, self.img_pts, 
                                      #self.cam, self.distort)#, 
                                   ##None, None, False,
                                   ##iterations, reproj_error, min_inliers)
        ##'''
        
        #print rv, tv, inliers
        
        ## check, print, return
        if rv is None or rv is None or not retval:
            raise Exception("Error occurred when calculating rotation and translation vectors.")
        
        print 'Rotation Vector:\n', rv
        print 'Translation Vector:\n', tv
        
        return cam, distort, rv, tv
    
    def isVisible(self, pt):
        pt = np.array(pt).reshape(3, 1)
        
        RxPt = self.R.dot(pt)
        tv = np.array(self.tv).reshape(3,1)
        pt_cam = np.add(RxPt, tv).reshape(1,3)[0]
        
        # spherical model
        r = np.linalg.norm(pt_cam)
        theta = np.arctan2(pt_cam[1], pt_cam[0])
        phi = np.arccos(-pt_cam[2]/r)
        inv_phi = phi
        
        #print np.degrees(inv_phi), "<", np.degrees(self.half_fov)
        return inv_phi < self.half_fov
        
    
    def reprojpts(self, obj_pts):
        if len(obj_pts) == 0:
            raise Exception('No points to project.')
        
        proj_imgpts, jac = cv2.projectPoints(np.asarray([obj_pts], dtype=np.float32), self.rv, self.tv, self.cam, self.distort)
        proj_imgpts = proj_imgpts.reshape((len(proj_imgpts), -1))
        
        return proj_imgpts[0]

