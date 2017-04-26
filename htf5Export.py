from eagleeye import Xmltrainer
from eagleeye import EasyArgs
from eagleeye import Theta
import numpy as np
import sys
import h5py

#Usage e.g.: python htf5Export.py -pointsXML D:\ACL\Session1\pnpWorkingFiles\trainPointsScattered.xml -outDir C:\Users\wjmea\Desktop\

args = EasyArgs(sys.argv)
inPath = args.pointsXML
outPath = args.outDir

def writePoints(side, strSide):

	#Gets lens points
	xml = Xmltrainer(inPath, side=side)
	objPts = np.asarray(xml.obj_pts())
	imgPts = np.asarray(xml.img_pts())

	print 'correspondences: '
	print objPts
	print imgPts

	h5f = h5py.File(outPath + strSide + '.h5', 'w')
	h5f.create_dataset('obj', data=objPts)
	h5f.create_dataset('img', data=imgPts)
	h5f.close()

	print 'READING IN (verify)'

	h5f = h5py.File(outPath + strSide + '.h5', 'r')
	readInObj = h5f['obj'][:]
	readInImg = h5f['img'][:]
	h5f.close()

	print readInObj
	print readInImg

print 'BUTTONSIDE'
writePoints(Theta.Buttonside, 'buttonside')
print '\nBACKSIDE'
writePoints(Theta.Backside, 'backside')