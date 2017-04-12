#
# Project Eagle Eye
# Group 15 - UniSA 2015
# Gwilyn Saunders
# version 0.2.7
# 
# Loads a whole CSV dataset into memory
# Provides flash detection and set restriction
# Include synchronisation techniques
#
import os, csv, re

class Memset:
    def __init__(self, path, delimiter=","):
        self.data = []
        self.total = 0
        self.start_set = 0
        self.end_set = 0
        self._at = 0.0
        
        m = re.match(".*\d_(.*)\.csv", os.path.basename(path))
        if m: self._name = m.group(1)
        #else: self._name = "NameError" #TODO: put this back in
        else: self._name = "EEWand"     #TODO: debugging, remove

        self._delim = delimiter
        self._path = path
        self._ratio = 1.0
        
        # determine set parameters
        flash = 0
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter)
            
            while True:
                try:
                    row = reader.next()
                except StopIteration:
                    break
                
                if row[1] == 'F' and flash == 0:
                    flash = 1
                    self.start_set = self.total + 1
                if row[1] != 'F' and flash == 1:
                    flash = 2
                if row[1] == 'F' and flash == 2:
                    flash = 3
                    self.end_set = self.total - 1
                    
                self.data.append(row)
                self.total += 1
        
    def restrict(self):
        # +1 because the set marks are indexes, not sizes
        self.data = self.data[self.start_set:self.end_set+1]
        self.total = self.end_set+1 - self.start_set
    
    # Gets current row, or a specific row if 'at' option is > 0
    def row(self, at=-1):
        if at == -1:
            return self.data[self.at()]
        else:
            return self.data[at]
        
    def next(self):
        i = self._at + self._ratio
        if i < self.total:
            self._at = i
            return True
        return False
    
    def back(self):
        i = self._at - self._ratio
        if i >= 0:
            self._at = i
            return True
        return False
    
    # calculate appropriate ratio from the matching video frames
    def setRatio(self, video_frames):
        self._ratio = self.total / float(video_frames)
        print "Ratio: ", self._ratio
    
    def resetRatio(self):
        self._ratio = 1.0
    
    def at(self):
        return int(round(self._at, 0))
    
    def eof(self):
        return self._at >= (self.total - self._ratio)
    
    def status(self):
        return "row: {:0>2d}/{:d}"\
            .format(self.at(), self.total)
    
    def name(self):
        return self._name


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print "please specify a csv file"
        exit(0)
    
    data = Memset(sys.argv[1])
    
    print "total", data.total, data.row()[1]
    
    print "start", data.start_set, data.row()[1]
    print "end", data.end_set, data.row()[1]
    
    print "restrict"
    data.restrict()
    
    print "start", data.at(), data.row()[1]
    print "total", data.total, data.row()[1]
    
    exit(0)
    
