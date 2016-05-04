import sys
import glob
import os
import utils.hdf5_getters as Getters
import utils.utils as Utils

#reuses code from http://labrosa.ee.columbia.edu/millionsong/sites/default/files/tutorial1.py.txt

class DataParser():
  def __init__(self, sqliteLocation, baseLocation, outDir):
    self.sqliteLocation = sqliteLocation
    self.baseLocation = baseLocation
    self.outDir = outDir

  def count_files(self):
   return 'number of song files:' + str(Utils.apply_to_all_files(self.baseLocation))


if __name__ == "__main__":
  kwargs = dict(x.split('=', 1) for x in sys.argv[1:])
  parser = DataParser(kwargs["--sqlite"], kwargs["--basedir"], kwargs["--outDir"])
  print "Files to read:" + parser.count_files();