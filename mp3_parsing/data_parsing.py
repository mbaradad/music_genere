import cPickle as pickle
import sys

import utils as utils
import numpy as np
import eyed3 as eyed3
import csv

NUMBER_OF_TAGS = 256

# reuses code from http://labrosa.ee.columbia.edu/millionsong/sites/default/files/tutorial1.py.txt

class DataParser():
  def __init__(self, baseLocation, outDir, stylesCsv):
    self.baseLocation = baseLocation
    self.outDir = outDir
    lines = np.genfromtxt(stylesCsv, delimiter=",", dtype=None)
    self.styles = dict()
    self.pitches_list = []

    self.timbres_list = []
    self.tags_list = []
    self.ids_list = []
    self.dft_list = []
    self.flushIndex = 0

  def process_info(self):
    songs = utils.apply_to_all_files(self.baseLocation, self.process_mp3_file, self.flushFunc, 100000)
    return 'number of song files:' + str(songs)

  def process_mp3_file(self, mp3file):
    """
    This function does 3 simple things:
    - open the song file
    - get info
    - close the file
    """

    audiofile = eyed3.load(mp3file)
    if not audiofile.tag or not audiofile.tag.genre:
      return 0
    genreName = audiofile.tag.genre.name
    if genreName in self.styles:
      self.styles[genreName]+=1
    else:
      self.styles[genreName]=1
    return 1

  def flushFunc(self):
    f = open(self.outDir + '/obj_' + "%02d" % (self.flushIndex,) + '.npz', 'wb')
    np.savez(f, x=self.timbres_list, y=self.tags_list)
    f.close()
    self.tags_list = []
    self.timbres_list = []
    self.flushIndex+=1


if __name__ == "__main__":
  kwargs = dict(x.split('=', 1) for x in sys.argv[1:])
  parser = DataParser(kwargs["--basedir"], kwargs["--outDir"], kwargs["--csvDir"])
  print "Files to read:" + parser.process_info();
  print parser.styles
  writer = csv.writer(open('out_styles.csv', 'wb'))
  for key, value in parser.styles.items():
    writer.writerow([key, value])
