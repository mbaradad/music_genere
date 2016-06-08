import cPickle as pickle
import sys

import utils as utils
import numpy as np
import eyed3 as eyed3
import csv
import mp3_utilities as mp3

NUMBER_OF_TAGS = 100

# reuses code from http://labrosa.ee.columbia.edu/millionsong/sites/default/files/tutorial1.py.txt

class DataParser():
  def __init__(self, baseLocation, outDir, stylesCsv):
    self.baseLocation = baseLocation
    self.outDir = outDir
    lines = np.genfromtxt(stylesCsv, delimiter=";", dtype=None)
    self.targetStyles = dict()
    for i in range(NUMBER_OF_TAGS):
      self.targetStyles[lines[i][0]] = i

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
    if genreName not in self.targetStyles.keys():
      return 0
    tag_list = np.zeros(NUMBER_OF_TAGS)
    tag_list[self.targetStyles[genreName]] = 1
    try:
      timbres_list = np.transpose(mp3.mp3ToDFT(mp3file))
    except:
      print "error producing in mp3"
      return 0
    if len(timbres_list) < 300:
      return 0
    created = 0
    # only take 5 at most
    for i in range(0, min((len(timbres_list) / 400), 10) * 400, 400):
      timbres_list_segment = timbres_list[i:(i + 300), ]
      self.ids_list.append(mp3file)
      self.tags_list.append(tag_list)
      self.timbres_list.append(timbres_list_segment)
      created += 1
    return created

    return 1

  def flushFunc(self):
    f = open(self.outDir + '/obj_' + "%02d" % (self.flushIndex,) + '.npz', 'wb')
    np.savez(f, x=self.timbres_list, y=self.tags_list, filenames=self.ids_list)
    f.close()
    self.tags_list = []
    self.timbres_list = []
    self.ids_list = []
    self.flushIndex+=1


if __name__ == "__main__":
  kwargs = dict(x.split('=', 1) for x in sys.argv[1:])
  parser = DataParser(kwargs["--basedir"], kwargs["--outDir"], kwargs["--csvDir"])
  print "Files to read:" + parser.process_info();
