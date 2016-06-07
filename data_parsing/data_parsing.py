import cPickle as pickle
import sys

import utils.utils as utils
import utils.hdf5_getters as Getters
import numpy as np

NUMBER_OF_TAGS = 256

# reuses code from http://labrosa.ee.columbia.edu/millionsong/sites/default/files/tutorial1.py.txt

class DataParser():
  def __init__(self, sqliteLocation, baseLocation, outDir, stylesCsv):
    self.sqliteLocation = sqliteLocation
    self.baseLocation = baseLocation
    self.outDir = outDir
    lines = np.genfromtxt(stylesCsv, delimiter=",", dtype=None)
    self.styles = dict()
    for i in range(NUMBER_OF_TAGS):
      self.styles[lines[i][0]] = i
    self.pitches_list = []

    self.timbres_list = []
    self.tags_list = []
    self.ids_list = []
    self.dft_list = []
    self.flushIndex = 0

  def process_info(self):
    songs = utils.apply_to_all_files(self.baseLocation, self.process_h5_file_info, self.flushFunc, 100000)
    return 'number of song files:' + str(songs)

  def process_h5_file_info(self, h5):
    """
    This function does 3 simple things:
    - open the song file
    - get info
    - close the file
    """
    try:
      trackId = Getters.get_track_id(h5)
      tags = Getters.get_artist_mbtags(h5);
      timbres_list = Getters.get_segments_timbre(h5)
    except:
      return 0
    if len(tags) == 0:
      return 0

    tag_list = np.zeros(NUMBER_OF_TAGS)
    someSeen = False
    for tag in tags:
      if tag in self.styles.keys():
        tag_list[self.styles[tag]] = 1
        someSeen = True
    if not someSeen:
      return 0
    if len(timbres_list) < 300:
      return 0
    created = 0
    #only take 5 at most
    for i in range(0,min((len(timbres_list)/400),5)*400, 400):
      timbres_list_segment = timbres_list[i:(i + 300), ]
      self.ids_list.append(trackId)
      self.tags_list.append(tag_list)
      self.timbres_list.append(timbres_list_segment)
      print(Getters.get_artist_name() + ": " + Getters.get_title(h5))
      created+=1
    return created

  def flushFunc(self):
    f = open(self.outDir + '/obj_' + "%02d" % (self.flushIndex,) + '.npz', 'wb')
    np.savez(f, x=self.timbres_list, y=self.tags_list)
    f.close()
    self.tags_list = []
    self.timbres_list = []
    self.flushIndex+=1


if __name__ == "__main__":
  kwargs = dict(x.split('=', 1) for x in sys.argv[1:])
  parser = DataParser(kwargs["--sqlite"], kwargs["--basedir"], kwargs["--outDir"], kwargs["--csvDir"])
  print "Files to read:" + parser.process_info();
  print parser.styles
