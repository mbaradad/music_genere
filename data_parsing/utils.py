import os
import glob
import hdf5_getters as Getters
import urllib2
from xml.dom import minidom

def apply_to_all_files(baseLocation, func=lambda x: x, flushFunc=lambda x: x, flushPeriodicity=1000, ext='.h5'):
  """
  From a base directory, go through all subdirectories,
  find all files with the given extension, apply the
  given function 'func' to all of them.
  If no 'func' is passed, we do nothing except counting.
  INPUT
     basedir  - base directory of the dataset
     func     - function to apply to all filenames
     ext      - extension, .h5 by default
  RETURN
     number of files
  """
  cnt = 0
  cnt_no_tags = 0
  # iterate over all files in all subdirectories
  for root, dirs, files in os.walk(baseLocation):
    files = glob.glob(os.path.join(root, '*' + ext))
    # count files

    # apply function to all files
    for f in files:
      hasRead = func(f)
      cnt+=hasRead
      cnt_no_tags+=1
      if hasRead == 1 and cnt%flushPeriodicity == 0:
        print "Processed" + str(cnt)
        print "Processed with tags " + str(cnt_no_tags)
        flushFunc()
  flushFunc()

  return cnt




def url_call(url):
    """
    Do a simple request to the 7digital API
    We assume we don't do intense querying, this function is not
    robust
    Return the answer as na xml document
    """
    stream = urllib2.urlopen(url)
    xmldoc = minidom.parse(stream).documentElement
    stream.close()
    return xmldoc


def get_preview(h5):
  """
    Ask for the preview to a particular track, get the XML answer
    After calling the API with a given track id,
    we get an XML response that looks like:

    <response status="ok" version="1.2" xsi:noNamespaceSchemaLocation="http://api.7digital.com/1.2/static/7digitalAPI.xsd">
      <url>
        http://previews.7digital.com/clips/34/6804688.clip.mp3
      </url>
    </response>
    We parse it for the URL that we return, or '' if a problem
    """
  trackid = Getters.get_track_7digitalid(h5)
  previewUrl =py7D.preview_url(trackid)

  return previewUrl