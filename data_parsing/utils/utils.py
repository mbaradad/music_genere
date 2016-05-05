import os
import glob

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
  return cnt
