import cPickle as pickle

class DataReader():
  def __init__(self, pickleLocation):
    self.pickleLocation = pickleLocation
    with open(pickleLocation, 'rb') as f:
        obj = pickle.load(f)
    self.x_train = obj[2]
    self.train_unprocessed = obj[1]
    self.processTrain()
  def getTrainX(self):
    return self.x_train

  def processTrain(self):
    #for i in 1:1000:
  #  self.y_train[i] = 1

  def getTrainY(self):
    return self.y_train

