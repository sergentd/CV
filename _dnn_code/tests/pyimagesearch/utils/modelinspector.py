# a class which list the layers and indexes of a model
class ModelInspector:
  def __init__(self, model):
    self.model = model
    
  def inspect_layers(self):
    # loop over the layers in the network and display them to the console
    print("[INFO] showing layers")
    for (i, layer) in enumerate(self.model.layers):
      print("[INFO] {}\t{}".format(i, layer.__class__.__name__))