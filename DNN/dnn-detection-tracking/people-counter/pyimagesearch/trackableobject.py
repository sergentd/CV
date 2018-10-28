class TrackableObject:
  def __init__(self, objectID, centroid):
    # Store the object ID, then init a list of centroids
    # using the current centroid
    self.objectID = objectID
    self.centroids = [centroid]
	
    # Init a boolean used to indicate if the object as already been counted
    self.counted  = False
