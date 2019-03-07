# import the necessary packages
import face_recognition
import pickle
import cv2

class FaceEncoder:
    def __init__(self, encodings=[], names=[], method="hog"):
        # initialize the encoder
        self.encodings = encodings
        self.names = names
        self.method = method
        self.rboxes = []
        self.rnames = []

    def load_encodings(self, file):
        # load the encodings stored on disk
        data = pickle.loads(open(file, "rb").read())
        self.encodings = data["encodings"]
        self.names = data["names"]
        self.method = data["method"]

    def locate(self, image):
        # return the faces locations in the image
        return face_recognition.face_locations(image, model=self.method)

    def quantify(self, image, boxes):
        # return the descriptor for face encodings
        return face_recognition.face_encodings(image, boxes)

    def compare(self, encoding):
        # return the matches between known encodings and the encoding
        return face_recognition.compare_faces(self.encodings, encoding)

    def describe(self, image):
        # convert the image to RGB color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # locate the faces in the image
        boxes = self.locate(image)

        # initialize the list of encodings for the image
        encodings = []

        # quantify the faces
        if len(boxes) > 0:
            encodings = self.quantify(image, boxes)

        # return the quantified vectors for faces
        return encodings

    def store(self, image, name=""):
        # describe the image by encoding the faces in it
        encodings = self.describe(image)

        # add the quantified face to the list
        # all faces are considered same person
        for e in encodings:
            self.encodings.append(e)
            self.names.append(name)

    def recognize(self, image):
        # convert the image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # localize the faces in the image
        self.rboxes = self.locate(image)

        # for each face, encode the properties
        encodings = self.quantify(image, boxes)

        # the list of all names
        self.rnames = []

        for e in encodings:
            # attempt to match each face in the input to the knowns
            # encodings
            matches = self.compare(e)
            name = "unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces and then initialize
                # an empty dictionnary to count the total number of times
                # each face was matched
                idx = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count
                # for each recognized face
                for i in idx:
                    name = self.names[i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes
                name = max(counts, key=count.get)

            # update the list of names
            self.rnames.append(name)

    def draw(self, image):
        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(self.rboxes, self.rnames):
            # draw the predicted face name on the image
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)

        # return the drawed image
        return image

    def save(self, filename):
        # prepare the datas
        data = {
            "encodings": self.encodings,
            "names": self.names,
            "method": self.method}

        # serialize the encodings to disk
        f = open(filename, "wb")
        f.write(pickle.dumps(data))
        f.close()
