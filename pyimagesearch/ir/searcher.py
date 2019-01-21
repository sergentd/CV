# import necessary packages
from .searchersult import SearchResult
from .dists import chi2chi2_distance
import numpy as np
import datetime
import h5py

class Searcher:
    def __init__(self, redisDB, bovwDBPath, featuresDBPath, idf=None,
        distanceMetric=chi2_distance):
        # store the redis database reference, the metruc and the idf array
        self.redisDB = redisDB
        self.distanceMetric = distanceMetric
        self.idf = idf

        # open both the bovw database and features database for reading
        self.bovwDB = h5py.File(bovwDBPath, mode="r")
        self.featuresDBPath = h5py.File(featuresDBPath, mode="r")

    def search(self, queryHist, numResults=10, maxCandidates=200):
        # start the timer to track how long the search took
        startTime = datetime.datetime.now()

        # determine the candidates and sort them in ascending order so
        # they can be read from the bovw database
        candidateIdxs = self.buildCandidates(queryHist, maxCandidates)
        candidateIdxs.sort()

        # grab the histogram for the candidates from the bovw
        # database and initialize the results dictionnary
        hists = self.bovw["bovw"][candidateIdxs]
        queryHist = queryHist.toarray()
        results = {}

        # if the inverse document frequency array as been supplied
        # multiply the query by it
        if self.idf is not None:
            queryHist *= self.idf

        # loop over the histograms
        for (candidates, hist) in zip(candidateIdxs, hists):
            # if the inverse document frequency array has been supplied
            # multiply the histogram by it
            if self.idf is not None:
                hist *= self.idf

            # compute the distance between the histograms and update the
            # results dictionnary
            d = self.distanceMetric(hist, queryHist)
            results[candidates] = d

        # sort the results by replacing the image indexes with the image
        # ids themselves
        results = sorted([(v, self.featuresDB["image_ids"][k], k)
            for (k, v) in results.items()])
        results = results[:numResults]

        # return the search result
        return SearchResult(results,
            (datetime.datetime.now() - startTime).total_seconds())

    def buildCandidates(self, hist, maxCandidates):
        # initialize the redis pipeline
        pipe = self.redisDB.pipeline()

        # loop over the columns of the sparse matric and create a query
        # to grab all images with an occurence of the current visual word
        for i in hist.col:
            pipe.lrange("vw:{}".format(i), 0, -1)

        # execute the pipeline and initialize the candidates list
        pipelineResults = pipe.execute()
        candidates = []

        # loop over the pipeline results, extract the image index and update
        # the candidates list
        for results in pipelineResults:
            results = [int(r) for r in results]
            candidates.extend(results)

        # count the occurence of each of the candidates and sort in descending
        # order
        (imageIdxs, counts) = np.unique(candidates, return_counts=True)
        imageIdxs = [i for (c, i) in sorted(zip(counts, imageIdxs),
            reverse=True)]

        # return the image indexes of the candidates
        return imageIdxs[:maxCandidates]

    def finish(self):
        # close the bovw database and the features database
        self.bovwDB.close()
        self.featuresDB.close()
