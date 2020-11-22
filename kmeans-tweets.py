import numpy as np
import random
import re


class KMeans:
    def __init__(self, data, epochs, K):
        self.data = self.preprocess(data)
        self.epochs = epochs
        self.K = K
        # Select K random tweets as initial centroids.
        self.centroids = random.sample(self.data, k=self.K)

    # Preprocess tweets before performing KMeans algorithm
    def preprocess(self, lines):
        tweets = [line[50:] for line in lines]                              # Remove the tweet id and timestamp
        tweets = [re.sub('@[^\s]+', '', tweet) for tweet in tweets]         # Remove any word that starts with the symbol @
        tweets = [re.sub(r'#([^\s]+)', r'\1', tweet) for tweet in tweets]   # Remove any hashtag symbols
        tweets = [re.sub('http://[^\s]+', '', tweet) for tweet in tweets]   # Remove any URL
        tweets = [tweet.rstrip('\n') for tweet in tweets]                   # Remove newline char(\n) from end of each tweet
        tweets = [tweet.lower() for tweet in tweets]                        # Convert every word to lowercase
        return tweets

    # Calculate Jaccard distance between two phrases.
    def jaccard_dist(self, a, b):
        a_words = set(a.split(sep=' '))
        b_words = set(b.split(sep=' '))
        i = a_words.intersection(b_words)
        u = a_words.union(b_words)
        return 1 - (len(i) / len(u))

    # Perform clustering.
    def cluster(self):
        same_centroids = 0
        # For each iteration, do the following.
        for epoch in range(self.epochs):
            # Create empty dictionary of clusters.
            clusters = {}

            # Add centroids to separate clusters.
            i = 0
            for c in self.centroids:
                clusters[i] = []
                clusters[i].append(self.centroids[i])
                i += 1

            # For each tweet, compare jaccard distance to each centroid and assign it to minimum distance cluster.
            for t in self.data:
                # If the tweet is already a centroids, skip it and go to next tweet.
                if t in self.centroids:
                    continue

                dist = []
                # Compare distance from tweet to each centroid and put in cluster with minimum distance.
                for c in self.centroids:
                    dist.append(self.jaccard_dist(t, c))
                m = np.argmin(dist)
                clusters[m].append(t)

            # Adjust the centroid of each cluster by setting it to
            #   tweet having minimum distance to all of the other tweets in a cluster.
            new_centroids = self.centroids.copy()
            for i in range(self.K):
                sums = []
                for a in clusters[i]:
                    sum = 0
                    for b in clusters[i]:
                        sum += self.jaccard_dist(a, b)
                    sums.append(sum)
                m = np.argmin(sums)
                new_centroids[i] = clusters[i][m]

            # Calculate SSE.
            sse = 0
            for i in range(self.K):
                for a in clusters[i]:
                    sse += self.jaccard_dist(self.centroids[i], a) ** 2



            # If the centroids are the same as the previous iteration, increment the counter.
            if new_centroids == self.centroids:
                same_centroids += 1
            # Otherwise, set the counter to 0.
            else:
                same_centroids = 0
                # break

            # If the centroids do not change for 10 consecutive iterations, stop the loop
            if same_centroids > 10:
                self.clusters = clusters.copy()
                self.sse = sse
                break
            # Otherwise, set the centroids to be the ones calculated in this iteration.
            else:
                self.centroids = new_centroids.copy()

        self.clusters = clusters.copy()
        self.sse = sse


if __name__ == "__main__":
    file = open('Health-Tweets/usnewshealth.txt', 'r')
    model = KMeans(file, 100, 5)
    model.cluster()
    print('Value of K: ',model.K)
    print('SSE: ', model.sse)
    print('Size of each cluster: ')
    for i in range(model.K):
        print('\t', i+1, ': ', len(model.clusters[i]))
