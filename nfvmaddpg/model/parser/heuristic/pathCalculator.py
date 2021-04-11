from __future__ import division
import pickle 
import pprint
import numpy
import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("Heuristic")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# fh = logging.FileHandler("logfile.log")
# fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh.setFormatter(formatter)
ch.setFormatter(formatter)

# logger.addHandler(fh)
logger.addHandler(ch)

class PathCalculator:
    def __init__(self, g):
        # Network graph
        self.g = g
        # V: list of network graph nodes
        self.V = g['nodes']
        # E: list of network graph edges
        self.E = g['edges']
        # D: dictionary of data rates for network graph edges
        self.D = g['edgeDatarate']
        # D: dictionary of latencies for network graph edges
        self.L = g['edgeLatency']

##############################################################################################

    # Single-Source Shortest Paths for All Flows 
    # T.-W. Shinn, T. Takaoka / Theoretical Computer Science 575 (2015) p10--16, Algorithm 2

    # D: dictionary of data rates for network graph edges
    # s: source node 
    # F: ordered set of maximal flows 

    def SSSP_AF(self, s, F):
        self.s = s 
        self.F = F 
        self.ssspaf = dict()


        # largestDatarateOnPathTo[v]: currently known largest bottleneck value from s to v
        largestDatarateOnPathTo = dict() # B
        # shortestDistTo[v]: currently possible shortest distance from s to v
        shortestDistTo = dict() # L
        # treeOptions[i]: list of vertices that may be added to SPT at distance i from s
        treeOptions = dict() # Q     
        # SPT: Shortest path spanning tree
        SPT = dict() # T

        # Initialize
        for v in self.V:
            largestDatarateOnPathTo[v] = 0
            shortestDistTo[v] = 0
            treeOptions[v] = [] 
            SPT[v] = dict()
            SPT[v]['inserted'] = False
            SPT[v]['parent'] = None
            self.ssspaf[v] = []
        largestDatarateOnPathTo[s] = float('Inf')
        SPT[s]['inserted'] = True 
        SPT[s]['parent'] = None

        for f in self.F:
            for v in self.V:
                if largestDatarateOnPathTo[v] < f:
                    if SPT[v]['inserted']:
                        SPT[v]['inserted'] = False
                        SPT[v]['parent'] = None
                    shortestDistTo[v] += 1 
                    if shortestDistTo[v] not in treeOptions.keys():
                        treeOptions[shortestDistTo[v]] = []
                    treeOptions[shortestDistTo[v]].append(v)
            for l in treeOptions.keys():
                if l < len(self.V) - 1:
                    while len(treeOptions[l]) > 0:
                        v = treeOptions[l].pop()
                        for (u,w) in self.E:
                            if u != w and w == v:
                                if shortestDistTo[u] == shortestDistTo[v] - 1:
                                    if min(self.D[(u,v)], largestDatarateOnPathTo[u]) > largestDatarateOnPathTo[v]:
                                        largestDatarateOnPathTo[v] = min(self.D[(u,v)], largestDatarateOnPathTo[u])
                                        SPT[v]['inserted'] = True
                                        SPT[v]['parent'] = u
                        if SPT[v]['inserted']:
                            self.ssspaf[v].append((shortestDistTo[v], largestDatarateOnPathTo[v]))
                        else:
                            shortestDistTo[v] += 1 
                            if shortestDistTo[v] not in treeOptions.keys():
                                treeOptions[shortestDistTo[v]] = []
                            treeOptions[shortestDistTo[v]].append(v)
        return self.ssspaf

##############################################################################################

    # Extract the path between a start and end point based on the results
    # calculated by SSSP-AF.
    # E: list of network graph edges
    # start: 

    def SSSP_AF_Path(self, start, dest, maxf, result, criteria = "min"):
        self.start = start 
        self.dest = dest 
        self.maxf = maxf 
        self.ssspafResult = result
        self.criteria = criteria
        self.ssspafPath = []

        # To end the while loop below without too much pain!
        self.ssspafResult[self.start] = [(0, self.maxf)]
        
        self.ssspafPath = []
        i = self.dest
        logger.debug("Distance options: %s", self.ssspafResult[i])
        logger.debug("Criteria: %s", self.criteria)
        if self.criteria == "min":
            if len(self.ssspafResult[i]) == 0:
                logger.error("NO PATH FOUND")
                return -1
            elif len([x[0] for x in self.ssspafResult[i] if x[1] >= self.maxf]) == 0:
                logger.error("NO PATH WITH SUFFICIENT DATARATE FOUND")
                return -1                
            else:    
                distance = min(x[0] for x in self.ssspafResult[i] if x[1] >= self.maxf)
                logger.debug("Distance: %s", distance)
        elif self.criteria == "max":
            distance = self.ssspafResult[self.dest][-1][0]
            logger.debug("Distance: %s", distance)
        while i != self.start:
            for (u,v) in self.E:
                if v == i and u != v:
                    r = self.ssspafResult[u]
                    # print "start, u, v, path, i, distance, maxf, r"
                    # print start, u, v, path, i, distance, maxf, r                
                    for (d,f) in r:
                        if f >= self.maxf and d == distance - 1:
                            distance = d
                            i = u
                            self.ssspafPath.append((u,v))
        self.ssspafPath.reverse()
        return self.ssspafPath 

##############################################################################################
