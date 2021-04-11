from __future__ import division
import pickle 
import pprint
import numpy
from datetime import datetime
from heuristic.pathCalculator import PathCalculator 
from heuristic.placementCalculator import PlacementCalculator
from optimization.create_network import CreateNetworkGraph
from heuristic.combinationBuilder import CombinationBuilder
from optimization.request import Request
from optimization.placement_input import PlacementInput
import logging

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger = logging.getLogger("Heuristic")

# fh = logging.FileHandler("logfile.log")
# fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh.setFormatter(formatter)
ch.setFormatter(formatter)

# logger.addHandler(fh)
logger.addHandler(ch)

day = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

g = pickle.load(open('netGraph.pickle','r'))

placementCalculator = PlacementCalculator(g)

requestList = pickle.load(open('requestList.pickle','r'))

logger.info("REQUEST LIST")
for r in requestList["reqs"]:
    logger.debug("%s", r)

combBuilder = CombinationBuilder(requestList, g)
combBuilder.parseAllRequests()
chosenCombs = combBuilder.chooseCombinations()

chosenCombsResults = dict()

for i,c in enumerate(chosenCombs):
    logger.debug("Chosen combination number %s", c)
    logger.debug("Combined dictionary")
    logger.debug("%s", combBuilder.combined_dict[c])
    placementInput = PlacementInput(dict(combBuilder.combined_dict[c]))
    chosenCombsResults[c] = placementCalculator.placeService(placementInput)
     

logger.debug("%s", chosenCombsResults)
filename = "heuristicResults/heuristic_" + str(day) + ".pickle"
pickle.dump(chosenCombsResults, open(filename,"wb"))

# F = []
# maxf = 1200
# for d in g['edgeDatarate'].values():
#     # if d <= maxf and d not in f:
#     if d not in F:
#         F.append(d)
# F.sort()
# print "*** F:", F
# pathCalculator = PathCalculator(g)
# ssspafResults = pathCalculator.SSSP_AF(4,F)
# print "*** SSSP-AF RESULTS", ssspafResults
# ssspafPath = pathCalculator.SSSP_AF_Path(4, 15, 900, ssspafResults, "min")
# print "*** f", maxf
# print "*** SSSP-AF PATH", ssspafPath
# # nextNode = ssspafPath[0][1]
# nextNode = placementCalculator.getNextNodeOnPath(ssspafPath)
# print "*** NEXT NODE:", nextNode
# pathLatency = placementCalculator.calculatePathLatency(ssspafPath)
# print "*** PATH LATENCY", pathLatency
