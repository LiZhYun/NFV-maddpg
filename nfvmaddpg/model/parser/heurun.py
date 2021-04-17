from __future__ import division
import pickle 
import pprint
import numpy
from datetime import datetime
from heuristic.pathCalculator import PathCalculator 
from heuristic.placementCalculator import PlacementCalculator
from optimization.create_network import CreateNetworkGraph
from heuristic.combinationBuilder import CombinationBuilder
from request import Request
from optimization.placement_input import PlacementInput
import logging
import sys

reqtype = sys.argv[1]
network = sys.argv[2]

day = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f'))

logFile = str(reqtype) + '_' + str(network) + '_' + str(day) + ".log"

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger = logging.getLogger("Heuristic")

fh = logging.FileHandler("heuristicResults/" + logFile)
fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

logger.addHandler(fh)
logger.addHandler(ch)

logger.debug("LOGFILE CREATED")

resultFile = str(reqtype) + '_' + str(network) + '_' + str(day)


g = pickle.load(open("netGraph_" + network + ".pickle","r"))

placementCalculator = PlacementCalculator(dict(g))

requestList = pickle.load(open("requestList_" + reqtype + ".pickle","r"))

logger.info("REQUEST LIST")
for r in requestList["reqs"]:
    logger.debug("%s", r)

combBuilder = CombinationBuilder(requestList, dict(g))
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
pprint.pprint(chosenCombsResults)

# filename = "heuristicResults/heuristic_" + str(day) + ".pickle"
filename = "heuristicResults/" + resultFile + ".pickle"
pickle.dump(chosenCombsResults, open(filename,"wb"))
