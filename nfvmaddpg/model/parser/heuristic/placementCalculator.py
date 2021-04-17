from __future__ import division
import pickle 
import pprint
import numpy
from pathCalculator import PathCalculator 
from optimization.create_network import CreateNetworkGraph
from combinationBuilder import CombinationBuilder
from optimization.request import Request
from optimization.placement_input import PlacementInput
import logging
from datetime import datetime
import operator

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

class PlacementCalculator:
    def __init__(self, g):

        # Network graph
        self.G = g
        # V: list of network graph nodes
        self.V = self.G['nodes']
        # C: list of network graph node capacities
        self.C = self.G['nodeCap']
        # E: list of network graph edges
        self.E = self.G['edges']
        # D: dictionary of data rates for network graph edges
        self.D = self.G['edgeDatarate']
        # D: dictionary of latencies for network graph edges
        self.L = self.G['edgeLatency']
        # F: list of virtual network functions
        self.F = self.G['F']
        # N_INS: dictionary of number of available instances for each function
        self.N_INS = self.G['n_ins']
        # N_USE: dictionary of number of requests that each instance can handle
        self.N_USE = self.G['n_use'] 
        # C_REQ: dictionary of computational requirements of functions per unit data rate
        self.C_REQ = self.G['c_req']
        # FINAL_C_REQ: dictionary of computational requirements of functions based on input data rate
        self.FINAL_C_REQ = dict()

        self.placementResults = dict()
        self.placementResults['vnf'] = dict()       
        self.placementResults['links'] = dict()
        self.placementResults['latencies'] = dict()
        self.placementResults['e2eLatencies'] = dict()  
        self.placementResults['meanE2eLatency'] = 0  
        self.placementResults['remainingDatarates'] = dict()  
        self.placementResults['meanRemainingDatarate'] = 0  
        self.placementResults['remainingNodeCapacities'] = dict()  
        self.placementResults['meanRemainingNodeCapacity'] = 0  
        self.placementResults['sumRemainingDatarate'] = 0
        self.placementResults['minRemainingDatarate'] = 0
        self.placementResults['maxRemainingDatarate'] = 0
        self.placementResults['sumRemainingNodeCapacity'] = 0
        self.placementResults['minRemainingNodeCapacity'] = 0
        self.placementResults['maxRemainingNodeCapacity'] = 0        

##############################################################################################

    def calculatePathLatency(self,path):
        latency = 0
        for (u,v) in path:
            logger.debug("(u,v) %s", (u,v))
            logger.debug("L[(u,v)] %s", self.L[(u,v)])
            latency += self.L[(u,v)]
        return latency

##############################################################################################

    # Extract simple paths out of pairs, for latency calculations
    # def findAllPathsFrom(self, node, tmppaths, tmp):
    #     tmp.append(node)
        
    #     if len(self.nodes[node]) == 0:
    #         tmppaths.append(tmp)
    #         return tmppaths
        
    #     for n in self.nodes[node]:
    #         newtmp = list(tmp)
    #         self.findAllPathsFrom(n, tmppaths, newtmp)
    #     #return []
    #     return tmppaths

##############################################################################################

    # def makePairsFromList(self, plist):
    #     ps = []
    #     for i in range(0, len(plist) -1):
    #         ps.append((plist[i], plist[i + 1]))
    #     return ps        

##############################################################################################

    def mapVirtualFunction(self, u, currentNode):
        logger.debug("%s %s %s", u, self.FINAL_C_REQ[u], self.C[currentNode])
        self.placementResults["vnf"][u] = currentNode
        self.C[currentNode] -= self.FINAL_C_REQ[u]
        logger.debug("CURRENT NODE CAPACITIES %s", self.C)


##############################################################################################

    def mapVirtualLink(self, u1, u2, tmpPath):
        self.placementResults["links"][(u1,u2)] = tmpPath
        self.placementResults['latencies'][(u1,u2)] = 0
        logger.debug("tmpPath %s", tmpPath)
        for (u,v) in tmpPath:
            if u != v:
                self.D[(u,v)] -= self.data.d_req[(u1,u2)]
                self.placementResults['latencies'][(u1,u2)] += self.L[(u,v)]
        logger.debug("CURRENT DATARATE CAPACITIES %s", self.D)


##############################################################################################

    def placePair(self, u1, u2, currentNode, endNode, pairs, blacklisting = False):
        remainingPairs = list(pairs)
        # import pdb; pdb.set_trace()
        logger.debug("REMAINING PAIRS %s", remainingPairs)
        logger.debug("PAIR %s %s", u1, u2)
        # Calculate flow values (f) for SSSP-AF based on current network state
        F = []
        for d in self.D.values():
            if d not in F:
                F.append(d)
        F.sort()
        logger.debug("F: %s", F)

        pathCalculator = PathCalculator(self.G)


        # Temp list to store edges traversed while looking for a node that can accommodate u2
        tmpPath = []

        pathIndex = 0

        maxdrReq = 0
        for (v1,v2) in remainingPairs:
            if self.data.d_req[(v1,v2)] > maxdrReq:
                maxdrReq = self.data.d_req[(v1,v2)]
        logger.debug("MAX DATARATE REQUIREMENT %s", maxdrReq)

        # Check if u2 can be placed in the current node (where u1 is placed)
        if self.FINAL_C_REQ[u2] <= self.C[currentNode]:
            tmpPath.append((currentNode, currentNode))
            self.mapVirtualFunction(u2, currentNode)
            self.mapVirtualLink(u1, u2, tmpPath)
            remainingPairs.remove((u1,u2))

            logger.debug(self.placementResults)
            logger.debug("CURRENT NODE CAPACITIES %s", self.C)
            logger.info("%s PLACED ON NODE %s", u2, currentNode)
            return remainingPairs   
        # If u2 cannot be placed in the current node, traverse the shortest path 
        # to find the next best node         
        else:
            logger.error("%s COULD NOT PLACE ON NODE %s ( %s > %s ), CONTINUE SEARCHING", u2, currentNode, self.FINAL_C_REQ[u2], self.C[currentNode])
            visited = [currentNode]

            F = []
            for d in self.D.values():
                if d not in F:
                    F.append(d)
            F.sort()
            logger.debug("F: %s", F)  

            maxdrReq = 0
            for (v1,v2) in remainingPairs:
                if self.data.d_req[(v1,v2)] > maxdrReq:
                    maxdrReq = self.data.d_req[(v1,v2)]
            logger.debug("MAX DATARATE REQUIREMENT %s", maxdrReq)

            if currentNode == endNode:
                logger.error("INFEASIBLE, ENDNODE REACHED")
                return -1 

            while (currentNode != endNode) and (u2 not in self.placementResults["vnf"].keys()):
                # Compute SSSP-AF from current node
                ssspafResults = pathCalculator.SSSP_AF(currentNode, F)
                logger.debug("SSSP-AF RESULTS from node %s %s", currentNode, ssspafResults)

                ssspafPath = pathCalculator.SSSP_AF_Path(
                    currentNode, 
                    endNode, 
                    maxdrReq, 
                    ssspafResults, 
                    "min")
                if ssspafPath == -1:
                    logger.error("INFEASIBLE, NO PATH FOUND")
                    return -2
                pathIndex = 0  

                logger.debug("PATHINDEX %s", pathIndex)
                nextNode = ssspafPath[pathIndex][1]

                if nextNode in visited:
                    logger.error("INFEASIBLE, %s ALREADY VISITED", nextNode)
                    return -1 
                else:                   
                    visited.append(nextNode)

                logger.debug("%s %s", currentNode, nextNode)
                tmpPath.append((currentNode, nextNode))
                currentNode = nextNode
                pathIndex += 1
                logger.debug("CURRENT NODE: %s", currentNode)

                if self.FINAL_C_REQ[u2] <= self.C[currentNode]:
                    self.mapVirtualFunction(u2, currentNode)
                    logger.debug("%s", tmpPath)
                    self.mapVirtualLink(u1, u2, tmpPath)
                    logger.debug("%s", tmpPath)
                    remainingPairs.remove((u1,u2))

                    logger.debug(self.placementResults)
                    logger.debug("CURRENT NODE CAPACITIES %s", self.C)
                    logger.info("%s PLACED ON NODE %s", u2, currentNode)
                    return remainingPairs   

                else:
                    logger.error("%s COULD NOT PLACE ON NODE %s ( %s > %s )", u2, currentNode, self.FINAL_C_REQ[u2], self.C[currentNode])
                    if currentNode == endNode:
                        logger.error("INFEASIBLE")
                        return -1


##############################################################################################


    def placeService(self, data):

        self.ts = datetime.now()

        self.data = data

        # Calculate flow values (f) for SSSP-AF
        F = []
        for d in self.G['edgeDatarate'].values():
            if d not in F:
                F.append(d)
        F.sort()
        logger.debug("F: %s", F)

        logger.debug("U_pairs %s", self.data.U_pairs)

        # Sum of the latencies for all edges
        lat_sum = 0
        maxdr = 0
        for (a,b) in self.E:
            lat_sum += self.L[(a,b)]
            if a != b:
                if self.D[(a,b)] > maxdr:
                    maxdr = self.D[(a,b)]
        # print maxdr
            
        # Number of edges except the self-loops 
        edgeCount = len(self.E) - len(self.V)
        
        # Number of pairs to be placed
        pairCount = len(self.data.U_pairs)

        # Extract simple paths out of pairs, for latency calculations
        def findAllPathsFrom(self, node, tmppaths, tmp):
            tmp.append(node)
            
            if len(nodes[node]) == 0:
                tmppaths.append(tmp)
                return tmppaths
            
            for n in nodes[node]:
                newtmp = list(tmp)
                findAllPathsFrom(self, n, tmppaths, newtmp)
            return tmppaths
            
        def makePairsFromList(self, plist):
            ps = []
            for i in range(0, len(plist) -1):
                ps.append((plist[i], plist[i + 1]))
            return ps        

        nodes = dict()
        for p in self.data.U_pairs:
            if p[0] in nodes:
                nodes[p[0]].append(p[1])
            else:
                nodes[p[0]] = [p[1]]
        for (start, end) in self.data.l_req.keys():
            nodes[end] = []

        # count paths
        pathCount = 0

        self.paths = dict()
        for (start, end) in self.data.l_req.keys():
            tmp = []
            tmppaths = []
            findAllPathsFrom(self, start, tmppaths, tmp)
            realpaths = []
            for tp in tmppaths:
                if end in tp:
                    realpaths.append(tp)
            #self.paths[(start, end)] = tmppaths
            self.paths[(start, end)] = realpaths
            pathCount += len(tmppaths)
        logger.debug("PATHS %s", self.paths)
        
        # count pathpairs
        pathPairCount = 0
        
        self.pathPairs = dict()
        self.pathPairsSumDatarate = dict()
        # self.pathPairsMaxDatarate = dict()
        for k in self.data.l_req.keys():
            self.pathPairs[k] = []
            self.pathPairsSumDatarate[k] = []
            # self.pathPairsMaxDatarate[k] = []
            for p in self.paths[k]:
                ps = makePairsFromList(self, p)
                self.pathPairs[k].append(ps)
                print ps
                tmpSum = 0
                # tmpMax = 0
                for x in ps:
                    tmpSum += self.data.d_req[(x)]
                    # if self.data.d_req[(x)] > tmpMax:
                        # tmpMax = self.data.d_req[(x)]
                self.pathPairsSumDatarate[k].append(tmpSum)
                # self.pathPairsMaxDatarate[k].append(tmpMax)
            pathPairCount += len(self.pathPairs[k])
        logger.debug("PATHPAIRS %s", self.pathPairs)
        logger.debug("pathPairsSumDatarate %s", self.pathPairsSumDatarate)
        # logger.debug("pathPairsMaxDatarate %s", self.pathPairsMaxDatarate)

        self.pathPairsSorted = dict()
        for kk in self.data.l_req.keys():
            s = list(self.pathPairsSumDatarate[kk])
            ss = list(sorted(range(len(s)), key=lambda k: s[k]))
            ss.reverse()
            self.pathPairsSorted[kk] = ss
            logger.debug("ss %s", ss)
        logger.debug("pathPairsSorted %s", self.pathPairsSorted)


        #~ print "PATHPAIRS", self.pathPairs
        
        # Actual computational requirements of functions based on data rate
        for u in self.data.U:
            self.FINAL_C_REQ[u] = self.data.in_rate[u] * self.C_REQ[self.data.UF[u]]
        logger.debug("FINAL_C_REQ %s", self.FINAL_C_REQ)

        endpointsSumDatarate = dict()
        for (a1,a2) in self.data.l_req:
            endpointsSumDatarate[(a1,a2)] = sum(self.pathPairsSumDatarate[(a1,a2)])
        logger.debug("endpointsSumDatarate %s", endpointsSumDatarate) 
        sortedEndpoints = sorted(endpointsSumDatarate.items(), key=operator.itemgetter(1))
        sortedEndpoints.reverse()
        logger.debug("sortedEndpoints %s", sortedEndpoints)




        # for (a1,a2) in self.data.l_req.keys():
        for ((a1,a2),aa) in sortedEndpoints:
            self.placementResults['e2eLatencies'][(a1,a2)] = dict()  
            for index in self.pathPairsSorted[(a1,a2)]:
                path = self.pathPairs[(a1,a2)][index]           
            # for index, path in enumerate(self.pathPairs[(a1,a2)]):
                self.placementResults['e2eLatencies'][(a1,a2)][index] = 0
                p = list(path)

                endpoint1 = self.data.A[a1]
                endpoint2 = self.data.A[a2]

                self.placementResults["vnf"][a1] = endpoint1
                self.placementResults["vnf"][a2] = endpoint2

    
                logger.debug("PATH NUMBER %s", index)
                logger.debug("PATH %s", path)
                logger.debug("BETWEEN ENDPOINTS %s and %s", endpoint1, endpoint2)

                for (u1,u2) in path:
                    pathCalculator = PathCalculator(self.G)

                    if u2 == a2 and (u1,u2) not in self.placementResults["links"]:
                        if self.placementResults['vnf'][u1] != endpoint2:
                            logger.debug("%s %s", u2, a2)
                            F = []
                            for d in self.D.values():
                                if d not in F:
                                    F.append(d)
                            F.sort()
                            logger.debug("F: %s", F) 

                            maxdrReq = self.data.d_req[(u1,u2)] 
                            # Compute SSSP-AF from current node
                            node = self.placementResults['vnf'][u1]
                            ssspafResults = pathCalculator.SSSP_AF(node, F)
                            logger.debug("SSSP-AF RESULTS from node %s %s", node, ssspafResults) 

                            ssspafPath = pathCalculator.SSSP_AF_Path(
                                node, 
                                endpoint2, 
                                maxdrReq, 
                                ssspafResults, 
                                "min")
                            self.mapVirtualLink(u1, u2, ssspafPath)
                            logger.debug("PATH TO DESTINATION")
                            logger.debug("%s", ssspafPath)
                        else:
                            path = [(endpoint2, endpoint2)]
                            self.mapVirtualLink(u1, u2, path)

                    elif u2 in self.placementResults["vnf"]:
                        logger.info("Pair %s %s is already placed", u1, u2)
                        p.remove((u1,u2))

                    else:
                        logger.debug("CALLING PLACEPAIR")
                        logger.debug("FOR PAIR %s %s", u1, u2)

                        last = self.placementResults['vnf'][u1]
                        logger.debug("Last %s", last)
                        currentPairIndex = p.index((u1,u2))
                        logger.debug("currentPairIndex %s", currentPairIndex)
                        currentEndpoint = endpoint2
                        for pi in xrange(currentPairIndex,len(p)):
                            if p[pi][1] in self.placementResults["vnf"]:
                                currentEndpoint = self.placementResults["vnf"][p[pi][1]]
                                break
                        logger.debug("currentEndpoint %s", currentEndpoint)
        

                        returnVal = self.placePair(u1, u2, last, currentEndpoint, p)
                        if returnVal == 1:
                            logger.debug("Pair %s %s has already been placed", u1, u2)
                        elif returnVal == -1:
                            logger.error("Placement Failed")
                            return -1
                        elif returnVal == -2:
                            logger.error("Placement Failed, no path found")
                            return -1
                        else:
                            p = returnVal

                    self.placementResults['e2eLatencies'][(a1,a2)][index] += self.placementResults['latencies'][(u1,u2)]

            if self.placementResults['e2eLatencies'][(a1,a2)][index] > self.data.l_req[(a1,a2)]:
                logger.error("Placement Failed, latency over path")
                return -1

        # Some statistics..
        self.placementResults['remainingDatarates'] = dict(self.D)
        self.placementResults['remainingNodeCapacities'] = dict(self.C)
        # self.placementResults['meanE2eLatency'] = sum(self.placementResults['e2eLatencies'].values()) / len(self.data.l_req.keys())

        tmpsum = 0
        self.placementResults['maxRemainingDatarate'] = 0
        for (a,b) in self.E:
            if a != b:
                tmpsum += self.D[(a,b)]
                if self.D[(a,b)] > self.placementResults['maxRemainingDatarate']:
                    self.placementResults['maxRemainingDatarate'] = self.D[(a,b)]
        self.placementResults['meanRemainingDatarate'] = tmpsum / edgeCount
        self.placementResults['sumRemainingDatarate'] = tmpsum

        self.placementResults['minRemainingDatarate'] = min(self.placementResults['remainingDatarates'].values())
        # self.placementResults['maxRemainingDatarate'] = max(self.placementResults['remainingDatarates'].values())

        self.placementResults['meanRemainingNodeCapacity'] = sum(self.placementResults['remainingNodeCapacities'].values()) / len(self.V)
        self.placementResults['sumRemainingNodeCapacity'] = sum(self.placementResults['remainingNodeCapacities'].values())

        self.placementResults['minRemainingNodeCapacity'] = min(self.placementResults['remainingNodeCapacities'].values())
        self.placementResults['maxRemainingNodeCapacity'] = max(self.placementResults['remainingNodeCapacities'].values())



        logger.info("FINAL PLACEMENT RESULTS %s", self.placementResults)
        self.te = datetime.now()
        td = self.te - self.ts
        self.placementResults["time_ms"] = td.total_seconds()*1000
        # pprint.pprint(self.placementResults)
        return self.placementResults

