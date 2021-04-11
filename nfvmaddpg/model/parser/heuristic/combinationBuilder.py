from __future__ import division
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.ioff()
from optimization.request import Request
from optimization.parser import Parser
from optimization.placement import Placement
# from network_data import NetworkData
from optimization.placement_input import PlacementInput
import pickle   
from itertools import permutations, islice, product, chain
import sys
from datetime import datetime
import random
import numpy, numpy.random
from operator import itemgetter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import pprint
import logging

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# logger = logging.getLogger("Heurisic")

# fh = logging.FileHandler("logfile.log")
# fh.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# fh.setFormatter(formatter)
ch.setFormatter(formatter)

# logger.addHandler(fh)
logger.addHandler(ch)


class CombinationBuilder:
    def __init__(self, request_list, network, objective_mode = "lex", network_name = "nobelGermany", seed = 123):
        self.request_list = request_list
        self.network = network
        self.results = dict()
        self.request_list = request_list["reqs"]
        self.placementInput_list = []
        self.reqtype = request_list["name"]
        self.objective_mode = objective_mode
        self.network_name = network_name
        self.seed = seed
        
        day = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.resultFile = str(self.reqtype) + '_all_' + str(self.objective_mode) + '_' + str(self.network_name) + '_' + str(day)

        random.seed(self.seed)
        numpy.random.seed(seed)

        # self.network = NetworkData()
        

        self.allReqsAllPermsDict = dict()
        self.allReqsAllPermMaxDataRate = dict()
        self.allReqsAllPermMinDataRate = dict()
        self.allReqsAllPermTotalDataRate = dict()
        self.allReqsAllPermDeploymentCost = dict()

        self.combinationTotalDataRate = dict()
        self.combinationAverageDataRate = dict()
        self.combinationDataRateCost = dict()
        self.combinationDeploymentCost = dict()
        self.combinationAverageComputationalReq = dict()
        self.combinationTotalComputationalReq = dict()


        self.combined_dict = dict()

        self.chosenCombsToPlace = []

    def parseAllRequests(self):
        logger.debug("Request type %s", self.reqtype)
        for i,req in enumerate(self.request_list):
            req.add_prefix(i)
            logger.debug(req)
            
            self.allReqsAllPermsDict[i] = []
            self.allReqsAllPermMaxDataRate[i] = []
            self.allReqsAllPermMinDataRate[i] = []
            self.allReqsAllPermTotalDataRate[i] = []
            self.allReqsAllPermDeploymentCost[i] = []

            req.forceOrder = dict()
            prsr = Parser(req)
            prsr.preparse()
            optords = prsr.optorderdict.copy()
            #~ print "%%%###%%%Req",i,optords 

            perms = dict()
            for v in optords.values():
                perms[",".join(v)] = []
                for x in permutations(v):
                    perms[",".join(v)].append(x)
            prod = list(product(*perms.values()))

            reqPlacementInputList = []
            for n,p in enumerate(prod):
                for j in range(len(p)):
                    for k in optords.keys():
                        if p[j] in perms[k]:
                            req.forceOrder[k] = p[j]
                # print "PERMUTATION",p
                # print "forceorder",req.forceOrder
                prsr = Parser(req)
                prsr.parse()
                thisperm = prsr.create_pairs()
                reqPlacementInputList.append(thisperm)

                # calculate "proxy metrics" for each permutation
                self.allReqsAllPermsDict[i].append(thisperm)
                self.allReqsAllPermMaxDataRate[i].append(max(thisperm['d_req'].values()))
                self.allReqsAllPermMinDataRate[i].append(min(thisperm['d_req'].values()))
                self.allReqsAllPermTotalDataRate[i].append(sum(thisperm['d_req'].values()))
                
                tmpcost = 0
                for u,f in thisperm['UF'].iteritems():
                    if f in self.network["F"]:
                        if self.network["n_ins"][f] == 0:
                            logger.debug("Request %s Permutation %s not feasible", i, n)
                        else:
                            tmpcost += self.network["c_req"][f] / self.network["n_ins"][f]
                self.allReqsAllPermDeploymentCost[i].append(tmpcost)
                
                prsr.print_results()
                logger.debug("Req %s Perm %s ", i, n)
                # print "%%%%%%%%%%%%","Req",i,"Perm",n,thisperm

            self.placementInput_list.append(reqPlacementInputList)

    def computeParetoFront(self, Xs, Ys, maxX, maxY):
        '''Pareto frontier selection process'''
        XsYs = [[Xs[i], Ys[i]] for i in range(len(Xs))]
        #~ sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxY)
        sorted_list = sorted(XsYs, reverse=maxY)
        combIndexesSortedByX = sorted(range(len(XsYs)),key=lambda a:XsYs[a])
        logger.debug("sorted index list %s", combIndexesSortedByX)
                
        pareto_front = [sorted_list[0]]
        paretoFrontCombNum = [combIndexesSortedByX[0]]
        logger.debug("sorted list %s ", sorted_list)
        for ind,tup in enumerate(sorted_list[1:]):
            if maxY:
                if tup[1] > pareto_front[-1][1]:
                    pareto_front.append(tup)
                    paretoFrontCombNum.append(combIndexesSortedByX[ind+1])
            else:
                if tup[1] < pareto_front[-1][1]:
                    pareto_front.append(tup)
                    paretoFrontCombNum.append(combIndexesSortedByX[ind+1])
        
        '''Plotting process'''
        plt.scatter(Xs,Ys)
        for i,x in enumerate(Xs):
            plt.annotate(i, (Xs[i], Ys[i]))
        pf_X = [tup[0] for tup in pareto_front]
        pf_Y = [tup[1] for tup in pareto_front]
        plt.plot(pf_X, pf_Y)
        plt.xlabel("Sum of data rates")
        #~ plt.xlabel("Average data rate")
        #~ plt.ylabel("Deployment cost")
        #~ plt.ylabel("Average computational requirement")
        plt.ylabel("Sum of computational requirements")
        logger.debug("Combinations on pareto front %s", paretoFrontCombNum)
        # plt.savefig("results/" + resultFile + 'Chosen.pdf')
        #~ plt.show()
        return paretoFrontCombNum    

    def chooseCombinations(self):
        # print "########", self.placementInput_list
        all_combinations = list(product(*self.placementInput_list))

        allReqsAllPermProduct = list(product(*self.allReqsAllPermsDict.values()))

        for num,c in enumerate(allReqsAllPermProduct):
            #~ print "COMBINATION",num,c
            self.combined_dict[num] = dict()
            self.combined_dict[num]['U_pairs'] = []
            for r in c:
                self.combined_dict[num]['U_pairs'].extend(r['U_pairs'])
            self.combined_dict[num]['U'] = []
            for r in c:
                self.combined_dict[num]['U'].extend(r['U'])
            self.combined_dict[num]['UF'] = dict()
            for r in c:
                self.combined_dict[num]['UF'].update(r['UF'])
            self.combined_dict[num]['d_req'] = dict()
            for r in c:
                self.combined_dict[num]['d_req'].update(r['d_req'])
            self.combined_dict[num]['A'] = dict()
            for r in c:
                self.combined_dict[num]['A'].update(r['A'])
            self.combined_dict[num]['l_req'] = dict()
            for r in c:
                self.combined_dict[num]['l_req'].update(r['l_req'])
            self.combined_dict[num]['in_rate'] = dict()
            for r in c:
                self.combined_dict[num]['in_rate'].update(r['in_rate'])

            logger.debug("PLACEMENT INPUT %s", num)
            # print "PLACEMENT INPUT",num, self.combined_dict[num]
            
            self.combinationTotalDataRate[num] = sum(self.combined_dict[num]['d_req'].values())
            self.combinationAverageDataRate[num] = sum(self.combined_dict[num]['d_req'].values()) / len(self.combined_dict[num]['U_pairs'])
            self.combinationDataRateCost[num] = self.combinationAverageDataRate[num] * self.combinationTotalDataRate[num]
            
            tmpcost = 0
            tmpsum = 0
            for u,f in self.combined_dict[num]['UF'].iteritems():
                if f in self.network["F"]:
                    tmpsum += self.network["c_req"][f]
                    if self.network["n_ins"][f] == 0:
                        logger.error("Combination %s not feasible", num)
                    else:
                        tmpcost += self.network["c_req"][f] / self.network["n_ins"][f]
            #~ combinationDeploymentCost[num]= tmpcost * combinationAverageDataRate[num]
            self.combinationDeploymentCost[num]= tmpcost 
            #~ combinationDeploymentCost[num]= tmpcost / min(combined_dict['l_req'].values())
            self.combinationAverageComputationalReq[num] = tmpsum / len(self.combined_dict[num]['U'])
            self.combinationTotalComputationalReq[num] = tmpsum
            
            
            #~ placementInput = PlacementInput(combined_dict)
            
            
        # print "combinationTotalDataRate", self.combinationTotalDataRate
        # print "combinationAverageDataRate", self.combinationAverageDataRate
        # print "combinationDataRateCost", self.combinationDataRateCost
        # print "combinationDeploymentCost", self.combinationDeploymentCost
        # print "combinationAverageComputationalReq", self.combinationAverageComputationalReq
        # print "combinationTotalComputationalReq", self.combinationTotalComputationalReq

        #~ chosenCombsToPlace = compute_pareto_front(combinationDataRateCost.values(), combinationDeploymentCost.values(), maxX = False, maxY = False)
        #~ chosenCombsToPlace = compute_pareto_front(combinationAverageDataRate.values(), combinationDeploymentCost.values(), maxX = False, maxY = False)
        #~ chosenCombsToPlace = compute_pareto_front(combinationAverageDataRate.values(), combinationAverageComputationalReq.values(), maxX = False, maxY = False)
        self.chosenCombsToPlace = self.computeParetoFront(self.combinationTotalDataRate.values(), self.combinationTotalComputationalReq.values(), maxX = False, maxY = False)
        #~ chosenCombsToPlace = compute_pareto_front(combinationTotalDataRate.values(), combinationDeploymentCost.values(), maxX = False, maxY = False)
        return self.chosenCombsToPlace

