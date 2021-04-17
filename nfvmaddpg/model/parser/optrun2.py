from __future__ import division
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.ioff()
from request import Request
from optimization.parser import Parser
from optimization.placement import Placement
from network_data import NetworkData
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

results = dict()
request_list = []
placementInput_list = []
# objective_mode = 'dr'
# reqtype = 'optorder'

# reqtype = sys.argv[1]
# netname = sys.argv[2]
# objective_mode = sys.argv[3]
reqtype = input("Enter request type (e.g., 'req150'): ")
netname = input("Enter network name (e.g., 'abilene'): ")
objective_mode = input("Enter optimization objective (e.g., 'lat' to minimize total latency, 'use' to minimize the number of used network nodes, 'dr' to maximize the total remaining link capacity in network): ")

# seed = int(sys.argv[3])
# network = 'abilene'
#~ network = 'atlanta'
day = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
resultFile = str(reqtype) + '_all_' + str(objective_mode) + '_' + str(netname) + '_' + str(day)

# random.seed(seed)
# numpy.random.seed(seed)

def compute_3dpareto_front(Xs, Ys, Zs, maxX, maxY, maxZ):
    '''Pareto frontier selection process'''
    XsYsZs = [[Xs[i], Ys[i], Zs[i]] for i in range(len(Xs))]
    sorted_list = sorted(XsYsZs, key=itemgetter(0), reverse=maxY)
    combIndexesSortedByX = sorted(range(len(XsYsZs)),key=lambda a:XsYsZs[a])
    #print "PARETO3D sorted index list", combIndexesSortedByX
            
    pareto_front = [sorted_list[0]]
    paretoFrontCombNum = [combIndexesSortedByX[0]]
    #print "PARETO3D sorted list",sorted_list
    for ind,tup in enumerate(sorted_list[1:]):
        nondominated = True
        if tup[0] == pareto_front[-1][0] and (tup[1] <= pareto_front[-1][1] and tup[2] <= pareto_front[-1][2]):
            pareto_front.pop(-1)
            paretoFrontCombNum.pop(-1)              
        else:
            for j in range(len(pareto_front)):
                if sum(tup[i] >= pareto_front[j][i] for i in range(len(tup))) == len(tup):
                    nondominated = False
                    break
        if nondominated:
            pareto_front.append(tup)
            paretoFrontCombNum.append(combIndexesSortedByX[ind+1])

    fig = plt.figure()
    c = []
    for i in range(len(Xs)):
        if i in paretoFrontCombNum:
            c.append('r')
        else:
            c.append('b')
    ax = fig.add_subplot(111, projection='3d')
    
    global labels_and_points

    labels_and_points = []


    ax.scatter(xs=Xs, ys=Ys, zs=Zs, s=20, c=c)
    ax.set_xlabel('Total data rate')
    ax.set_ylabel('Total computational requirement')
    ax.set_zlabel('Total number of instances')
    
    labels = []
    for i,x in enumerate(Xs):
        labels.append(i)        
    for txt, x, y, z in zip(labels, Xs, Ys, Zs):
        x2, y2, _ = proj3d.proj_transform(x,y,z, ax.get_proj())
        label = plt.annotate(
            txt, xy = (x2, y2), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
        labels_and_points.append((label, x, y, z))

    def update_pos(e):
        for label, x, y, z in labels_and_points:
            x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
            label.xy = x2,y2
            label.update_positions(fig.canvas.renderer)
        fig.canvas.draw()


    fig.canvas.mpl_connect('motion_notify_event', update_pos)
                
    #print "PARETO3D Combinations on pareto front", paretoFrontCombNum
    plt.savefig("optimizationResults/" + resultFile + 'Chosen.pdf')
    #~ plt.show()

    return paretoFrontCombNum


def compute_pareto_front(Xs, Ys, maxX, maxY):
    '''Pareto frontier selection process'''
    XsYs = [[Xs[i], Ys[i]] for i in range(len(Xs))]
    #~ sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxY)
    sorted_list = sorted(XsYs, reverse=maxY)
    combIndexesSortedByX = sorted(range(len(XsYs)),key=lambda a:XsYs[a])
    #print "sorted index list", combIndexesSortedByX
            
    pareto_front = [sorted_list[0]]
    paretoFrontCombNum = [combIndexesSortedByX[0]]
    #print "sorted list",sorted_list
    for ind,tup in enumerate(sorted_list[1:]):
        if maxY:
            if tup[1] > pareto_front[-1][1]:
                pareto_front.append(tup)
                paretoFrontCombNum.append(combIndexesSortedByX[ind+1])
        else:
            if tup[1] < pareto_front[-1][1]:
                pareto_front.append(tup)
                paretoFrontCombNum.append(combIndexesSortedByX[ind+1])
    #~ #print "Combinations on pareto front", paretoFrontCombNum
    
    '''Plotting process'''
    plt.scatter(Xs,Ys)
    for i,x in enumerate(Xs):
        plt.annotate(i, (Xs[i], Ys[i]))
    pf_X = [tup[0] for tup in pareto_front]
    pf_Y = [tup[1] for tup in pareto_front]
    plt.plot(pf_X, pf_Y)
    plt.xlabel("Sum of data rates")
    #~ plt.xlabel("Average data rate")
    plt.ylabel("Deployment cost")
    #~ plt.ylabel("Average computational requirement")
    #~ plt.ylabel("Sum of computational requirements")
    #print "Combinations on pareto front", paretoFrontCombNum
    plt.savefig("optimizationResults/" + resultFile + 'Chosen.pdf')
    #~ plt.show()
        
    return paretoFrontCombNum

class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()
request_list_import = pickle.load(StrToBytes(open("requestList_" + reqtype + ".pickle","r")))
request_list = list(request_list_import["reqs"])

# num_requests = len(request_list)
# ratios = numpy.random.dirichlet(numpy.ones(num_requests),size=1)[0]
# for i,r in enumerate(request_list):
#     r.input_datarate = round(ratios[i]*1000,3)
    
network = NetworkData(netname)

# allOptions = []
# for i,req in enumerate(request_list):
#   req.add_prefix(i)
#   # #print req

#   req.forceOrder = dict()
#   prsr = Parser(req)
#   prsr.preparse()

#   optords = prsr.optorderdict.copy()
#   allOptions.append(optords)
# #print '*********************************** allOptions '
# p#print.p#print(allOptions)

allOptions = []
for i,req in enumerate(request_list):
    req.add_prefix(i)
    #print req

    req.forceOrder = dict()
    prsr = Parser(req)
    prsr.preparse()

    optords = prsr.optorderdict.copy()

    #print "##################################PREPARSE###"
    pprint.pprint(optords)


    perms = dict()
    for v in optords.values():
        perms[",".join(v)] = []
        for x in permutations(v):
            perms[",".join(v)].append(x)

    allOptions.append(dict(perms))

    prod = list(product(*perms.values()))
    #print 'prod', prod

    # allOptions[req] = []
    reqPlacementInputList = []
    for p in prod:
        for i in range(len(p)):
            for k in optords.keys():
                if p[i] in perms[k]:
                    req.forceOrder[k] = p[i]
        #print "PERMUTATION",p
        #print "forceorder",req.forceOrder
        prsr = Parser(req)
        prsr.parse()
        reqPlacementInputList.append(prsr.create_pairs())

        # allOptions[req].append(dict(req.forceOrder).values())

        prsr.print_results()


    placementInput_list.append(reqPlacementInputList)
    
#print "########", placementInput_list
all_combinations = list(product(*placementInput_list))

#print '*********************************** allOptions'
pprint.pprint(allOptions)

########################################################################

""" Compute chosen combinations """
combinationTotalDataRate = dict()
#~ combinationMaxDataRate = dict()
combinationTotalComputationalReq = dict()
#~ combinationMaxComputationalReq = dict()
combinationNumInstances = dict()
#~ combinationDeploymentCost = dict()

for num,c in enumerate(all_combinations):
    combinationTotalDataRate[num] = 0
    combinationTotalComputationalReq[num] = 0
    #~ combinationMaxDataRate[num] = 0
    #~ combinationMaxComputationalReq[num] = 0
    combinationNumInstances[num] = 0
    for r in c:
        combinationNumInstances[num] += len(r['U'])
        combinationTotalDataRate[num] += round(sum(r['d_req'].values()),3)
        #~ if max(r['d_req'].values()) > combinationMaxDataRate[num]:
            #~ combinationMaxDataRate[num] = round(max(r['d_req'].values()),3)
        for u,f in r['UF'].iteritems():
            #~ #print u,f
            if f in network.F:
                #~ combinationTotalComputationalReq[num] += max(network.p_d[f], network.p_s[f])
                combinationTotalComputationalReq[num] += round(network.c_req[f] * r['in_rate'][u],3)
                #~ if network.c_req[f] * r['in_rate'][u] > combinationMaxComputationalReq[num]:
                    #~ combinationMaxComputationalReq[num] = round(network.c_req[f] * r['in_rate'][u],3)
    #~ combinationMaxDataRate[num] = sum(r['d_req'].values())
    #~ combinationMeanComputationalReq[num] = combinationTotalComputationalReq[num] / len(r['U'])
    #~ combinationDeploymentCost[num] = round(combinationTotalComputationalReq[num] * combinationNumInstances[num],3)
                
    combinationTotalDataRate[num] = round(combinationTotalDataRate[num],3) 
    combinationTotalComputationalReq[num] = round(combinationTotalComputationalReq[num],3)
    
#print "combinationTotalDataRate", combinationTotalDataRate
#print "combinationTotalComputationalReq", combinationTotalComputationalReq
#~ #print "combinationMaxDataRate", combinationMaxDataRate
#~ #print "combinationMaxComputationalReq", combinationMaxComputationalReq
#print "combinationNumInstances", combinationNumInstances
#~ #print "combinationDeploymentCost", combinationDeploymentCost


chosenCombsToPlace = compute_3dpareto_front(combinationTotalDataRate.values(), combinationTotalComputationalReq.values(), combinationNumInstances.values(), maxX = False, maxY = False, maxZ = False)
#~ chosenCombsToPlace = compute_pareto_front(combinationTotalDataRate.values(), combinationDeploymentCost.values(), maxX = False, maxY = False)
#~ chosenCombsToPlace = compute_pareto_front(combinationTotalDataRate.values(), combinationTotalComputationalReq.values(), maxX = False, maxY = False)
#print  "@@@@@@@", chosenCombsToPlace
########################################################################

for num,c in enumerate(all_combinations):
    #~ #print "COMBINATION",c
    combined_dict = dict()
    combined_dict['U_pairs'] = []
    for r in c:
        combined_dict['U_pairs'].extend(r['U_pairs'])
    combined_dict['U'] = []
    for r in c:
        combined_dict['U'].extend(r['U'])
    combined_dict['UF'] = dict()
    for r in c:
        combined_dict['UF'].update(r['UF'])
    combined_dict['d_req'] = dict()
    for r in c:
        combined_dict['d_req'].update(r['d_req'])
    combined_dict['A'] = dict()
    for r in c:
        combined_dict['A'].update(r['A'])
    combined_dict['l_req'] = dict()
    for r in c:
        combined_dict['l_req'].update(r['l_req'])
    combined_dict['in_rate'] = dict()
    for r in c:
        combined_dict['in_rate'].update(r['in_rate'])

    #print "PLACEMENT INPUT:", combined_dict
    #~ 
    #placementInput = PlacementInput(combined_dict)
    placementInput = PlacementInput(dict(combined_dict))
    
    g = pickle.load(open("netGraph_" + netname + ".pickle","r"))

    # network = NetworkData()

    result = Placement(num, chosenCombsToPlace, dict(g), placementInput, objective_mode).solve()
    name = str(combined_dict['U_pairs'])
    #print 'RESULT'
    # for r in result.iteritems():
    #     #print r
    results[name] = result

pickle.dump(results,open("optimizationResults/" + resultFile + ".pickle","wb"))

result = open("optimizationResults/" + resultFile + ".log", "w")
for r in request_list:
    result.write(str(r) + "\n\n")
    
    

