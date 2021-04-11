import pickle 
from optimization.create_network import CreateNetworkGraph
import logging
import sys


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# netName = sys.argv[1]


class NetworkData:
	def __init__(self, netName = None):
		self.netName = netName

		# Available types of VNFs
		self.F = {'FW','CACHE','DPI','PCTL','WAPGW','VOPT','HHE','IDS','LB','PRX','AV','WOPT','IPS'}

		# Number of available instances for each VNF type
		#~ self.n_ins = {'FW':9,'CACHE':10,'DPI':6,'PCTL':5,'WAPGW':5,'VOPT':4,'HHE':5,'IDS':10,'LB':20,'BNG':100,'CR':100,'REG':100,'SRV':100,'PRX':100,'AV':20,'WOPT':20,'IPS':10}
		self.n_ins = {'FW':100,'CACHE':100,'DPI':100,'PCTL':100,'WAPGW':100,'VOPT':100,'HHE':100,'IDS':100,'LB':100,'BNG':100,'CR':100,'REG':100,'SRV':100,'PRX':100,'AV':100,'WOPT':100,'IPS':100}

		# Number of usage requests that each instance of every VNF type can handle
		#~ self.n_use = {'FW':5,'CACHE':10,'DPI':5,'PCTL':5,'WAPGW':5,'VOPT':5,'HHE':5,'IDS':5,'LB':5,'BNG':100,'CR':100,'GGSN':100,'WWW':100,'REG':100,'SRV':100,'PRX':100,'AV':100,'WOPT':100,'IPS':5}
		self.n_use = {'FW':1,'CACHE':1,'DPI':1,'PCTL':1,'WAPGW':1,'VOPT':1,'HHE':1,'IDS':1,'LB':1,'BNG':100,'CR':100,'GGSN':100,'WWW':100,'REG':100,'SRV':100,'PRX':1,'AV':1,'WOPT':1,'IPS':1}
		
		# Required computational power of each intance of VNF types when mapped
		# to a data center node 
		#~ self.p_d = {'FW':50,'CACHE':35,'DPI':40,'PCTL':55,'WAPGW':55,'VOPT':70,'HHE':5,'IDS':10,'LB':45,'BNG':0,'CR':0,'GGSN':0,'WWW':0,'REG':0,'SRV':0,'PRX':5,'AV':10,'WOPT':10,'IPS':10}
		self.c_req = {'FW':0.55,'CACHE':0.3,'DPI':0.8,'PCTL':2.0,'WAPGW':0.5,'VOPT':0.7,'HHE':0.5,'IDS':0.6,'LB':0.3,'BNG':0,'CR':0,'GGSN':0,'WWW':0,'REG':0,'SRV':0,'PRX':0.5,'AV':0.7,'WOPT':0.6,'IPS':0.6}
		#~ self.c_req = {'FW':1,'CACHE':1,'DPI':1,'PCTL':1,'WAPGW':1,'VOPT':1,'HHE':1,'IDS':1,'LB':1,'BNG':0,'CR':0,'GGSN':0,'WWW':0,'REG':0,'SRV':0,'PRX':1,'AV':1,'WOPT':1,'IPS':1}
		
		# Required computational power of each instance of VNF types when mapped
		# to a switch node 
		#~ self.p_s = {'FW':11,'CACHE':0,'DPI':20,'PCTL':0,'WAPGW':0,'VOPT':0,'HHE':0,'IDS':20,'LB':10,'BNG':0,'CR':0,'GGSN':0,'WWW':0,'REG':0,'SRV':0,'PRX':0,'AV':0,'WOPT':0,'IPS':2}

	def makeNetwork(self):
		if self.netName == None:
			self.netName = sys.argv[1]


		# networkCreator = CreateNetworkGraph(inputFile)
		networkCreator = CreateNetworkGraph("./tops/tmp/sndlib_" + str(self.netName) + ".nxgraph.pickle")
		logger.debug("Creating substrate network...")

		self.V, self.c = networkCreator.getNodes()
		self.E, self.d, self.l = networkCreator.getEdges()
		logger.debug("c %s", self.c)

		self.g = dict()
		self.g['nodes'] = list(self.V)
		self.g['edges'] = list(self.E)
		self.g['nodeCap'] = dict(self.c)
		self.g['edgeDatarate'] = dict(self.d)
		self.g['edgeLatency'] = dict(self.l)
		self.g['F'] = list(self.F)
		self.g['n_ins'] = dict(self.n_ins)
		self.g['n_use'] = dict(self.n_use)
		self.g['c_req'] = dict(self.c_req)

	def exportNetwork(self):
		pickle.dump(self.g,open("netGraph_" + str(self.netName) + ".pickle","wb"))
		logger.debug("g %s", self.g)
		print (self.g)


if __name__ == '__main__':
	net = NetworkData()
	net.makeNetwork()
	net.exportNetwork()
