from __future__ import division
import math
from collections import defaultdict
from gurobipy import *
from datetime import datetime

class Placement:
	def __init__(self, combnum, chosencombs, network, data, objective_mode='lex', bounds={}):
		self.tsi = datetime.now()
		self.combinationNumber = combnum
		self.chosenCombsToPlace = chosencombs
		# self.network = network
		self.data = data
		self.objective_mode = objective_mode

		self.G = network

		# Network graph
		# self.network = dict(network)
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
		
		self.model = Model()
		self.model.setParam("Threads", 1)
		self.model.setParam("NodefileStart", 0.5)
		self.model.setParam("LogToConsole", 1)
		#self.model.setParam("OutputFlag", 0)
		#self.model.setParam("Presolve", 0)
		#self.model.setParam("Aggregate", 0)
		#self.model.setParam("FeasibilityTol", 1e-9)
		
		#self.model.setParam("IntFeasTol",1e-9)
		#self.model.setParam("PSDTol",0)

		#self.bigM = 100000
		self.bigM = 1000

		# Sum of the latencies for all edges
		lat_sum = 0
		maxdr = 0
		for (a,b) in self.E:
			lat_sum += self.L[(a,b)]
			if a != b:
				if self.D[(a,b)] > maxdr:
					maxdr = self.D[(a,b)]
		print (maxdr)
			
		# Number of edges except the self-loops	
		edge_count = len(self.E) - len(self.V)
		
		# Number of pairs to be placed
		pair_count = len(self.data.U_pairs)


		# Extract simple paths out of pairs, for latency calculations
		def findAllPathsFrom(self, node, tmppaths, tmp):  # 深度优先遍历 找到每个简单的路径
			tmp.append(node)
			
			if len(nodes[node]) == 0:
				tmppaths.append(tmp)
				return tmppaths
			
			for n in nodes[node]:
				newtmp = list(tmp)
				findAllPathsFrom(self, n, tmppaths, newtmp)
			#return []
			return tmppaths
			
		def makePairsFromList(self, plist):
			print ("PLIST",plist)
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
			nodes[end] = []   # 每个usage后面要跟的usage

		# count paths
		path_count = 0

		self.paths = dict()
		for (start, end) in self.data.l_req.keys():
			tmp = []
			tmppaths = []
			findAllPathsFrom(self, start, tmppaths, tmp)
			print ("XXX",tmppaths)
			realpaths = []
			for tp in tmppaths:
				if end in tp:
					realpaths.append(tp)
			#self.paths[(start, end)] = tmppaths
			self.paths[(start, end)] = realpaths
			path_count += len(tmppaths)
		#~ print "PATHS", self.paths
		
		# count pathpairs
		pathpair_count = 0
		
		self.pathPairs = dict()
		print ("PLIST1",self.data.l_req.keys())
		for k in self.data.l_req.keys():
			self.pathPairs[k] = []
			print ("PLIST2",self.paths[k])
			for p in self.paths[k]:
				ps = makePairsFromList(self, p)
				self.pathPairs[k].append(ps)
			pathpair_count += len(self.pathPairs[k])

		#~ print "PATHPAIRS", self.pathPairs
		
		for u in self.data.U:
			self.FINAL_C_REQ[u] = self.data.in_rate[u] * self.C_REQ[self.data.UF[u]]
		print ("TOTALC_REQ", self.FINAL_C_REQ)
		
		print ("PATHPAIRS", self.pathPairs)

		#1 / 0

		### Variables ###
		#################

		print ("Variables...")
		
		"""
		# e[a,b,x,y,u1,u2] is a binary variable that shows if edge (a,b) belongs
		# to the path between nodes x and y where usage requests u1 and f2 are
		# mapped to
		"""
		self.e = dict()
		for a in self.V:
			for b in self.V:
				if (a,b) in self.E:
					for x in self.V:
						for y in self.V:
							for (u1,u2) in self.data.U_pairs:
								self.e[a,b,x,y,u1,u2] = self.model.addVar(vtype = GRB.BINARY, name = "e_%s_%s_%s_%s_%s_%s" % (a,b,x,y,u1,u2))

		"""
		# m[u,v] is a binary variable that shows if the usage request u is mapped to the node v
		# ms[u,v] is a binary variable that shows usage request u is mapped to node v as a switch function
		# md[u,v] is a binary variable that shows usage request u is mapped to node v as a data center function
		# mms[u,v] is a helper variable that shows usage request u is mapped to v and is mapped as a switch function
		# mmd[u,v] is a helper variable that shows usage request u is mapped to v and is mapped as a data center function55555555555555555555
		"""
		#~ self.m = dict()
		#~ for u in self.data.U:
			#~ for v in self.V:
				#~ self.m[u,v] = self.model.addVar(vtype = GRB.BINARY, name = "m_%s_%s" % (u,v))

		self.m = dict()
		#~ self.ms = dict()
		#~ self.md = dict()
		#~ self.mms = dict()
		#~ self.mmd = dict()
		for u in self.data.U:
			for v in self.V:
				self.m[u,v] = self.model.addVar(vtype = GRB.BINARY, name = "m_%s_%s" % (u,v))
				#~ if self.p_d[self.data.UF[u]] <> 0 and self.p_s[self.data.UF[u]] <> 0:
					#~ self.ms[u,v] = self.model.addVar(vtype = GRB.BINARY, name = "ms_%s_%s" % (u,v))
					#~ self.md[u,v] = self.model.addVar(vtype = GRB.BINARY, name = "md_%s_%s" % (u,v))
				#~ elif self.p_d[self.data.UF[u]] == 0:
					#~ self.ms[u,v] = 1
					#~ self.md[u,v] = 0
				#~ else:
					#~ self.ms[u,v] = 0
					#~ self.md[u,v] = 1
				#~ self.mms[u,v] = self.model.addVar(vtype = GRB.BINARY, name = "mms_%s_%s" % (u,v))
				#~ self.mmd[u,v] = self.model.addVar(vtype = GRB.BINARY, name = "mmd_%s_%s" % (u,v))


		"""
		# i[f,v] is a binary variable that shows if node v contains an
		# instance of VNF type f
		"""
		self.i = dict()
		for v in self.V:
			for f in self.F:
				self.i[f,v] = self.model.addVar(vtype = GRB.BINARY, name = "i_%s_%s" % (f,v))

		"""
		# q[x,y,u1,u2] is a helper variable that shows if usage requests u1 and 
		# u2 are mapped to nodes x and y respectively. q_inv is the inverse of q
		"""
		self.q = dict()
		self.q_inv = dict()
		for x in self.V:
			for y in self.V:
				for (u1,u2) in self.data.U_pairs:
					self.q[x,y,u1,u2] = self.model.addVar(vtype = GRB.BINARY, name = "q_%s_%s_%s_%s" % (x,y,u1,u2))
					self.q_inv[x,y,u1,u2] = self.model.addVar(vtype = GRB.BINARY, name = "q_inv_%s_%s_%s_%s" % (x,y,u1,u2))
					
		"""
		# lat[u1,u2] is the latency of the path between u1 and u2
		"""
		self.lat = dict()
		for (u1,u2) in self.data.U_pairs:
			self.lat[u1,u2] = self.model.addVar(vtype = GRB.CONTINUOUS, lb = 0, ub = 100, name = "lat_%s_%s" % (u1,u2))
			

		"""
		# used[v] shows if request is mapped to node v
		# util[v] shows the utilization of node v
		"""
		self.used = dict()
		self.util = dict()
		self.remaining_node_cap = dict()
		for v in self.V:
			self.used[v] = self.model.addVar(vtype = GRB.BINARY, name = "used_%s" % (v))
			self.util[v] = self.model.addVar(vtype = GRB.CONTINUOUS, lb = 0, ub = 1, name = "util_%s" % (v))
			self.remaining_node_cap[v] = self.model.addVar(vtype = GRB.CONTINUOUS, name = "remaining_node_cap_%s" % (v))
		"""
		# remaining_datarate[a,b] shows the amount of free link capacity on link (a,b)
		# datarate_util[a,b] shows the utilization of link (a,b)
		"""
		self.remaining_datarate = dict()
		for (a,b) in self.E:
			self.remaining_datarate[a,b] = self.model.addVar(vtype = GRB.CONTINUOUS, name = "remaining_datarate_%s_%s" % (a,b))
			
		self.datarate_util = dict()
		for (a,b) in self.E:
			self.datarate_util[a,b] = self.model.addVar(vtype = GRB.CONTINUOUS, lb = 0, ub = 1, name = "datarate_util_%s_%s" % (a,b))
			
		self.model.update()

		for u in self.data.U:
			sos = []
			for v in self.V:
				sos.append(self.m[u,v])
			self.model.addSOS(GRB.SOS_TYPE1,sos)

		for (u1,u2) in self.data.U_pairs:
			sos = []
			for x in self.V:
				for y in self.V:
					sos.append(self.q[x,y,u1,u2])
			self.model.addSOS(GRB.SOS_TYPE1,sos)

		### Constraints ###
		###################
		
		print ("Constraints..."	)
				
		
		"""
		# q[x,y,u1,u2] should be set to 1 iff u1 is mapped to x and u2 to y, and
		# q_inv should be set to 1-q 
		"""
		for x in self.V:
			for y in self.V:
				for (u1,u2) in self.data.U_pairs:
					 self.model.addConstr(self.q[x,y,u1,u2], GRB.EQUAL, self.m[u1,x] * self.m[u2,y])
					 
					 self.model.addConstr(self.q_inv[x,y,u1,u2] + self.q[x,y,u1,u2], GRB.EQUAL, 1)
					 
		"""
		x
		# Latency of the path between every two usage request u1 and u2 is
		# the sum of latencies of all edges that build the path
		"""			 
		for (u1,u2) in self.data.U_pairs:
			self.model.addConstr(self.lat[u1,u2], GRB.EQUAL, quicksum(self.e[a,b,x,y,u1,u2] * self.L[(a,b)] for (a,b) in self.E for x in self.V for y in self.V))

		"""
		x
		# A node should be marked as used if there is at least one instance 
		# of some VNF type running on it 
		"""
		for v in self.V:
			self.model.addConstr(quicksum(self.i[f,v] for f in self.F), GRB.LESS_EQUAL, self.bigM * self.used[v])
			
			self.model.addConstr(self.used[v], GRB.LESS_EQUAL, quicksum(self.i[f,v] for f in self.F))
			
		"""
		x
		# The remaining data rate on each network link is calculated by 
		# subtracting the passing data rate from the maximum available 
		# data rate on that link
		"""	
		for (a,b) in self.E:
			self.model.addConstr(self.remaining_datarate[a,b], GRB.EQUAL, self.D[(a,b)] - quicksum(self.e[a,b,x,y,u1,u2] * self.data.d_req[(u1,u2)] for (u1,u2) in self.data.U_pairs for x in self.V for y in self.V)) 

		"""
		# Data rate utilization on each network link is calculated by 
		# dividing the passing data rate by initial data rate of that 
		# link
		"""
		for (a,b) in self.E:
			self.model.addConstr(self.datarate_util[a,b], GRB.EQUAL, quicksum(self.e[a,b,x,y,u1,u2] * self.data.d_req[(u1,u2)] for (u1,u2) in self.data.U_pairs for x in self.V for y in self.V) * (1 / self.D[(a,b)]))
		
		""" 
		x
		# There should be an instance of VNF type f on a node v, if there is 
		# at least one request for VNF type f on node v
		"""
		for f in self.F:
			for v in self.V:
				self.model.addConstr(quicksum(self.m[u,v] for u in self.data.U if self.data.UF[u] == f), GRB.LESS_EQUAL, self.bigM * self.i[f,v])
				
				self.model.addConstr(self.i[f,v], GRB.LESS_EQUAL, quicksum(self.m[u,v] for u in self.data.U if self.data.UF[u] == f))
				
		"""		
		x	
		# Each VNF usage request is mapped to exactly 1 node			 
		"""
		for u in self.data.U:
			self.model.addConstr(quicksum(self.m[u,v] for v in self.V), GRB.EQUAL, 1)

		"""
		x
		# There are a limited number of instances available for each VNF type
		"""
		for f in self.F:
			self.model.addConstr(quicksum(self.i[f,v] for v in self.V), GRB.LESS_EQUAL, self.N_INS[f])
			
		"""
		x	
		# Each VNF instance can handle a limited number of usage requests
		"""
		for v in self.V:
			for f in self.F:
				self.model.addConstr(quicksum(self.m[u,v] for u in self.data.U if self.data.UF[u] == f), GRB.LESS_EQUAL, self.N_USE[f])

		"""	
		x
		# Computational resource requirements of all requests mapped to a node should
		# be less than or equal to the available computational resources of that node
		"""
		#~ for v in self.V:
			#~ self.model.addConstr(quicksum(self.m[u,v] * self.p_d[self.data.UF[u]] for u in self.data.U), GRB.LESS_EQUAL, self.c_d[v])
			#~ 
			#~ self.model.addConstr(quicksum(self.m[u,v] * self.p_s[self.data.UF[u]] for u in self.data.U), GRB.LESS_EQUAL, self.c_s[v])
			
		#~ for u in self.data.U:
			#~ for v in self.V:
				#~ self.model.addConstr(self.ms[u,v] + self.md[u,v], GRB.EQUAL, 1)
				#~ self.model.addConstr(self.md[u,v] * self.m[u,v], GRB.EQUAL, self.mmd[u,v])
				#~ self.model.addConstr(self.ms[u,v] * self.m[u,v], GRB.EQUAL, self.mms[u,v])
				
		for v in self.V:
			#~ self.model.addConstr(quicksum(self.mmd[u,v] * self.p_d[self.data.UF[u]] for u in self.data.U), GRB.LESS_EQUAL, self.c_d[v])
			#~ self.model.addConstr(quicksum(self.m[u,v] * self.C_REQ[self.data.UF[u]] * self.data.in_rate[u] for u in self.data.U), GRB.LESS_EQUAL, self.C[v])
			self.model.addConstr(quicksum(self.m[u,v] * self.FINAL_C_REQ[u] for u in self.data.U), GRB.LESS_EQUAL, self.C[v])
			
			#~ self.model.addConstr(quicksum(self.mms[u,v] * self.p_s[self.data.UF[u]] for u in self.data.U), GRB.LESS_EQUAL, self.c_s[v])
			
			#self.model.addConstr(quicksum(self.m[u,v] * self.p_d[self.data.UF[u]] for u in self.data.U), GRB.LESS_EQUAL, 
			#self.c_d[v])
			
			#self.model.addConstr(quicksum(self.m[u,v] * self.p_s[self.data.UF[u]] for u in self.data.U), GRB.LESS_EQUAL, 
			#self.c_s[v])
			
		"""
		# Utilization of nodes
		"""
		for v in self.V:
			#~ if self.c_d[v] <> 0:
				#~ self.model.addConstr(quicksum(self.mmd[u,v] * self.p_d[self.data.UF[u]] for u in self.data.U) * (1 / self.c_d[v]), GRB.EQUAL, self.util[v])
			#~ self.model.addConstr(quicksum(self.m[u,v] * self.C_REQ[self.data.UF[u]] * self.data.in_rate[u] for u in self.data.U) * (1 / self.C[v]), GRB.EQUAL, self.util[v])
			self.model.addConstr(quicksum(self.m[u,v] * self.FINAL_C_REQ[u] for u in self.data.U) * (1 / self.C[v]), GRB.EQUAL, self.util[v])
			#~ elif self.c_s[v] <> 0:
				#~ self.model.addConstr(quicksum(self.mms[u,v] * self.p_s[self.data.UF[u]] for u in self.data.U) * (1 / self.c_s[v]), GRB.EQUAL, self.util[v])

		"""
		# Remaining node capacities
		"""
		for v in self.V:
			self.model.addConstr(self.C[v] - quicksum(self.m[u,v] * self.FINAL_C_REQ[u] for u in self.data.U), GRB.EQUAL, self.remaining_node_cap[v])

		"""	
		x
		# An edge can belong to a path between two nodes, only if there are VNF
		# usage requests mapped to those nodes and a path needs to be created 
		# between them.
		"""
		for (a,b) in self.E:
			for x in self.V:
				for y in self.V:
					for (u1,u2) in self.data.U_pairs:
						self.model.addConstr(self.e[a,b,x,y,u1,u2], GRB.LESS_EQUAL, self.m[u1,x] * self.m[u2,y])
			
		"""		
		x			
		# Total latency of all paths created between two application tiers  
		# should not exceed the maximum latency limit of that chain
		"""
		for (a1,a2) in self.data.l_req:
			for path in self.pathPairs[(a1,a2)]:
				self.model.addConstr(quicksum(self.e[a,b,x,y,u1,u2] * self.L[(a,b)] for (a,b) in self.E for x in self.V for y in self.V for (u1,u2) in path), GRB.LESS_EQUAL, self.data.l_req[(a1,a2)])
			
		"""	
		x
		# Sum of required data rates of all logical links mapped to an edge 
		# should be less than or equal to the available data rate on that edge
		"""
		for (a,b) in self.E:
			self.model.addConstr(quicksum(self.e[a,b,x,y,u1,u2] * self.data.d_req[(u1,u2)] for (u1,u2) in self.data.U_pairs for x in self.V for y in self.V), GRB.LESS_EQUAL, self.D[(a,b)])

		"""
		x
		# Flows should start at a node
		"""
		for (u1,u2) in self.data.U_pairs:
			self.model.addConstr(quicksum(self.e[x,b,x,y,u1,u2] * self.q[x,y,u1,u2] for y in self.V for (x,b) in self.E), GRB.EQUAL, 1)

			self.model.addConstr(quicksum(self.e[x,b,x,y,u1,u2] * self.q_inv[x,y,u1,u2] for y in self.V for (x,b) in self.E), GRB.EQUAL, 0)

		"""
		x	
		# Flows should end in a node
		"""
		for (u1,u2) in self.data.U_pairs:
			self.model.addConstr(quicksum(self.e[a,y,x,y,u1,u2] * self.q[x,y,u1,u2] for x in self.V for (a,y) in self.E), GRB.EQUAL, 1) 

			self.model.addConstr(quicksum(self.e[a,y,x,y,u1,u2] * self.q_inv[x,y,u1,u2] for x in self.V for (a,y) in self.E), GRB.EQUAL, 0) 
			
		"""	
		x
		# Flows should be preserved while traversing nodes
		"""
		for (u1,u2) in self.data.U_pairs:
			for v in self.V:
				for x in self.V:
					for y in self.V:
						self.model.addConstr(quicksum(self.e[a,v,x,y,u1,u2] for a in self.V if (a,v) in self.E if v != y), GRB.EQUAL, quicksum(self.e[v,b,x,y,u1,u2] for b in self.V if (v,b) in self.E if v != x))

		"""
		x
		# Prevent getting stuck in self-loops if the start and end nodes of the 
		# path are different
		"""
		for (u1,u2) in self.data.U_pairs:
			for v in self.V:
				for x in self.V:
					for y in self.V:	
						if x != y:
							self.model.addConstr(self.e[v,v,x,y,u1,u2], GRB.EQUAL, 0)
			
		"""				
		x
		# Prevent loops 				
		"""
		for (u1,u2) in self.data.U_pairs:
			for (a,b) in self.E:
				if (b,a) in self.E and a != b:
					for x in self.V:
						for y in self.V:
							self.model.addConstr(self.e[a,b,x,y,u1,u2] + self.e[b,a,x,y,u1,u2], GRB.LESS_EQUAL, 1)
		
		"""
		x
		# Map the start and end points of the chain to the requested nodes
		"""
		for a in self.data.A:
			self.model.addConstr(self.m[a, self.data.A[a]], GRB.EQUAL, 1)
					
		### Objective ###
		#################

		#self.model.setObjective(quicksum(quicksum(m[f,v] * p_d[f] for f in F) + quicksum(m[f,v] * p_s[f] for f in F) for v in V), GRB.MAXIMIZE)
		
		#self.model.setObjective(quicksum(quicksum(self.e[a,b,x,y,u1,u2] * self.L[(a,b)] for (a,b) in self.E for x in self.V for y in self.V) for (u1,u2) in self.data.U_pairs), GRB.MINIMIZE)

		#self.model.setObjective(quicksum(self.lat[u1,u2] for (u1,u2) in self.data.U_pairs), GRB.MINIMIZE)

		#self.model.setObjective(quicksum(self.passed[(a,b)] for (a,b) in self.E), GRB.MAXIMIZE)
		
		#~ self.model.setObjective(quicksum(self.remaining_datarate[a,b] * self.passed[(a,b)] for (a,b) in self.E if a <> b), GRB.MAXIMIZE)

		#~ self.model.setObjective(quicksum(self.lat[u1,u2] for path in (u1,u2) in self.pathPairs[(a1,a2)] for (a1,a2) in self.data.l_req), GRB.MINIMIZE)

		# Utilization, latency
		#self.model.setObjective(1000 * quicksum(self.used[v] for v in self.V) + quicksum(self.lat[u1,u2] for (u1,u2) in self.data.U_pairs), GRB.MINIMIZE)

		# Utilization, latency, data rate
		#self.model.setObjective((10000 * quicksum(self.used[v] for v in self.V) + 1000 * quicksum(self.lat[u1,u2] for (u1,u2) in self.data.U_pairs) - quicksum(self.passed[(a,b)] for (a,b) in self.E)), GRB.MINIMIZE)
		
		# Used nodes, data rate, latency
		#~ self.model.setObjective((100000 * quicksum(self.used[v] for v in self.V) - quicksum(self.remaining_datarate[a,b] * self.passed[(a,b)] for (a,b) in self.E if a <> b) + quicksum(self.lat[u1,u2] for (u1,u2) in self.data.U_pairs)), GRB.MINIMIZE)
		
		#~ self.mode.setObjective((10000 * quicksum(self.remaining_datarate[a,b] * self.passed[(a,b)] for (a,b) in self.E if a <> b) - quicksum(self.used[v] for v in self.V)), GRB.MAXIMIZE)
		
		#wlat = 1
		#wdr = wlat * pair_count * lat_sum
		#wnode = wdr * maxdr * edge_count
		#self.model.setObjective(wnode * quicksum(self.used[v] for v in self.V)
		#						- wdr * quicksum(self.remaining_datarate[a,b] for (a,b) in self.E if a <> b)
		#						+ wlat * quicksum(quicksum(quicksum(self.lat[u1,u2] for (u1,u2) in path) for path in self.pathPairs[(a1,a2)]) for (a1,a2) in self.data.l_req), GRB.MINIMIZE) 
		
		# Minimize the total latency in the chain.
		if objective_mode == 'lat':
			self.model.setObjective(quicksum(quicksum(quicksum(self.lat[u1,u2] for (u1,u2) in path) for path in self.pathPairs[(a1,a2)]) for (a1,a2) in self.data.l_req), GRB.MINIMIZE)
		
		# Minimize the number of nodes that have a VNF instance running 
		elif objective_mode == 'use':
			self.model.setObjective(quicksum(self.used[v] for v in self.V), GRB.MINIMIZE)
		
		# Maximize the remaining data rate capacity in network links,  over all links except self-loops
		elif objective_mode == 'dr':
			self.model.setObjective(quicksum(self.remaining_datarate[a,b] for (a,b) in self.E if a != b), GRB.MAXIMIZE)

		# Combined objectives
		elif objective_mode == 'lex':
			wlat = 1
			wnode = wlat * pathpair_count * lat_sum
			wdr = wnode * len(self.V)
			self.model.setObjective(wdr * quicksum(self.remaining_datarate[a,b] for (a,b) in self.E if a != b)
									- wnode * quicksum(self.used[v] for v in self.V)
									- wlat * quicksum(quicksum(quicksum(self.lat[u1,u2] for (u1,u2) in path) for path in self.pathPairs[(a1,a2)]) for (a1,a2) in self.data.l_req), GRB.MAXIMIZE)

		# (1) minimize number of used nodes, (2) maximize latency 
		elif objective_mode == 'use-lat+':
			wlat = 1
			wnode = wlat * pathpair_count * lat_sum
			self.model.setObjective(wnode * quicksum(self.used[v] for v in self.V)
									- wlat * quicksum(quicksum(quicksum(self.lat[u1,u2] for (u1,u2) in path) for path in self.pathPairs[(a1,a2)]) for (a1,a2) in self.data.l_req), GRB.MINIMIZE)	

		# (1) minimize latency, (2) maximize number of used nodes
		elif objective_mode == 'lat-use+':
			wnode = 1
			wlat = wnode * len(self.V)
			self.model.setObjective(wlat * quicksum(quicksum(quicksum(self.lat[u1,u2] for (u1,u2) in path) for path in self.pathPairs[(a1,a2)]) for (a1,a2) in self.data.l_req)
									- wnode * quicksum(self.used[v] for v in self.V), GRB.MINIMIZE)	

		# Minimize (1) latency and (2) number of used nodes
		elif objective_mode == 'lat-use-':
			wnode = 1
			wlat = wnode * len(self.V)
			self.model.setObjective(wlat * quicksum(quicksum(quicksum(self.lat[u1,u2] for (u1,u2) in path) for path in self.pathPairs[(a1,a2)]) for (a1,a2) in self.data.l_req)
									+ wnode * quicksum(self.used[v] for v in self.V), GRB.MINIMIZE)	

		# Minimize (1) number of used nodes and (2) latency
		elif objective_mode == 'use-lat-':
			wlat = 1
			wnode = wlat * pathpair_count * lat_sum			
			self.model.setObjective(wlat * quicksum(quicksum(quicksum(self.lat[u1,u2] for (u1,u2) in path) for path in self.pathPairs[(a1,a2)]) for (a1,a2) in self.data.l_req)
									+ wnode * quicksum(self.used[v] for v in self.V), GRB.MINIMIZE)												

		# Maximize the remaining data rate capacity in network links, 
		# over all network links except self-loops and minimize the mean utilization
		# over all network nodes
		elif objective_mode == 'DR+util+':
			wutil = 1
			wdr = 10
			self.model.setObjective(wdr * quicksum(self.remaining_datarate[a,b] for (a,b) in self.E if a != b)
									+ wutil * quicksum(self.util[v] for v in self.V), GRB.MAXIMIZE)

		elif objective_mode == 'DR+util-':
			wutil = 1
			wdr = 10
			self.model.setObjective(wdr * quicksum(self.remaining_datarate[a,b] for (a,b) in self.E if a != b)
									- wutil * quicksum(self.util[v] for v in self.V), GRB.MAXIMIZE)

		elif objective_mode == 'dr+UTIL-':
			wutil = (len(self.E) - len(self.V)) * 200
			wdr = 1
			self.model.setObjective(- wdr * quicksum(self.remaining_datarate[a,b] for (a,b) in self.E if a != b)
									+ wutil * quicksum(self.util[v] for v in self.V), GRB.MINIMIZE)

		elif objective_mode == 'dr+UTIL+':
			wutil = (len(self.E) - len(self.V)) * 200
			wdr = 1
			self.model.setObjective(- wdr * quicksum(self.remaining_datarate[a,b] for (a,b) in self.E if a != b)
									+ wutil * quicksum(self.util[v] for v in self.V), GRB.MAXIMIZE)
		elif objective_mode == 'maxutil-':
			self.model.setObjective(max(self.util[v] for v in self.V), GRB.MINIMIZE)

		elif objective_mode == 'mindr+':
			self.model.setObjective(min(self.remaining_datarate[a,b] for (a,b) in self.E if a != b), GRB.MAXIMIZE)

		elif objective_mode == 'MINDR+maxutil-':
			wdr = 1 / ((len(self.E) - len(self.V)) * 200)
			self.model.setObjective(10 * wdr * min(self.remaining_datarate[a,b] for (a,b) in self.E if a != b)
									- max(self.util[v] for v in self.V), GRB.MAXIMIZE)

		elif objective_mode == 'mindr+maxutil-':
			wdr = 1 / ((len(self.E) - len(self.V)) * 200)
			self.model.setObjective(0.5 * wdr * min(self.remaining_datarate[a,b] for (a,b) in self.E if a != b)
									- 0.5 * max(self.util[v] for v in self.V), GRB.MAXIMIZE)
																		
		elif objective_mode == 'dr+maxutil-':
			wdr = 1 / ((len(self.E) - len(self.V)) * 200)
			self.model.setObjective(0.5 * wdr * quicksum(self.remaining_datarate[a,b] for (a,b) in self.E if a != b)
									- 0.5 * max(self.util[v] for v in self.V), GRB.MAXIMIZE)
		elif objective_mode == 'dr+maxutil+':
			wdr = 1 / ((len(self.E) - len(self.V)) * 200)
			self.model.setObjective(0.5 * wdr * quicksum(self.remaining_datarate[a,b] for (a,b) in self.E if a != b)
									+ 0.5 * max(self.util[v] for v in self.V), GRB.MAXIMIZE)
		elif objective_mode == 'maxnodeutil-':
			self.model.setObjective(max(self.util[v] for v in self.V), GRB.MINIMIZE)							
		elif objective_mode == 'maxlinkutil-':
			self.model.setObjective(max(self.datarate_util[a,b] for (a,b) in self.E if a != b), GRB.MINIMIZE)	
		#~ elif objective_mode == 'bothmaxutil-':
			#~ self.model.setObjective(0.5 * max(self.datarate_util[a,b] for (a,b) in self.E if a <> b)
									#~ + 0.5 * max(self.util[v] for v in self.V), GRB.MINIMIZE)	
		elif objective_mode == 'prodnodeutil':
			self.sumnodeutil = self.model.addVar(vtype = GRB.CONTINUOUS, name = "sumnodeutil")
			self.sumnodeused = self.model.addVar(vtype = GRB.CONTINUOUS, name = "sumnodeused")
			#~ self.sumnodeusedinv = self.model.addVar(vtype = GRB.CONTINUOUS, name = "sumnodeusedinv")
			self.model.update()
			self.model.addConstr(self.sumnodeutil, GRB.EQUAL, quicksum(self.util[v] for v in self.V))
			self.model.addConstr(self.sumnodeused, GRB.EQUAL, quicksum(self.used[v] for v in self.V))
			#~ self.model.addConstr(self.sumnodeusedinv, GRB.EQUAL, (1 / self.sumnodeused))
			self.model.update()
			self.model.setObjective(self.sumnodeutil * self.sumnodeused, GRB.MINIMIZE)
		elif objective_mode == 'varmaxnodeutil-':
			self.maxnodeutil = self.model.addVar(vtype = GRB.CONTINUOUS, name = "maxnodeutil")
			self.model.update()
			for v in self.V:
				self.model.addConstr(self.maxnodeutil, GRB.GREATER_EQUAL, self.util[v])
			self.model.setObjective(self.maxnodeutil, GRB.MINIMIZE)
		elif objective_mode == 'varmaxlinkutil-':
			self.maxlinkutil = self.model.addVar(vtype = GRB.CONTINUOUS, name = "maxlinkutil")
			self.model.update()
			for (a,b) in self.E:
				if a != b:
					self.model.addConstr(self.maxlinkutil, GRB.GREATER_EQUAL, self.datarate_util[a,b])
			self.model.setObjective(self.maxlinkutil, GRB.MINIMIZE)	
		elif objective_mode == 'bothmaxutil-':
			self.maxnodeutil = self.model.addVar(vtype = GRB.CONTINUOUS, name = "maxnodeutil")
			self.maxlinkutil = self.model.addVar(vtype = GRB.CONTINUOUS, name = "maxlinkutil")
			self.model.update()
			for v in self.V:
				self.model.addConstr(self.maxnodeutil, GRB.GREATER_EQUAL, self.util[v])
			for (a,b) in self.E:
				if a != b:
					self.model.addConstr(self.maxlinkutil, GRB.GREATER_EQUAL, self.datarate_util[a,b])
			self.model.setObjective(((0.5 * self.maxlinkutil) + (0.5 * self.maxnodeutil)), GRB.MINIMIZE)

							
		# Pareto
		elif objective_mode == 'pareto':
			if bounds['lat'] == -1:
				# minimize lat, bound use and dr
				self.model.addConstr(quicksum(self.used[v] for v in self.V), GRB.LESS_EQUAL, bounds['use'])
				self.model.addConstr(quicksum(self.remaining_datarate[a,b] for (a,b) in self.E if a != b), GRB.GREATER_EQUAL, bounds['dr'])
				
				self.model.setObjective(quicksum(quicksum(quicksum(self.lat[u1,u2] for (u1,u2) in path) for path in self.pathPairs[(a1,a2)]) for (a1,a2) in self.data.l_req), GRB.MINIMIZE)
				
			elif bounds['use'] == -1:
				# minimize use, bound lat and dr
				self.model.addConstr(quicksum(quicksum(quicksum(self.lat[u1,u2] for (u1,u2) in path) for path in self.pathPairs[(a1,a2)]) for (a1,a2) in self.data.l_req), GRB.LESS_EQUAL, bounds['lat'])
				self.model.addConstr(quicksum(self.remaining_datarate[a,b] for (a,b) in self.E if a != b), GRB.GREATER_EQUAL, bounds['dr'])
				
				self.model.setObjective(quicksum(self.used[v] for v in self.V), GRB.MINIMIZE)
			
			elif bounds['dr'] == -1:
				# maximize dr, bound lat and use
				self.model.addConstr(quicksum(quicksum(quicksum(self.lat[u1,u2] for (u1,u2) in path) for path in self.pathPairs[(a1,a2)]) for (a1,a2) in self.data.l_req), GRB.LESS_EQUAL, bounds['lat'])
				self.model.addConstr(quicksum(self.used[v] for v in self.V), GRB.LESS_EQUAL, bounds['use'])
				
				self.model.setObjective(quicksum(self.remaining_datarate[a,b] for (a,b) in self.E if a != b), GRB.MAXIMIZE)
				
		self.model.update()
		self.tei = datetime.now()
		
	"""
	# Check if the optimal solution was found without constraint violations
	"""
	def checkSolution(self):
		if self.model.status != GRB.status.OPTIMAL:
			return False
		if self.model.getAttr(GRB.attr.ConstrVio) > 0.1:
			print ("ConstrVio",self.model.getAttr(GRB.attr.ConstrVio))
			return False
		return True

	### Solve the optimization problem and print the results ###
	############################################################

	def solve(self):
		self.tss = datetime.now()
		print ("Optimize...")
		self.model.optimize()
		self.tes = datetime.now()

		results = {}
		
		td = (self.tei - self.tsi) + (self.tes - self.tss)
		results["time_ms"] = td.total_seconds()*1000

		results['combination_number'] = self.combinationNumber
		
		solved = self.checkSolution()
		
		results['solved'] = solved
		if not solved:
			return results
		
		print (str(self.model.getObjective()) + " value=1")
		
		#~ print "Edges used in paths between pairs of usage requests:"	
		results['edges'] = dict()	
		for (u1,u2) in self.data.U_pairs:
			results['edges'][(u1,u2)] = []
			for x in self.V:
				for y in self.V:
					for (a,b) in self.E:
						if int(round(self.model.getVarByName("e_%s_%s_%s_%s_%s_%s" % (a,b,x,y,u1,u2)).x)) == 1:
							#~ print str(self.model.getVarByName("e_%s_%s_%s_%s_%s_%s" % (a,b,x,y,u1,u2))) + " value=" + str(int(round(self.model.getVarByName("e_%s_%s_%s_%s_%s_%s" % (a,b,x,y,u1,u2)).x)))
							results['edges'][(u1,u2)].append((a,b))

		#~ print "Mapping of VNF instances to nodes:"
		results['vnf_to_node'] = dict()
		for f in self.F:
			results['vnf_to_node'][f] = []
			for v in self.V:
				if int(round(self.model.getVarByName("i_%s_%s" % (f,v)).x)) == 1:
					#~ print str(self.model.getVarByName("i_%s_%s" % (f,v))) + " value=" + str(int(round(self.model.getVarByName("i_%s_%s" % (f,v)).x)))
					results['vnf_to_node'][f].append(v)
		
		#~ print "Mapping of usage requests to nodes:"
		results['ureq_to_node'] = dict()		
		for u in self.data.U:
			results['ureq_to_node'][u] = []
			for v in self.V:
				if int(round(self.model.getVarByName("m_%s_%s" % (u,v)).x)) == 1:
					#~ print str(self.model.getVarByName("m_%s_%s" % (u,v))) + " value=" + str(int(round(self.model.getVarByName("m_%s_%s" % (u,v)).x)))
					results['ureq_to_node'][u].append(v)
		
		#~ print "Latency between pairs of usage requests:"	
		#~ latsum = 0	
		#~ for (u1,u2) in self.data.U_pairs:
			#~ print str(self.model.getVarByName("lat_%s_%s" % (u1,u2)))
			#~ latsum += self.model.getVarByName("lat_%s_%s" % (u1,u2)).x
		#~ results['mean_latency'] = latsum / len(self.data.U_pairs)
		#~ print "Latency for chains:"	
		latsum = 0	
		results['max_latency'] = dict()
		for (a1,a2) in self.data.l_req:
			chainlat = -1
			for path in self.pathPairs[(a1,a2)]:
				pathlat = 0
				for (u1,u2) in path:
					pathlat += self.model.getVarByName("lat_%s_%s" % (u1,u2)).x
				if pathlat > chainlat:
					chainlat = pathlat
				latsum += pathlat
			results['max_latency'][(a1,a2)] = chainlat
		results['mean_latency'] = latsum / len(self.data.l_req.keys())
		results['pareto_lat'] = latsum

		# Mean data rate over all edges except self-loops
		results['remaining_datarate'] = dict()
		results['datarate_util'] = dict()
		rdrsum = 0
		drutilsum = 0
		for (a,b) in self.E:
			if a != b:
				rdrsum += self.model.getVarByName("remaining_datarate_%s_%s" % (a,b)).x
				results['remaining_datarate'][(a,b)] = self.model.getVarByName("remaining_datarate_%s_%s" % (a,b)).x
				drutilsum += self.model.getVarByName("datarate_util_%s_%s" % (a,b)).x
				results['datarate_util'][(a,b)] = self.model.getVarByName("datarate_util_%s_%s" % (a,b)).x
		results['mean_remaining_datarate'] = rdrsum / (len(self.E) - len(self.V))
		results['sum_remaining_datarate'] = rdrsum
		results['mean_datarate_util'] = drutilsum / (len(self.E) - len(self.V))
		results['pareto_dr'] = rdrsum

		results['max_remaining_datarate'] = max(results['remaining_datarate'].values())
		results['min_remaining_datarate'] = min(results['remaining_datarate'].values())
		
		usednodes = 0
		for v in self.V:
			usednodes += int(round(self.model.getVarByName("used_%s" % (v)).x))
		results['used_nodes'] = usednodes
		
		# Number of instances of each VNF type that are used
		results['used_instances'] = 0
		results['num_instances'] = dict()
		for f in self.F:
			results['num_instances'][f] = 0
			for v in self.V:
				results['num_instances'][f] += int(round(self.model.getVarByName("i_%s_%s" % (f,v)).x))
				results['used_instances'] += int(round(self.model.getVarByName("i_%s_%s" % (f,v)).x))

		# Utilization
		sumutil = 0
		sumremnodecap = 0
		results['node_utilization'] = dict()
		results['remaining_node_cap'] = dict()
		for v in self.V:
			results['node_utilization'][v] = self.model.getVarByName("util_%s" % (v)).x
			results['remaining_node_cap'][v] = self.model.getVarByName("remaining_node_cap_%s" % (v)).x
			sumutil += self.model.getVarByName("util_%s" % (v)).x
			sumremnodecap += self.model.getVarByName("remaining_node_cap_%s" % (v)).x
		results['mean_utilization'] = sumutil / len(self.V)
		results['mean_remaining_node_cap'] = sumremnodecap / len(self.V)
		results['sum_remaining_node_cap'] = sumremnodecap

		results['max_remaining_node_cap'] = max(results['remaining_node_cap'].values())
		results['min_remaining_node_cap'] = min(results['remaining_node_cap'].values())


		results['num_usage_req'] = len(self.data.U) - len(self.data.A)
		results['num_network_nodes'] = len(self.V)
		results['num_network_edges'] = len(self.E)
		
		if self.objective_mode == 'varmaxnodeutil-':
			results['maxnodeutil'] = self.model.getVarByName("maxnodeutil").x
		if self.objective_mode == 'varmaxlinkutil-':
			results['maxlinkutil'] = self.model.getVarByName("maxlinkutil").x
		if self.objective_mode == 'bothmaxutil-':
			results['maxnodeutil'] = self.model.getVarByName("maxnodeutil").x
			results['maxlinkutil'] = self.model.getVarByName("maxlinkutil").x
		
		results['chosencombs'] = self.chosenCombsToPlace

		# Check which constraint makes the model infeasible, if it is infeasible
		#~ if self.model.status == GRB.status.INFEASIBLE:        
			#~ self.model.computeIIS()
			#~ print "IIS Contraints:"
			#~ for x in self.model.getConstrs():
				#~ if x.getAttr(GRB.attr.IISConstr) > 0:
					#~ print x.getAttr(GRB.attr.ConstrName),x.sense,x.rhs
			#~ print "IIS Variables:"
			#~ for x in self.model.getVars():
				#~ if x.getAttr(GRB.attr.IISLB) > 0 or x.getAttr(GRB.attr.IISUB) > 0:
					#~ print x.getAttr(GRB.attr.VarName)
	
		return results
