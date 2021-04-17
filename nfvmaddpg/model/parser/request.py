from copy import deepcopy


class Request:
	def __init__(self, req):
		# Usage requests for an instance of a VNF type and the VNF type they require
		self.UF = deepcopy(req['UF'])

		# Usage requests
		self.U = deepcopy(list(req['UF'].keys()))  # a1 a2 u1 u2
		# Ratio of the outgoing data rate to incoming data rate for each branch
		# of each VNF instance that is requested
		self.r = deepcopy(req['r'])

		# Application tier instances and the nodes they're mapped to
		self.A = deepcopy(req['A'])

		# Maximum acceptable latency between the application tiers
		self.l_req = deepcopy(req['l_req'])

		# Incoming data rate to the entrance point of the requested chain
		self.input_datarate = deepcopy(req['input_datarate'])

		# Requested chain of VM and VNF instances
		self.chain = deepcopy(req['chain'])

		self.prefix = ''

		self.optords = None

		# The set of permutations that should be used for the optional
		# orders in the request, if any
		#self.forceOrder = dict()
	
	def __str__(self):
		return "REQUEST:\n" + "UF: " + str(self.UF) + "\n" + "r: " + str(self.r) + "\n" + "A: " + str(self.A) + "\n" + "l_req: " + str(self.l_req) + "\n" + "Input datarate: " + str(self.input_datarate) + "\n" + "Chain: " + str(self.chain)

	def add_prefix(self, i):
		self.prefix = "req" + str(i)
		UF_tmp = {}
		for uf in list(self.UF.keys()):
			UF_tmp["req" + str(i) + "_" + uf] = self.UF.pop(uf)
		self.UF = UF_tmp
		for x,u in enumerate(self.U):
			self.U[x] = "req" + str(i) + "_" + u
		r_tmp = {}
		for r in list(self.r.keys()):
			r_tmp["req" + str(i) + "_" + r] = self.r.pop(r)
		self.r = r_tmp
		A_tmp = {}
		for a in list(self.A.keys()):
			A_tmp["req" + str(i) + "_" + a] = self.A.pop(a)
		self.A = A_tmp
		l_req_tmp = {}
		for (x,y) in list(self.l_req.keys()):
			l_req_tmp[("req" + str(i) + "_" + x, "req" + str(i) + "_" + y)] = self.l_req.pop((x,y))
		self.l_req = l_req_tmp
		self.chain = self.chain.replace("u", "req" + str(i) + "_u")
		self.chain = self.chain.replace("a", "req" + str(i) + "_a")
