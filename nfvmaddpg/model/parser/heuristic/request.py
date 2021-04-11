
class Request:
	def __init__(self, req):
		# Usage requests for an instance of a VNF type and the VNF type they require
		self.UF = req['UF']	

		# Usage requests
		self.U = self.UF.keys()

		# Ratio of the outgoing data rate to incoming data rate for each branch 
		# of each VNF instance that is requested
		self.r = req['r']

		# Application tier instances and the nodes they're mapped to
		self.A = req['A']
		
		# Maximum acceptable latency between the application tiers
		self.l_req = req['l_req']

		# Incoming data rate to the entrance point of the requested chain
		self.input_datarate = req['input_datarate']

		# Requested chain of VM and VNF instances
		self.chain = req['chain']

		# The set of permutations that should be used for the optional
		# orders in the request, if any
		#self.forceOrder = dict()
	
	def __str__(self):
		return "REQUEST:\n" + "UF: " + str(self.UF) + "\n" + "r: " + str(self.r) + "\n" + "A: " + str(self.A) + "\n" + "l_req: " + str(self.l_req) + "\n" + "Input datarate: " + str(self.input_datarate) + "\n" + "Chain: " + str(self.chain)

	def add_prefix(self, i):
		for uf in self.UF.keys():
			self.UF["req" + str(i) + "_" + uf] = self.UF.pop(uf)
		for x,u in enumerate(self.U):
			self.U[x] = "req" + str(i) + "_" + u
		for r in self.r.keys():
			self.r["req" + str(i) + "_" + r] = self.r.pop(r)
		for a in self.A.keys():
			self.A["req" + str(i) + "_" + a] = self.A.pop(a)
		for (x,y) in self.l_req.keys():
			self.l_req[("req" + str(i) + "_" + x, "req" + str(i) + "_" + y)] = self.l_req.pop((x,y))
		self.chain = self.chain.replace("u", "req" + str(i) + "_u")
		self.chain = self.chain.replace("a", "req" + str(i) + "_a")
