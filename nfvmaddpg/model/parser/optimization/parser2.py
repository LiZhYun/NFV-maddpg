"""
Parser that parses different modules and combinations of them.
"""
from pyparsing import *
import sys, os
sys.path.append('./optimization')  
from placement_input import PlacementInput
from itertools import product, permutations
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Parser:
	def __init__(self, req):	
		self.optorderdict = dict()
			
		# Request data
		self.req = req
		# Data that will be passed from parser to placement
#		self.placementInput = PlacementInput()
		self.placementInput = dict()
		# A list to store the pairs of VM/VNF instances that should be placed
		# and connected to each other
		self.pairs = []
		# A dictionary to store the modules after parsing the request chain
		self.parseDict = dict()
		# Required data rate for pairs of VM and VNF instances
		self.d_req = dict()
		# A dictionary to store the total incoming data rate for each VM/VNF instance
		self.in_rate = dict()
		# A list that consists of all the usage requests and the VM instance 
		# where the request ends
		self.inst_queue = []
		# A list for storing the updated usage requests. Consists of the
		# input usage request list, plus the copies of usage requests that
		# are created while processing parallel modules
		self.U_out = list(self.req.U)
		# A dictionary for storing the updated mapping of usage requests to VNF
		# types. Consists of the input UF list, plus the mappings for copies
		# of usage requests that are created while processing parallel modules
		self.UF_out = self.req.UF.copy()
		# A dictionary for storing the updated ratios of usage requests to VNF
		# types. Consists of the input r list, plus the values for copies
		# of usage requests that are created while processing parallel modules		
		self.r_out = self.req.r.copy()

		# Grammar definition
#		self.instance = Word(alphas + "_" + nums)
		self.instance = Word(alphanums + "_" + alphanums)
		self.instance.setParseAction(self.parseInstance)
		
		self.number = Word(nums).setParseAction(lambda t: int(t[0]))

		self.module = Forward()
		self.order = Forward()
		self.optorder = Forward()
		self.split = Forward()
		self.parallel = Forward()

		self.module << ( self.optorder | self.split | self.parallel | self.instance )

		self.order <<  self.module + "."
		self.order.setParseAction(self.parseOrder)

		self.optorder << "(" + self.instance + ZeroOrMore("," + self.instance) + ")"
		self.optorder.setParseAction(self.parseOptorder)

		self.split << self.instance + "[" + ZeroOrMore(self.order) + self.module + ZeroOrMore("|" + ZeroOrMore(self.order) + self.module) + "]"
		self.split.setParseAction(self.parseSplit)

		self.parallel << self.instance + "{" + self.instance + ZeroOrMore("," + self.instance) + ";" + ZeroOrMore(self.order) + self.module + ";" + self.number + "}"
		self.parallel.setParseAction(self.parseParallel)
		
		self.start = ZeroOrMore(self.order) + self.module
		self.start.setParseAction(self.parseStart)

	"""
	setParseAction( *fn ) - specify one or more functions to call after successful matching of the element; each function is defined as fn( s, loc, toks ), where:
	s is the original parse string
	loc is the location in the string where matching started
	toks is the list of the matched tokens, packaged as a ParseResults_ object

	http://pyparsing.wikispaces.com/HowToUsePyparsing
	"""

	"""
	# For an Order module, store the VNF instance in parseDict and compute
	# the index in parseDict for the module that comes after this module
	"""
	def parseOrder(self, s, loc, toks):
		##print "Order", loc, toks
		ts = toks.copy()
		jump = loc + sum(len(str(t)) for t in ts)
		self.parseDict[loc].jump = jump
		return toks
		
	"""	
	# For an OptionalOrder module, store the functions in the dictionary and 
	# compute the index in parseDict for the module that comes after this module
	"""
	def parseOptorder(self, s, loc, toks):
		##print "OptionalOrder Module", loc, toks
		ts = toks.copy()
		funcs = []
		for i,t in enumerate(ts):
			if t != "(" and t != ")" and t != ",":
				funcs.append(loc + sum(len(s) for s in ts[0:i]))
		#jump = loc + sum(len(x) for x in ts) + 1
		jump = -1
		self.parseDict[loc] = storeOptorder(funcs, jump)
		return toks

	"""
	# For a Split module, store the splitting function in the dictionary along
	# with the module on each branch going out of the splitter function, and 
	# compute the index in parseDict for the module that comes after this module
	"""
	def parseSplit(self, s, loc, toks):
		##print "Split Module", loc, toks
		ts = toks.copy()
		func = ts[0]
		positions = []
		level = -1
		# Skip the branches of splits that might be defined inside this split
		for i,t in enumerate(ts):
			if t == "[":
				level += 1
			if t == "]":
				level -= 1
			if level == 0:
				if t == "|" or t == "[":
					positions.append(i + 1)

		mods = []
		for p in positions:
			tpos = 0
			for i in range(0,p):
				tpos += len(ts[i])
			mods.append(loc + tpos)
		#jump = loc + sum(len(str(x)) for x in ts) + 1
		jump = -1
		self.parseDict[loc] = storeSplit(func, mods, jump)
		return toks

	"""	
	# For a Parallel splitting module, store the splitting function and the 
	# optional functions that can have an optional order with the splitter
	# function, along with the module that should be replicated over a given
	# number of branches, and compute the index in parseDict for the module
	# that comes after this module
	"""
	def parseParallel(self, s, loc, toks):
		##print "Parallel Module", loc, toks
		ts = toks.copy()
		func = ts[0]
		delimiters = []
		level = -1
		# Skip the sections of parallels that might be defined inside this parallel
		for i,t in enumerate(ts):
			if t == "{":
				level += 1
			if t == "}":
				level -= 1
			if level == 0:
				if t == ";":
					delimiters.append(i)
		funcs = []
		x = 2
		while x < delimiters[0]:
			#funcs.append(ts[x])
			funcs.append(loc + sum(len(s) for s in ts[0:x]))
			x += 2
		mod = loc + sum(len(x) for x in ts[0 : delimiters[0] + 1])
		num = ts[delimiters[1] + 1]
		#jump = loc + sum(len(str(x)) for x in ts) + 1
		jump = -1
		self.parseDict[loc] = storeParallel(func, num, funcs, mod, jump)
		return toks

	"""
	# Store the individual functions
	"""
	def parseInstance(self, s, loc, toks):
		##print "Instance Module", loc, toks	
		ts = toks.copy()
		#jump = loc + len(ts[0]) + 1
		jump = -1
		if loc not in self.parseDict:
			self.parseDict[loc] = storeInstance(ts[0],jump)
		return toks

	"""	
	# Store the first and last element in the parsed request as the VMs that
	# are the beginning and end of the requested chain of VNFs
	"""
	def parseStart(self, s, loc, toks):
		##print "Request", loc, toks
		ts = toks.copy()
		begin = ts[0]
		end = ts[-1]
		# Jump value for the starting point of the chain
		jump = loc + 1 + len(ts[0])
		#jump = -1
		self.parseDict[loc] = storeRequest(begin, end, jump)
		return toks
		
	def reverseOptOrderTest(self, funcs):
		origfuncs = funcs
		revfuncs = list(reversed(origfuncs))
		tempPD = dict()
		for f in origfuncs:
			tempPD[f] = self.parseDict[f]
		for i,f in enumerate(origfuncs):
			self.parseDict[f] = tempPD[revfuncs[i]]
	
	"""
	# Sort the functions with optional order based on the ratios		
	"""
	def minDatarateOrder(self, funcs):
		origfuncs = funcs
		orderedfuncs = []
		funcqueue = list(funcs)
		while len(funcqueue) > 0:
			minratio = 1e9
			minfunc = None
			for f in funcqueue:
				#~ if self.req.r[self.parseDict[f].func][0] < minratio:
				if min(self.req.r[self.parseDict[f].func]) < minratio:
					#~ minratio = self.req.r[self.parseDict[f].func][0]
					minratio = min(self.req.r[self.parseDict[f].func])
					minfunc = f
			orderedfuncs.append(minfunc)
			funcqueue.remove(minfunc)
		tempPD = dict()
		for f in origfuncs:
			tempPD[f] = self.parseDict[f]
		for i,f in enumerate(origfuncs):
			self.parseDict[f] = tempPD[orderedfuncs[i]]
			
	"""
	# Use a specific order for the optional order functions
	"""
	def forceOrder(self, funcs):
		origfuncs = funcs
		key = ",".join(self.parseDict[f].func for f in origfuncs)
			
		forcefuncs = self.req.forceOrder[key]
		forder = []
		for f in forcefuncs:
			for g in origfuncs:
				if self.parseDict[g].func == f:
					forder.append(g)
		tempPD = dict()
		for f in origfuncs:
			tempPD[f] = self.parseDict[f]
		for i,f in enumerate(origfuncs):
			self.parseDict[f] = tempPD[forder[i]]
			
	"""
	# Process the optional order and parallel modules and create possible
	# orderings for the instances with optional order
	"""
	def fixOptionalOrders(self, begin, end):
		for pos in range(begin, end):
			if pos in self.parseDict:
				s = self.parseDict[pos]
				if isinstance(s, storeOptorder):
					# Do reordering here
					#self.reverseOptOrderTest(s.funcs)
					if hasattr(self.req, 'forceOrder'):
						self.forceOrder(s.funcs)
					else:
						self.minDatarateOrder(s.funcs)
					# Fix the jumps
					j = s.jump
					for f in reversed(s.funcs):
						self.parseDict[f].jump = j
						j = f
					s.jump = f
				if isinstance(s, storeParallel):
					# Do reordering
					#self.reverseOptOrderTest(s.funcs)
					if hasattr(self.req, 'forceOrder'):
						self.forceOrder(s.funcs)
					else:
						self.minDatarateOrder(s.funcs)
					# Fix the jumps
					j = s.mod
					for f in reversed(s.funcs):
						self.parseDict[f].jump = j
						j = f
	
	"""				
	# Assign correct jump values to modules and instances that could not get
	# correct values during parsing
	"""
	def fixNexts(self, jump, begin, end):
		pos = begin
		while pos < end:
			s = self.parseDict[pos]
			if isinstance(s, storeInstance) and s.jump == -1:
				s.jump = jump
				return
			if isinstance(s, storeSplit):
				if s.jump == -1:
					s.jump = jump
				for c in s.mods:
					self.fixNexts(s.jump, c, s.jump)
			if isinstance(s, storeParallel):
				#print "jjj",jump,s.jump
				if s.jump == -1:
					s.jump = jump
				self.fixNexts(s.jump, s.mod, s.jump)
			pos = s.jump
	
	"""
	# Create pairs of VNFs in the modules that starts at position 'pos' 
	# of parseDict. 'end' is the end of this module, 'prev' is the list
	# of elements that will build a pair with the starting element of
	# the current module. 'suffix' is used for building copies of instances.
	# 'rindex' is the index of the branch going out of the previous VNF 
	# that this pair is placed on. For everything other than VNFs that 
	# come after a splitter module every other pairs needs rindex = 0
	"""	
	def createPairs(self, pos, end, prev, suffix='', rindex=0):
		while pos < end and pos != -1:
			s = self.parseDict[pos]
			if isinstance(s, storeInstance):
				for k,p in enumerate(prev):
					# Add the newly created copies of usage requests to 
					# U and r and UF and remove the original one from U
					if s.func + suffix not in self.U_out:
						self.U_out.append(s.func + suffix)
						self.r_out[s.func + suffix] = self.req.r[s.func]
						self.UF_out[s.func + suffix] = self.req.UF[s.func]
						self.in_rate[s.func + suffix] = 0
						if s.func + suffix != s.func and s.func in self.U_out:
							self.U_out.remove(s.func)
					
					# Create the pair
					self.pairs.append((p, s.func + suffix))
					# Compute the data rate of the new pair
					self.d_req[(p, s.func + suffix)] = self.r_out[p][rindex] * self.in_rate[p]
					# Calculate the incoming data rate for every VNF that
					# has all its incoming pairs processed
					self.in_rate[s.func + suffix] += self.r_out[p][rindex] * self.in_rate[p]
				
					rindex = 0
				prev = [s.func + suffix]
			
			if isinstance(s, storeSplit):
				for k,p in enumerate(prev):
					# Add the newly created copies of usage requests to 
					# U and r and UF and remove the original one from U
					if s.func + suffix not in self.U_out:
						self.U_out.append(s.func + suffix)
						self.r_out[s.func + suffix] = self.req.r[s.func]
						self.UF_out[s.func + suffix] = self.req.UF[s.func]
						self.in_rate[s.func + suffix] = 0
						if s.func + suffix != s.func and s.func in self.U_out:
							self.U_out.remove(s.func)
					
					# Create the pair
					self.pairs.append((p, s.func + suffix))
					# Compute the data rate of the new pair
					self.d_req[(p, s.func + suffix)] = self.r_out[p][rindex] * self.in_rate[p]
					# Calculate the incoming data rate for every VNF that
					# has all its incoming pairs processed
					self.in_rate[s.func + suffix] += self.r_out[p][rindex] * self.in_rate[p]

					rindex = 0
				prevs = []
				
				for i,m in enumerate(s.mods):
					if m == s.mods[-1]:
						e = s.jump
					else:
						e = s.mods[i+1]
					p = self.createPairs(m, e, [s.func + suffix], suffix, i)
					for q in p:
						prevs.append(q)
				
				prev = prevs
			
			if isinstance(s, storeParallel):
				for p in prev:
					# Add the newly created copies of usage requests to 
					# U and r and UF and remove the original one from U
					if self.parseDict[s.funcs[0]].func + suffix not in self.U_out:
						self.U_out.append(self.parseDict[s.funcs[0]].func + suffix)
						self.r_out[self.parseDict[s.funcs[0]].func + suffix] = self.req.r[self.parseDict[s.funcs[0]].func]
						self.UF_out[self.parseDict[s.funcs[0]].func + suffix] = self.req.UF[self.parseDict[s.funcs[0]].func]
						self.in_rate[self.parseDict[s.funcs[0]].func + suffix] = 0
						if self.parseDict[s.funcs[0]].func + suffix != self.parseDict[s.funcs[0]].func and self.parseDict[s.funcs[0]].func in self.U_out:
							self.U_out.remove(self.parseDict[s.funcs[0]].func)
					
					# Create the pair
					self.pairs.append((p, self.parseDict[s.funcs[0]].func + suffix))
					# Compute the data rate of the new pair
					self.d_req[(p, self.parseDict[s.funcs[0]].func + suffix)] = self.r_out[p][rindex] * self.in_rate[p]
					# Calculate the incoming data rate for every VNF that
					# has all its incoming pairs processed
					self.in_rate[self.parseDict[s.funcs[0]].func + suffix] += self.r_out[p][rindex] * self.in_rate[p]

					rindex = 0

				# Part 0 of the parallel module are the pairs that are
				# placed before the splitter instance. These pairs should
				# be simply paired up with each other. 
				part = 0
				if self.parseDict[s.funcs[0]].func == s.func:
					part = 1
				for i,f in enumerate(s.funcs[1:]):
					# Part 1 of the parallel module are the pairs that 
					# start at the splitter instance (a) and end in the
					# beginning of the branches (b's). For every such
					# (a,b), copies of b should be made for the number
					# of outgoing branches from a and a should be paired
					# up with each of these copies of b's. 
					if part == 1:
						for q in range(0, s.num):
							# Add the newly created copies of usage requests to 
							# U and r and UF and remove the original one from U
							if self.parseDict[s.funcs[i]].func + suffix not in self.U_out:
								self.U_out.append(self.parseDict[s.funcs[i]].func + suffix)
								self.r_out[self.parseDict[s.funcs[i]].func + suffix] = self.req.r[self.parseDict[s.funcs[i]].func]
								self.UF_out[self.parseDict[s.funcs[i]].func + suffix] = self.req.UF[self.parseDict[s.funcs[i]].func]
								self.in_rate[self.parseDict[s.funcs[i]].func + suffix] = 0
								if self.parseDict[s.funcs[i]].func + suffix != self.parseDict[s.funcs[i]].func and self.parseDict[s.funcs[i]].func in self.U_out:
									self.U_out.remove(self.parseDict[s.funcs[i]].func)

							if self.parseDict[s.funcs[i+1]].func + suffix + "_" + str(q) not in self.U_out:
								self.U_out.append(self.parseDict[s.funcs[i+1]].func + suffix + "_" + str(q))
								self.r_out[self.parseDict[s.funcs[i+1]].func + suffix + "_" + str(q)] = self.req.r[self.parseDict[s.funcs[i+1]].func + suffix]
								self.UF_out[self.parseDict[s.funcs[i+1]].func + suffix + "_" + str(q)] = self.req.UF[self.parseDict[s.funcs[i+1]].func + suffix]
								self.in_rate[self.parseDict[s.funcs[i+1]].func + suffix + "_" + str(q)] = 0
								if self.parseDict[s.funcs[i+1]].func + suffix + "_" + str(q) != self.parseDict[s.funcs[i+1]].func + suffix and self.parseDict[s.funcs[i+1]].func + suffix in self.U_out:
									self.U_out.remove(self.parseDict[s.funcs[i+1]].func + suffix)

							# Create the pair
							self.pairs.append((self.parseDict[s.funcs[i]].func + suffix, self.parseDict[s.funcs[i+1]].func + suffix + "_" + str(q)))
							# Compute the data rate of the new pair
							self.d_req[(self.parseDict[s.funcs[i]].func + suffix, self.parseDict[s.funcs[i+1]].func + suffix + "_" + str(q))] = self.r_out[self.parseDict[s.funcs[i]].func + suffix][rindex] * self.in_rate[self.parseDict[s.funcs[i]].func + suffix]
							# Calculate the incoming data rate for every VNF that
							# has all its incoming pairs processed
							self.in_rate[self.parseDict[s.funcs[i+1]].func + suffix + "_" + str(q)] += self.r_out[self.parseDict[s.funcs[i]].func + suffix][rindex] * self.in_rate[self.parseDict[s.funcs[i]].func + suffix]
							rindex = 0

						part = 2
					# Part 2 of the parallel module are the pairs on the
					# branches that are copies of the module in the request chain
					elif part == 2:
						for q in range(0, s.num):
						# Add the newly created copies of usage requests to 
						# U and r and UF and remove the original one from U
							if self.parseDict[s.funcs[i]].func + suffix + "_" + str(q) not in self.U_out:
								self.U_out.append(self.parseDict[s.funcs[i]].func + suffix + "_" + str(q))
								self.r_out[self.parseDict[s.funcs[i]].func + suffix + "_" + str(q)] = self.req.r[self.parseDict[s.funcs[i]].func + suffix]
								self.UF_out[self.parseDict[s.funcs[i]].func + suffix + "_" + str(q)] = self.req.UF[self.parseDict[s.funcs[i]].func + suffix]
								self.in_rate[self.parseDict[s.funcs[i]].func + suffix + "_" + str(q)] = 0
								if self.parseDict[s.funcs[i]].func + suffix + "_" + str(q) != self.parseDict[s.funcs[i]].func + suffix and self.parseDict[s.funcs[i]].func + suffix in self.U_out:
									self.U_out.remove(self.parseDict[s.funcs[i]].func + suffix)
							
							if self.parseDict[s.funcs[i+1]].func + suffix + "_" + str(q) not in self.U_out:
								self.U_out.append(self.parseDict[s.funcs[i+1]].func + suffix + "_" + str(q))
								self.r_out[self.parseDict[s.funcs[i+1]].func + suffix + "_" + str(q)] = self.req.r[self.parseDict[s.funcs[i+1]].func + suffix]
								self.UF_out[self.parseDict[s.funcs[i+1]].func + suffix + "_" + str(q)] = self.req.UF[self.parseDict[s.funcs[i+1]].func + suffix]
								self.in_rate[self.parseDict[s.funcs[i+1]].func + suffix + "_" + str(q)] = 0
								if self.parseDict[s.funcs[i+1]].func + suffix + "_" + str(q) != self.parseDict[s.funcs[i+1]].func + suffix and self.parseDict[s.funcs[i+1]].func + suffix in self.U_out:
									self.U_out.remove(self.parseDict[s.funcs[i+1]].func + suffix)

							# Create the pair
							self.pairs.append((self.parseDict[s.funcs[i]].func + suffix + "_" + str(q), self.parseDict[s.funcs[i+1]].func + suffix + "_" + str(q)))
							# Compute the data rate of the new pair
							self.d_req[(self.parseDict[s.funcs[i]].func + suffix + "_" + str(q), self.parseDict[s.funcs[i+1]].func + suffix + "_" + str(q))] = self.r_out[self.parseDict[s.funcs[i]].func + suffix + "_" + str(q)][rindex] * self.in_rate[self.parseDict[s.funcs[i]].func + suffix + "_" + str(q)]
							# Calculate the incoming data rate for every VNF that
							# has all its incoming pairs processed
							#print self.parseDict[s.funcs[i+1]].func + suffix + "_" + str(q)
							#print self.parseDict[s.funcs[i]].func + suffix + "_" + str(q)
							self.in_rate[self.parseDict[s.funcs[i+1]].func + suffix + "_" + str(q)] += self.r_out[self.parseDict[s.funcs[i]].func + suffix + "_" + str(q)][rindex] * self.in_rate[self.parseDict[s.funcs[i]].func + suffix + "_" + str(q)]
							rindex = 0

					else:
						# Add the newly created copies of usage requests to 
						# U and r and UF and remove the original one from U
						if self.parseDict[s.funcs[i]].func + suffix not in self.U_out:
							self.U_out.append(self.parseDict[s.funcs[i]].func + suffix)
							self.r_out[self.parseDict[s.funcs[i]].func + suffix] = self.req.r[self.parseDict[s.funcs[i]].func]
							self.UF_out[self.parseDict[s.funcs[i]].func + suffix] = self.req.UF[self.parseDict[s.funcs[i]].func]
							self.in_rate[self.parseDict[s.funcs[i]].func + suffix] = 0
							if self.parseDict[s.funcs[i]].func + suffix != self.parseDict[s.funcs[i]].func and self.parseDict[s.funcs[i]].func in self.U_out:
								self.U_out.remove(self.parseDict[s.funcs[i]].func)
					
						if self.parseDict[s.funcs[i+1]].func + suffix not in self.U_out:
							self.U_out.append(self.parseDict[s.funcs[i+1]].func + suffix)
							self.r_out[self.parseDict[s.funcs[i+1]].func + suffix] = self.req.r[self.parseDict[s.funcs[i+1]].func]
							self.UF_out[self.parseDict[s.funcs[i+1]].func + suffix] = self.req.UF[self.parseDict[s.funcs[i+1]].func]
							self.in_rate[self.parseDict[s.funcs[i+1]].func + suffix] = 0
							if self.parseDict[s.funcs[i+1]].func + suffix != self.parseDict[s.funcs[i+1]].func and self.parseDict[s.funcs[i+1]].func in self.U_out:
								self.U_out.remove(self.parseDict[s.funcs[i+1]].func)

						# Create the pair
						self.pairs.append((self.parseDict[s.funcs[i]].func + suffix, self.parseDict[s.funcs[i+1]].func + suffix))
						# Compute the data rate of the new pair
						self.d_req[(self.parseDict[s.funcs[i]].func + suffix, self.parseDict[s.funcs[i+1]].func + suffix)] = self.r_out[self.parseDict[s.funcs[i]].func + suffix][rindex] * self.in_rate[self.parseDict[s.funcs[i]].func + suffix]
						# Calculate the incoming data rate for every VNF that
						# has all its incoming pairs processed
						self.in_rate[self.parseDict[s.funcs[i+1]].func + suffix] += self.r_out[self.parseDict[s.funcs[i]].func + suffix][rindex] * self.in_rate[self.parseDict[s.funcs[i]].func + suffix]
						rindex = 0
							
					if self.parseDict[f].func == s.func:
						part = 1
				
				prevs = []
				for q in range(0, s.num):
					if part == 2:
						p = self.createPairs(s.mod, s.jump, [self.parseDict[s.funcs[-1]].func + suffix + "_" + str(q)], suffix + "_" + str(q))
					else:
						p = self.createPairs(s.mod, s.jump, [self.parseDict[s.funcs[-1]].func + suffix], suffix + "_" + str(q))
					for r in p:
						prevs.append(r)
				
				prev = prevs
			pos = s.jump
		return prev
	
	"""
	# Go through the dictionary of parsed request and create the pairs
	"""	
	def create_pairs(self):
		s = self.parseDict[0]
		#self.U_out.append(s.func)
		# Initialize the incoming data rate for every instance to 0
		for u in self.U_out:
			self.in_rate[u] = 0
		self.in_rate[s.func] = self.req.input_datarate
		self.createPairs(s.jump, len(self.req.chain), [s.func])

		for dreqk in self.d_req.keys():
			self.d_req[dreqk] = round(self.d_req[dreqk],3)
		for lreqk in self.req.l_req.keys():
			self.req.l_req[lreqk] = round(self.req.l_req[lreqk],3)
		for irk in self.in_rate.keys():
			self.in_rate[irk] = round(self.in_rate[irk],3)
		# Pass the pairs and their data rates to placement input
		self.placementInput['U_pairs'] = self.pairs
		self.placementInput['U'] = self.U_out
		self.placementInput['UF'] = self.UF_out
		self.placementInput['d_req'] = self.d_req
		self.placementInput['A'] = self.req.A
		self.placementInput['l_req'] = self.req.l_req
		self.placementInput['in_rate'] = self.in_rate
		return self.placementInput

	"""
	# #print out the results of parsing, creating pairs, and computing the 
	# data rates between pairs
	"""		
	def print_results(self):
		logger.debug("Parsed chain:")
		logger.debug("%s", self.parsed_chain)
		#~ #print "Modules:"
		#~ #print self.parseDict
		#~ for i in range(0, len(self.req.chain)):
			#~ if i in self.parseDict:
				#~ #print i,self.parseDict[i]
		logger.debug("Pairs, data rates:")
		for p in self.pairs:
			logger.debug("%s %s", p, self.d_req[p])

	"""
	# Extract all optional orders in the request before the actual parsing and pair creation
	"""
	def preparse(self):
		self.parsed_chain = self.start.parseString(self.req.chain)
		for i in range(0, len(self.req.chain)):
			if i in self.parseDict:
				if isinstance(self.parseDict[i], storeOptorder) or isinstance(self.parseDict[i], storeParallel):
					fs = ",".join(self.parseDict[p].func for p in self.parseDict[i].funcs)
					self.optorderdict[fs] = []
					for p in self.parseDict[i].funcs:
						self.optorderdict[fs].append(self.parseDict[p].func)
					
	"""
	# Parse the request chain
	"""	
	def parse(self):
		self.parsed_chain = self.start.parseString(self.req.chain)
		self.fixOptionalOrders(0, len(self.req.chain))
		self.fixNexts(len(self.req.chain), 0, len(self.req.chain))

			
class storeInstance():
	def __init__(self, func, jump):
		self.func = func
		self.jump = jump
	def __str__(self):
		return "Instance: " + self.func + " Next: " + str(self.jump)

class storeOptorder():
	def __init__(self, funcs, jump):
		self.funcs = funcs
		self.jump = jump
	def __str__(self):
		return "OptionalOrder: " + ",".join(str(x) for x in self.funcs) + " Next: " + str(self.jump) 

class storeSplit():
	def __init__(self, func, mods, jump):
		self.func = func 
		self.mods = mods
		self.jump = jump 
	def __str__(self):
		return "Split at " + self.func + ": " + ",".join(str(x) for x in self.mods) + " Next: " + str(self.jump)

class storeParallel():
	def __init__(self, func, num, funcs, mod, jump):
		self.func = func
		self.num = num
		self.funcs = funcs 
		self.mod = mod 
		self.jump = jump 
	def __str__(self):
		return "Parallel splitting at " + str(self.func) + ": " + str(self.num) + " branches containing optional order among " + ",".join(str(x) for x in self.funcs) + " and module " + str(self.mod) + " Next: " + str(self.jump) 
		
class storeRequest():
	def __init__(self, begin, end, jump):
		self.begin = begin
		self.end = end
		self.func = begin
		self.jump = jump
	def __str__(self):
		return "This is a request to place the VNFs between application tier VMs " + self.begin + " and " + self.end + ":\n0 Instance: " + self.begin + " Next: " + str(self.jump) 
