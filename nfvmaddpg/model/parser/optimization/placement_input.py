import copy

class PlacementInput:
	def __init__(self, p):
		#~ self.U = p['U']
		#~ self.UF = p['UF']
		#~ self.U_pairs = p['U_pairs']
		#~ self.d_req = p['d_req']
		#~ self.A = p['A']
		#~ self.l_req = p['l_req']
		self.U = copy.deepcopy(p['U'])
		self.UF = copy.deepcopy(p['UF'])
		self.U_pairs = copy.deepcopy(p['U_pairs'])
		self.d_req = copy.deepcopy(p['d_req'])
		self.A = copy.deepcopy(p['A'])
		self.l_req = copy.deepcopy(p['l_req'])
		self.in_rate = copy.deepcopy(p['in_rate'])
	
