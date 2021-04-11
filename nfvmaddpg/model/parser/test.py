# -*- coding:utf-8 -*-
# import numpy as np

# # 概率分布之Beta分布与Dirichlet分布 每个参数对应每一个维度，每个生成的样本维度元素和为1
# # s = np.random.dirichlet((10, 5, 3), 20)
# # print s

# from gurobipy import *

# try:

#     # Create a new model
#     m = Model("mip1")

#     # Create variables
#     x = m.addVar(vtype=GRB.BINARY, name="x")
#     y = m.addVar(vtype=GRB.BINARY, name="y")
#     z = m.addVar(vtype=GRB.BINARY, name="z")

#     # Set objective
#     m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)

#     # Add constraint: x + 2 y + 3 z <= 4
#     m.addConstr(x + 2 * y + 3 * z <= 4, "c0")

#     # Add constraint: x + y >= 1
#     m.addConstr(x + y >= 1, "c1")

#     m.optimize()

#     for v in m.getVars():
#         print(v.varName, v.x)

#     print('Obj:', m.objVal)

# except GurobiError:
#     print('Error reported')

# reqtype = raw_input("Enter request type (e.g., 'req150'): ") 
# s = "requestList_" + reqtype.rstrip() + ".pickle"
# print (s)

# from pyparsing import *

# integer  = Word(nums)            # simple unsigned integer 
# variable = Char(alphas)          # single letter variable, such as x, z, m, etc.
# arithOp  = oneOf("+ - * /")      # arithmetic operators 
# equation = variable + "=" + integer + arithOp + integer    # will match "x=2+2", etc.

# identifier = Word(alphas, alphanums + '_')
# number = Word(nums + '.')  

# expr = identifier + '=' + (identifier | number)

# result = expr.parseString("a=.2")     
# print(result)

import pickle
import json
from nfvmaddpg.model.parser.optimization.parser import Parser, storeParallel, storeOptorder
import pprint
from itertools import permutations, islice, product, chain 

# request_list_import = pickle.load(open("requestList_" + "req110" + ".pickle","r")) 
# Chain: req0_a1.req0_u1{req0_u1;req0_a2;3}
# class StrToBytes:
#     def __init__(self, fileobj):
#         self.fileobj = fileobj
#     def read(self, size):
#         return self.fileobj.read(size).encode()
#     def readline(self, size=-1):
#         return self.fileobj.readline(size).encode()
# request_list_import = pickle.load(StrToBytes(open("requestList_" + "complex" + ".pickle","r"))) 
# request_list = list(request_list_import["reqs"])

# allOptions = []
# for i,req in enumerate(request_list):
#     print(req)
#     req.add_prefix(i)
#     print(req)

#     req.forceOrder = dict()     
#     prsr = Parser(req)     
#     prsr.preparse()

#     print ("##################################parseResult###")               
#     pprint.pprint(prsr.parsed_chain) 

#     parseDict = prsr.parseDict.copy()
#     optords = prsr.optorderdict.copy()
#     parallel_num = prsr.parallel_num.copy()

#     print ("##################################parseDict###")          
#     pprint.pprint(parseDict)
#     print(parseDict[0])  
#     print(parseDict[8])  

#     print ("##################################PREPARSE###")     
#     pprint.pprint(optords)

#     for key, value in parseDict.items():  # Chain: req0_a1.req0_u1{req0_u1,req0_u2,req0_u3;req0_u4.req0_u5;3}.req0_u6.req0_a2
#         print("key: ", key)               #            0      8       16     24      32      40        48        59     67
#         if not isinstance(value, storeOptorder): 
#             print("func: ", value.func)
#         else:           
#             print("funcs: ", value.funcs)
#         print("jump: ", value.jump)       # Chain: req0_a1.req0_u1{req0_u1,req0_u2,req0_u3;req0_u4[req0_u5|(req0_u6,req0_u7)|req0_u8.req0_u9|req0_u10].req0_u11.req0_a2;4}
#         if isinstance(value, storeParallel): #         0   8        16      24     32        40    48      56 57    65       74      82      90        100      109
#             print("last_modjump: ", value.jump_lastmod)

#     perms = dict()
#     for v in optords.values():
#         perms[",".join(v)] = []
#         for x in permutations(v):
#             perms[",".join(v)].append(x)

#     allOptions.append(dict(perms))

#     prod = list(product(*perms.values()))[0]

#     # prod = (('req0_u3', 'req0_u2', 'req0_u1'), ('req0_u6', 'req0_u7'))
#     # print 'prod', prod    # 当前请求中每个optorder部分的所有可能顺序的组合('req0_u1', 'req0_u2', 'req0_u3'),)

#     # allOptions[req] = []
#     reqPlacementInputList = []   # 根据所有optorder可能的顺序组合  构造每个确定顺序的排列
#     for i in range(len(prod)):
#         for k in optords.keys():
#             if prod[i] in perms[k]:
#                 req.forceOrder[k] = prod[i]
#     prsr = Parser(req)
#     prsr.parse()
#     # prsr.fixOptionalOrders(0, len(req.chain))
#     # prsr.fixNexts(len(req.chain), 0, len(req.chain))
#     reqPlacementInput = prsr.create_pairs()

#         # allOptions[req].append(dict(req.forceOrder).values())c

#     prsr.print_results()
    

    
def posEncode(reqPlacementInput, parseDict, optords):
    funclist = []
    previsopt = False
    posencode = {}
    lastpair = ''
    firstpair = True
    for key, value in optords.items():
        funclist.extend(value)
    for pair in reqPlacementInput['U_pairs']:
        if firstpair:
            posencode[pair[0]] = 0
            lastpair = pair[0]
            firstpair = False
            if pair[0] in funclist or '_'.join(pair[0].split('_')[:-1]) in funclist: # 此对的前一个在optlist中
                previsopt = True
            else:  # 第一个不在optlist中，则不论上一对的最后是否在optlist中，其pos均为上一对后一个的pos+1
                previsopt = False
        if posencode.get(pair[0], None) == None:
            if posencode.get('_'.join(pair[0].split('_')[:-1]) + '_0', None) != None:
                # posencode[pair[0]] = posencode['_'.join(pair[0].split('_')[:-1]) + '_0']
                continue
            if pair[0] in funclist or '_'.join(pair[0].split('_')[:-1]) in funclist: # 此对的前一个在optlist中
                if previsopt == True: # 如果此对的前一对的后一个在optlist中，则此对前一个的pos与上一个相同
                    posencode[pair[0]] =  posencode[lastpair]
                else: # 否则为+1
                    posencode[pair[0]] = posencode[lastpair] + 1
                previsopt = True
            else:  # 第一个不在optlist中，则不论上一对的最后是否在optlist中，其pos均为上一对后一个的pos+1
                posencode[pair[0]] = posencode[lastpair] + 1
                previsopt = False
        if posencode.get(pair[1], None) != None:
            if posencode.get('_'.join(pair[1].split('_')[:-1]) + '_0', None) != None:
                if pair[1] in funclist or '_'.join(pair[1].split('_')[:-1]) in funclist:
                    if previsopt == True:
                        if posencode['_'.join(pair[1].split('_')[:-1]) + '_0'] >= posencode[pair[0]]:
                            posencode[pair[1]] = posencode['_'.join(pair[1].split('_')[:-1]) + '_0']
                        else:
                            posencode[pair[1]] = posencode[pair[0]]
                    else:
                        if posencode['_'.join(pair[1].split('_')[:-1]) + '_0'] >= posencode[pair[0]] + 1:
                            posencode[pair[1]] = posencode['_'.join(pair[1].split('_')[:-1]) + '_0']
                        else:
                            posencode[pair[1]] = posencode[pair[0]] + 1
                    previsopt = True
                else:
                    if posencode['_'.join(pair[1].split('_')[:-1]) + '_0'] >= posencode[pair[0]] + 1:
                        posencode[pair[1]] = posencode['_'.join(pair[1].split('_')[:-1]) + '_0']
                    else:
                        posencode[pair[1]] = posencode[pair[0]] + 1
                    previsopt = False
                continue
        if posencode.get(pair[1], None) == None:
            if posencode.get('_'.join(pair[1].split('_')[:-1]) + '_0', None) != None:
                if pair[1] in funclist or '_'.join(pair[1].split('_')[:-1]) in funclist:
                    if previsopt == True:
                        if posencode['_'.join(pair[1].split('_')[:-1]) + '_0'] >= posencode[pair[0]]:
                            posencode[pair[1]] = posencode['_'.join(pair[1].split('_')[:-1]) + '_0']
                        else:
                            posencode[pair[1]] = posencode[pair[0]]
                    else:
                        if posencode['_'.join(pair[1].split('_')[:-1]) + '_0'] >= posencode[pair[0]] + 1:
                            posencode[pair[1]] = posencode['_'.join(pair[1].split('_')[:-1]) + '_0']
                        else:
                            posencode[pair[1]] = posencode[pair[0]] + 1
                    previsopt = True
                else:
                    if posencode['_'.join(pair[1].split('_')[:-1]) + '_0'] >= posencode[pair[0]] + 1:
                        posencode[pair[1]] = posencode['_'.join(pair[1].split('_')[:-1]) + '_0']
                    else:
                        posencode[pair[1]] = posencode[pair[0]] + 1
                    previsopt = False
                continue
            if pair[1] in funclist or '_'.join(pair[1].split('_')[:-1]) in funclist:
                if previsopt == True: # 如果此对的前一个在optlist中，则此对前一个的pos与上一个相同
                    posencode[pair[1]] = posencode[pair[0]]
                else: # 否则为+1
                    posencode[pair[1]] = posencode[pair[0]] + 1
                previsopt = True
            else:
                posencode[pair[1]] = posencode[pair[0]] + 1
                previsopt = False
        lastpair = pair[1]
    return posencode

# posencode = posEncode(reqPlacementInput, parseDict, optords)

# print ("##################################PosEncode###")
# pprint.pprint(posencode)

# # {'req0_a1': 'REG', 'req0_a2': 'SRV', 'req0_a2_0': 'SRV', 'req0_a2_1': 'SRV', 'req0_a2_2': 'SRV', 'req0_a2_3': 'SRV', 'req0_u1': 'LB', 'req0_u10': 'IDS', 'req0_u10_0': 'IDS', 'req0_u10_1': 'IDS', 'req0_u10_2': 'IDS', 'req0_u10_3': 'IDS', 'req0_u11': 'IPS', 'req0_u11_0': 'IPS', ...}

# transformer_input = {}
# for k,v in posencode.items():
#     for m,n in reqPlacementInput['UF'].items():
#         if k == m:
#             transformer_input[k] = []
#             transformer_input[k].append(posencode[k])
#             transformer_input[k].append(reqPlacementInput['UF'][k])
#             posencode[k] = transformer_input[k]
#             reqPlacementInput['UF'][k] = transformer_input[k]
#         else:
#             transformer_input[k] = posencode[k]
#             transformer_input[m] = reqPlacementInput['UF'][m]
# print(transformer_input)

# for k in list(transformer_input.keys()):
#     if type(transformer_input[k]) != list:
#         del transformer_input[k]
# print(transformer_input)

# train_vnf = ''
# tran_pos = ''
# for k,v in transformer_input.items():
#     train_vnf += v[1] + ' '
#     tran_pos += str(v[0]) + ' '
# # with open("train_vnf", "w", encoding="UTF-8") as f:
# #     f.write(train_vnf.strip() + '\n')
# # with open("train_pos", "w", encoding="UTF-8") as f:
# #     f.write(tran_pos.strip() + '\n')

# reqPlacementInput['r'] = req.r
# reqPlacementInput['opt'] = req.forceOrder
# reqPlacementInput['parallel_num'] = parallel_num
# for k in list(reqPlacementInput['UF'].keys()):
#     if type(reqPlacementInput['UF'][k]) != list:
#         del reqPlacementInput['UF'][k]
# pickle.dump(reqPlacementInput, open(
#     "sfcRequest_" + "complex" + ".pickle", "wb"))
# with open("train", "w", encoding="UTF-8") as f:
#     s_dump = json.dumps(transformer_input)
#     json.dump(s_dump, f)

# with open("train", "r", encoding="UTF-8") as f:
#     s_dump = json.load(f)
#     print(type(json.loads(s_dump)))


# {'req0_a1': [0, 'REG'], 'req0_u1': [1, 'LB'], 'req0_u3': [1, 'AV'], 'req0_u2': [1, 'WOPT'], 'req0_u4_0': [2, 'DPI'], 'req0_u5_0': [3, 'WAPGW'], 'req0_u6_0': [3, 'PCTL'], 'req0_u7_0': [3, 'FW'], 'req0_u8_0': [3, 'VOPT'], 'req0_u9_0': [4, 'CACHE'], 'req0_u10_0': [3, 'IDS'], 'req0_u11_0': [5, 'IPS'], 'req0_a2_0': [6, 
# 'SRV'], 'req0_u4_1': [2, 'DPI'], 'req0_u5_1': [3, 'WAPGW'], 'req0_u6_1': [3, 'PCTL'], 'req0_u7_1': [3, 'FW'], 'req0_u8_1': [3, 'VOPT'], 'req0_u9_1': [4, 'CACHE'], 'req0_u10_1': [3, 'IDS'], 'req0_u11_1': [5, 'IPS'], 'req0_a2_1': [6, 'SRV'], 'req0_u4_2': [2, 'DPI'], 'req0_u5_2': [3, 'WAPGW'], 'req0_u6_2': [3, 'PCTL'], 'req0_u7_2': [3, 'FW'], 'req0_u8_2': [3, 'VOPT'], 'req0_u9_2': [4, 'CACHE'], 'req0_u10_2': [3, 'IDS'], 'req0_u11_2': [5, 'IPS'], 'req0_a2_2': [6, 'SRV'], 'req0_u4_3': [2, 'DPI'], 'req0_u5_3': [3, 'WAPGW'], 'req0_u6_3': [3, 'PCTL'], 'req0_u7_3': [3, 'FW'], 'req0_u8_3': [3, 'VOPT'], 'req0_u9_3': [4, 'CACHE'], 'req0_u10_3': [3, 'IDS'], 'req0_u11_3': [5, 'IPS'], 'req0_a2_3': [6, 'SRV']}

# Chain: req0_a1.req0_u1{req0_u1,req0_u2,req0_u3;req0_u4[req0_u5|(req0_u6,req0_u7)|req0_u8.req0_u9|req0_u10].req0_u11.req0_a2;4}
# prsr.undofixforopt(len(req.chain), 0, len(req.chain))

    # print ("##################################afterparseDict###")             
    # pprint.pprint(parseDict)

    # for key, value in parseDict.items():  # Chain: req0_a1.req0_u1{req0_u1,req0_u2,req0_u3;req0_u4.req0_u5;3}.req0_u6.req0_a2
    #     print("key: ", key)               #            0      8       16     24      32      40        48        59     67
    #     if not isinstance(value, storeOptorder): 
    #         print("func: ", value.func)
    #     else:           
    #         print("funcs: ", value.funcs)
    #     print("jump: ", value.jump)       # Chain: req0_a1.req0_u1{req0_u1,req0_u2,req0_u3;req0_u4[req0_u5|(req0_u6,req0_u7)|req0_u8.req0_u9|req0_u10].req0_u11.req0_a2;4}
    #     if isinstance(value, storeParallel): #         0   8        16      24     32        40    48      56 57    65       74      82      90        100      109
    #         print("last_modjump: ", value.jump_lastmod)

    # perms = dict()
    # for v in optords.values():
    #     perms[",".join(v)] = []
    #     for x in permutations(v):
    #         perms[",".join(v)].append(x)

    # # def myproduct(*iter):
    # #     print iter

    # # myproduct(perms.values())

    # allOptions.append(dict(perms))

    # prod = list(product(*perms.values()))
    # print 'prod', prod

    # reqPlacementInputList = []   # 根据所有optorder可能的顺序组合  构造每个确定顺序的排列
    # for p in prod:
    #     for i in range(len(p)):
    #         for k in optords.keys():
    #             # print p[i]
    #             # print perms[k]
    #             if p[i] in perms[k]:
    #                 req.forceOrder[k] = p[i]
    #     print "PERMUTATION",p
    #     print "forceorder",req.forceOrder
    #     prsr = Parser(req)
    #     prsr.parse()
    #     print "##################################create_pairs###"   
    #     reqPlacementInputList.append(prsr.create_pairs())

    #     # allOptions[req].append(dict(req.forceOrder).values())

    #     prsr.print_results()
