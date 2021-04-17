# str1 = 'req0_u11_0'
# print('_'.join(str1.split('_')[:-1]))
request_list = []
### abilene ###
###############
reqtest = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'LB','u2':'FW','u3':'VOPT'}, 
        'r' : {'a1':[1.0],'a2':[1.0],'a2_0':[1.0],'a2_1':[1.0],'a2_2':[1.0],'u1':[0.3],'u2':[1.2],'u3':[4.0]},
        'A': {'a1': 2,'a2_0': 0,'a2_1': 7,'a2_2': 10},
        'l_req' : {('a1','a2'): 5,('a1','a2_0'): 3,('a1','a2_1'): 4,('a1','a2_2'): 5},
        'input_datarate' : 100,
        'chain' : "a1.u1{u1;a2;3}"}

### Mixed request set ###
#########################

req140 = {'UF' : {'a1':'BNG', 'a2':'CR','u1':'FW','u2':'CACHE'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0]},
        'A': {'a1': 9, 'a2': 2},
        'l_req' : {('a1','a2'): 4},
        'input_datarate' : 100,
        'chain' : "a1.(u1,u2).a2"}
        
req150 = {'UF' : {'a1':'CR', 'a2':'BNG','u1':'FW','u2':'DPI','u3':'PCTL'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0],'u3':[0.6]},
        'A': {'a1': 2, 'a2': 9},
        'l_req' : {('a1','a2'): 5.5},
        'input_datarate' : 100,
        'chain' : "a1.u1.u2.u3.a2"}     
        
# req160 = {'UF' : {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE','u6':'FW'}, 
#         'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.3,0.5,0.2],'u2':[1.0],'u3':[2.0],'u4':[0.4],'u5':[1.0],'u6':[0.5]},
#         'A': {'a1': 9, 'a2': 3},
#         'l_req' : {('a1','a2'): 7},
#         'input_datarate' : 100,
#         'chain' : "a1.u1[u2|u3.(u4,u5)|u6].a2"} 

req160 = {'UF' : {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE','u6':'FW'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.3,0.5,0.2],'u2':[1.0],'u3':[2.0],'u4':[0.4],'u5':[1.0],'u6':[0.5]},
        'A': {'a1': 0, 'a2': 5},
        'l_req' : {('a1','a2'): 7},
        'input_datarate' : 100,
        'chain' : "a1.u1[u2|u3.u4.u5|u6].a2"} 

req161 = {'UF' : {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE','u6':'FW'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.3,0.5,0.2],'u2':[1.0],'u3':[2.0],'u4':[0.4],'u5':[1.0],'u6':[0.5]},
        'A': {'a1': 9, 'a2': 8},
        'l_req' : {('a1','a2'): 7},
        'input_datarate' : 100,
        'chain' : "a1.u1[u2|u3.u4.u5|u6].a2"}  

req162 = {'UF' : {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE','u6':'FW'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.3,0.5,0.2],'u2':[1.0],'u3':[2.0],'u4':[0.4],'u5':[1.0],'u6':[0.5]},
        'A': {'a1': 2, 'a2': 7},
        'l_req' : {('a1','a2'): 7},
        'input_datarate' : 100,
        'chain' : "a1.u1[u2|u3.u4.u5|u6].a2"}         


req170 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'LB','u2':'FW','u3':'VOPT'}, 
        'r' : {'a1':[1.0],'a2':[1.0],'a2_0':[1.0],'a2_1':[1.0],'a2_2':[1.0],'u1':[0.3],'u2':[1.2],'u3':[4.0]},
        'A': {'a1': 2,'a2_0': 0,'a2_1': 7,'a2_2': 10},
        'l_req' : {('a1','a2'): 5,('a1','a2_0'): 3,('a1','a2_1'): 4,('a1','a2_2'): 5},
        'input_datarate' : 100,
        'chain' : "a1.u1{u1,u2,u3;a2;3}"}

req171 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'LB','u2':'FW','u3':'VOPT'}, 
        'r' : {'a1':[1.0],'a2':[1.0],'a2_0':[1.0],'a2_1':[1.0],'a2_2':[1.0],'u1':[0.3],'u2':[0.7],'u3':[4.0]},
        'A': {'a1': 2,'a2_0': 0,'a2_1': 7,'a2_2': 10},
        'l_req' : {('a1','a2'): 5,('a1','a2_0'): 6,('a1','a2_1'): 5,('a1','a2_2'): 7},
        'input_datarate' : 200,
        'chain' : "a1.u1{u1,u2,u3;a2;3}"}

req172 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'LB','u2':'FW','u3':'VOPT'}, 
        'r' : {'a1':[1.0],'a2':[1.0],'a2_0':[1.0],'a2_1':[1.0],'a2_2':[1.0],'u1':[0.25],'u2':[0.1],'u3':[3.0]},
        'A': {'a1': 2,'a2_0': 0,'a2_1': 7,'a2_2': 10},
        'l_req' : {('a1','a2'): 8,('a1','a2_0'): 5,('a1','a2_1'): 3,('a1','a2_2'): 5},
        'input_datarate' : 150,
        'chain' : "a1.u1{u1,u2,u3;a2;3}"}
        
req180 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'DPI','u2':'FW'}, 
        'r' : {'a1':[1.0],'a2':[1.0],'a2_0':[1.0],'a2_1':[1.0],'a2_2':[1.0],'u1':[0.3],'u2':[0.1]},
        'A': {'a1': 2,'a2_0': 0,'a2_1': 7,'a2_2': 10},
        'l_req' : {('a1','a2'): 5,('a1','a2_0'): 6,('a1','a2_1'): 5.5,('a1','a2_2'): 5},
        'input_datarate' : 200,
        'chain' : "a1.u1{u1,u2;a2;3}"}  
            
req181 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'DPI','u2':'FW'}, 
        'r' : {'a1':[1.0],'a2':[1.0],'a2_0':[1.0],'a2_1':[1.0],'a2_2':[1.0],'u1':[0.3],'u2':[0.7]},
        'A': {'a1': 2,'a2_0': 0,'a2_1': 7,'a2_2': 10},
        'l_req' : {('a1','a2'): 4,('a1','a2_0'): 5.5,('a1','a2_1'): 5,('a1','a2_2'): 7.5},
        'input_datarate' : 150,
        'chain' : "a1.u1{u1,u2;a2;3}"}      

### Broadband network ###
#########################

# tenant1
req00 = {'UF' : {'a1':'BNG', 'a2':'CR','u1':'FW','u2':'CACHE'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0]},
        'A': {'a1': 9, 'a2': 2},
        'l_req' : {('a1','a2'): 5},
        'input_datarate' : 100,
        'chain' : "a1.(u1,u2).a2"}
req01 = {'UF' : {'a1':'BNG', 'a2':'CR','u1':'FW','u2':'CACHE'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0]},
        'A': {'a1': 4, 'a2': 2},
        'l_req' : {('a1','a2'): 5},
        'input_datarate' : 100,
        'chain' : "a1.(u1,u2).a2"}
req02 = {'UF' : {'a1':'BNG', 'a2':'CR','u1':'FW','u2':'CACHE'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0]},
        'A': {'a1': 10, 'a2': 2},
        'l_req' : {('a1','a2'): 5},
        'input_datarate' : 100,
        'chain' : "a1.(u1,u2).a2"}
req03 = {'UF' : {'a1':'BNG', 'a2':'CR','u1':'FW','u2':'CACHE'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0]},
        'A': {'a1': 3, 'a2': 2},
        'l_req' : {('a1','a2'): 5},
        'input_datarate' : 100,
        'chain' : "a1.(u1,u2).a2"}

req10 = {'UF' : {'a1':'CR', 'a2':'BNG','u1':'FW','u2':'DPI','u3':'PCTL'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0],'u3':[0.6]},
        'A': {'a1': 2, 'a2': 8},
        'l_req' : {('a1','a2'): 5},
        'input_datarate' : 700,
        'chain' : "a1.u1.u2.u3.a2"}
req11 = {'UF' : {'a1':'CR', 'a2':'BNG','u1':'FW','u2':'DPI','u3':'PCTL'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0],'u3':[0.6]},
        'A': {'a1': 2, 'a2': 4},
        'l_req' : {('a1','a2'): 5},
        'input_datarate' : 10,
        'chain' : "a1.u1.u2.u3.a2"}
req12 = {'UF' : {'a1':'CR', 'a2':'BNG','u1':'FW','u2':'DPI','u3':'PCTL'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0],'u3':[0.6]},
        'A': {'a1': 2, 'a2': 10},
        'l_req' : {('a1','a2'): 5},
        'input_datarate' : 10,
        'chain' : "a1.u1.u2.u3.a2"}
req13 = {'UF' : {'a1':'CR', 'a2':'BNG','u1':'FW','u2':'DPI','u3':'PCTL'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0],'u3':[0.6]},
        'A': {'a1': 2, 'a2': 3},
        'l_req' : {('a1','a2'): 5},
        'input_datarate' : 10,
        'chain' : "a1.u1.u2.u3.a2"}

# tenant 2
req20 = {'UF' : {'a1':'BNG', 'a2':'CR','u1':'CACHE','u2':'FW'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[1.0],'u2':[0.5]},
        'A': {'a1': 3, 'a2': 4},
        'l_req' : {('a1','a2'): 5},
        'input_datarate' : 150,
        'chain' : "a1.u1.u2.a2"}
req21 = {'UF' : {'a1':'BNG', 'a2':'CR','u1':'CACHE','u2':'FW'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[1.0],'u2':[0.5]},
        'A': {'a1': 7, 'a2': 2},
        'l_req' : {('a1','a2'): 5},
        'input_datarate' : 100,
        'chain' : "a1.u1.u2.a2"}
req22 = {'UF' : {'a1':'BNG', 'a2':'CR','u1':'CACHE','u2':'FW'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[1.0],'u2':[0.5]},
        'A': {'a1': 11, 'a2': 9},
        'l_req' : {('a1','a2'): 5},
        'input_datarate' : 150,
        'chain' : "a1.u1.u2.a2"}
req23 = {'UF' : {'a1':'BNG', 'a2':'CR','u1':'CACHE','u2':'FW'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[1.0],'u2':[0.5]},
        'A': {'a1': 10, 'a2': 2},
        'l_req' : {('a1','a2'): 5},
        'input_datarate' : 150,
        'chain' : "a1.u1.u2.a2"}


req30 = {'UF' : {'a1':'CR', 'a2':'BNG','u1':'FW','u2':'DPI','u3':'PCTL'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0],'u3':[0.8]},
        'A': {'a1': 2, 'a2': 3},
        'l_req' : {('a1','a2'): 5},
        'input_datarate' : 10,
        'chain' : "a1.(u1,u2).u3.a2"}
req31 = {'UF' : {'a1':'CR', 'a2':'BNG','u1':'FW','u2':'DPI','u3':'PCTL'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0],'u3':[0.8]},
        'A': {'a1': 2, 'a2': 7},
        'l_req' : {('a1','a2'): 5},
        'input_datarate' : 10,
        'chain' : "a1.(u1,u2).u3.a2"}
req32 = {'UF' : {'a1':'CR', 'a2':'BNG','u1':'FW','u2':'DPI','u3':'PCTL'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0],'u3':[0.8]},
        'A': {'a1': 2, 'a2': 11},
        'l_req' : {('a1','a2'): 5},
        'input_datarate' : 10,
        'chain' : "a1.(u1,u2).u3.a2"}
req33 = {'UF' : {'a1':'CR', 'a2':'BNG','u1':'FW','u2':'DPI','u3':'PCTL'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.4],'u2':[1.0],'u3':[0.8]},
        'A': {'a1': 2, 'a2': 1},
        'l_req' : {('a1','a2'): 5},
        'input_datarate' : 10,
        'chain' : "a1.(u1,u2).u3.a2"}
        
### Mobile core network ###
###########################

# tenant 3
req40 = {'UF' : {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE','u6':'FW'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.3,0.5,0.2],'u2':[1.0],'u3':[2.0],'u4':[0.4],'u5':[1.0],'u6':[0.5]},
        'A': {'a1': 9, 'a2': 3},
        'l_req' : {('a1','a2'): 5},
        'input_datarate' : 100,
        'chain' : "a1.u1[u2|u3.(u4,u5)|u6].a2"}     
req41 = {'UF' : {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE','u6':'FW'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.5,0.2,0.3],'u2':[1.0],'u3':[2.0],'u4':[0.4],'u5':[1.0],'u6':[0.5]},
        'A': {'a1': 4, 'a2': 3},
        'l_req' : {('a1','a2'): 5},
        'input_datarate' : 100,
        'chain' : "a1.u1[u2|u3.(u4,u5)|u6].a2"}     
req42 = {'UF' : {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE','u6':'FW'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.2,0.5,0.3],'u2':[1.0],'u3':[2.0],'u4':[0.4],'u5':[1.0],'u6':[0.5]},
        'A': {'a1': 1, 'a2': 3},
        'l_req' : {('a1','a2'): 5},
        'input_datarate' : 100,
        'chain' : "a1.u1[u2|u3.(u4,u5)|u6].a2"}     
req43 = {'UF' : {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE','u6':'FW'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.2,0.5,0.3],'u2':[1.0],'u3':[2.0],'u4':[0.4],'u5':[1.0],'u6':[0.5]},
        'A': {'a1': 8, 'a2': 3},
        'l_req' : {('a1','a2'): 5},
        'input_datarate' : 100,
        'chain' : "a1.u1[u2|u3.(u4,u5)|u6].a2"}     

req50 = {'UF' : {'a1':'WWW', 'a2':'GGSN','a3':'GGSN','a4':'GGSN','a5':'GGSN','u1':'VOPT','u2':'DPI'}, 
        'r' : {'a1':[1.0],'a2':[1.0],'a3':[1.0],'a4':[1.0],'a5':[1.0],'u1':[2.0],'u2':[0.25,0.25,0.25,0.25]},
        'A': {'a1': 3, 'a2': 9, 'a3': 4, 'a4': 1, 'a5': 8},
        'l_req' : {('a1','a2'):5,('a1','a3'):5,('a1','a4'):5,('a1','a5'):5},
        'input_datarate':100,
        'chain' : "a1.u1.u2[a2|a3|a4|a5]"}

req51 = {'UF' : {'a1':'WWW', 'a2':'GGSN','u1':'VOPT','u2':'DPI', 'u3':'FW', 'u4':'FW'}, 
        'r' : {'a1':[1.0],'a2':[1.0],'u1':[2.0],'u2':[0.5,0.5], 'u3':[1.0], 'u4':[1.0]},
        'A': {'a1': 7, 'a2': 11},
        'l_req' : {('a1','a2'):5},
        'input_datarate':200,
        'chain' : "a1.u1.u2[u3|u4].a2"}

req52 = {'UF' : {'a1':'WWW', 'a2':'GGSN','u1':'VOPT','u2':'DPI', 'u3':'FW', 'u4':'FW'}, 
        'r' : {'a1':[1.0],'a2':[1.0],'u1':[2.0],'u2':[0.5,0.5], 'u3':[1.0], 'u4':[1.0]},
        'A': {'a1': 0, 'a2': 7},
        'l_req' : {('a1','a2'):5},
        'input_datarate':200,
        'chain' : "a1.u1.u2[u3|u4].a2"}

# tenant 4
req60 = {'UF' : {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.5,0.5],'u2':[1.0],'u3':[3.0],'u4':[0.5],'u5':[1.0]},
        'A': {'a1': 10, 'a2': 3},
        'l_req' : {('a1','a2'): 5},
        'input_datarate' : 5,
        'chain' : "a1.u1[u2|u3.(u4,u5)].a2"}        
req61 = {'UF' : {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.5,0.5],'u2':[1.0],'u3':[3.0],'u4':[0.5],'u5':[1.0]},
        'A': {'a1': 9, 'a2': 3},
        'l_req' : {('a1','a2'): 5},
        'input_datarate' : 5,
        'chain' : "a1.u1[u2|u3.(u4,u5)].a2"}        
req62 = {'UF' : {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.5,0.5],'u2':[1.0],'u3':[3.0],'u4':[0.5],'u5':[1.0]},
        'A': {'a1': 0, 'a2': 3},
        'l_req' : {('a1','a2'): 5},
        'input_datarate' : 5,
        'chain' : "a1.u1[u2|u3.(u4,u5)].a2"}        
req63 = {'UF' : {'a1':'GGSN', 'a2':'WWW','u1':'DPI','u2':'WAPGW','u3':'VOPT','u4':'FW','u5':'CACHE'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.5,0.5],'u2':[1.0],'u3':[3.0],'u4':[0.5],'u5':[1.0]},
        'A': {'a1': 3, 'a2': 3},
        'l_req' : {('a1','a2'): 5},
        'input_datarate' : 5,
        'chain' : "a1.u1[u2|u3.(u4,u5)].a2"}        

req70 = {'UF' : {'a1':'WWW', 'a2':'GGSN','a3':'GGSN','a4':'GGSN','a5':'GGSN','u1':'DPI','u2':'VOPT'}, 
        'r' : {'a1':[1.0],'a2':[1.0],'a3':[1.0],'a4':[1.0],'a5':[1.0],'u1':[1.0],'u2':[2.0]},
        'A': {'a1': 6, 'a2': 10},
        'l_req' : {('a1','a2'):5},
        'input_datarate':5,
        'chain' : "a1.u1.u2.a2"}
req71 = {'UF' : {'a1':'WWW', 'a2':'GGSN','u1':'DPI','u2':'VOPT'}, 
        'r' : {'a1':[1.0],'a2':[1.0],'u1':[1.0],'u2':[2.0]},
        'A': {'a1': 11, 'a2': 9},
        'l_req' : {('a1','a2'):5},
        'input_datarate':5,
        'chain' : "a1.u1.u2.a2"}
req72 = {'UF' : {'a1':'WWW', 'a2':'GGSN','u1':'DPI','u2':'VOPT'}, 
        'r' : {'a1':[1.0],'a2':[1.0],'u1':[1.0],'u2':[2.0]},
        'A': {'a1': 5, 'a2': 0},
        'l_req' : {('a1','a2'):5},
        'input_datarate':5,
        'chain' : "a1.u1.u2.a2"}
req73 = {'UF' : {'a1':'WWW', 'a2':'GGSN','u1':'DPI','u2':'VOPT'}, 
        'r' : {'a1':[1.0],'a2':[1.0],'u1':[1.0],'u2':[2.0]},
        'A': {'a1': 7, 'a2': 8},
        'l_req' : {('a1','a2'):5},
        'input_datarate':5,
        'chain' : "a1.u1.u2.a2"}

### Data center ###
###################

# tenant 5
req80 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'LB','u2':'FW'}, 
        'r' : {'a1':[1.0],'a2':[1.0],'a2_0':[1.0],'a2_1':[1.0],'a2_2':[1.0],'u1':[0.3],'u2':[0.8]},
        'A': {'a1': 2,'a2_0': 0,'a2_1': 7,'a2_2': 10},
        'l_req' : {('a1','a2'): 5,('a1','a2_0'): 5,('a1','a2_1'): 5,('a1','a2_2'): 5},
        'input_datarate' : 200,
        'chain' : "a1.u1{u1,u2;a2;3}"}
req81 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'LB','u2':'FW'}, 
        'r' : {'a1':[1.0],'a2':[1.0],'a2_0':[1.0],'a2_1':[1.0],'a2_2':[1.0],'u1':[0.3],'u2':[1.2]},
        'A': {'a1': 4,'a2_0': 0,'a2_1': 7,'a2_2': 10},
        'l_req' : {('a1','a2'): 5,('a1','a2_0'): 5,('a1','a2_1'): 5,('a1','a2_2'): 5},
        'input_datarate' : 100,
        'chain' : "a1.u1{u1,u2;a2;3}"}
req82 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'LB','u2':'FW'}, 
        'r' : {'a1':[1.0],'a2':[1.0],'a2_0':[1.0],'a2_1':[1.0],'a2_2':[1.0],'u1':[0.3],'u2':[0.4]},
        'A': {'a1': 8,'a2_0': 0,'a2_1': 7,'a2_2': 10},
        'l_req' : {('a1','a2'): 5,('a1','a2_0'): 5,('a1','a2_1'): 5,('a1','a2_2'): 5},
        'input_datarate' : 150,
        'chain' : "a1.u1{u1,u2;a2;3}"}
req83 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'LB','u2':'FW'}, 
        'r' : {'a1':[1.0],'a2':[1.0],'a2_0':[1.0],'a2_1':[1.0],'a2_2':[1.0],'u1':[0.2],'u2':[0.8]},
        'A': {'a1': 9,'a2_0': 0,'a2_1': 7,'a2_2': 10},
        'l_req' : {('a1','a2'): 5,('a1','a2_0'): 5,('a1','a2_1'): 5,('a1','a2_2'): 5},
        'input_datarate' : 200,
        'chain' : "a1.u1{u1,u2;a2;3}"}

### Complex  ###
#######################

req9 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'LB','u2':'WOPT','u3':'AV','u4':'DPI','u5':'WAPGW','u6':'PCTL','u7':'FW','u8':'VOPT','u9':'CACHE','u10':'IDS','u11':'IPS'}, 
        'r' : {'a1':[1.0],'a2':[1.0],'a2_0':[1.0],'a2_1':[1.0],'a2_2':[1.0],'a2_3':[1.0],'u1':[0.25],'u2':[1.5],'u3':[0.75],'u4':[0.25,0.1,0.6,0.05],'u5':[1.0],'u6':[1.4],'u7':[0.8],'u8':[2.5],'u9':[1.0],'u10':[0.75],'u11':[0.9]},
        'A' : {'a1': 4 ,'a2_0': 10,'a2_1': 7,'a2_2': 0,'a2_3' : 8},
        'l_req' : {('a1','a2'): 10,('a1','a2_0'): 10,('a1','a2_1'): 10,('a1','a2_2'): 10,('a1','a2_3'): 10},
        'input_datarate' : 150,
        'chain' : "a1.u1{u1,u2,u3;u4[u5|(u6,u7)|u8.u9|u10].u11.a2;4}"}
        
### Optional Order ###
######################

req100 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'DPI','u2':'VOPT','u3':'FW'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[1.0],'u2':[4.0],'u3':[0.4]},
        'A' : {'a1': 9, 'a2': 8},
        'l_req' : {('a1','a2'): 1},
        'input_datarate' : 150,
        'chain' : "a1.(u1,u2,u3).a2"}       
        
req110 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'LB','u2':'WOPT','u3':'AV','u4':'DPI','u5':'WAPGW','u6':'FW'}, 
        'r' : {'a1':[1.0],'a2':[1.0],'u1':[0.3],'u2':[2.5],'u3':[0.8],'u4':[1.0],'u5':[1.0],'u6':[0.4]},
        'A' : {'a1': 3 ,'a2': 0},
        'l_req' : {('a1','a2'): 1},
        'input_datarate' : 200,
        'chain' : "a1.u1{u1,u2,u3;u4.u5;3}.u6.a2"}

req120 = {'UF' : {'a1':'REG', 'a2':'SRV','u1':'FW','u2':'VOPT'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.8],'u2':[2.0]},
        'A' : {'a1': 1, 'a2': 10},
        'l_req' : {('a1','a2'): 1},
        'input_datarate' : 100,
        'chain' : "a1.(u1,u2).a2"}
        
req130 = {'UF' : {'a1':'BNG', 'a2':'CR','u1':'FW','u2':'CACHE','u3':'AV','u4':'DPI','u5':'VOPT'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.5],'u2':[1.0],'u3':[0.5],'u4':[1.0],'u5':[3.0]},
        'A' : {'a1': 11, 'a2': 9},
        'l_req' : {('a1','a2'): 1},
        'input_datarate' : 150,
        'chain' : "a1.(u1,u2).u3.(u4,u5).a2"}

request_list.append((req140))
request_list.append((req150))
request_list.append((req160))
request_list.append((req170))
request_list.append((req171))
request_list.append((req172))
request_list.append((req180))
request_list.append((req181))
request_list.append((req150))
request_list.append((req20))
request_list.append((req20))
request_list.append((req21))
request_list.append((req22))
request_list.append((req23))
request_list.append((req50))
request_list.append((req51))
request_list.append((req52))
request_list.append((req160))
request_list.append((req161))
request_list.append((req162))
request_list.append((req70))
request_list.append((req71))
request_list.append((req72))    
request_list.append((req73))    
request_list.append((req20))
request_list.append((req21))
request_list.append((req22))
request_list.append((req23))
request_list.append((req50))
request_list.append((req51))
request_list.append((req52))
request_list.append((req20))
request_list.append((req21))
request_list.append((req22))
request_list.append((req23))
request_list.append((req70))
request_list.append((req71))
request_list.append((req72))
request_list.append((req73))
request_list.append((req20))
request_list.append((req21))
request_list.append((req22))
request_list.append((req23))
request_list.append((req50))
request_list.append((req51))
request_list.append((req52))
request_list.append((req160))
request_list.append((req161))
request_list.append((req162))
request_list.append((req70))
request_list.append((req71))
request_list.append((req72))    
request_list.append((req73))    
request_list.append((req20))
request_list.append((req21))
request_list.append((req22))
request_list.append((req23))
request_list.append((req50))
request_list.append((req51))
request_list.append((req52))
request_list.append((req160))
request_list.append((req161))
request_list.append((req162))
request_list.append((req70))
request_list.append((req71))
request_list.append((req72))    
request_list.append((req73))    
request_list.append((req20))
request_list.append((req21))
request_list.append((req22))
request_list.append((req23))
request_list.append((req50))
request_list.append((req51))
request_list.append((req52))
request_list.append((req20))
request_list.append((req21))
request_list.append((req22))
request_list.append((req23))
request_list.append((req70))
request_list.append((req71))
request_list.append((req72))
request_list.append((req73))
request_list.append((req160))
#~ request_list.append((req140))
#~ request_list.append((req150))
#~ request_list.append((req160))
request_list.append((req170))
#~ request_list.append((req171))
#~ request_list.append((req172))
#~ request_list.append((req180))
#~ request_list.append((req140))
#~ request_list.append((req150))
#~ request_list.append((req160))
#~ request_list.append((req171))
#~ request_list.append((req172))
request_list.append((req180))
#~ request_list.append((req181))
#~ request_list.append((req140))
#~ request_list.append((req150))
#~ request_list.append((req160))
request_list.append((req170))
request_list.append((req171))
request_list.append((req172))
#~ request_list.append((req180))
#~ request_list.append((req181))
#~ request_list.append((req140))
#~ request_list.append((req150))
#~ request_list.append((req160))
#~ request_list.append((req170))
#~ request_list.append((req171))
#~ request_list.append((req172))
request_list.append((req180))
request_list.append((req181))
request_list.append((req140))
#~ request_list.append((req150))
#~ request_list.append((req160))
request_list.append((req170))
#~ request_list.append((req171))
#~ request_list.append((req172))
request_list.append((req180))
#~ request_list.append((req181))
request_list.append((req10))
request_list.append((req50))
request_list.append((req51))
request_list.append((req52))
request_list.append((req80))    
request_list.append((req00))
request_list.append((req01))
request_list.append((req02))
request_list.append((req03))
request_list.append((req10))
request_list.append((req11))
request_list.append((req12))
request_list.append((req13))
request_list.append((req20))
request_list.append((req21))
request_list.append((req22))
request_list.append((req23))
request_list.append((req30))
request_list.append((req31))
request_list.append((req32))
request_list.append((req33))
request_list.append((req00))
request_list.append((req01))
request_list.append((req02))
request_list.append((req03))
request_list.append((req10))
request_list.append((req11))
request_list.append((req12))
request_list.append((req13))
request_list.append((req00))
request_list.append((req01))
request_list.append((req02))
request_list.append((req03))
request_list.append((req40))
request_list.append((req41))
request_list.append((req42))
request_list.append((req43))
request_list.append((req50))
request_list.append((req60))
request_list.append((req61))
request_list.append((req62))
request_list.append((req63))
request_list.append((req70))
request_list.append((req71))
request_list.append((req72))
request_list.append((req73))
request_list.append((req40))
request_list.append((req41))
request_list.append((req42))
request_list.append((req43))
request_list.append((req50))
request_list.append((req40))
request_list.append((req41))
request_list.append((req42))
request_list.append((req43))
request_list.append((req80))
request_list.append((req81))
request_list.append((req82))
request_list.append((req83))
request_list.append((req80))
request_list.append((req81))
#~ request_list.append((req82))
#~ request_list.append((req83))  
#~ request_list.append((req110)) 
request_list.append((req130))    
request_list.append((req170))    
request_list.append((req80))
request_list.append((req81))
request_list.append((req9))
request_list.append((req00))
request_list.append((req01))
request_list.append((req02))
request_list.append((req03))
request_list.append((req10))
request_list.append((req11))
request_list.append((req12))
request_list.append((req13))
request_list.append((req20))
request_list.append((req21))
request_list.append((req22))
request_list.append((req23))
request_list.append((req30))
request_list.append((req31))
request_list.append((req32))
request_list.append((req33))
request_list.append((req40))
request_list.append((req41))
request_list.append((req42))
request_list.append((req43))
request_list.append((req50))
request_list.append((req60))
request_list.append((req61))
request_list.append((req62))
request_list.append((req63))
request_list.append((req70))
request_list.append((req71))
request_list.append((req72))
request_list.append((req73))
request_list.append((req80))
request_list.append((req81))
request_list.append((req82))
request_list.append((req83))
request_list.append((req100))
request_list.append((req110))
request_list.append((req120))
request_list.append((req130))
request_list.append((req100))
request_list.append((req110))
request_list.append((req120))
request_list.append((req130))
request_list.append((req140))  
request_list.append((req150))    
request_list.append((req100))
request_list.append((req110))
request_list.append((req100))
request_list.append((req120))
request_list.append((req100))
request_list.append((req130))
request_list.append((req100))
request_list.append((req110))
request_list.append((req120))
request_list.append((req110))
request_list.append((req120))
request_list.append((req110))
request_list.append((req130))
request_list.append((req110))
request_list.append((req120))
request_list.append((req130))
request_list.append((req120))
request_list.append((req130))
request_list.append((req100))
request_list.append((req110))
request_list.append((req130))
request_list.append((reqtest))

req130 = {'UF' : {'a1':'BNG', 'a2':'CR','u1':'FW','u2':'CACHE','u3':'AV','u4':'DPI','u5':'VOPT'}, 
        'r' : {'a1':[1.0], 'a2':[1.0],'u1':[0.5],'u2':[1.0],'u3':[0.5],'u4':[1.0],'u5':[3.0]},
        'A' : {'a1': 11, 'a2': 9},
        'l_req' : {('a1','a2'): 1},
        'input_datarate' : 150,
        'chain' : "a1.(u1,u2).u3.(u4,u5).a2"}

vnfset = set()

for req in request_list:
    for key, value in req['UF'].items():
        vnfset.add(value)

writelist = []
for n, value in enumerate(vnfset):
    writelist.append(value + '\t' + '-' + str(n))

with open('vnf.vocab1', 'w', encoding='utf-8', errors='ignore') as fout:
    fout.write("\n".join(writelist))
