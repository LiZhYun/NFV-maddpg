# Specification, Composition, and Placement of Network Services with Flexible Structures

The code used for the simulation of the following paper:

* S. Dräxler, H. Karl. Specification, Composition, and Placement of Network Services with Flexible Structures. International Journal of Network Management. 2017;e1963. DOI:10.1002/nem.1963.

The optimization approach is also relevant for our following papers:

* S. Mehraghdam, M. Keller, H. Karl. Specifying and Placing Chains of Virtual Network Functions. In IEEE 3rd International Conference on Cloud Networking (CloudNet 2014).
* S. Mehraghdam, H. Karl. Placement of Services with Flexible Structures Specified by a YANG Data Model. In IEEE 2nd Conference on Network Softwarization (NetSoft 2016).

You need Python2.7 and Gurobi.

* Create service definitions to place by 'makeRequestList.py'. Ready-to-use example services (req20, req150, complex) in the main directoy.
* Example network topologies from SNDlib in 'tops' folder. Ready-to-use example network (abilene) in the main directory.
* Run the optimization by 'python2.7 optrun.py' and enter the required input.
