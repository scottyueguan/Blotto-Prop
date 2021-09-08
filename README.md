# Blotto Prop Algorithm
## Installation
Create a conda environment in the terminal 
``conda create --name blotto-prop``.

Activate the environment 
``conda activate blotto-prop``.

Install the required packages
``pip install -r requirements.txt``

If error "fatal error: gmp.h: No such file or directory" appears, gmp is not properly installed. 
For Linux run `sudo apt-get install libgmp3-dev libmpfr-dev`. 
For Mac try `sudo apt-get install libgmp3-dev libmpfr-dev`.



## Limitations
(1) The convexhull is generated using scipy ConvexHull. 
Due to the fact that the polygon lives on the simplex, the polygon is degerated. 
ConvexHull does not handle degeneracy, and we have to project the polygon to N-1 space. 

(2) The intersection is handled using shapely Polygon, which only handle 2D geometries.


## Winning condition for Attacker
Let x_t and y_t be the distribution for the Defender and the Attacker.
The Attacker wins if it overwhelm the defender at one location. Formally,  [y_t]_i >= [x_t]_i at some time step t and at some node i.


## Constraints on Attacker's action
Aside from the underlying graph constraints, we enforce additional restrictions on Attacker's actions.
* Given Defender's allocation x_t, the attacker must maintain at least eta * [x_t]_i
 at node i. This is implemented in ```utils.compute_y_req_v1```.

* Given Defender's allocation x_t, the attacker must maintain at least eta fraction of its resource at node i. 
  This is implemented in ```utils.compute_y_req_v2```.
  
Change the code in ```main.py``` to select different restriction on Attacker.