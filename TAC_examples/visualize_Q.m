load("3_node_q_prop_example.mat")

% data[t, i, n, k, :] gives the coordinate of step t, node i, n-th face of
% the Qset, and the k-th point of this face. 
data(2, 3, 1, 2, :)

% if it returns [-1, -1, -1] then do not consider this point. 
% This element is added to preserve the dimension of the data.