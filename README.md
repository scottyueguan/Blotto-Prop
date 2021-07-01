# Blotto Prop Algorithm



## Limitations
(1) The convexhull is generated using scipy ConvexHull. 
Due to the fact that the polygon lives on the simplex, the polygon is degerated. 
ConvexHull does not handle degeneracy, and we have to project the polygon to N-1 space. 

(2) The intersection is handled using shapely Polygon, which only handle 2D geometries.

