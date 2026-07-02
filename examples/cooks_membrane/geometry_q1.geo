////////////////////////////////////////////////////////////
// Cook's membrane (Q1 quadrilateral mesh)
//
// Geometry:
// (0,44) ----------- (48,60)
//   |                   |
//   |                   |
// (0,0) ------------ (48,44)
////////////////////////////////////////////////////////////

SetFactory("OpenCASCADE");

// Geometry
Point(1) = {0,  0, 0};
Point(2) = {48,44, 0};
Point(3) = {48,60, 0};
Point(4) = {0,44, 0};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Curve Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};

// Structured mesh
nx = 32;
ny = 32;

Transfinite Curve{1,3} = nx + 1;
Transfinite Curve{2,4} = ny + 1;

Transfinite Surface{1};
Recombine Surface{1};

// Physical groups
Physical Surface("Domain") = {1};

Physical Curve("Left")   = {4};
Physical Curve("Right")  = {2};
Physical Curve("Bottom") = {1};
Physical Curve("Top")    = {3};

Mesh.ElementOrder = 1;
Mesh.SecondOrderIncomplete = 0;

Mesh 2;