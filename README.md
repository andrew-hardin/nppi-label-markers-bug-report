# Bug in nppiLabelMarkersUF_8u32u_C1R
This project reproduces a correctness bug in `nppiLabelMarkersUF_8u32u_C1R` under CUDA 11.8.

The input matrix is this:
```
0 1 1 1 0 0
0 1 1 1 0 0
0 1 1 1 0 0
0 0 0 0 0 0
1 1 0 0 1 1
1 1 0 0 1 1
1 1 0 0 1 1
```

The output matrix is this:
```
0 1 1 1 0 0
0 1 1 1 0 0
0 1 1 1 0 0
0 0 0 0 0 0
24 24 0 0 24 24
24 24 0 0 24 24
24 24 0 0 24 24
^  ^      ^  ^

Bug! The left and right lower corners share the same label ID, but they're not connected!
```