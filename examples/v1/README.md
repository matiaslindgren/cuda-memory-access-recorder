v1 from http://ppc.cs.aalto.fi/ch4/v1/ is basically the same as [v0](http://ppc.cs.aalto.fi/ch4/v0/), but indexes i and j have been swapped when accessing `d` and writing the result:
```cuda
	float x = d[n*j + k];
	float y = d[n*k + i];
	// ...
r[n*j + i] = v;
```
