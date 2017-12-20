def minArea(x,y,k):
	n = len(x)
	xmin = min(x)
	xmax = max(x)
	ymin - min(x)
	ymax = max(y)

	delx = xmax-xmin
	dely = ymax-ymin
	side = max(delx, dely)

	sidex1 = xmin-1
	sidey1 = ymin-1
	diff = abs(delx-dely) + 1

	sidex2 = 0
	sidey2 = 0

	if delx > dely:
		sidex2 = xmax+1
		sidey2 = ymax+diff
	else:
		sidex2 = xmax+diff
		sidey2 = ymax+1
	size = (sx2 - sx1) * (sx2 - sx1);

	p = 0
	q = 0
	r = 0
	s = 0
	minSize = k**(1/2) + 1
