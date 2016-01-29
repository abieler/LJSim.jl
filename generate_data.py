confile = open("configuration.dat","w")
dimension = 4.0
bounds = int(dimension/2.0)
for i in range(-bounds,bounds):
 for j in range(-bounds,bounds):
   for k in range(-bounds,bounds):
     confile.write(" " + str(i) + " " + str(j) + " " + str(k) + "\n")
