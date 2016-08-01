from numba import jit
import numpy as np
import random as rng
import math
import time

@jit(cache=True)
def main():
    boxl = 22.5
    boxl2 = boxl/2.0
    coords = np.loadtxt("configuration.dat")
    print(coords.shape)
    npart = coords.shape[0]
    temp = 1.8
    beta = 1.0/temp
    print(npart)
    ncycle = int(10**5 * npart)
    E_T = 0.0
    E_T = detailed_ecalc(npart, coords, boxl, boxl2)
    traj = open("Traj.xyz","w")
    print(npart,file=traj)
    print("\n",file=traj)
    for i in range(0,npart):
        print("Ar",coords[i][0],coords[i][1],coords[i][2],file=traj)
    print("Initial Energy: ", E_T)
    print("Simulation Start!")
    start_time = time.time()
    coords = montecarlo_loop(ncycle,coords, beta, E_T, boxl, boxl2)
    E_Final = detailed_ecalc(npart, coords, boxl, boxl2)
    elapsed_time = time.time() - start_time
    print("Total Time: ", elapsed_time)
    finalconfig = open("final_configuration.dat", "w")
    for i in range(0,npart):
        print(coords[i][0],coords[i][1],coords[i][2],file=finalconfig)
    print(npart,file=traj)
    print("\n",file=traj)
    for i in range(0,npart):
        print("Ar",coords[i][0],coords[i][1],coords[i][2],file=traj)

@jit(cache=True)
def montecarlo_loop(ncycle,coords, beta, E_T,  boxl, boxl2):
    npart = coords.shape[0]
    disp = np.ndarray(shape=[1,3], dtype=float)
    e_diff = 0.0
    nmove = 0
    for indx in range(0,ncycle+1):
        nmove = math.floor(npart*rng.random())
        for i in range(0,3):
            disp[0][i] = 0.2*(2.0*rng.random()-1.0)
            if math.fabs(disp[0][i] + coords[nmove][i]) > boxl:
                disp[0][i] = disp[0][i] - math.copysign(boxl,disp[0][i])
        e_diff = shift_ecalc(npart,coords,nmove,disp, boxl, boxl2)
        if e_diff < 0.0:
            E_T += e_diff
            coords[nmove][0] += disp[0][0]
            coords[nmove][1] += disp[0][1]
            coords[nmove][2] += disp[0][2]
        else:
            if math.exp(-beta*e_diff) > rng.random():
                E_T += e_diff
                coords[nmove][0] += disp[0][0]
                coords[nmove][1] += disp[0][1]
                coords[nmove][2] += disp[0][2]
#        if indx%10000 == 0:
#            fortprint.screenprint(indx,E_T)
#            print(indx,E_T)
#        if indx%100000 == 0:
#            print(npart,file=traj)
#            print(file=traj)
#            for i in range(0,npart):
#                print("Ar",coords[i][0],coords[i][1],coords[i][2],file=traj)
    print("Culmaltive Energy: ", E_T)
    E_Final = detailed_ecalc(npart, coords, boxl, boxl2)
    print("Final Energy: ", E_Final)
    return coords

@jit(nopython=True,cache=True)
def shift_ecalc(npart,coords,nmove,disp, boxl, boxl2):
    energy = 0.0
    for j in range(0,npart):
        if j != nmove:
            rx = coords[nmove][0]+disp[0][0] - coords[j][0]
            ry = coords[nmove][1]+disp[0][1] - coords[j][1]
            rz = coords[nmove][2]+disp[0][2] - coords[j][2]
            if math.fabs(rx) > boxl2:
                rx -= math.copysign(boxl,rx)
            if math.fabs(ry) > boxl2:
                ry -= math.copysign(boxl,ry)
            if math.fabs(rz) > boxl2:
                rz -= math.copysign(boxl,rz)

            r_sq = rx*rx + ry*ry + rz*rz
            if r_sq < 49.0:
                LJ = (1.0/r_sq)
                LJ = LJ*LJ*LJ
                LJ = 4.0 * LJ * (LJ-1.0)
                energy = energy + LJ

            rx = coords[nmove][0] - coords[j][0]
            ry = coords[nmove][1] - coords[j][1]
            rz = coords[nmove][2] - coords[j][2]
            if math.fabs(rx) > boxl2:
                rx -= math.copysign(boxl,rx)
            if math.fabs(ry) > boxl2:
                ry -= math.copysign(boxl,ry)
            if math.fabs(rz) > boxl2:
                rz -= math.copysign(boxl,rz)
            r_sq = rx*rx + ry*ry + rz*rz
            if r_sq < 49.0:
                LJ = (1.0/r_sq)
                LJ = LJ*LJ*LJ
                LJ = 4.0*LJ*(LJ-1.0)
                energy = energy - LJ
    return energy




@jit(nopython=True,cache=True)
def detailed_ecalc(npart,coords, boxl, boxl2):
    energy = 0.0
    for i in range(0,npart-1):
        for j in range(i+1,npart):
            rx = coords[i][0] - coords[j][0]
            ry = coords[i][1] - coords[j][1]
            rz = coords[i][2] - coords[j][2]
            if math.fabs(rx) > boxl2:
                rx -= math.copysign(boxl,rx)
            if math.fabs(ry) > boxl2:
                ry -= math.copysign(boxl,ry)
            if math.fabs(rz) > boxl2:
                rz -= math.copysign(boxl,rz)

            r_sq = rx*rx+ ry*ry + rz*rz
            if r_sq < 49.0:
                LJ = (1.0/r_sq)
                LJ = LJ*LJ*LJ
                LJ = 4.0*LJ*(LJ-1.0)
                energy = energy + LJ
    return energy

main()
