
using DataFrames

function main()
    boxl = 22.5
    boxl2 = boxl/2.0
    coords = readtable("configuration.dat", separator=' ', header=false)
    coords = convert(Matrix{Float64}, coords)
    println(size(coords))
    npart = size(coords, 1)
    temp = 1.8
    beta = 1.0 / temp
    println(npart)

    ncycle = round(Int, 10^5 * npart)
    E_T = detailed_ecalc(npart, coords, boxl, boxl2)

    traj = open("Traj_jl.xyz","w")
    write(traj, string(npart) * "\n\n\n")
    for i in 1:npart
      @printf traj "Ar %.3f %.3f %.3f\n" coords[i,1] coords[i,2] coords[i,3]
    end

    println("Initial Energy: ", E_T)
    println("Simulation Start!")
    tic()
    montecarlo_loop(ncycle, coords, beta, E_T, boxl, boxl2)
    E_Final = detailed_ecalc(npart, coords, boxl, boxl2)
    toc()


    finalconfig = open("final_configuration_jl.dat", "w")
    for i in 1:npart
      @printf finalconfig "%.3f %.3f %.3f" coords[i,1] coords[i,2] coords[i,3]
    end

    write(traj, npart, "\n\n")
    for i in 1:npart
      @printf traj "Ar %.3f %.3f %.3f\n" coords[i,1] coords[i,2] coords[i,3]
    end
    close(finalconfig)
    close(traj)
end

function montecarlo_loop(ncycle, coords, beta, E_T, boxl, boxl2)
    npart = size(coords, 1)-1
    disp = zeros(Float64, 3)
    e_diff = 0.0
    nmove = 0
    for indx in 1:ncycle+1
        nmove = 1 + floor(Int, npart * rand())
        for i in 1:3
          disp[i] = 0.2 * (2.0 * rand()-1.0)
          if abs(disp[i] + coords[nmove,i]) > boxl
            disp[i] = disp[i] - copysign(boxl,disp[i])
          end
        end
        e_diff = shift_ecalc(npart, coords, nmove, disp, boxl, boxl2)
        if e_diff < 0.0
          E_T += e_diff
          coords[nmove,1] += disp[1]
          coords[nmove,2] += disp[2]
          coords[nmove,3] += disp[3]
        else
          if exp(-beta*e_diff) > rand()
            E_T += e_diff
            coords[nmove,1] += disp[1]
            coords[nmove,2] += disp[2]
            coords[nmove,3] += disp[3]
          end
        end
    end
    println("Culmaltive Energy: ", E_T)
    E_Final = 0.0
    E_Final = detailed_ecalc(npart, coords, boxl, boxl2)
    println("Final Energy: ", E_Final)
    nothing
end


function detailed_ecalc(npart, coords, boxl, boxl2)
  energy = 0.0
  for i in 1:npart-1
    for j in i+1:npart
      rx = coords[i, 1] - coords[j, 1]
      ry = coords[i, 2] - coords[j, 2]
      rz = coords[i, 3] - coords[j, 3]
      if abs(rx) > boxl2
        rx -= copysign(boxl, rx)
      end
      if abs(ry) > boxl2
        ry -= copysign(boxl,ry)
      end
      if abs(rz) > boxl2
        rz -= copysign(boxl,rz)
      end

      r_sq = rx*rx + ry*ry + rz*rz
      if r_sq < 49.0
        LJ = 1.0/r_sq
        LJ = LJ*LJ*LJ
        LJ = 4.0*LJ*(LJ-1.0)
        energy = energy + LJ
      end
    end
  end
  return energy
end

function shift_ecalc(npart,coords,nmove,disp, boxl, boxl2)
  energy = 0.0
  for j in 1:npart
    if j != nmove
      @inbounds rx = coords[nmove,1] + disp[1] - coords[j,1]
      @inbounds ry = coords[nmove,2] + disp[2] - coords[j,2]
      @inbounds rz = coords[nmove,3] + disp[3] - coords[j,3]
      if abs(rx) > boxl2
        rx -= copysign(boxl, rx)
      end
      if abs(ry) > boxl2
        ry -= copysign(boxl, ry)
      end
      if abs(rz) > boxl2
        rz -= copysign(boxl, rz)
      end

      r_sq = rx*rx + ry*ry + rz*rz
      if r_sq < 49.0
          LJ = 1.0/r_sq
          LJ = LJ*LJ*LJ
          LJ = 4.0 * LJ * (LJ-1.0)
          energy = energy + LJ
      end

      @inbounds rx = coords[nmove,1] - coords[j,1]
      @inbounds ry = coords[nmove,2] - coords[j,2]
      @inbounds rz = coords[nmove,3] - coords[j,3]
      if abs(rx) > boxl2
          rx -= copysign(boxl,rx)
      end
      if abs(ry) > boxl2
          ry -= copysign(boxl,ry)
      end
      if abs(rz) > boxl2
        rz -= copysign(boxl,rz)
      end
      r_sq = rx*rx + ry*ry + rz*rz
      if r_sq < 49.0
          LJ = 1.0/r_sq
          LJ = LJ*LJ*LJ
          LJ = 4.0*LJ*(LJ-1.0)
          energy = energy - LJ
      end
    end
  end
  return energy
end

main()
