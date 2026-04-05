using HDF5
using StableRNGs

file = h5open("matrices-and-rhs.h5", "cw")

matrix_sizes = [128, 143, 256, 263, 512, 1024, 2048, 2057, 4096, 8192, 16384, 16397]
nmat = 10
nrhs = 10

rng = StableRNG(6543)

for m ∈ matrix_sizes
    println("m=$m")
    for imat ∈ 1:nmat
        mat = rand(rng, m, m)
        name = "matrix-$m-$imat"
        create_dataset(file, name, mat)
        write(file[name], mat)
    end
    rhs = rand(rng, m, nrhs)
    name = "rhs-$m"
    create_dataset(file, name, rhs)
    write(file[name], rhs)
end

close(file)
