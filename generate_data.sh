cd graphs; ./rmat_gen 22 64; cd -
cd graphs; ./rmat_gen 23 32; cd -
cd graphs; ./rmat_gen 24 16; cd -
cd spgemm; julia make_scale.jl; cd -
