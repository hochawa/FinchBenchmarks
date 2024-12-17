using Finch
using BenchmarkTools


function permute_split_rows_dynamic_grain_kernel_mul(y, A, x)
        _y = Tensor(Dense(Element(0.0)), y)
        _A = swizzle(Tensor(Dense(SparseList(Element(0.0))), permutedims(A)), 2, 1)
        _x = Tensor(Dense(Element(0.0)), x)
        time = @belapsed begin
                (_y, _A, _x) = $(_y, _A, _x)
                permute_split_rows_dynamic_grain_kernel(_y, _A, _x)
        end
        return (; time=time, y=_y)
end

function permute_split_rows_dynamic_grain_kernel(_y::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}, _A::Finch.SwizzleArray{(2, 1),Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}}, _x::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}})
        @inbounds @fastmath(begin
                _y_lvl = _y.lvl
                _y_lvl_val = _y_lvl.lvl.val
                tns_lvl = _A.body.lvl
                tns_lvl_2 = tns_lvl.lvl
                tns_lvl_ptr = tns_lvl_2.ptr
                tns_lvl_idx = tns_lvl_2.idx
                tns_lvl_2_val = tns_lvl_2.lvl.val
                _x_lvl = _x.lvl
                _x_lvl_val = _x_lvl.lvl.val
                _x_lvl.shape == tns_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(_x_lvl.shape) != $(tns_lvl_2.shape))"))
                Finch.resize_if_smaller!(_y_lvl_val, tns_lvl.shape)
                Finch.fill_range!(_y_lvl_val, 0.0, 1, tns_lvl.shape)
                val = _y_lvl_val
                _y_lvl_val = (Finch).moveto(_y_lvl_val, CPU(Threads.nthreads()))
                _x_lvl_val = (Finch).moveto(_x_lvl_val, CPU(Threads.nthreads()))
                tns_lvl_ptr = (Finch).moveto(tns_lvl_ptr, CPU(Threads.nthreads()))
                tns_lvl_idx = (Finch).moveto(tns_lvl_idx, CPU(Threads.nthreads()))
                tns_lvl_2_val = (Finch).moveto(tns_lvl_2_val, CPU(Threads.nthreads()))
                Threads.@threads for i_4 = 1:Threads.nthreads()
                        Finch.@barrier begin
                                @inbounds @fastmath(begin
                                        phase_start_2 = max(1, 1 + fld(tns_lvl.shape * (i_4 + -1), Threads.nthreads()))
                                        phase_stop_2 = min(tns_lvl.shape, fld(tns_lvl.shape * i_4, Threads.nthreads()))
                                        if phase_stop_2 >= phase_start_2
                                                for i_7 = phase_start_2:phase_stop_2
                                                        _y_lvl_q = (1 - 1) * tns_lvl.shape + i_7
                                                        tns_lvl_q = (1 - 1) * tns_lvl.shape + i_7
                                                        tns_lvl_2_q = tns_lvl_ptr[tns_lvl_q]
                                                        tns_lvl_2_q_stop = tns_lvl_ptr[tns_lvl_q+1]
                                                        if tns_lvl_2_q < tns_lvl_2_q_stop
                                                                tns_lvl_2_i1 = tns_lvl_idx[tns_lvl_2_q_stop-1]
                                                        else
                                                                tns_lvl_2_i1 = 0
                                                        end
                                                        phase_stop_3 = min(_x_lvl.shape, tns_lvl_2_i1)
                                                        if phase_stop_3 >= 1
                                                                if tns_lvl_idx[tns_lvl_2_q] < 1
                                                                        tns_lvl_2_q = Finch.scansearch(tns_lvl_idx, 1, tns_lvl_2_q, tns_lvl_2_q_stop - 1)
                                                                end
                                                                while true
                                                                        tns_lvl_2_i = tns_lvl_idx[tns_lvl_2_q]
                                                                        if tns_lvl_2_i < phase_stop_3
                                                                                tns_lvl_3_val = tns_lvl_2_val[tns_lvl_2_q]
                                                                                _x_lvl_q = (1 - 1) * _x_lvl.shape + tns_lvl_2_i
                                                                                _x_lvl_2_val = _x_lvl_val[_x_lvl_q]
                                                                                _y_lvl_val[_y_lvl_q] = tns_lvl_3_val * _x_lvl_2_val + _y_lvl_val[_y_lvl_q]
                                                                                tns_lvl_2_q += 1
                                                                        else
                                                                                phase_stop_5 = min(phase_stop_3, tns_lvl_2_i)
                                                                                if tns_lvl_2_i == phase_stop_5
                                                                                        tns_lvl_3_val = tns_lvl_2_val[tns_lvl_2_q]
                                                                                        _x_lvl_q = (1 - 1) * _x_lvl.shape + phase_stop_5
                                                                                        _x_lvl_2_val_2 = _x_lvl_val[_x_lvl_q]
                                                                                        _y_lvl_val[_y_lvl_q] += tns_lvl_3_val * _x_lvl_2_val_2
                                                                                        tns_lvl_2_q += 1
                                                                                end
                                                                                break
                                                                        end
                                                                end
                                                        end
                                                end
                                        end
                                        phase_start_6 = max(1, 1 + fld(tns_lvl.shape * i_4, Threads.nthreads()))
                                        phase_stop_7 = tns_lvl.shape
                                        if phase_stop_7 >= phase_start_6
                                                phase_stop_7 + 1
                                        end
                                end)
                                nothing
                        end
                end
                result = ()
                resize!(val, tns_lvl.shape)
                result
        end)
end
