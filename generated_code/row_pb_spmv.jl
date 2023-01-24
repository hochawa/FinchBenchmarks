function run_row_pb_spmv(ex, Tv)
    @inbounds begin
          y_lvl = ex.body.body.lhs.tns.tns.lvl
          y_lvl_2 = y_lvl.lvl
          y_lvl_2_val_alloc = length(y_lvl.lvl.val)
          y_lvl_2_val = 0.0
          A_lvl = (ex.body.body.rhs.args[1]).tns.tns.lvl
          A_lvl_2 = A_lvl.lvl
          A_lvl_2_headers_alloc = length(A_lvl_2.headers)
          A_lvl_2_pos_alloc = length(A_lvl_2.pos)
          A_lvl_2_values_alloc = length(A_lvl_2.values)
          x_lvl = (ex.body.body.rhs.args[2]).tns.tns.lvl
          x_lvl_2 = x_lvl.lvl
          x_lvl_2_val_alloc = length(x_lvl.lvl.val)
          x_lvl_2_val = 0.0
          j_stop = A_lvl_2.I
          i_stop = A_lvl.I
          y_lvl_2_val_alloc = (Finch).refill!(y_lvl_2.val, 0.0, 0, 4)
          y_lvl_2_val_alloc < (*)(1, A_lvl.I) && (y_lvl_2_val_alloc = (Finch).refill!(y_lvl_2.val, 0.0, y_lvl_2_val_alloc, (*)(1, A_lvl.I)))
          for i = 1:i_stop
              y_lvl_q = (1 - 1) * A_lvl.I + i
              A_lvl_q = (1 - 1) * A_lvl.I + i
              y_lvl_2_val = y_lvl_2.val[y_lvl_q]
              A_lvl_2_pos = A_lvl_2.pos[A_lvl_q]
              A_lvl_2_end = A_lvl_2.pos[A_lvl_q + 1]
              A_lvl_2_hpos = A_lvl_2.hpos[A_lvl_q]
              A_lvl_2_hend = A_lvl_2.hpos[A_lvl_q + 1]
              A_lvl_2_i = (MyInt)(1)
              A_lvl_2_next_i = (MyInt)(1)
              if A_lvl_2_hpos < A_lvl_2_hend
                  if A_lvl_2.headers[A_lvl_2_hpos] >> 15 == 1
                      A_lvl_2_next_i += A_lvl_2.headers[A_lvl_2_hpos] & ~((UInt16)(0x01) << 15)
                  else
                      A_lvl_2_next_i += A_lvl_2.headers[A_lvl_2_hpos]
                  end
              end
              j = 1
              while A_lvl_2_pos + (MyInt)(1) < A_lvl_2_end && A_lvl_2_i < 1
                  A_lvl_2_header_value = A_lvl_2.headers[A_lvl_2_hpos]
                  if A_lvl_2_header_value >> 15 == 1
                      A_lvl_2_pos += (MyInt)(1)
                      A_lvl_2_i = A_lvl_2_next_i
                      A_lvl_2_next_i += A_lvl_2.headers[A_lvl_2_hpos] & ~((UInt16)(0x01) << 15)
                  else
                      A_lvl_2_pos += ~((UInt16)(0x01) << 15) & A_lvl_2_header_value
                      A_lvl_2_i = A_lvl_2_next_i
                      A_lvl_2_next_i += A_lvl_2.headers[A_lvl_2_hpos]
                  end
                  A_lvl_2_hpos += (MyInt)(1)
              end
              while j <= j_stop
                  j_start = j
                  A_lvl_2_header_value = A_lvl_2.headers[A_lvl_2_hpos]
                  if A_lvl_2_header_value >> 15 == 1
                      phase_start = j_start
                      phase_stop = (min)(A_lvl_2_next_i - 1, j_stop)
                      j = j
                      if A_lvl_2_next_i - 1 == phase_stop
                          A_lvl_2_value_temp = A_lvl_2.values[A_lvl_2_pos]
                          if A_lvl_2_value_temp == zero(Tv)
                          else
                              temp_acc = zero(Tv)
                              for j_2 = phase_start:phase_stop
                                  x_lvl_q = (1 - 1) * x_lvl.I + j_2
                                  x_lvl_2_val = x_lvl_2.val[x_lvl_q]
                                  temp_acc += x_lvl_2_val
                              end
                              y_lvl_2_val = (+)(y_lvl_2_val, (*)(A_lvl_2_value_temp[], temp_acc))
                          end
                          A_lvl_2_pos += (MyInt)(1)
                      else
                          A_lvl_2_value_temp = A_lvl_2.values[A_lvl_2_pos]
                          if A_lvl_2_value_temp == zero(Tv)
                          else
                              temp_acc = zero(Tv)
                              for j_3 = phase_start:phase_stop
                                  x_lvl_q = (1 - 1) * x_lvl.I + j_3
                                  x_lvl_2_val = x_lvl_2.val[x_lvl_q]
                                  temp_acc += x_lvl_2_val
                              end
                              y_lvl_2_val = (+)(y_lvl_2_val, (*)(A_lvl_2_value_temp[], temp_acc))

                          end
                      end
                      j = phase_stop + 1
                  else
                      phase_start_2 = j_start
                      phase_stop_2 = (min)(A_lvl_2_next_i - 1, j_stop)
                      j_4 = j
                      if A_lvl_2_next_i - 1 == phase_stop_2
                          for j_5 = phase_start_2:phase_stop_2
                              x_lvl_q = (1 - 1) * x_lvl.I + j_5
                              A_lvl_2_lookup_index = (A_lvl_2_pos + j_5) - A_lvl_2_i
                              x_lvl_2_val = x_lvl_2.val[x_lvl_q]
                              y_lvl_2_val = (+)(y_lvl_2_val, (*)((A_lvl_2.values[A_lvl_2_lookup_index])[], x_lvl_2_val))
                          end
                          A_lvl_2_pos += ~((UInt16)(0x01) << 15) & A_lvl_2_header_value
                      else
                          for j_6 = phase_start_2:phase_stop_2
                              x_lvl_q = (1 - 1) * x_lvl.I + j_6
                              A_lvl_2_lookup_index = (A_lvl_2_pos + j_6) - A_lvl_2_i
                              x_lvl_2_val = x_lvl_2.val[x_lvl_q]
                              y_lvl_2_val = (+)(y_lvl_2_val, (*)((A_lvl_2.values[A_lvl_2_lookup_index])[], x_lvl_2_val))
                          end
                      end
                      j = phase_stop_2 + 1
                  end
                  A_lvl_2_hpos += (MyInt)(1)
                  A_lvl_2_i = A_lvl_2_next_i
                  if A_lvl_2_hpos < A_lvl_2_hend
                      if A_lvl_2.headers[A_lvl_2_hpos] >> 15 == 1
                          A_lvl_2_next_i += A_lvl_2.headers[A_lvl_2_hpos] & ~((UInt16)(0x01) << 15)
                      else
                          A_lvl_2_next_i += A_lvl_2.headers[A_lvl_2_hpos]
                      end
                  end
              end
              y_lvl_2.val[y_lvl_q] = y_lvl_2_val
          end
          qos = 1 * A_lvl.I
          resize!(y_lvl_2.val, qos)
          (y = Fiber((Finch.DenseLevel){Int64}(A_lvl.I, y_lvl_2), (Finch.Environment)(; )),)
      end
end