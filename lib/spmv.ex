defmodule MVMulSerial do

  # a is sparse in CSR format (using lists for values, colidx, and rowptr)
  def mv_multiply(%CSR0{values: v, colidx: c, rowptr: r, n: n}, x, _t0) do
     #x = Nx.to_list(x)
     CSR0.mv_multiply(%CSR0{values: v, colidx: c, rowptr: r, n: n}, x)
  end

  # a is sparse in CSR format (using tensors for values, colidx, and rowptr)
  def mv_multiply(%CSR1{values: v, colidx: c, rowptr: r, n: n}, x, _t0) do
     CSR1.mv_multiply(%CSR1{values: v, colidx: c, rowptr: r, n: n}, x)
  end

  # a is sparse in CSR format, with rowidx (i.e., COO-based mv_multiply)
  def mv_multiply(%CSR2{values: v, colidx: c, rowidx: ri, n: n}, x, _t0) do
     CSR2.mv_multiply(%CSR2{values: v, colidx: c, rowidx: ri, n: n}, x)
  end

  # a is sparse in COO format
  def mv_multiply(%COO{values: v, rowidx: ri, colidx: c, n: n}, x, t0) do
    COO.mv_multiply(%COO{values: v, rowidx: ri, colidx: c, n: n}, x, t0)
    #COO.mv_multiply_defn(v, ri, c, x, t0)
  end

  # a is dense
  def mv_multiply(a, x, _t0) do
#    IO.inspect({:dense_mv, Nx.shape(a), Nx.shape(x)}, label: "mv_multiply dense")
    Nx.dot(a, x)
  end

end
