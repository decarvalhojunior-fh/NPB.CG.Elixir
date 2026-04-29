import Nx.Defn

defmodule COO do

  defstruct values: nil,
            rowidx: nil,
            colidx: nil,
            n: 0

  def spmv(%COO{values: v, rowidx: r, colidx: c, n: n}, x, t0) do

    # pega x[colidx]
    xcol = Nx.take(x, c)

    # multiplica pelos valores
    prod = Nx.multiply(v, xcol)

    # soma por linha
    Nx.indexed_add(
      t0,
      Nx.new_axis(r, -1),
      prod
    )

  end

  defn spmv_defn(v, r, c, x, t0) do

    # pega x[colidx]
    xcol = Nx.take(x, c)

    # multiplica pelos valores
    prod = Nx.multiply(v, xcol)

    # soma por linha
    Nx.indexed_add(
      t0,
      Nx.new_axis(r, -1),
      prod
    )

  end

end
