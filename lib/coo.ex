defmodule COO do

  defstruct values: nil,
            rowidx: nil,
            colidx: nil,
            n: 0

  def spmv(%COO{values: v, rowidx: r, colidx: c, n: n}, x) do

    # IO.inspect(%COO{values: v, rowidx: r, colidx: c, n: n}, label: "a")
    # IO.inspect(x, label: "x")

    # pega x[colidx]
    xcol = Nx.take(x, c)

    # multiplica pelos valores
    prod = Nx.multiply(v, xcol)

    # soma por linha
    w = Nx.indexed_add(
      Nx.broadcast(0.0, {n}),
      Nx.new_axis(r, -1),
      prod
    )

    # IO.inspect(w, label: "w")

    w
  end
end
