defmodule CSR2 do

  defstruct values: nil,   # Nx tensor [nnz]
            colidx: nil,   # Nx tensor [nnz] (s32)
            rowptr: nil,   # Nx tensor [n+1]  (s32)
            rowidx: nil,   # Nx tensor [nnz]  (s32) -> linhas de cada não-zero
            n: 0

  def build_rowidx(rowptr) do
    r = Nx.to_flat_list(rowptr)
    n = length(r) - 1

    rows =
      for i <- 0..(n-1),
          _k <- Enum.at(r, i)..(Enum.at(r, i+1) - 1),
          do: i

    Nx.tensor(rows, type: :s32)
  end

  def spmv(%CSR2{values: v, colidx: c, rowidx: ri, n: n}, x) do

    #IO.inspect(%CSR2{values: v, colidx: c, rowidx: ri, n: n}, label: "a")
    #IO.inspect(x, label: "x")

    # pega x[colidx]
    xcol = Nx.take(x, c)

    # multiplica pelos valores
    prod = Nx.multiply(v, xcol)

    # soma por linha
    w = Nx.indexed_add(
      Nx.broadcast(0.0, {n}),
      Nx.new_axis(ri, -1),
      prod
    )

    #IO.inspect(w, label: "w")

    w
  end

end
