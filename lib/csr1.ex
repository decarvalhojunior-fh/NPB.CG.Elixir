defmodule CSR1 do

  defstruct values: nil,
            colidx: nil,
            rowptr: nil,
            n: 0

  def spmv(%CSR1{values: v, colidx: c, rowptr: r, n: n}, x) do

    #IO.inspect(%CSR1{values: v, colidx: c, rowptr: r, n: n}, label: "a")
    #IO.inspect(x, label: "x")

    result =
      for i <- 0..(n-1) do
        start = Nx.take(r, i) |> Nx.to_number()
        stop = (Nx.take(r, i+1) |> Nx.to_number()) - 1
        #IO.inspect({i, start, stop}, label: "spmv_row")

        Enum.reduce(start..stop//1, 0.0, fn k, acc ->
          val = Nx.take(v, k) |> Nx.to_number()
          col = Nx.take(c, k) |> Nx.to_number()
         # IO.inspect({i, k, val, col}, label: "spmv")
          acc + val * Nx.to_number(Nx.take(x, col))
        end)
      end

    w = Nx.tensor(result)

    #IO.inspect(w, label: "w")

    w
  end

end
