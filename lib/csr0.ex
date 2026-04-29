defmodule CSR0 do

  defstruct values: nil,
            colidx: nil,
            rowptr: nil,
            n: 0

  def spmv(%CSR0{values: v, colidx: c, rowptr: r, n: n}, x) do

    result =
      for i <- 0..(n-1) do
        start = Enum.at(r, i)
        stop = Enum.at(r, i+1) - 1

        Enum.reduce(start..stop//1, 0.0, fn k, acc ->
          val = Enum.at(v, k)
          col = Enum.at(c, k)
         # acc + val * Enum.at(x, col)
          acc + val * Nx.to_number(Nx.take(x, col))
        end)
      end

    w = Nx.tensor(result)

    #IO.inspect(w, label: "w")

    w
  end

end
