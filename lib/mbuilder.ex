defmodule MBuilder do

   def generate_spd_sparse(n, density \\ 0.05, shift \\ 5.0) do
    key = Nx.Random.key(System.unique_integer())

    {t, key} = Nx.Random.uniform(key, shape: {n, n})

    mask = Nx.less(t, density)

    {t, _} = Nx.Random.normal(key, shape: {n, n})

    values = Nx.multiply(t, mask)

    # tornar simétrica
    sym =
      Nx.divide(
        Nx.add(values, Nx.transpose(values)),
        2
      )

    # adicionar shift diagonal
    Nx.add(sym, Nx.multiply(shift, Nx.eye(n)))
  end

  def generate_spd_dense(n, shift \\ 10.0) do
    key = Nx.Random.key(System.unique_integer())

    {m, _} =
      Nx.Random.normal(key, shape: {n, n})

    mt = Nx.transpose(m)

    a =
      Nx.dot(mt, m)
      |> Nx.add(Nx.multiply(shift, Nx.eye(n)))

    a
  end

  def generate_spd_dense_npb_like(n, shift \\ 10.0) do

    key = Nx.Random.key(System.unique_integer())

    {m, _} = Nx.Random.normal(key, shape: {n,n})

    sym = Nx.divide(
            Nx.add(m, Nx.transpose(m)),
            2
          )

    diag = Nx.sum(Nx.abs(sym), axes: [1])

    Nx.add(sym, Nx.make_diagonal(Nx.add(diag, shift)))
  end

  def generate_rhs(n) do
    key = Nx.Random.key(System.unique_integer())

    {t, _} = Nx.Random.normal(key, shape: {n})
    t
  end

  def generate_spd_sparse_npb_like(n, nz_per_row \\ 10, shift \\ 10.0) do
    key = Nx.Random.key(System.unique_integer())
    rows =
      for i <- 0..(n-1) do
        cols =
          Enum.take_random(0..(n-1), nz_per_row)

        Enum.map(cols, fn j ->
          {val, _} = Nx.Random.normal(key, shape: {})

          val |> Nx.to_number()
        end)
      end

    dense =
      Nx.tensor(
        for {row, i} <- Enum.with_index(rows) do
          r = List.duplicate(0.0, n)

          Enum.reduce(Enum.with_index(row), r, fn {v, j}, acc ->
            List.replace_at(acc, j, v)
          end)
        end
      )

     # IO.inspect(dense, label: "dense")

    sym = Nx.divide(Nx.add(dense, Nx.transpose(dense)), 2)

    Nx.add(sym, Nx.multiply(shift, Nx.eye(n)))
 end

end
