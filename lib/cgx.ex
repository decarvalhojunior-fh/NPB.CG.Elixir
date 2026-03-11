defmodule CGx do
  @moduledoc """
  Documentation for `CGx`.
  """

  @doc """
  Hello world.

  ## Examples

      iex> CGx.hello()
      :world

  """

  use Application

  def main_loop(z, _, _, _, rnorm, 0, _), do: {z, Nx.to_number(rnorm)}
  def main_loop(_, shift, a, x, _, it, tol) do
        {z, rnorm} = conjgrad(a, x, tol)
        zeta = Nx.add(shift, Nx.divide(1, Nx.dot(x, z)))
        IO.puts("it=#{it}, rnorm=#{rnorm |> Nx.to_number}, zeta=#{zeta |> Nx.to_number} ---")

        if Nx.to_number(rnorm) < tol do
          {z, Nx.to_number(rnorm)}
        else
          x = Nx.divide(z, enorm(z))          # update_x
          main_loop(z, shift, a, x, rnorm, it-1, tol)
        end
  end

  def conjgrad(a, x, tol) do

      {s} = Nx.shape(x)

      z = Nx.broadcast(0.0, {s})            # init_conj_grad
      r = x                                 # init_conj_grad
      p = r                                 # init_conj_grad

      rho = Nx.dot(r, r) # ok

      bnorm = enorm(x)

      z = conjgrad_loop(a, z, r, rho, p, 25, tol, bnorm)

      rnorm = enorm(Nx.subtract(x, Nx.dot(a,z)))

      {z, rnorm}
  end

  def enorm(y) do
    Nx.sqrt(Nx.dot(y,y))
  end

  def conjgrad_loop(_, z, _, _, _, 0, _, _), do: z

  def conjgrad_loop(a, z, r, rho, p, i, tol, bnorm) do

      q = Nx.dot(a, p)
      alpha = Nx.divide(rho, Nx.dot(p, q))
      z = Nx.add(z, Nx.multiply(alpha, p))
      r = Nx.subtract(r, Nx.multiply(alpha, q))

      rnorm = enorm(r)

      if Nx.to_number(Nx.divide(rnorm, bnorm)) < tol do
        z
      else
        rho0 = rho
        rho = Nx.dot(r, r)
        beta = Nx.divide(rho, rho0)
        p = Nx.add(r, Nx.multiply(beta, p))
        conjgrad_loop(a, z, r, rho, p, i - 1, tol, bnorm)
      end
  end

  def simple_example(niter, tol, shift) do

    a = Nx.tensor([
            [4.0, 1.0],
            [1.0, 3.0]
      ])

    x = Nx.tensor([1.0, 2.0])

    CGx.main_loop(nil, shift, a, x, nil, niter, tol)

  end

  def laplacian_1d_matrix(niter, tol, shift) do

    a = Nx.tensor([
      [ 2.0, -1.0,  0.0,  0.0],
      [-1.0,  2.0, -1.0,  0.0],
      [ 0.0, -1.0,  2.0, -1.0],
      [ 0.0,  0.0, -1.0,  2.0]
    ])

    x = Nx.tensor([1.0, 0.0, 0.0, 1.0])

    CGx.main_loop(nil, shift, a, x, nil, niter, tol)
  end

  def randomic_sparse_matrix(niter, tol, shift) do

    n = 500

    a = generate_spd_sparse(n, 0.05, 10.0)
    x = generate_rhs(n)

    CGx.main_loop(nil, shift, a, x, nil, niter, tol)
  end

  def npb_like_matrix(niter, tol, shift) do

    n = 500

    a = generate_npb_like_matrix(n, 10, 10.0)
    x = generate_rhs(n)

    CGx.main_loop(nil, shift, a, x, nil, niter, tol)
  end

  def start(_type, _args) do
    children = []

    opts = [strategy: :one_for_one, name: CGx.Supervisor]
    Supervisor.start_link(children, opts)
  end

  def main(_args \\ []) do

    niter = 15
    tol = 1.0e-5
    shift = Nx.tensor(10)   # shift is a scalar

    {z, rnorm} = randomic_sparse_matrix(niter, tol, shift)


    IO.inspect(z, label: "solution")
    IO.inspect(rnorm, label: "solution residual")

    IO.inspect(rnorm <= tol, label: "residual less than tol?")

    System.halt(0)

  end


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

  def generate_rhs(n) do
    key = Nx.Random.key(System.unique_integer())

    {t, _} = Nx.Random.normal(key, shape: {n})
    t
  end

  def generate_npb_like_matrix(n, nz_per_row \\ 10, shift \\ 10.0) do
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
