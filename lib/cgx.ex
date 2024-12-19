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

  def main_loop(_, _, _, 0), do: :ok
  def main_loop(shift, a, x, it) do
        {z, rnorm} = cgsolve(a, x)
        zeta = Nx.add(shift, Nx.divide(1, Nx.dot(x, z)))
        IO.puts("it=#{it}, rnorm=#{rnorm |> Nx.to_number}, zeta=#{zeta |> Nx.to_number} ---")
        IO.inspect(z)

        x = Nx.divide(z, enorm(z))

        main_loop(shift, a, x, it-1)
  end

  def cgsolve(a, x) do

      {s} = Nx.shape(x)
      z = Nx.tensor(for _ <- 1..s, do: 0.0)
      r = x
      rho = Nx.dot(r, r)
      p = r
      z = cgsolve_loop(a, z, r, rho, p, 25)

      rnorm = enorm(Nx.subtract(x, Nx.dot(a,z)))

      {z, rnorm}
  end

  def enorm(y) do
    Nx.sqrt(Nx.dot(y,y))
  end

  def cgsolve_loop(_, z, _, _, _, 0), do: z

  def cgsolve_loop(a, z, r, rho, p, i) do

      q = Nx.dot(a, p)
      alpha = Nx.divide(rho, Nx.dot(p, q))
      z = Nx.add(z, Nx.multiply(alpha, p))
      rho0 = rho
      r = Nx.subtract(r, Nx.multiply(alpha, q))
      rho = Nx.dot(r, r)
      beta = Nx.divide(rho, rho0)

      p = Nx.add(r, Nx.multiply(beta, p))

      cgsolve_loop(a, z, r, rho, p, i-1)
  end

  def start(_type, _args) do

    IO.puts("started")

    dim = 4
    niter = 15

    #a = Nx.tensor(for _ <- 1..dim, do: (for _ <- 1..dim, do: 1.0))   # a is a 2D tensor
    #x = Nx.tensor(for _ <- 1..dim, do: 1.0)                          # x is a 1D tensors

    a = Nx.tensor([[1, 3, 2, 1], [2, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]])
    x = Nx.tensor([-3, 0, 4, -2])

    shift = Nx.tensor(1)                                           # shift is a  scalar


    main_loop(shift, a, x, niter)

    IO.puts("finished")
  end

end
