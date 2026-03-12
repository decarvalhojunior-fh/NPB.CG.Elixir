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

  # a is sparse in CSR format (using lists for values, colidx, and rowptr)
  def matvecmul(%CSR0{values: v, colidx: c, rowptr: r, n: n}, x) do
     #x = Nx.to_list(x)
     CSR0.spmv(%CSR0{values: v, colidx: c, rowptr: r, n: n}, x)
  end

  # a is sparse in CSR format (using tensors for values, colidx, and rowptr)
  def matvecmul(%CSR1{values: v, colidx: c, rowptr: r, n: n}, x) do
     CSR1.spmv(%CSR1{values: v, colidx: c, rowptr: r, n: n}, x)
  end

  # a is sparse in CSR format, with rowidx (i.e., COO-based spmv)
  def matvecmul(%CSR2{values: v, colidx: c, rowidx: ri, n: n}, x) do
     CSR2.spmv(%CSR2{values: v, colidx: c, rowidx: ri, n: n}, x)
  end

  # a is sparse in COO format
  def matvecmul(%COO{values: v, rowidx: r, colidx: c, n: n}, x) do
    COO.spmv(%COO{values: v, rowidx: r, colidx: c, n: n}, x)
  end

  # a is dense
  def matvecmul(a, x) do
    Nx.dot(a, x)
  end

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

      rnorm = enorm(Nx.subtract(x, matvecmul(a, z)))

      {z, rnorm}
  end

  def enorm(y) do
    Nx.sqrt(Nx.dot(y,y))
  end

  def conjgrad_loop(_, z, _, _, _, 0, _, _), do: z

  def conjgrad_loop(a, z, r, rho, p, i, tol, bnorm) do

      q = matvecmul(a, p)
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



  def start(_type, _args) do
    children = []

    opts = [strategy: :one_for_one, name: CGx.Supervisor]
    Supervisor.start_link(children, opts)
  end

  def main(_args \\ []) do

    niter = 15
    tol = 1.0e-5
    shift = Nx.tensor(10)   # shift is a scalar

    {z, rnorm} = CGxExamples.randomic_sparse_matrix(niter, tol, shift)


    IO.inspect(z, label: "solution")
    IO.inspect(rnorm, label: "solution residual")

    IO.inspect(rnorm <= tol, label: "residual less than tol?")

    System.halt(0)

  end



end
