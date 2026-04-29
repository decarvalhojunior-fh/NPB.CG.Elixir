defmodule CGxww do
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
  def matvecmul(%COO{values: v, rowidx: ri, colidx: c, n: n}, x) do
    COO.spmv(%COO{values: v, rowidx: ri, colidx: c, n: n}, x)
  end

  # a is dense
  def matvecmul(a, x) do
    Nx.dot(a, x)
  end

  def main_loop(z, _, _, _, rnorm, zeta, 0, _, mvtime), do: {z, Nx.to_number(rnorm), Nx.to_number(zeta), mvtime}
  def main_loop(_, shift, a, x, _, _, it, tol, mvtime) do
        {z, rnorm, mvtime} = conjgrad(a, x, tol, mvtime)
        zeta = Nx.add(shift, Nx.divide(1, Nx.dot(x, z)))
        IO.puts("it=#{it}, rnorm=#{rnorm |> Nx.to_number}, zeta=#{zeta |> Nx.to_number} ---")

        if Nx.to_number(rnorm) < tol do
          {z, Nx.to_number(rnorm), zeta, mvtime}
        else
          x = Nx.divide(z, enorm(z))          # update_x
          main_loop(z, shift, a, x, rnorm, zeta, it-1, tol, mvtime)
        end
  end

  def conjgrad(a, x, tol, mvtime) do

      {s} = Nx.shape(x)

      z = Nx.broadcast(0.0, {s}) |> Nx.as_type(:f64)  # init_conj_grad
      r = x                                           # init_conj_grad
      p = r                                           # init_conj_grad

      rho = Nx.dot(r, r) # ok

      bnorm = enorm(x)

      {z, mvtime} = conjgrad_loop(a, z, r, rho, p, 25, tol, bnorm, mvtime)

      {mvtime0, r} = :timer.tc(fn -> matvecmul(a, z) end)
      mvtime = mvtime + mvtime0

      rnorm = enorm(Nx.subtract(x, r))

      {z, rnorm, mvtime}
  end

  def enorm(y) do
    Nx.sqrt(Nx.dot(y,y))
  end

  def conjgrad_loop(_, z, _, _, _, 0, _, _, mvtime), do: {z, mvtime}

  def conjgrad_loop(a, z, r, rho, p, i, tol, bnorm, mvtime) do


      {mvtime0, q} = :timer.tc(fn -> matvecmul(a, p) end)

      mvtime = mvtime + mvtime0

      alpha = Nx.divide(rho, Nx.dot(p, q))
      z = Nx.add(z, Nx.multiply(alpha, p))
      r = Nx.subtract(r, Nx.multiply(alpha, q))

      rnorm = enorm(r)
      #IO.inspect(Nx.to_number(Nx.divide(rnorm, bnorm)), label: "relative residual")

      if Nx.to_number(Nx.divide(rnorm, bnorm)) < tol do
        {z, mvtime}
      else
        rho0 = rho
        rho = Nx.dot(r, r)
        beta = Nx.divide(rho, rho0)
        p = Nx.add(r, Nx.multiply(beta, p))
        conjgrad_loop(a, z, r, rho, p, i - 1, tol, bnorm, mvtime)
      end
  end



  def start(_type, _args) do
    children = []

    opts = [strategy: :one_for_one, name: CGx.Supervisor]
    Supervisor.start_link(children, opts)
  end

  def main_serial(tol \\ 0.0) do


    {z, rnorm, zeta} = CGxExamples.npb_like_coo_matrix_cpu_1(tol)

    IO.inspect(z, label: "solution")
    IO.inspect(rnorm, label: "solution residual")
    IO.inspect(zeta, label: "eigenvalue estimate")

    IO.inspect(rnorm <= tol, label: "residual less than tol?")

    System.halt(0)

  end

  def main_parallel(hostname) do

    #hostname = "heron-Inspiron-7460"
    nodes = Enum.map(["p0", "p1", "p2"], fn sname -> String.to_atom(sname <> "@" <> hostname) end)

    pids = [
      self()
      | Enum.map(nodes, fn node ->
          Node.spawn_link(node, CollectiveElixir, :perform_computation, [])
        end)
    ]

    for pid <- pids do
      send(pid, {:pids, pids})
    end

    g2x2 = Group.new_group(pids, topology: [2, 2])

    niter = 15
    tol = 1.0e-5
    shift = Nx.tensor(10)   # shift is a scalar

    {z, rnorm, zeta} = CGxExamples.parallel_example(niter, shift, tol, g2x2)

    IO.inspect(z, label: "solution")
    IO.inspect(rnorm, label: "solution residual")
    IO.inspect(zeta, label: "eigenvalue estimate")

    IO.inspect(rnorm <= tol, label: "residual less than tol?")

    System.halt(0)

  end


end
