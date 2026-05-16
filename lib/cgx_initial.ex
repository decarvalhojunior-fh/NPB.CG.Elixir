defmodule CGx3 do
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



  def main_loop(z, _, _, _, rnorm, zeta, 0, _), do: {z, Nx.to_number(rnorm), Nx.to_number(zeta)}
  def main_loop(_, shift, a, x, _, _, it, t0) do
        {z, rnorm} = conjgrad(a, x, t0)
        zeta = Nx.add(shift, Nx.divide(1, Nx.dot(x, z)))
        IO.puts("it=#{it}, rnorm=#{rnorm |> Nx.to_number}, zeta=#{zeta |> Nx.to_number} ---")

        x = Nx.divide(z, enorm(z))          # update_x
        main_loop(z, shift, a, x, rnorm, zeta, it-1, t0)
  end

  def main(z, shift, a, x, rnoem, zeta, it) do
    t0 = Nx.broadcast(0.0, {a.n}) |> Nx.as_type(:f64)  # dummy tensor to pass to matvecmul
    main_loop(z, shift, a, x, rnoem, zeta, it, t0)
  end

  def conjgrad(a, x, t0) do

      {s} = Nx.shape(x)

      z = Nx.broadcast(0.0, {s}) |> Nx.as_type(:f64)  # init_conj_grad
      r = x                                           # init_conj_grad
      p = r                                           # init_conj_grad

      rho = Nx.dot(r, r) # ok

      bnorm = enorm(x)

      z = conjgrad_loop(a, z, r, rho, p, 25, bnorm, t0)

      r = MVMulSerial.mv_multiply(a, z, t0)

      rnorm = enorm(Nx.subtract(x, r))

      {z, rnorm}
  end

  def enorm(y) do
    Nx.sqrt(Nx.dot(y,y))
  end

  def conjgrad_loop(_, z, _, _, _, 0, _, _), do: z

  def conjgrad_loop(a, z, r, rho, p, i, bnorm, t0) do

      q = MVMulSerial.mv_multiply(a, p, t0)

      alpha = Nx.divide(rho, Nx.dot(p, q))
      z = Nx.add(z, Nx.multiply(alpha, p))
      r = Nx.subtract(r, Nx.multiply(alpha, q))

      rho0 = rho
      rho = Nx.dot(r, r)
      beta = Nx.divide(rho, rho0)
      p = Nx.add(r, Nx.multiply(beta, p))
      conjgrad_loop(a, z, r, rho, p, i - 1, bnorm, t0)
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
    shift = Nx.tensor(10.0, type: :f64)   # shift is a scalar

    {z, rnorm, zeta} = CGxExamples.parallel_example(niter, shift, tol, g2x2)

    IO.inspect(z, label: "solution")
    IO.inspect(rnorm, label: "solution residual")
    IO.inspect(zeta, label: "eigenvalue estimate")

    IO.inspect(rnorm <= tol, label: "residual less than tol?")

    System.halt(0)

  end


end
