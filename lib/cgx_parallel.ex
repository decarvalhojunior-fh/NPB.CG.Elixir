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



  def main_loop(z, _, _, _, rnorm, zeta, 0, _, _, _), do: {z, Nx.to_number(rnorm), Nx.to_number(zeta)}
  def main_loop(_, shift, a, x, _, _, it, t0, group_row, group_solve) do

        {z, rnorm} = conjgrad(a, x, t0, group_row, group_solve)

        norm1 = VVMulParallel.vv_multiply(x, z, group_row: group_row) # all reduce
        norm2 = VVMulParallel.vv_multiply(z, z, group_row: group_row) # all reduce

        zeta = Nx.add(shift, Nx.divide(1.0, norm1))

        (is_nil(group_solve) || group_solve.rank == 0)
              && IO.puts("it=#{it}, rnorm=#{rnorm |> Nx.to_number}, zeta=#{zeta |> Nx.to_number} ---")

        x = Nx.divide(z, Nx.sqrt(norm2))          # update_x
        main_loop(z, shift, a, x, rnorm, zeta, it-1, t0, group_row, group_solve)
  end

  def main(z, shift, a, x, rnorm, zeta, it, opts \\ []) do
    t0 = Nx.broadcast(0.0, {a.n}) |> Nx.as_type(:f64)  # dummy tensor to pass to matvecmul
    group_row = opts[:group_row] || nil
    group_solve = opts[:group_solve] || nil
    main_loop(z, shift, a, x, rnorm, zeta, it, t0, group_row, group_solve)
  end

  def conjgrad(a, x, t0, group_row, group_solve) do

      {s} = Nx.shape(x)

      z = Nx.broadcast(0.0, {s}) |> Nx.as_type(:f64)  # init_conj_grad
      r = x                                           # init_conj_grad
      p = r                                           # init_conj_grad

      rho = VVMulParallel.vv_multiply(r, r, group_row: group_row) # all reduce

      z = conjgrad_loop(a, z, r, rho, p, 25, t0, group_row, group_solve)

      r = MVMulParallel.mv_multiply(a, z, t0, group_row: group_row, group_solve: group_solve) # all reduce -> all to all v

      d0 = Nx.subtract(x, r)

      d = VVMulParallel.vv_multiply(d0, d0, group_row: group_row) # all reduce

      rnorm = Nx.sqrt(d)

      {z, rnorm}
  end


  def conjgrad_loop(_, z, _, _, _, 0, _, _, _), do: z

  def conjgrad_loop(a, z, r, rho, p, i, t0, group_row, group_solve) do

      q = MVMulParallel.mv_multiply(a, p, t0, group_row: group_row, group_solve: group_solve) # all reduce -> all to all v

       #i == 24 && exit(0)

      d = VVMulParallel.vv_multiply(p, q, group_row: group_row) # all reduce

      alpha = Nx.divide(rho, d)
      z = Nx.add(z, Nx.multiply(alpha, p))
      r = Nx.subtract(r, Nx.multiply(alpha, q))

      rho0 = rho
      rho = VVMulParallel.vv_multiply(r, r, group_row: group_row) # all reduce
      beta = Nx.divide(rho, rho0)
      p = Nx.add(r, Nx.multiply(beta, p))

      conjgrad_loop(a, z, r, rho, p, i - 1, t0, group_row, group_solve)
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

  def get_active_procs(pkind, nprocs) do

    if pkind <= 1 do
      np2 = :math.sqrt(nprocs) |> trunc
      np1 = np2
      {np1, np2}
    else
      np1 = :math.log(nprocs) / :math.log(2) |> trunc
      np2 = div(np1, 2)
      np1 = np1 - np2
      np1 = 2 ** np1
      np2 = 2 ** np2
      {np1, np2}
    end

  end


  def setup_submatrix_info(n, group) do
    # partition matrix into blocks of rows for each process

    proc_row = group.coord |> elem(0)
    proc_col = group.coord |> elem(1)

    nprows = group.topology |> elem(0)
    npcols = group.topology |> elem(1)

    if (div(n,npcols)*npcols == n) do
      col_size = div(n, npcols)
      firstcol = proc_col*col_size + 1
      lastcol  = firstcol - 1 + col_size
      row_size = div(n, nprows)
      firstrow = proc_row*row_size + 1
      lastrow  = firstrow - 1 + row_size
      {firstrow, lastrow, firstcol, lastcol}
    else
      IO.puts("n is not evenly divisible by npcols")
      exit(0)
    end

  end


end
