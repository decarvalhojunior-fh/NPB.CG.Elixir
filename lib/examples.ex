defmodule CGxExamples do

  def simple_example(niter, shift, tol) do

    a = Nx.tensor([
            [4.0, 1.0],
            [1.0, 3.0]
      ], type: :f64)

    x = Nx.tensor([1.0, 2.0], type: :f64)

    CGx1.main_loop(shift, a, x, niter, tol)

  end

  def laplacian_1d_matrix(niter, shift, tol) do

    a = Nx.tensor([
      [ 2.0, -1.0,  0.0,  0.0],
      [-1.0,  2.0, -1.0,  0.0],
      [ 0.0, -1.0,  2.0, -1.0],
      [ 0.0,  0.0, -1.0,  2.0]
    ], type: :f64)

    x = Nx.tensor([1.0, 0.0, 0.0, 1.0], type: :f64)

    CGx1.main_loop(shift, a, x, niter, tol)
  end

  def randomic_dense_matrix(niter, shift, tol) do

    n = 7000

    IO.inspect(n, label: "matrix size n")
    a = MBuilder.generate_spd_dense_npb_like(n, 10.0)

    IO.puts("starting untimed iteration...")
    # untimed iteration
    x = Nx.broadcast(1.0, {n}) |> Nx.as_type(:f64)  # generate_rhs(n)
    CGx1.main_loop(shift, a, x, 1, tol)

    IO.puts("starting timed iterations...")

    x = Nx.broadcast(1.0, {n}) |> Nx.as_type(:f64)  # generate_rhs(n)

    {timed_us, {z, rnorm, zeta}} = :timer.tc(fn ->
      CGx1.main_loop(shift, a, x, niter, tol)
    end)

    IO.puts("Tempo: #{timed_us / 1_000_000} segundos")
  #  IO.puts("Tempo de multiplicação de vetor: #{mvtime / 1_000_000} segundos")

    {z, rnorm, zeta}
  end


  def randomic_sparse_matrix(niter, shift, tol) do

    n = 1400

    a = MBuilder.generate_spd_sparse(n, 0.05, 10.0)

    IO.puts("starting untimed iteration...")
    x = Nx.broadcast(1.0, {n}) |> Nx.as_type(:f64) # generate_rhs(n)
    CGx1.main_loop(shift, a, x, 1, tol)

    IO.puts("starting timed iterations...")
    x = Nx.broadcast(1.0, {n}) |> Nx.as_type(:f64) # generate_rhs(n)

    {timed_us, {z, rnorm, zeta}} = :timer.tc(fn ->
      CGx1.main_loop(shift, a, x, niter, tol)
    end)

    IO.puts("Tempo: #{timed_us / 1_000_000} segundos")

    {z, rnorm, zeta}
  end

  def npb_like_matrix(niter, shift, tol) do

    n = 1000

#    a = MBuilder.generate_spd_sparse_npb_like(n, 10, 10.0)

    a = Nx.with_default_backend(Nx.BinaryBackend, fn ->
        MBuilder.generate_spd_sparse_npb_like(n, 10, 10.0)
      end)

    IO.puts("starting untimed iteration...")
    x = Nx.broadcast(1.0, {n}) |> Nx.as_type(:f64) # generate_rhs(n)
    CGx1.main_loop(shift, a, x, 1, tol)

    IO.puts("starting timed iterations...")
    x = Nx.broadcast(1.0, {n}) |> Nx.as_type(:f64) # generate_rhs(n)

    {timed_us, {z, rnorm, zeta}} = :timer.tc(fn ->
      CGx1.main_loop(shift, a, x, niter, tol)
    end)

    IO.puts("Tempo: #{timed_us / 1_000_000} segundos")

    {z, rnorm, zeta}
  end

  def simple_csr_matrix_0(niter, shift, _tol) do

    values =[
                 2.0, -1.0,
                -1.0,  2.0, -1.0,
                -1.0, 2.0, -1.0,
                -1.0, 2.0
    ]
    colidx =[
                 0, 1,
                 0, 1, 2,
                 1, 2, 3,
                 2, 3
    ]
    rowptr = [
                 0,
                 2,
                 5,
                 8,
                 10
             ]

    a = %CSR0{
      values: values,
      colidx: colidx,
      rowptr: rowptr,
      n: 4
    }

    x = Nx.tensor([1.0, 0.0, 0.0, 1.0]) |> Nx.as_type(:f64)

    CGx.main(nil, shift, a, x, nil, nil, niter)

  end

  def simple_csr_matrix_1(niter, shift, _tol) do

    values = Nx.tensor([
                 2.0, -1.0,
                -1.0,  2.0, -1.0,
                -1.0, 2.0, -1.0,
                -1.0, 2.0
    ], type: :f64)

    colidx = Nx.tensor([
                 0, 1,
                 0, 1, 2,
                 1, 2, 3,
                 2, 3
    ], type: :s32)

    rowptr = Nx.tensor([
                 0,
                 2,
                 5,
                 8,
                 10
             ], type: :s32)

    a = %CSR1{
      values: values,
      colidx: colidx,
      rowptr: rowptr,
      n: 4
    }

    x = Nx.tensor([1.0, 0.0, 0.0, 1.0])  |> Nx.as_type(:f64)

    CGx.main(nil, shift, a, x, nil, nil, niter)

  end

  def simple_csr_matrix_2(niter, shift, _tol) do

    values = Nx.tensor([
                 2.0, -1.0,
                -1.0,  2.0, -1.0,
                -1.0, 2.0, -1.0,
                -1.0, 2.0
    ])
    colidx = Nx.tensor([
                 0, 1,
                 0, 1, 2,
                 1, 2, 3,
                 2, 3
    ], type: :s32)
    rowptr = Nx.tensor([
                 0,
                 2,
                 5,
                 8,
                 10
             ], type: :s32)

    rowidx = CSR2.build_rowidx(rowptr)

    a = %CSR2{
      values: values,
      colidx: colidx,
      rowptr: rowptr,
      rowidx: rowidx,
      n: 4
    }

    x = Nx.tensor([1.0, 0.0, 0.0, 1.0])  |> Nx.as_type(:f64)

    CGx.main(nil, shift, a, x, nil, nil, niter)

  end

  def simple_coo_matrix(niter, shift, _tol) do

    values = Nx.tensor([
      2.0,
      -1.0,
      -1.0,
      2.0,
      -1.0,
      -1.0,
      2.0,
      -1.0,
      -1.0,
      2.0
    ], type: :f64)

    rowidx = Nx.tensor([
      0,
      0,
      1,
      1,
      1,
      2,
      2,
      2,
      3,
      3
    ], type: :s32)

    colidx = Nx.tensor([
      0,
      1,
      0,
      1,
      2,
      1,
      2,
      3,
      2,
      3
    ], type: :s32)

#    a = %COO{
#      values: values,
#      rowidx: rowidx,
#      colidx: colidx,
#      n: 4
#    }

    IO.puts("starting untimed iteration...")
    x = Nx.tensor([1.0, 0.0, 0.0, 1.0], type: :f64)
    CGx2.main(shift, values, colidx, rowidx, x, 1)

    IO.puts("starting timed iterations...")
    x = Nx.tensor([1.0, 0.0, 0.0, 1.0], type: :f64)
     {timed_us, {z, rnorm, zeta}} = :timer.tc(fn ->
        CGx2.main(shift, values, colidx, rowidx, x, niter)
     end)

    IO.puts("Tempo: #{timed_us / 1_000_000} segundos")

    {z, rnorm, zeta}

  end

  def npb_like_csr1_matrix(_tol) do

    params = Params.npb_cg_params(:W)

    {timed_makea, {values, colidx, rowidx}} = :timer.tc(fn ->
        Nx.with_default_backend(Nx.BinaryBackend, fn ->
                 Makea.makea_coo(params.n, params.nonzer, params.shift)
        end)
    end)

    IO.puts("Tempo makea: #{timed_makea / 1_000_000} segundos")

    {values, colidx, rowptr} = Makea.coo_to_csr(values, colidx, rowidx)

    a = %CSR1{
      values: values,
      colidx: colidx,
      rowptr: rowptr,
      n: params.n
    }

    IO.puts("starting untimed iteration...")
    x = Nx.broadcast(1.0, {params.n})  |> Nx.as_type(:f64) # generate_rhs(params.n)
    CGx.main(nil, params.shift, a, x, nil, nil, 1)

    IO.puts("starting timed iterations...")
    x = Nx.broadcast(1.0, {params.n})  |> Nx.as_type(:f64) # generate_rhs(params.n)
    {timed_us, {z, rnorm, zeta}} = :timer.tc(fn ->
      CGx.main(nil, params.shift, a, x, nil, nil, params.niter)
    end)

    IO.puts("Tempo: #{timed_us / 1_000_000} segundos")

    {z, rnorm, zeta}
  end

  def npb_like_csr2_matrix(_tol) do

    params = Params.npb_cg_params(:B)

    {timed_makea, {values, colidx, rowidx}} = :timer.tc(fn ->
                    Nx.with_default_backend(Nx.BinaryBackend, fn ->
                            Makea.makea_coo(params.n, params.nonzer, params.shift)
                    end)
    end)

    IO.puts("Tempo makea: #{timed_makea / 1_000_000} segundos")

  #  IO.inspect(values, label: "1-values")
  #  IO.inspect(colidx, label: "1-colidx")
  #  IO.inspect(rowidx, label: "1-rowidx")

    {values, colidx, rowptr} = Makea.coo_to_csr(values, colidx, rowidx)

    a = %CSR2{
      values: values,
      colidx: colidx,
      rowptr: rowptr,
      rowidx: CSR2.build_rowidx(rowptr),
      n: params.n
    }

  #  IO.inspect(a.values, label: "2-values")
  #  IO.inspect(a.colidx, label: "2-colidx")
  #  IO.inspect(a.rowidx, label: "2-rowidx")



    IO.puts("starting untimed iteration...")
    x = Nx.broadcast(1.0, {params.n})  |> Nx.as_type(:f64) # generate_rhs(params.n)
    CGx.main(nil, params.shift, a, x, nil, nil, 1)

    IO.puts("starting timed iterations...")
    x = Nx.broadcast(1.0, {params.n})  |> Nx.as_type(:f64) # generate_rhs(params.n)
    {timed_us, {z, rnorm, zeta}} = :timer.tc(fn ->
      CGx.main(nil, params.shift, a, x, nil, nil, params.niter)
    end)

    IO.puts("Tempo: #{timed_us / 1_000_000} segundos")

    {z, rnorm, zeta}

  end


  def npb_like_coo_matrix_exla(_tol) do

    main_jit = Nx.Defn.jit(&CGx2.main/6)

    params = Params.npb_cg_params(:B)

    IO.puts("generating matrix in COO format...")
    {timed_makea, {values, colidx, rowidx}} = :timer.tc(fn ->
                    Nx.with_default_backend(Nx.BinaryBackend, fn ->
                            Makea.makea_coo(params.n, params.nonzer, params.shift)
                    end)
    end)

    IO.puts("Tempo makea: #{timed_makea / 1_000_000} segundos")

    {values, rowidx, colidx} = Makea.sort_coo(values, rowidx, colidx)

    IO.puts("starting untimed iteration... defn")
    x = Nx.broadcast(1.0, {params.n})  |> Nx.as_type(:f64) # generate_rhs(params.n)
    main_jit.(params.shift, values, colidx, rowidx, x, 1)

    IO.puts("starting timed iterations... defn")
    x = Nx.broadcast(1.0, {params.n})  |> Nx.as_type(:f64) # generate_rhs(params.n)

    {timed_us, {z, rnorm, zeta, it}} = :timer.tc(fn ->
        main_jit.(params.shift, values, colidx, rowidx, x, params.niter)
     end)

    IO.puts("Tempo: #{timed_us / 1_000_000} segundos - it=#{it |> Nx.to_number}")

    {z, rnorm, zeta}
  end

  def benchmark_cgx2_exla_types do

    main_jit = Nx.Defn.jit(&CGx2.main/6)

    params = Params.npb_cg_params(:B)

    IO.puts("generating matrix in COO format...")
    {timed_makea, {values, colidx, rowidx}} = :timer.tc(fn ->
                    Nx.with_default_backend(Nx.BinaryBackend, fn ->
                            Makea.makea_coo(params.n, params.nonzer, params.shift)
                    end)
    end)

    IO.puts("Tempo makea: #{timed_makea / 1_000_000} segundos")

    {values, rowidx, colidx} = Makea.sort_coo(values, rowidx, colidx)

    [
      {:f64, values |> Nx.as_type(:f64), Nx.broadcast(1.0, {params.n}) |> Nx.as_type(:f64)},
      {:f32, values |> Nx.as_type(:f32), Nx.broadcast(1.0, {params.n}) |> Nx.as_type(:f32)}
    ]
    |> Enum.map(fn {type, typed_values, x} ->
      IO.puts("warming up #{type}...")
      main_jit.(params.shift, typed_values, colidx, rowidx, x, 1)

      IO.puts("timing #{type}...")

      {timed_us, {_z, rnorm, zeta, it}} = :timer.tc(fn ->
        main_jit.(params.shift, typed_values, colidx, rowidx, x, params.niter)
      end)

      seconds = timed_us / 1_000_000

      IO.puts(
        "Tempo #{type}: #{seconds} segundos - it=#{it |> Nx.to_number} - rnorm=#{rnorm |> Nx.to_number} - zeta=#{zeta |> Nx.to_number}"
      )

      {type, seconds}
    end)
  end

  def npb_like_coo_matrix_cpu_1(_tol) do

    params = Params.npb_cg_params(:B)

   IO.puts("generating matrix in COO format...")
    {timed_makea, {values, colidx, rowidx}} = :timer.tc(fn ->
                    Nx.with_default_backend(Nx.BinaryBackend, fn ->
                            Makea.makea_coo(params.n, params.nonzer, params.shift)
                    end)
    end)

    IO.puts("Tempo makea: #{timed_makea / 1_000_000} segundos")

    {values, rowidx, colidx} = Makea.sort_coo(values, rowidx, colidx)

    IO.inspect(Nx.size(values), label: "#values")

    a = %COO{
      values: values,
      rowidx: rowidx,
      colidx: colidx,
      n: params.n
    }

    #Nx.with_default_backend(Nx.BinaryBackend, fn ->
      IO.puts("starting untimed iteration... no defn")
      x = Nx.broadcast(1.0, {params.n})  |> Nx.as_type(:f64) # generate_rhs(params.n)
      CGx3.main(nil, params.shift, a, x, nil, nil, 1)

      IO.puts("starting timed iterations... no defn")
      x = Nx.broadcast(1.0, {params.n})  |> Nx.as_type(:f64) # generate_rhs(params.n)

      {timed_us, {z, rnorm, zeta}} = :timer.tc(fn ->
          CGx3.main(nil, params.shift, a, x, nil, nil, params.niter)
       end)

      IO.puts("Tempo: #{timed_us / 1_000_000} segundos")

      {z, rnorm, zeta}
    #end)

  end

def npb_like_coo_matrix_parallel_launcher(nodes, problem_size, is_sparse) do

    IO.puts("Spawning processes on nodes: #{inspect(nodes)}")

    pid_workers = Enum.map(nodes, fn node ->
        IO.puts("Spawning process on node #{node}...")
          spawn_worker(node, problem_size, is_sparse)
        end)

    IO.inspect(pid_workers, label: "spawned pids")

    for pid <- pid_workers do
      send(pid, {:pid_manager, self()})
      send(pid, {:pid_workers, pid_workers})
    end

    IO.puts("Sent pids to all processes. Waiting for results...")

    receive do
      {:result, result} -> result
    end

  end


def npb_like_coo_matrix_parallel(problem_size, is_sparse) do

   IO.puts("Process #{inspect(self())} started and waiting for pids...")

   pid_manager = receive do
            {:pid_manager, pid} -> pid
       end

   pid_workers = receive do
            {:pid_workers, pids} -> pids
       end

   IO.puts("Process #{inspect(self())} received pids: #{inspect(pid_workers)}")

   nprocs = length(pid_workers)

   {npcols, nprows} =  CGx.get_active_procs(3, nprocs)

   # build the groups for parallel execution

   group_solve = Group.new_group(pid_workers, topology: {nprows, npcols})
   group_row = Group.group_split(group_solve, [0])

   IO.inspect(group_row, label: "group_row")

   params = Params.npb_cg_params(problem_size)

   {firstrow, lastrow, firstcol, lastcol} =
          if is_nil(group_solve) do
              {1, params.n, 1, params.n}
          else
              CGx.setup_submatrix_info(params.n, group_solve)
          end

   IO.puts("generating matrix in COO format...")
    {timed_makea, {values, colidx, rowidx}} = :timer.tc(fn ->
                    Nx.with_default_backend(Nx.BinaryBackend, fn ->
                        Makea.makea_coo(params.n, params.nonzer, params.shift, firstrow: firstrow, lastrow: lastrow, firstcol: firstcol, lastcol: lastcol)
                    end)
    end)

    IO.puts("Tempo makea: #{timed_makea / 1_000_000} segundos")

    {values, rowidx, colidx} = Makea.sort_coo(values, rowidx, colidx)

    IO.inspect(Nx.size(values), label: "#values")

    n_rows = lastrow - firstrow + 1
    n_cols = lastcol - firstcol + 1

    IO.inspect({group_solve.coord, firstrow, lastrow, firstcol, lastcol}, label: "===> ")


    a = if is_sparse do
      %COO{
        values: values,
        rowidx: rowidx,
        colidx: Nx.subtract(colidx, firstcol - 1),
        n: n_rows
      }
    else
      IO.inspect({n_rows, n_cols}, label: "dense n = ")
      Makea.coo_to_dense(values, rowidx, Nx.subtract(colidx, firstcol - 1), n_rows, n_cols)
    end

    IO.inspect(n_rows, label: "n = ")

    IO.puts("starting untimed iteration... no defn")
    x = Nx.broadcast(1.0, {lastcol - firstcol + 1})  |> Nx.as_type(:f64) # generate_rhs(params.n)
    CGx.main(nil, params.shift, n_rows, a, x, nil, nil, 1, group_row: group_row, group_solve: group_solve)

    IO.puts("starting timed iterations... no defn")
    x = Nx.broadcast(1.0, {lastcol - firstcol + 1})  |> Nx.as_type(:f64) # generate_rhs(params.n)

    {timed_us, {z, rnorm, zeta}} = :timer.tc(fn ->
        CGx.main(nil, params.shift, n_rows, a, x, nil, nil, params.niter, group_row: group_row, group_solve: group_solve)
      end)

    IO.puts("Tempo: #{timed_us / 1_000_000} segundos")

    if self() == hd(pid_workers) do
      send_result(pid_manager, {z, rnorm, zeta})
    end

    IO.puts("Process #{inspect(self())} finished computation and sent results to #{inspect(pid_manager)}")

  end

  defp spawn_worker(worker_node, problem_size, is_sparse) when is_atom(worker_node) do
    #if worker_node == Node.self() do
    #  spawn_link(CGxExamples, :npb_like_coo_matrix_parallel, [tol])
    #else
      Node.spawn_link(worker_node, CGxExamples, :npb_like_coo_matrix_parallel, [problem_size, is_sparse])
    #end
  end

  defp spawn_worker(_worker_node, problem_size, is_sparse) do
    spawn_link(CGxExamples, :npb_like_coo_matrix_parallel, [problem_size, is_sparse])
  end

  defp send_result(pid_manager, result) do
    result =
      if node(pid_manager) == Node.self() do
        result
      else
        transfer_tensors_to_binary(result)
      end

    send(pid_manager, {:result, result})
  end

  defp transfer_tensors_to_binary({z, rnorm, zeta}) do
    {
      transfer_tensor_to_binary(z),
      transfer_tensor_to_binary(rnorm),
      transfer_tensor_to_binary(zeta)
    }
  end

  defp transfer_tensor_to_binary(%Nx.Tensor{} = tensor) do
    Nx.backend_transfer(tensor, Nx.BinaryBackend)
  end

  defp transfer_tensor_to_binary(value), do: value

end
