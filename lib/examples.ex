defmodule CGxExamples do

  def simple_example(niter, shift) do

    a = Nx.tensor([
            [4.0, 1.0],
            [1.0, 3.0]
      ], type: :f64)

    x = Nx.tensor([1.0, 2.0], type: :f64)

    tol = 1.0e-7
    CGx1.main_loop(shift, a, x, niter, tol)

  end

  def laplacian_1d_matrix(niter, shift) do

    a = Nx.tensor([
      [ 2.0, -1.0,  0.0,  0.0],
      [-1.0,  2.0, -1.0,  0.0],
      [ 0.0, -1.0,  2.0, -1.0],
      [ 0.0,  0.0, -1.0,  2.0]
    ], type: :f64)

    x = Nx.tensor([1.0, 0.0, 0.0, 1.0], type: :f64)

    tol = 1.0e-7
    CGx1.main_loop(shift, a, x, niter, tol)
  end

  def randomic_dense_matrix(niter, shift) do

    n = 7000

    tol = 0.0

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

  def simple_csr_matrix_0(niter, shift, tol) do

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

    CGx.main_loop(nil, shift, a, x, nil, niter, tol, 0.0)

  end

  def simple_csr_matrix_1(niter, shift, tol) do

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

    CGx.main_loop(nil, shift, a, x, nil, niter, tol, 0.0)

  end

  def simple_csr_matrix_2(niter, shift, tol) do

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

    CGx.main_loop(nil, shift, a, x, nil, niter, tol, 0.0)

  end

  def simple_coo_matrix(niter, shift, tol) do

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
    CGx2.main_loop(shift, values, colidx, rowidx, x, 1, tol)

    IO.puts("starting timed iterations...")
    x = Nx.tensor([1.0, 0.0, 0.0, 1.0], type: :f64)
     {timed_us, {z, rnorm, zeta}} = :timer.tc(fn ->
        CGx2.main_loop(shift, values, colidx, rowidx, x, niter, tol)
     end)

    IO.puts("Tempo: #{timed_us / 1_000_000} segundos")

    {z, rnorm, zeta}

  end

  def npb_like_csr1_matrix(tol) do

    params = Params.npb_cg_params(:S)

    {values, colidx, rowidx} = Nx.with_default_backend(Nx.BinaryBackend, fn ->
      Makea.makea_coo(params.n, params.nonzer, params.shift)
    end)

    {values, colidx, rowptr} = Makea.coo_to_csr(values, colidx, rowidx)

    a = %CSR1{
      values: values,
      colidx: colidx,
      rowptr: rowptr,
      n: params.n
    }

    IO.puts("starting untimed iteration...")
    x = Nx.broadcast(1.0, {params.n})  |> Nx.as_type(:f64) # generate_rhs(params.n)
    CGx.main_loop(nil, params.shift, a, x, nil, 1, tol, 0.0)

    IO.puts("starting timed iterations...")
    x = Nx.broadcast(1.0, {params.n})  |> Nx.as_type(:f64) # generate_rhs(params.n)
    {timed_us, {z, rnorm, zeta}} = :timer.tc(fn ->
      CGx.main_loop(nil, params.shift, a, x, nil, params.niter, tol, 0.0)
    end)

    IO.puts("Tempo: #{timed_us / 1_000_000} segundos")

    {z, rnorm, zeta}
  end

  def npb_like_csr2_matrix(tol) do

    params = Params.npb_cg_params(:S)

    {values, colidx, rowidx} = Nx.with_default_backend(Nx.BinaryBackend, fn ->
      Makea.makea_coo(params.n, params.nonzer, params.shift)
    end)

    {values, colidx, rowptr} = Makea.coo_to_csr(values, colidx, rowidx)

    a = %CSR2{
      values: values,
      colidx: colidx,
      rowptr: rowptr,
      rowidx: CSR2.build_rowidx(rowptr),
      n: params.n
    }

    IO.puts("starting untimed iteration...")
    x = Nx.broadcast(1.0, {params.n})  |> Nx.as_type(:f64) # generate_rhs(params.n)
    CGx.main_loop(nil, params.shift, a, x, nil, 1, tol, 0.0)

    IO.puts("starting timed iterations...")
    x = Nx.broadcast(1.0, {params.n})  |> Nx.as_type(:f64) # generate_rhs(params.n)
    {timed_us, {z, rnorm, zeta}} = :timer.tc(fn ->
      CGx.main_loop(nil, params.shift, a, x, nil, params.niter, tol, 0.0)
    end)

    IO.puts("Tempo: #{timed_us / 1_000_000} segundos")

    {z, rnorm, zeta}

  end


  def npb_like_coo_matrix(tol) do

    params = Params.npb_cg_params(:B)

    IO.puts("generating matrix in COO format...")
    {values, colidx, rowidx} = Nx.with_default_backend(Nx.BinaryBackend, fn ->
      Makea.makea_coo(params.n, params.nonzer, params.shift)
    end)

    IO.puts("starting untimed iteration...")
    x = Nx.broadcast(1.0, {params.n})  |> Nx.as_type(:f64) # generate_rhs(params.n)
    CGx2.main_loop(params.shift, values, colidx, rowidx, x, 1, tol)

    IO.puts("starting timed iterations...")
    x = Nx.broadcast(1.0, {params.n})  |> Nx.as_type(:f64) # generate_rhs(params.n)

    {timed_us, {z, rnorm, zeta}} = :timer.tc(fn ->
        CGx2.main_loop(params.shift, values, colidx, rowidx, x, params.niter, tol)
     end)

    IO.puts("Tempo: #{timed_us / 1_000_000} segundos")

    {z, rnorm, zeta}
  end

end
