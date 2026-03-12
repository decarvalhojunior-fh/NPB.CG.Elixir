defmodule CGxExamples do

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

  def randomic_dense_matrix(niter, tol, shift) do

    n = 500

    a = MBuilder.generate_spd_dense_npb_like(n, 10.0)
    x = MBuilder.generate_rhs(n)

    CGx.main_loop(nil, shift, a, x, nil, niter, tol)
  end


  def randomic_sparse_matrix(niter, tol, shift) do

    n = 500

    a = MBuilder.generate_spd_sparse(n, 0.05, 10.0)
    x = MBuilder.generate_rhs(n)

    CGx.main_loop(nil, shift, a, x, nil, niter, tol)
  end

  def npb_like_matrix(niter, tol, shift) do

    n = 500

    a = MBuilder.generate_spd_sparse_npb_like(n, 10, 10.0)
    x = MBuilder.generate_rhs(n)

    CGx.main_loop(nil, shift, a, x, nil, niter, tol)
  end

  def simple_csr_matrix_0(niter, tol, shift) do

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

    x = Nx.tensor([1.0, 0.0, 0.0, 1.0])

    CGx.main_loop(nil, shift, a, x, nil, niter, tol)

  end

  def simple_csr_matrix_1(niter, tol, shift) do

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

    a = %CSR1{
      values: values,
      colidx: colidx,
      rowptr: rowptr,
      n: 4
    }

    x = Nx.tensor([1.0, 0.0, 0.0, 1.0])

    CGx.main_loop(nil, shift, a, x, nil, niter, tol)

  end

  def simple_csr_matrix_2(niter, tol, shift) do

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

    x = Nx.tensor([1.0, 0.0, 0.0, 1.0])

    CGx.main_loop(nil, shift, a, x, nil, niter, tol)

  end

  def simple_coo_matrix(niter, tol, shift) do

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
    ])

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

    a = %COO{
      values: values,
      rowidx: rowidx,
      colidx: colidx,
      n: 4
    }

    x = Nx.tensor([1.0, 0.0, 0.0, 1.0])

    CGx.main_loop(nil, shift, a, x, nil, niter, tol)

  end

  def npb_like_csr1_matrix(tol) do

    params = Params.npb_cg_params(:S)

    a = Makea.makea_coo(params.n, params.nonzer, params.shift) |> Makea.coo_to_csr()

    x = MBuilder.generate_rhs(params.n)

    CGx.main_loop(nil, params.shift, a, x, nil, params.niter, tol)

  end

  def npb_like_csr2_matrix(tol) do

    params = Params.npb_cg_params(:S)

    a_ = Makea.makea_coo(params.n, params.nonzer, params.shift) |> Makea.coo_to_csr()

    a = %CSR2{
      values: a_.values,
      colidx: a_.colidx,
      rowptr: a_.rowptr,
      rowidx: CSR2.build_rowidx(a_.rowptr),
      n: params.n
    }

    x = MBuilder.generate_rhs(params.n)

    CGx.main_loop(nil, params.shift, a, x, nil, params.niter, tol)

  end


  def npb_like_coo_matrix(tol) do

    params = Params.npb_cg_params(:S)

    a = Makea.makea_coo(params.n, params.nonzer, params.shift)
    x = MBuilder.generate_rhs(params.n)

    CGx.main_loop(nil, params.shift, a, x, nil, params.niter, tol)

  end

end
