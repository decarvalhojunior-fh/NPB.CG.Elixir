defmodule CGxTest do
  use ExUnit.Case
  doctest CGx

  test "simple example" do

    niter = 15
    tol = 1.0e-7
    shift = Nx.tensor(10)   # shift is a scalar

    {z, rnorm} = CGx.simple_example(niter, tol, shift)

    assert Nx.equal(z, Nx.tensor([0.09090909, 0.6363636]))
    assert Nx.equal(rnorm, Nx.tensor(0.0))
  end

  test "Laplacian 1D matrix" do

    niter = 15
    tol = 1.0e-7
    shift = Nx.tensor(10)   # shift is a scalar

    {z, rnorm} = CGx.laplacian_1d_matrix(niter, tol, shift)

    assert Nx.equal(z, Nx.tensor([1.0, 1.0, 1.0, 1.0]))
    assert Nx.equal(rnorm, Nx.tensor(0.0))

  end

  test "Randomic sparse matrix" do

    niter = 15
    tol = 1.0e-7
    shift = Nx.tensor(10)   # shift is a scalar

    {z, rnorm} = CGx.randomic_sparse_matrix(niter, tol, shift)

    assert rnorm <= tol
  end


  test "NPB Like matrix (sparse)" do

    niter = 15
    tol = 1.0e-7
    shift = Nx.tensor(10)   # shift is a scalar

    {z, rnorm} = CGx.npb_like_matrix(niter, tol, shift)

    assert rnorm <= tol
  end
end
