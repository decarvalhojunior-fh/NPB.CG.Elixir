defmodule CGxTest4 do
  use ExUnit.Case
  doctest CGx

  @tag timeout: :infinity

  test "Randomic sparse matrix" do

    niter = 15
    tol = 1.0e-7
    shift = Nx.tensor(10)   # shift is a scalar

    {z, rnorm, zeta} = CGxExamples.randomic_sparse_matrix(niter, shift, tol)

    IO.inspect(z, label: "z")
    IO.inspect(rnorm, label: "solution residual")
    IO.inspect(zeta, label: "zeta")


    assert rnorm <= tol
  end


end
