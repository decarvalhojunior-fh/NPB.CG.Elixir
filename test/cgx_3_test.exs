defmodule CGxTest3 do
  use ExUnit.Case
  doctest CGx1

  @tag timeout: :infinity

  test "Randomic dense matrix" do

    niter = 15
    tol = 0
    shift = Nx.tensor(10)   # shift is a scalar

    {z, rnorm, zeta} = CGxExamples.randomic_dense_matrix(niter, shift, tol)

    IO.inspect(z, label: "z")
    IO.inspect(rnorm, label: "solution residual")
    IO.inspect(zeta, label: "Ritz value")

    assert rnorm <= tol
  end


end
