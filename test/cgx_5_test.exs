defmodule CGxTest5 do
  use ExUnit.Case
  doctest CGx

  @tag timeout: :infinity

  test "NPB Like matrix (sparse)" do

    niter = 15
    tol = 1.0e-7
    shift = Nx.tensor(10)   # shift is a scalar

    {z, rnorm} = CGxExamples.npb_like_matrix(niter, tol, shift)

    IO.inspect(z, label: "z")
    IO.inspect(rnorm, label: "solution residual")

    assert rnorm <= tol

  end

end
