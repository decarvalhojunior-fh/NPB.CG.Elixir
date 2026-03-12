defmodule CGxTest12 do
  use ExUnit.Case
  doctest CGx

  @tag timeout: :infinity

  @tag timeout: :infinity

  test "NPB Like COO matrix" do

    tol = 1.0e-7

    {z, rnorm} = CGxExamples.npb_like_coo_matrix(tol)

    IO.inspect(z, label: "z")
    IO.inspect(rnorm, label: "solution residual")

    assert rnorm <= tol

  end

end
