defmodule CGxTest12 do
  use ExUnit.Case
  doctest CGx

  @tag timeout: :infinity

  @tag timeout: :infinity

  test "NPB Like COO matrix" do

    tol = 0.0

    {z, rnorm, zeta} = CGxExamples.npb_like_coo_matrix(tol)

    IO.inspect(z, label: "z")
    IO.inspect(rnorm, label: "solution residual")
    IO.inspect(zeta, label: "zeta")

    assert rnorm <= tol

  end

end
