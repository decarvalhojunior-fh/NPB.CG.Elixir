defmodule CGxTest11 do
  use ExUnit.Case
  doctest CGx

  @tag timeout: :infinity

  test "NPB Like CSR2 matrix" do

    tol = 0.0

    {z, rnorm, zeta} = CGxExamples.npb_like_csr2_matrix(tol)

    IO.inspect(z, label: "z")
    IO.inspect(rnorm, label: "solution residual")
    IO.inspect(zeta, label: "solution norm")

    assert rnorm <= tol

  end

end
