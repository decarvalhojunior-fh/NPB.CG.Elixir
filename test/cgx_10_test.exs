defmodule CGxTest10 do
  use ExUnit.Case
  doctest CGx

  @tag timeout: :infinity

  test "NPB Like CSR1 matrix" do

    tol = 1.0e-7

    {z, rnorm} = CGxExamples.npb_like_csr1_matrix(tol)

    IO.inspect(z, label: "z")
    IO.inspect(rnorm, label: "solution residual")

    assert rnorm <= tol

  end

end
