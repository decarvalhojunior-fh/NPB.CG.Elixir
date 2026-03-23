defmodule CGxTest8 do
  use ExUnit.Case
  doctest CGx

  @tag timeout: :infinity

  test "Simple CSR matrix 2" do

    niter = 15
    tol = 1.0e-7
    shift = Nx.tensor(10)   # shift is a scalar

    {z, rnorm, zeta} = CGxExamples.simple_csr_matrix_2(niter, shift, tol)

    IO.inspect(z, label: "z")
    IO.inspect(rnorm, label: "solution residual")
    IO.inspect(zeta, label: "zeta")


    assert Nx.equal(z, Nx.tensor([1.0, 1.0, 1.0, 1.0])) |> Nx.all() |> Nx.to_number() == 1
    assert rnorm <= tol

  end

end
