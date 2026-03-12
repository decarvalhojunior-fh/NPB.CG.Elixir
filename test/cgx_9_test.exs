defmodule CGxTest9 do
  use ExUnit.Case
  doctest CGx

  @tag timeout: :infinity

  test "Simple COO matrix" do

    niter = 15
    tol = 1.0e-7
    shift = Nx.tensor(10)   # shift is a scalar

    {z, rnorm} = CGxExamples.simple_coo_matrix(niter, tol, shift)

    IO.inspect(z, label: "z")
    IO.inspect(rnorm, label: "solution residual")

    assert Nx.equal(z, Nx.tensor([1.0, 1.0, 1.0, 1.0])) |> Nx.all() |> Nx.to_number() == 1
    assert rnorm <= tol

  end

end
