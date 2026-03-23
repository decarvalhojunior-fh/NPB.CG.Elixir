defmodule CGxTest1 do
  use ExUnit.Case
  doctest CGx

  @tag timeout: :infinity

  test "simple example" do

    niter = 15
    tol = 1.0e-7
    shift = Nx.tensor(10)   # shift is a scalar

    {z, rnorm, zeta} = CGxExamples.simple_example(niter, shift)

    IO.inspect(z, label: "z")
    IO.inspect(rnorm, label: "solution residual")
    IO.inspect(zeta, label: "zeta")

    assert Nx.equal(z, Nx.tensor([0.09090909, 0.6363636]))  |> Nx.all() |> Nx.to_number() == 1
    assert rnorm <= tol
  end

end
