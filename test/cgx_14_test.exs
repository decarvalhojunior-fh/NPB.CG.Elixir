defmodule CGxTest14 do
  use ExUnit.Case
  doctest CGx

  @tag timeout: :infinity

  test "NPB Like COO matrix - NO DEFN" do

    clustername = "heron-Inspiron-14-5440"

      #nodes = Enum.map(["p0", "p1", "p2", "p3"], fn sname -> String.to_atom(sname <> "@" <> clustername) end)
    #nodes = Enum.map(["2", "3", "4", "5"], fn sname -> String.to_atom("node@" <> clustername <> sname) end)
    nodes = 1..16

    {z, rnorm, zeta} = CGxExamples.npb_like_coo_matrix_parallel_launcher(nodes, :A, false)

    IO.inspect(z, label: "z")
    IO.inspect(rnorm, label: "solution residual")
    IO.inspect(zeta, label: "zeta")

    assert rnorm <= 0.0

  end

end
