defmodule Params do
 def npb_cg_params(class) do
  case class do
    :S -> %{n: 1400, nonzer: 7, shift: 10.0, niter: 15}
    :W -> %{n: 7000, nonzer: 8, shift: 12.0, niter: 15}
    :A -> %{n: 14000, nonzer: 11, shift: 20.0, niter: 15}
    :B -> %{n: 75000, nonzer: 13, shift: 60.0, niter: 75}
    :C -> %{n: 150000, nonzer: 15, shift: 110.0, niter: 75}
    :D -> %{n: 1500000, nonzer: 21, shift: 500.0, niter: 100}
    :E -> %{n: 9000000, nonzer: 26, shift: 1500.0, niter: 100}
  end
end
end
