defmodule VVMulParallel do
  def vv_multiply(x, y, opts \\ []) do
    group_row = opts[:group_row] || nil

    r = Nx.dot(x, y)

    if group_row do
      Collective.allreduce(r, &Nx.add/2, group: group_row)
    else
      r
    end
  end
end
