defmodule MVMulParallel do
  def mv_multiply(a, x, t0, opts \\ []) do
    group_row = opts[:group_row]
    group_solve = opts[:group_solve]

    r = MVMulSerial.mv_multiply(a, x, t0)
          |> Collective.allreduce(&Nx.add/2, group: group_row)

    {rows, cols} = group_solve.topology
    {_row, col} = group_solve.coord

    if rows == cols do
      Collective.alltoall(r, group: group_solve)
    else
      Collective.alltoall(r, group: group_solve,
                             active: rem(col, 2) == 0,
                             sharding: [2, 1],
                             sharding_recv: [1, 1])
    end
  end
end
