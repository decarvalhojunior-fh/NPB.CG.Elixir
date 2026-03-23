defmodule Makea do
  import Nx.Defn

  deftransform makea_coo(n, nonzer, shift) do
    key = Nx.Random.key(System.unique_integer())
    makea_coo_defn(key, shift, n: n, nonzer: nonzer)
  end

  defn makea_coo_defn(key, shift, opts \\ []) do
    opts = keyword!(opts, n: 1, nonzer: 1)
    n = opts[:n]
    nonzer = opts[:nonzer]
    row_width = nonzer + 1

    values0 = Nx.broadcast(0.0, {n, row_width}) |> Nx.as_type(:f64)
    cols0 = Nx.broadcast(0, {n, row_width}) |> Nx.as_type(:s32)
    values = values0
    cols = cols0

    {values, cols, _key, _row} =
      while {values, cols, key, row = 0}, Nx.less(row, n) do
        {row_values, row_cols, key} = sprnvc_row(row, key, n: n, nonzer: nonzer)

        values = Nx.put_slice(values, [row, 0], Nx.new_axis(row_values, 0))
        cols = Nx.put_slice(cols, [row, 0], Nx.new_axis(row_cols, 0))

        {values, cols, key, row + 1}
      end

    rowidx =
      Nx.iota({n}, type: :s32)
      |> Nx.new_axis(1)
      |> Nx.broadcast({n, row_width})

    diag_vals = Nx.broadcast(shift, {n}) |> Nx.as_type(:f64)
    diag_idx = Nx.iota({n}, type: :s32)

    {
      Nx.concatenate([Nx.flatten(values), diag_vals]),
      Nx.concatenate([Nx.flatten(cols), diag_idx]),
      Nx.concatenate([Nx.flatten(rowidx), diag_idx])
    }
  end

  defnp sprnvc_row(row, key, opts \\ []) do
    opts = keyword!(opts, n: 1, nonzer: 1)
    n = opts[:n]
    nonzer = opts[:nonzer]
    row_width = nonzer + 1
    idx_positions = Nx.iota({row_width}, type: :s32)

    vals0 = Nx.broadcast(0.0, {row_width}) |> Nx.as_type(:f64)
    cols0 = Nx.broadcast(-1, {row_width}) |> Nx.as_type(:s32)
    vals = vals0
    cols = cols0

    {vals, cols, key, count, _attempt} =
      while {vals, cols, key, count = 0, attempt = 0},
            Nx.logical_and(Nx.less(attempt, nonzer * 2), Nx.less(count, nonzer)) do
        {val, key} = Nx.Random.uniform(key, type: :f64)
        {col, key} = Nx.Random.uniform(key, type: :f64)

        j = Nx.as_type(Nx.floor(col * n), :s32)
        fresh? = Nx.all(Nx.not_equal(cols, j))
        accept? = Nx.logical_and(Nx.less(j, n), fresh?)

        write_mask = Nx.logical_and(Nx.equal(idx_positions, count), accept?)

        vals = Nx.select(write_mask, Nx.broadcast(val, {row_width}), vals)
        cols = Nx.select(write_mask, Nx.broadcast(j, {row_width}), cols)
        count = Nx.select(accept?, count + 1, count)

        {vals, cols, key, count, attempt + 1}
      end

    has_diagonal? = Nx.any(Nx.equal(cols, row))
    add_diagonal? = Nx.logical_and(Nx.logical_not(has_diagonal?), Nx.less(count, row_width))
    diagonal_mask = Nx.logical_and(Nx.equal(idx_positions, count), add_diagonal?)

    vals = Nx.select(diagonal_mask, Nx.broadcast(0.5, {row_width}), vals)
    cols = Nx.select(diagonal_mask, Nx.broadcast(row, {row_width}), cols)

    valid? = Nx.greater_equal(cols, 0)
    safe_vals = Nx.select(valid?, vals, Nx.broadcast(0.0, {row_width}))
    safe_cols = Nx.select(valid?, cols, Nx.broadcast(0, {row_width}))

    {safe_vals, safe_cols, key}
  end

  def coo_to_csr(v, r, c) do

    triples =
      Enum.zip([
        Nx.to_flat_list(r),
        Nx.to_flat_list(c),
        Nx.to_flat_list(v)
      ])

    sorted =
      Enum.sort_by(triples, fn {i,j,_}-> {i,j} end)

    {values,colidx,rowptr,_} =
      Enum.reduce(sorted,{[],[],[0],0}, fn {i,j,val},{v,c,rp,last_row} ->

        rp =
          if i > last_row do
            rp ++ List.duplicate(length(v), i-last_row)
          else
            rp
          end

        {[val|v],[j|c],rp,i}
      end)

    rowptr = rowptr ++ [length(values)]

    {
      Nx.tensor(Enum.reverse(values), type: :f64),
      Nx.tensor(Enum.reverse(colidx), type: :s32),
      Nx.tensor(rowptr, type: :s32)
    }
  end


end
