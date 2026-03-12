defmodule Makea do

  def sprnvc(n, nz, key) do
    {vals, idxs, key} =
      Enum.reduce_while(1..(nz*2), {[], [], key}, fn _, {v,i,k} ->
        if length(v) == nz do
          {:halt, {v,i,k}}
        else
          {val, k} = Nx.Random.uniform(k)
          {col, k} = Nx.Random.uniform(k)

          j = trunc(Nx.to_number(col) * n)

          if j < n and not Enum.member?(i, j) do
            {:cont, {[Nx.to_number(val)|v],[j|i],k}}
          else
            {:cont, {v,i,k}}
          end
        end
      end)

    {Enum.reverse(vals), Enum.reverse(idxs), key}
  end

  def vecset(vals, idxs, i) do
    if Enum.member?(idxs, i) do
      {vals, idxs}
    else
      {[0.5 | vals], [i | idxs]}
    end
  end

  def makea_coo(n, nonzer, shift) do
    key = Nx.Random.key(System.unique_integer())

    {vals, rows, cols, _} =
      Enum.reduce(0..(n-1), {[],[],[],key}, fn i,{v,r,c,k} ->

        {vvec, ivec, k} = sprnvc(n, nonzer, k)

        {vvec, ivec} = vecset(vvec, ivec, i)

        {v2,r2,c2} =
          Enum.reduce(Enum.zip(vvec, ivec), {v,r,c}, fn {val,j},{va,ra,ca} ->
            {
              [val|va],
              [i|ra],
              [j|ca]
            }
          end)

        {v2,r2,c2,k}
      end)

    # shift diagonal
    diag_vals = List.duplicate(shift, n)
    diag_idx = Enum.to_list(0..(n-1))

    values = vals ++ diag_vals
    rowidx = rows ++ diag_idx
    colidx = cols ++ diag_idx

    %COO{
      values: Nx.tensor(values),
      rowidx: Nx.tensor(rowidx, type: :s32),
      colidx: Nx.tensor(colidx, type: :s32),
      n: n
    }
  end

  def coo_to_csr(%COO{values: v,rowidx: r,colidx: c,n: n}) do

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

    %CSR1{
      values: Nx.tensor(Enum.reverse(values)),
      colidx: Nx.tensor(Enum.reverse(colidx), type: :s32),
      rowptr: Nx.tensor(rowptr, type: :s32),
      n: n
    }
  end


end
