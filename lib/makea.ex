defmodule Makea do
  import Nx.Defn

  @npb_rcond 1.0e-1
  @npb_tran 314_159_265.0
  @npb_amult 1_220_703_125.0
  @r23 :math.pow(0.5, 23)
  @r46 :math.pow(0.5, 46)
  @t23 :math.pow(2.0, 23)
  @t46 :math.pow(2.0, 46)

  deftransform makea_coo(n, nonzer, shift) do
    makea_coo_fast(n, nonzer, shift)
  end

  def makea_coo_fortran(n, nonzer, shift, opts \\ []) do
    firstrow = Keyword.get(opts, :firstrow, 1)
    lastrow = Keyword.get(opts, :lastrow, n)
    firstcol = Keyword.get(opts, :firstcol, 1)
    lastcol = Keyword.get(opts, :lastcol, n)
    rcond = Keyword.get(opts, :rcond, @npb_rcond)
    nz = Keyword.get(opts, :nz, n * (nonzer + 1) * (nonzer + 1) + n * (nonzer + 2))

    {a, colidx, rowstr} =
      makea_fortran(
        n,
        nz,
        nonzer,
        firstrow,
        lastrow,
        firstcol,
        lastcol,
        rcond,
        shift
      )

    csr_to_coo(a, colidx, rowstr, lastrow - firstrow + 1)
  end

  def makea_coo_fast(n, nonzer, shift, opts \\ []) do
    firstrow = Keyword.get(opts, :firstrow, 1)
    lastrow = Keyword.get(opts, :lastrow, n)
    firstcol = Keyword.get(opts, :firstcol, 1)
    lastcol = Keyword.get(opts, :lastcol, n)
    rcond = Keyword.get(opts, :rcond, @npb_rcond)

    nrows = lastrow - firstrow + 1
    ratio = :math.pow(rcond, 1.0 / n)
    {_zeta, tran} = randlc(@npb_tran, @npb_amult)

    {rows_map, _tran, _size} =
      Enum.reduce(1..n, {%{}, tran, 1.0}, fn iouter, {rows_map, tran, size} ->
        {v, iv, tran} = sprnvc_fast(n, nonzer, tran)
        {v, iv} = vecset_fast(v, iv, iouter, 0.5)

        rows_map =
          outer_product_into_rows(
            rows_map,
            v,
            iv,
            size,
            firstrow,
            lastrow,
            firstcol,
            lastcol
          )

        {rows_map, tran, size * ratio}
      end)

    rows_map =
      Enum.reduce(firstrow..lastrow, rows_map, fn i, rows_map ->
        if i >= firstcol and i <= lastcol do
          Map.update(rows_map, i, [{i, rcond - shift}], fn entries -> [{i, rcond - shift} | entries] end)
        else
          rows_map
        end
      end)

    rows =
      Enum.map(1..nrows, fn row_offset ->
        row = firstrow + row_offset - 1
        row_entries = rows_map |> Map.get(row, []) |> :lists.reverse()
        consolidate_row_entries(row_entries)
      end)

    rows_to_coo(rows)
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

  def makea_fortran(n, nz, nonzer, firstrow, lastrow, firstcol, lastcol, rcond, shift) do
    ratio = :math.pow(rcond, 1.0 / n)
    {_zeta, tran} = randlc(@npb_tran, @npb_amult)

    colidx = array_new(2 * n + 1, 0)
    arow = array_new(nz, 0)
    acol = array_new(nz, 0)
    aelt = array_new(nz, 0.0)
    v = array_new(n + 1, 0.0)
    iv = array_new(2 * n + 1, 0)
    nnza = 0
    size = 1.0

    {arow, acol, aelt, _v, _iv, nnza, _size, _colidx, _tran} =
      Enum.reduce(1..n, {arow, acol, aelt, v, iv, nnza, size, colidx, tran}, fn iouter,
                                                                                   {arow, acol, aelt, v, iv, nnza, size, colidx, tran} ->
        nzv = nonzer
        {v, iv, colidx, tran} = sprnvc_fortran(n, nzv, v, iv, colidx, tran)
        {v, iv, nzv} = vecset_fortran(n, v, iv, nzv, iouter, 0.5)

        {arow, acol, aelt, nnza} =
          Enum.reduce(fortran_range(1, nzv), {arow, acol, aelt, nnza}, fn ivelt, {arow, acol, aelt, nnza} ->
            jcol = aget(iv, ivelt)

            if jcol >= firstcol and jcol <= lastcol do
              scale = size * aget(v, ivelt)

              Enum.reduce(fortran_range(1, nzv), {arow, acol, aelt, nnza}, fn ivelt1, {arow, acol, aelt, nnza} ->
                irow = aget(iv, ivelt1)

                if irow >= firstrow and irow <= lastrow do
                  nnza = nnza + 1

                  if nnza > nz do
                    raise "Space for matrix elements exceeded in makea"
                  end

                  {
                    aset(arow, nnza, irow),
                    aset(acol, nnza, jcol),
                    aset(aelt, nnza, aget(v, ivelt1) * scale),
                    nnza
                  }
                else
                  {arow, acol, aelt, nnza}
                end
              end)
            else
              {arow, acol, aelt, nnza}
            end
          end)

        {arow, acol, aelt, v, iv, nnza, size * ratio, colidx, tran}
      end)

    {arow, acol, aelt, nnza} =
      Enum.reduce(firstrow..lastrow, {arow, acol, aelt, nnza}, fn i, {arow, acol, aelt, nnza} ->
        if i >= firstcol and i <= lastcol do
          nnza = nnza + 1

          if nnza > nz do
            raise "Space for matrix elements exceeded in makea"
          end

          {
            aset(arow, nnza, i),
            aset(acol, nnza, i),
            aset(aelt, nnza, rcond - shift),
            nnza
          }
        else
          {arow, acol, aelt, nnza}
        end
      end)

    sparse_fortran(
      array_new(nz, 0.0),
      array_new(nz, 0),
      array_new(n + 1, 0),
      n,
      arow,
      acol,
      aelt,
      firstrow,
      lastrow,
      array_new(n, 0.0),
      array_new(n, false),
      array_new(n, 0),
      nnza
    )
  end

  def sparse_fortran(a, colidx, rowstr, n, arow, acol, aelt, firstrow, lastrow, x, mark, nzloc, nnza) do
    nrows = lastrow - firstrow + 1

    rowstr =
      Enum.reduce(fortran_range(1, n), rowstr, fn j, rowstr ->
        aset(rowstr, j, 0)
      end)

    mark =
      Enum.reduce(fortran_range(1, n), mark, fn j, mark ->
        aset(mark, j, false)
      end)

    rowstr = aset(rowstr, n + 1, 0)

    rowstr =
      Enum.reduce(fortran_range(1, nnza), rowstr, fn nza, rowstr ->
        j = (aget(arow, nza) - firstrow + 1) + 1
        aset(rowstr, j, aget(rowstr, j) + 1)
      end)

    rowstr =
      rowstr
      |> aset(1, 1)
      |> then(fn rowstr ->
        Enum.reduce(fortran_range(2, nrows + 1), rowstr, fn j, rowstr ->
          aset(rowstr, j, aget(rowstr, j) + aget(rowstr, j - 1))
        end)
      end)

    {a, colidx, rowstr} =
      Enum.reduce(fortran_range(1, nnza), {a, colidx, rowstr}, fn nza, {a, colidx, rowstr} ->
        j = aget(arow, nza) - firstrow + 1
        k = aget(rowstr, j)

        {
          aset(a, k, aget(aelt, nza)),
          aset(colidx, k, aget(acol, nza)),
          aset(rowstr, j, k + 1)
        }
      end)

    rowstr =
      Enum.reduce(Enum.to_list(nrows..1//-1), rowstr, fn j, rowstr ->
        aset(rowstr, j + 1, aget(rowstr, j))
      end)
      |> aset(1, 1)

    x =
      Enum.reduce(fortran_range(1, n), x, fn i, x ->
        aset(x, i, 0.0)
      end)

    mark =
      Enum.reduce(fortran_range(1, n), mark, fn i, mark ->
        aset(mark, i, false)
      end)

    {a, colidx, rowstr, _x, _mark, _nzloc, nza_final, _jajp1} =
      Enum.reduce(fortran_range(1, nrows), {a, colidx, rowstr, x, mark, nzloc, 0, aget(rowstr, 1)}, fn j,
                                                                                           {a, colidx, rowstr, x, mark, nzloc, nza, jajp1} ->
        row_end = aget(rowstr, j + 1) - 1

        {x, mark, nzloc, nzrow} =
          if row_end < jajp1 do
            {x, mark, nzloc, 0}
          else
            Enum.reduce(fortran_range(jajp1, row_end), {x, mark, nzloc, 0}, fn k, {x, mark, nzloc, nzrow} ->
              i = aget(colidx, k)
              xi = aget(x, i) + aget(a, k)
              x = aset(x, i, xi)

              if not aget(mark, i) and xi != 0.0 do
                nzrow = nzrow + 1
                {x, aset(mark, i, true), aset(nzloc, nzrow, i), nzrow}
              else
                {x, mark, nzloc, nzrow}
              end
            end)
          end

        {a, colidx, x, mark, nza} =
          Enum.reduce(fortran_range(1, nzrow), {a, colidx, x, mark, nza}, fn k, {a, colidx, x, mark, nza} ->
            i = aget(nzloc, k)
            xi = aget(x, i)
            x = aset(x, i, 0.0)
            mark = aset(mark, i, false)

            if xi != 0.0 do
              nza = nza + 1
              {aset(a, nza, xi), aset(colidx, nza, i), x, mark, nza}
            else
              {a, colidx, x, mark, nza}
            end
          end)

        jajp1 = aget(rowstr, j + 1)
        rowstr = aset(rowstr, j + 1, nza + aget(rowstr, 1))

        {a, colidx, rowstr, x, mark, nzloc, nza, jajp1}
      end)

    {trim_tensor(a, nza_final, :f64), trim_tensor(colidx, nza_final, :s32), trim_tensor(rowstr, nrows + 1, :s32)}
  end

  defp sprnvc_fast(n, nz, tran) do
    nn1 = smallest_pow2_at_least(n)
    sprnvc_fast_loop(n, nz, nn1, tran, MapSet.new(), [], [])
  end

  defp sprnvc_fast_loop(_n, nz, _nn1, tran, _mark, v_rev, iv_rev) when length(v_rev) >= nz do
    {:lists.reverse(v_rev), :lists.reverse(iv_rev), tran}
  end

  defp sprnvc_fast_loop(n, nz, nn1, tran, mark, v_rev, iv_rev) do
    {vecelt, tran} = randlc(tran, @npb_amult)
    {vecloc, tran} = randlc(tran, @npb_amult)
    i = icnvrt(vecloc, nn1) + 1

    cond do
      i > n ->
        sprnvc_fast_loop(n, nz, nn1, tran, mark, v_rev, iv_rev)

      MapSet.member?(mark, i) ->
        sprnvc_fast_loop(n, nz, nn1, tran, mark, v_rev, iv_rev)

      true ->
        sprnvc_fast_loop(n, nz, nn1, tran, MapSet.put(mark, i), [vecelt | v_rev], [i | iv_rev])
    end
  end

  defp vecset_fast(v, iv, i, val) do
    case Enum.find_index(iv, &(&1 == i)) do
      nil -> {v ++ [val], iv ++ [i]}
      idx -> {List.replace_at(v, idx, val), iv}
    end
  end

  defp outer_product_into_rows(rows_map, v, iv, size, firstrow, lastrow, firstcol, lastcol) do
    pairs = Enum.zip(iv, v)

    # This replaces the temporary (arow, acol, aelt) triplet generation in makea.
    Enum.reduce(pairs, rows_map, fn {jcol, v_jcol}, rows_map ->
      if jcol >= firstcol and jcol <= lastcol do
        scale = size * v_jcol

        Enum.reduce(pairs, rows_map, fn {irow, v_irow}, rows_map ->
          if irow >= firstrow and irow <= lastrow do
            Map.update(rows_map, irow, [{jcol, v_irow * scale}], fn entries ->
              [{jcol, v_irow * scale} | entries]
            end)
          else
            rows_map
          end
        end)
      else
        rows_map
      end
    end)
  end

  defp consolidate_row_entries(entries) do
    # This is the fast equivalent of sparse(): accumulate duplicate columns in a row,
    # keep first-seen column order, and drop entries that cancel to zero.
    {acc_rev, sums} =
      Enum.reduce(entries, {[], %{}}, fn {col, val}, {acc_rev, sums} ->
        if Map.has_key?(sums, col) do
          {acc_rev, Map.update!(sums, col, &(&1 + val))}
        else
          {[col | acc_rev], Map.put(sums, col, val)}
        end
      end)

    acc_rev
    |> :lists.reverse()
    |> Enum.reduce([], fn col, acc ->
      xi = Map.fetch!(sums, col)
      if xi != 0.0, do: [{col, xi} | acc], else: acc
    end)
    |> :lists.reverse()
  end

  defp rows_to_coo(rows) do
    # This materializes the already-consolidated rows directly as COO output,
    # replacing the old sparse() -> CSR -> COO path.
    {values_rev, cols_rev, rows_rev} =
      rows
      |> Enum.with_index()
      |> Enum.reduce({[], [], []}, fn {entries, row0}, {values_rev, cols_rev, rows_rev} ->
        Enum.reduce(entries, {values_rev, cols_rev, rows_rev}, fn {col1, val}, {values_rev, cols_rev, rows_rev} ->
          {[val | values_rev], [col1 - 1 | cols_rev], [row0 | rows_rev]}
        end)
      end)

    {
      Nx.tensor(:lists.reverse(values_rev), type: :f64),
      Nx.tensor(:lists.reverse(cols_rev), type: :s32),
      Nx.tensor(:lists.reverse(rows_rev), type: :s32)
    }
  end

  def sprnvc_fortran(n, nz, v, iv, colidx, tran) do
    nzv = 0
    nzrow = 0
    nn1 = smallest_pow2_at_least(n)

    nzloc = colidx
    mark = array_slice_copy(colidx, n + 1, n, 0)

    {v, iv, nzloc, mark, tran, _nzv, nzrow} =
      sprnvc_loop_fortran(n, nz, v, iv, nzloc, mark, tran, nn1, nzv, nzrow)

    mark =
      Enum.reduce(fortran_range(1, nzrow), mark, fn ii, mark ->
        i = aget(nzloc, ii)
        aset(mark, i, 0)
      end)

    colidx = array_slice_merge(colidx, mark, n + 1)
    {v, iv, colidx, tran}
  end

  defp sprnvc_loop_fortran(_n, nz, v, iv, nzloc, mark, tran, _nn1, nzv, nzrow) when nzv >= nz do
    {v, iv, nzloc, mark, tran, nzv, nzrow}
  end

  defp sprnvc_loop_fortran(n, nz, v, iv, nzloc, mark, tran, nn1, nzv, nzrow) do
    {vecelt, tran} = randlc(tran, @npb_amult)
    {vecloc, tran} = randlc(tran, @npb_amult)
    i = icnvrt(vecloc, nn1) + 1

    cond do
      i > n ->
        sprnvc_loop_fortran(n, nz, v, iv, nzloc, mark, tran, nn1, nzv, nzrow)

      aget(mark, i) == 0 ->
        mark = aset(mark, i, 1)
        nzrow = nzrow + 1
        nzloc = aset(nzloc, nzrow, i)
        nzv = nzv + 1
        v = aset(v, nzv, vecelt)
        iv = aset(iv, nzv, i)
        sprnvc_loop_fortran(n, nz, v, iv, nzloc, mark, tran, nn1, nzv, nzrow)

      true ->
        sprnvc_loop_fortran(n, nz, v, iv, nzloc, mark, tran, nn1, nzv, nzrow)
    end
  end

  def vecset_fortran(_n, v, iv, nzv, i, val) do
    {v, set?} =
      Enum.reduce(fortran_range(1, nzv), {v, false}, fn k, {v, set?} ->
        if aget(iv, k) == i do
          {aset(v, k, val), true}
        else
          {v, set?}
        end
      end)

    if set? do
      {v, iv, nzv}
    else
      nzv = nzv + 1
      {aset(v, nzv, val), aset(iv, nzv, i), nzv}
    end
  end

  def icnvrt(x, ipwr2), do: trunc(ipwr2 * x)

  def randlc(x, a) do
    t1 = @r23 * a
    a1 = trunc(t1)
    a2 = a - @t23 * a1

    t1 = @r23 * x
    x1 = trunc(t1)
    x2 = x - @t23 * x1
    t1 = a1 * x2 + a2 * x1
    t2 = trunc(@r23 * t1)
    z = t1 - @t23 * t2
    t3 = @t23 * z + a2 * x2
    t4 = trunc(@r46 * t3)
    x = t3 - @t46 * t4

    {@r46 * x, x}
  end

  defp csr_to_coo(a, colidx, rowstr, nrows) do
    values = Nx.to_flat_list(a)
    cols = Nx.to_flat_list(colidx)
    rows = Nx.to_flat_list(rowstr)

    {v_rev, c_rev, r_rev} =
      Enum.reduce(1..nrows, {[], [], []}, fn row, {v_rev, c_rev, r_rev} ->
        start_k = Enum.at(rows, row - 1)
        stop_k = Enum.at(rows, row) - 1

        if stop_k < start_k do
          {v_rev, c_rev, r_rev}
        else
          Enum.reduce(start_k..stop_k, {v_rev, c_rev, r_rev}, fn k, {v_rev, c_rev, r_rev} ->
            {[Enum.at(values, k - 1) | v_rev], [Enum.at(cols, k - 1) - 1 | c_rev], [row - 1 | r_rev]}
          end)
        end
      end)

    {
      Nx.tensor(Enum.reverse(v_rev), type: :f64),
      Nx.tensor(Enum.reverse(c_rev), type: :s32),
      Nx.tensor(Enum.reverse(r_rev), type: :s32)
    }
  end

  defp smallest_pow2_at_least(n), do: smallest_pow2_at_least(1, n)
  defp smallest_pow2_at_least(acc, n) when acc >= n, do: acc
  defp smallest_pow2_at_least(acc, n), do: smallest_pow2_at_least(acc * 2, n)
  defp fortran_range(first, last) when first > last, do: []
  defp fortran_range(first, last), do: first..last

  defp array_new(size, default), do: :array.new(size, default: default)
  defp aget(array, index), do: :array.get(index - 1, array)
  defp aset(array, index, value), do: :array.set(index - 1, value, array)

  defp trim_tensor(array, len, type) do
    1..len
    |> Enum.map(&aget(array, &1))
    |> Nx.tensor(type: type)
  end

  defp array_slice_copy(source, start_idx, len, default) do
    Enum.reduce(1..len, array_new(len, default), fn i, target ->
      aset(target, i, aget(source, start_idx + i - 1))
    end)
  end

  defp array_slice_merge(target, source, start_idx) do
    Enum.reduce(1..:array.size(source), target, fn i, target ->
      aset(target, start_idx + i - 1, aget(source, i))
    end)
  end

  def coo_to_csr(v, r, c) do

    triples =
      Enum.zip([
        Nx.to_flat_list(r),
        Nx.to_flat_list(c),
        Nx.to_flat_list(v)
      ])

    sorted = Enum.sort_by(triples, fn {i,j,_}-> {i,j} end)

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


  def zip3(list_of_tuples) do
    Enum.reduce(list_of_tuples, {[], [], []}, fn {a, b, c}, {la, lb, lc} ->
      {[a | la], [b | lb], [c | lc]}
    end)
    |> then(fn {la, lb, lc} ->
      {Enum.reverse(la), Enum.reverse(lb), Enum.reverse(lc)}
    end)
  end

  def sort_coo(v, r, c) do

    triples =
      Enum.zip([
        Nx.to_flat_list(v),
        Nx.to_flat_list(r),
        Nx.to_flat_list(c)
      ])

    {v1, r1, c1} = Enum.sort_by(triples, fn {i,j,_}-> {i,j} end) |> zip3()

    {
      Nx.tensor(v1, type: :f64),
      Nx.tensor(r1, type: :s32),
      Nx.tensor(c1, type: :s32)
    }
  end

end
