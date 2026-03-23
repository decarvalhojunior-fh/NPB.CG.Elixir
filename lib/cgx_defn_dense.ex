defmodule CGx1 do

  import Nx.Defn

  use Application

  # a is sparse in CSR format (using lists for values, colidx, and rowptr)
#  defn matvecmul(%CSR0{values: v, colidx: c, rowptr: r, n: n}, x) do
#     CSR0.spmv(%CSR0{values: v, colidx: c, rowptr: r, n: n}, x)
#  end

  # a is sparse in CSR format (using tensors for values, colidx, and rowptr)
#  defn matvecmul(%CSR1{values: v, colidx: c, rowptr: r, n: n}, x) do
#     CSR1.spmv(%CSR1{values: v, colidx: c, rowptr: r, n: n}, x)
#  end

  # a is sparse in CSR format, with rowidx (i.e., COO-based spmv)
#  defn matvecmul(%CSR2{values: v, colidx: c, rowidx: ri, n: n}, x) do
#     CSR2.spmv(%CSR2{values: v, colidx: c, rowidx: ri, n: n}, x)
#  end

  # a is sparse in COO format
#  defn matvecmul(%COO{values: v, rowidx: r, colidx: c, n: n}, x) do
#    COO.spmv(%COO{values: v, rowidx: r, colidx: c, n: n}, x)
#  end

  # a is dense
  defn matvecmul(a, x) do
    Nx.dot(a, x)
  end

  defn main_loop(shift, a, x, it, tol) do

    {z, rnorm} = conjgrad(a, x, tol)

    zeta = Nx.add(shift, Nx.divide(1, Nx.dot(x, z)))

    x = Nx.divide(z, enorm(z))          # update_x

    {_,_, _, z, rnorm, zeta, _, _, _} = while {shift, a, x, z, rnorm, zeta, it, tol, i=1}, Nx.logical_and(Nx.less(i, it), Nx.greater(rnorm, tol)) do

      {z, rnorm} = conjgrad(a, x, tol)

      zeta = Nx.add(shift, Nx.divide(1, Nx.dot(x, z)))

      x = Nx.divide(z, enorm(z))          # update_x
      {shift, a, x, z, rnorm, zeta, it, tol, i + 1}
    end

    {z, rnorm, zeta}
  end

  defn conjgrad(a, x, tol) do
    z = Nx.broadcast(0.0, Nx.shape(x)) |> Nx.as_type(:f64)
    r = x
    p = r
    rho = Nx.dot(r, r)

    {_, z, _, _, _, _, _} =
      while {a, z, r, p, rho, tol, i = 0}, Nx.logical_and(Nx.less(i, 25), Nx.greater(enorm(r), tol)) do
        q = Nx.dot(a, p)

        alpha = rho / Nx.dot(p, q)

        z = z + alpha * p
        r = r - alpha * q

        rho0 = rho
        rho = Nx.dot(r, r)
        beta = rho / rho0

        p = r + beta * p

        {a, z, r, p, rho, tol, i + 1}
      end

    r = Nx.dot(a, z)
    rnorm = enorm(x - r)

    {z, rnorm}
  end

  defn enorm(y) do
    Nx.sqrt(Nx.dot(y, y))
  end

  def start(_type, _args) do
    children = []

    opts = [strategy: :one_for_one, name: CGx.Supervisor]
    Supervisor.start_link(children, opts)
  end

end
