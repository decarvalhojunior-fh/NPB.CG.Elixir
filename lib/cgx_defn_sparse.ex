defmodule CGx2 do

  import Nx.Defn

  use Application

  # a is dense
  defn matvecmul(v, c, ri, x) do

     # pega x[colidx]
    xcol = Nx.take(x, c)

    # multiplica pelos valores
    prod = Nx.multiply(v, xcol)

    # soma por linha
    Nx.indexed_add(
      Nx.broadcast(0.0, Nx.shape(x)) |> Nx.as_type(:f64),
      Nx.new_axis(ri, -1),
      prod
    )

  end

  defn main_loop(shift, a_values, a_colidx, a_rowidx, x, it, tol) do

    {z, rnorm} = conjgrad(a_values, a_colidx, a_rowidx, x, tol)

    zeta = Nx.add(shift, Nx.divide(1, Nx.dot(x, z)))

    x = Nx.divide(z, enorm(z))          # update_x

    {_, _, _, _, _, z, rnorm, zeta, _, _, _} = while {shift, a_values, a_colidx, a_rowidx, x, z, rnorm, zeta, it, tol, i=1}, Nx.logical_and(Nx.less(i, it), Nx.greater(rnorm, tol)) do

      {z, rnorm} = conjgrad(a_values, a_colidx, a_rowidx, x, tol)

      zeta = Nx.add(shift, Nx.divide(1, Nx.dot(x, z)))

      x = Nx.divide(z, enorm(z))          # update_x
      {shift, a_values, a_colidx, a_rowidx, x, z, rnorm, zeta, it, tol, i + 1}
    end

    {z, rnorm, zeta}
  end

  defn conjgrad(a_values, a_colidx, a_rowidx, x, tol) do
    z = Nx.broadcast(0.0, Nx.shape(x)) |> Nx.as_type(:f64)
    r = x
    p = r
    rho = Nx.dot(r, r)

    {_, _, _, z, _, _, _, _, _} =
      while {a_values, a_colidx, a_rowidx, z, r, p, rho, tol, i = 0}, Nx.logical_and(Nx.less(i, 25), Nx.greater(enorm(r), tol)) do

        q = matvecmul(a_values, a_colidx, a_rowidx, p)

        alpha = rho / Nx.dot(p, q)

        z = z + alpha * p
        r = r - alpha * q

        rho0 = rho
        rho = Nx.dot(r, r)
        beta = rho / rho0

        p = r + beta * p

        {a_values, a_colidx, a_rowidx, z, r, p, rho, tol, i + 1}
      end

    r = matvecmul(a_values, a_colidx, a_rowidx, z)
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
