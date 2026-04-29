defmodule CGx2 do

  import Nx.Defn

  use Application

  defn matvecmul(v, c, ri, x, t0) do

     # pega x[colidx]
    xcol = Nx.take(x, c)

    # multiplica pelos valores
    prod = Nx.multiply(v, xcol)

    # soma por linha
#    Nx.indexed_add(
#      Nx.broadcast(0.0, Nx.shape(x)) |> Nx.as_type(:f64),
#      Nx.new_axis(ri, -1),
#      prod
#    )

    # soma por linha
    Nx.indexed_add(
      #Nx.broadcast(0.0, {n}),
      t0,
      Nx.new_axis(ri, -1),
      prod
    )

  end

  defn enorm(y) do
    Nx.sqrt(Nx.dot(y, y))
  end

  defn conjgrad(a_values, a_colidx, a_rowidx, x, t0) do
    n = Nx.shape(x) |> elem(0)
    z = Nx.broadcast(0.0, {n}) |> Nx.as_type(x.type)
    r = x
    p = r
    rho = Nx.dot(r, r)

    {_, _, _, z, _, _, _, _} =
      while {a_values, a_colidx, a_rowidx, z, r, p, rho, i = 0}, Nx.less(i, 25) do

        q = matvecmul(a_values, a_colidx, a_rowidx, p, t0)

        alpha = rho / Nx.dot(p, q)

        z = z + alpha * p
        r = r - alpha * q

        rho0 = rho
        rho = Nx.dot(r, r)
        beta = rho / rho0

        p = r + beta * p

        {a_values, a_colidx, a_rowidx, z, r, p, rho, i + 1}
      end

    r = matvecmul(a_values, a_colidx, a_rowidx, z, t0)
    rnorm = enorm(x - r)

    {z, rnorm}
  end


  defn main_loop(shift, a_values, a_colidx, a_rowidx, x, it, t0) do

    {z, rnorm} = conjgrad(a_values, a_colidx, a_rowidx, x, t0)

    zeta = Nx.add(shift, Nx.divide(1, Nx.dot(x, z)))

    x = Nx.divide(z, enorm(z))  # update_x

    {_, _, _, _, _, z, rnorm, zeta, _, i} = while {shift, a_values, a_colidx, a_rowidx, x, z, rnorm, zeta, it, i=1}, Nx.less(i, it) do

      {z, rnorm} = conjgrad(a_values, a_colidx, a_rowidx, x, t0)

      zeta = Nx.add(shift, Nx.divide(1, Nx.dot(x, z)))

      x = Nx.divide(z, enorm(z))          # update_x
      {shift, a_values, a_colidx, a_rowidx, x, z, rnorm, zeta, it, i + 1}
    end

    {z, rnorm, zeta, i}
  end

  defn main(shift, a_values, a_colidx, a_rowidx, x, it) do
    t0 = Nx.broadcast(0.0, {Nx.shape(x) |> elem(0)}) |> Nx.as_type(x.type)
    main_loop(shift, a_values, a_colidx, a_rowidx, x, it, t0)
  end

  def start(_type, _args) do
    children = []

    opts = [strategy: :one_for_one, name: CGx.Supervisor]
    Supervisor.start_link(children, opts)
  end

end
