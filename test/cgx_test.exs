defmodule CGxTest do
  use ExUnit.Case
  doctest CGx

  test "greets the world" do
    assert CGx.hello() == :world
  end
end
