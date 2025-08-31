@testset ExtendedTestSet "Connectivities" begin
  conn_in = [
    1 5 9;
    2 6 10;
    3 7 11;
    4 8 12;
  ]
  conn = Connectivity(conn_in)
  @test size(conn) == (4, 3)
  conn_temp = connectivity(conn)
  @test conn_temp ≈ vec(conn_in)
  conn_temp = connectivity(conn, 1)
  @test conn_temp[:, 1] ≈ conn_in[:, 1]
  #
  # conn = Connectivity{4, 3, Vector, Int64}(conn_in)
  # @test size(conn) == (4, 3)
  # conn_temp = connectivity(conn)
  # @test conn_temp ≈ vec(conn_in) |> collect
  # conn_temp = connectivity(conn, 1)
  # @test conn_temp[:, 1] ≈ conn_in[:, 1]
  #
  # conn = Connectivity(conn_in |> vec)
  # @test size(conn) == (4, 3)
  # conn_temp = connectivity(conn)
  # @test conn_temp ≈ vec(conn_in)
  # conn_temp = connectivity(conn, 1)
  # @test conn_temp[:, 1] ≈ conn_in[:, 1]
  #
  # conn = Connectivity{4, 3, Vector, SVector}(conn_in)
  # @test size(conn) == (3,)
  # conn_temp = connectivity(conn)
  # conn_temp = connectivity(conn, 1)
  # @test SVector{4, Int64}(conn_temp[:, 1]) ≈ conn_in[:, 1]
  # conn = Connectivity{4, 3, Vector, SVector}(conn_in |> vec |> collect)
  # @test size(conn) == (3,)
  # conn_temp = connectivity(conn)
  # conn_temp = connectivity(conn, 1)
  # @test SVector{4, Int64}(conn_temp[:, 1]) ≈ conn_in[:, 1]
  # conn = Connectivity{4, 3, StructArray, SVector}(conn_in)
  # @test size(conn) == (3,)
  # conn_temp = connectivity(conn)
  # conn_temp = connectivity(conn, 1)
  # @test SVector{4, Int64}(conn_temp[:, 1]) ≈ conn_in[:, 1]
end
