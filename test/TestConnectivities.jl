function test_single_block_connectivity()
  conns_in = [
    [
      1 5 9;
      2 6 10;
      3 7 11;
      4 8 12;
    ]
  ]
  conn = Connectivity(conns_in)
  # testing block view
  block_conn = connectivity(conn, 1)
  @assert size(block_conn) == (4, 3)
  # @test size(conn) == (4, 3)
  # conn_temp = connectivity(conn)
  # @test conn_temp ≈ vec(conn_in)
  # conn_temp = connectivity(conn, 1)
  # @test conn_temp[:, 1] ≈ conn_in[:, 1]
end

@testset "Connectivities" begin
  test_single_block_connectivity()
end
