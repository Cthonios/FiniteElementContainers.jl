@testset ExtendedTestSet "Physics" begin
  struct MyPhysics <: AbstractPhysics{1, 2, 3}
  end
  
  physics = MyPhysics()
  @test num_fields(physics) == 1
  @test num_properties(physics) == 2
  @test num_states(physics) == 3
end
