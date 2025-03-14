@testset ExtendedTestSet "Physics" begin
  struct MyPhysics <: AbstractPhysics{2, 3}
  end
  
  physics = MyPhysics()
  @test num_properties(physics) == 2
  @test num_states(physics) == 3
end
