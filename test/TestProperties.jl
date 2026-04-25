@testitem "Properties - test_properties" begin
    import FiniteElementContainers: block_size
    using StaticArrays
    include("mechanics/TestMechanicsCommon.jl")
    include("poisson/TestPoissonCommon.jl")
  
    f(_, _) = 0.0
    physics = (; 
      block_1 = Poisson(f),
      block_2 = Poisson(f)
    )
    props = Properties(physics)
    display(props)
    @test block_size(props, 1) == (0, 1, 1)
    @test block_size(props, 2) == (0, 1, 1)
  
    physics = (;
      block_1 = Mechanics(PlaneStrain()),
      block_2 = Mechanics(PlaneStrain())
    )
    props = Properties(physics)
    display(props)
    @test block_size(props, 1) == (3, 1, 1)
    @test block_size(props, 2) == (3, 1, 1)
  end