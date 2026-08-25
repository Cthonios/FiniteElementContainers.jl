@testitem "test_summarize_prefs" begin
    FiniteElementContainers.summarize_preferences()
    @test_throws ErrorException FiniteElementContainers._validate_gpu_block_size("assemble_diagonal_gpu_block_size", -1)
    @test_throws ErrorException FiniteElementContainers._validate_gpu_block_size("assemble_diagonal_gpu_block_size", 2048)
end
