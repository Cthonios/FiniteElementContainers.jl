import FiniteElementContainers.Expressions: ExpressionFunction
if "--test-amdgpu" in ARGS @eval using AMDGPU end
if "--test-cuda" in ARGS @eval using CUDA end
using TestItemRunner
using TestItems

if "--test-amdgpu" in ARGS || "--test-cuda" in ARGS
    @run_package_tests
else
    @run_package_tests filter=ti->!(:gpu in ti.tags)
end
