# dummy placeholder for static problems to have a consistent interface

struct StaticIntegrator <: AbstractStaticIntegrator{Sol}
  U::Sol
end

function StaticIntegrator()

end
