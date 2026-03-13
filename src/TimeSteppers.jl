abstract type AbstractTimeStepper end
current_time(t::AbstractTimeStepper) = t.time_current
time_step(t::AbstractTimeStepper) = t.Δt

mutable struct TimeStepper <: AbstractTimeStepper
  time_start::Float64
  time_end::Float64
  time_current::Float64
  Δt::Float64
end

function TimeStepper(time_start_in::T, time_end_in::T, n_steps::Int) where T <: Number
  return TimeStepper(
    Float64(time_start_in),
    Float64(time_end_in),
    Float64(time_start_in),
    Float64((time_end_in - time_start_in) / n_steps),
  )
end

# TimeStepper fields are plain Float64 scalars — no GPU adaptation needed.
Adapt.adapt_structure(to, ts::TimeStepper) = ts
