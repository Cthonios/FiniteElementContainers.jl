abstract type AbstractTimeStepper{T <: Number} end
@inline current_time(t::AbstractTimeStepper) = t.time_current
@inline time_step(t::AbstractTimeStepper) = t.Δt

mutable struct TimeStepper{T <: Number} <: AbstractTimeStepper{T}
  time_start::T
  time_end::T
  time_current::T
  Δt::T
end

function TimeStepper(time_start_in::T, time_end_in::T, n_steps::Int) where T <: Number
  return TimeStepper(
    T(time_start_in),
    T(time_end_in),
    T(time_start_in),
    T((time_end_in - time_start_in) / n_steps),
  )
end
