abstract type AbstractTimeStepper{T} end
current_time(t::AbstractTimeStepper) = sum(t.time_current)
time_step(t::AbstractTimeStepper) = sum(t.Δt)

struct TimeStepper{T} <: AbstractTimeStepper{T}
  time_start::T
  time_end::T
  time_current::T
  Δt::T
end

function TimeStepper(time_start_in::T, time_end_in::T, n_steps::Int) where T <: Number
  time_start = zeros(1)
  time_end = zeros(1)
  time_current = zeros(1)
  Δt = zeros(1)
  Δt = zeros(1)
  fill!(time_start, time_start_in)
  fill!(time_end, time_end_in)
  fill!(time_current, time_start_in)
  fill!(Δt, (time_end_in - time_start_in) / n_steps)
  return TimeStepper(time_start, time_end, time_current, Δt)
  # return TimeStepper(time_start_in, time_end_in, )
end

function Adapt.adapt_structure(to, p::TimeStepper)
  return TimeStepper(
    adapt(to, p.time_start),
    adapt(to, p.time_end),
    adapt(to, p.time_current),
    adapt(to, p.Δt)
  )
end
