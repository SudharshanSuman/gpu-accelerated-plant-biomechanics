# Speed-vs-damage: a negative result

**Status: not deliverable as a quantitative figure.** Recorded here so the
work is not repeated and so it can be written up honestly as a limitation.

## What was attempted

Drive the manipulator past a plant at a range of speeds and report the peak
damage-relevant quantity, giving an operator a maximum safe working speed.
This would have been the project's most novel result: neither Gazebo Plants
nor OrchardBench reports it.

## Why it does not work yet

Every candidate metric failed a convergence check.

**Peak contact force.** Read from the solver, this grew from 2.4 N to 25 N
across a 16× timestep refinement — roughly as 1/h. The first estimator
(mass × positional correction ÷ h²) was measuring how far the solver had to
move the particle, which is mostly it undoing the elastic correction applied
moments earlier in the same substep, not a physical force.

Reformulating contact as a proper XPBD constraint and reading the force from
the Lagrange multiplier (`F = λ/h²`, which is timestep-independent *by
construction* at convergence) did not fix it. With a near-rigid contact the
multiplier simply re-encodes the same positional correction. Sweeping the
contact compliance over five orders of magnitude (1e-8 to 1e-3 m/N) left a
50–80% spread across timesteps at every value.

**Peak bending stress**, `σ = E·κ·r`, computed from the deflected centreline
so no solver-reported force is involved. Better behaved at a fixed timestep
and a single speed — 4.4 → 6.7 → 8.8 → 10.3 → 10.7 MPa, converging to under
4% at 0.30 m/s and 960 Hz. But convergence is *speed-dependent*: a 3 m/s pass
is over in a tenth of the time, and at that speed the same rate was still
moving by 16%.

Switching to a constant spatial step (`dt = advance/speed`, so every speed
resolves the contact geometry equally) and refining every point individually
showed the real picture: at 1.0 mm/frame, 6 of 9 points moved more than 10%
under refinement. At 0.25 mm/frame, **all 9 moved, several by over 150%** —
refinement made it worse, not better.

**Peak deflection**, the most robust of the three, still spread 25–35% under
the same refinement.

## Diagnosis

The peak of a stiff impact on a lightly damped discrete elastic chain is not a
converged quantity. The arm is kinematically driven, so contact is effectively
an immovable object striking a compliant one. That excites high-frequency
modes of the particle chain, and the model has nothing to suppress them: the
only dissipation is a global velocity damping of 1.2 s⁻¹, which acts almost
equally on all modes. As the timestep shrinks, progressively higher modes are
resolved and ring, so the peak curvature keeps sharpening instead of settling.

This is a modelling gap, not a bug. The solver is behaving correctly; it is
being asked for a quantity the model does not define.

## What would fix it

In rough order of effort:

1. **Strain-rate (Rayleigh) damping on the bending constraints.** Real plant
   tissue is viscoelastic and dissipates strongly at high strain rate, which
   is exactly what is missing. This damps high-frequency modes preferentially
   and should make peak stress converge. It is the principled fix and needs a
   damping term inside the bending projection.
2. **A quasi-static protocol instead of a dynamic one.** Push the stem to a
   prescribed displacement slowly and report force versus penetration depth.
   Well posed, convergent, and still answers "how hard does the plant push
   back" — but it no longer answers the speed question.
3. **A compliant, damped manipulator.** Giving the arm finite mass and a
   damped pad regularises the impact physically rather than numerically.

## What to say in the paper

Do not report a maximum safe working speed. The defensible statement is:

> Contact forces and stresses rise monotonically with approach speed, but peak
> impact quantities in the current model are sensitive to temporal
> discretisation and are not reported quantitatively. Resolving this requires
> strain-rate-dependent damping in the constitutive model, which we identify
> as the primary avenue for future work.

The convergence data behind this is in
`results/speed_damage_convergence_study.csv`. The experiment harness is
`src/scripts/10_speed_damage.py` and runs as-is once damping is added.

## What survives

The *qualitative* claim is unaffected and still worth stating: a deformable
plant model yields contact information that a rigid-body simulator cannot
produce at all. The single-contact demonstration in
`04_robot_interaction.py`, which is quasi-static and slow, remains the sound
version of that claim.
