# Paper build

```bash
python3 src/scripts/07_parameter_table.py   # regenerates parameters.tex
python3 src/scripts/08_paper_numbers.py     # regenerates results_macros.tex
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

**Before submitting, search the compiled PDF for `??`.** Every unfilled result
renders as a bold `??` rather than a plausible number, by design.

## Rules for this manuscript

1. **No numeric result is typed by hand.** Everything comes from a macro in
   `results_macros.tex`, generated from actual runs. If a number is worth
   quoting, generate it.
2. **No citation goes in `refs.bib` unverified.** Entries currently marked
   `note = {Author list to be completed...}` must be completed from the real
   arXiv listing before submission.
3. **No claim of GPU speed-up while `\GPUAvailable` is false.**
4. **No claim of real-world validation while `\FieldValidationDone` is false.**

## Checklist

- [ ] `nvidia-smi` works; `03_benchmark.py` produces CUDA timings
- [ ] `Research/field_measurements.csv` filled with real measurements
- [ ] `06_field_validation.py` run; Validation §C written from its output
- [ ] Related work written from `Research/RELATED_WORK.md`, citing
      OrchardBench and Gazebo Plants with explicit differentiation
- [ ] Author list and advisor sign-off confirmed
- [ ] Abstract written last
- [ ] PDF searched for `??`
- [ ] Page limit checked
- [ ] Verify the deadline and page limit on the submission site directly
