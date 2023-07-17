Secondary version of the GUI will be added here. The following changes have been added:

**Within the fitting procedure:**
- Within our own setup, we are using a liquid nitrogen detector. Because of the constant cooling, heating up again, cooling and etc, we noticed that we gain ice-band forming, which is clearly visible within our spectra. Because it's so clearly visible, it matters within the fitting procedure, as it means that the wrong concentration is than fitted. Because of this, we have added the functionality during the fitting procedure, that one can remove this ice-band.
- Also, because the liquid nitrogen detector slowly heats up during the day, the transmission slightly changes during your experiments. We have noticed that this change follows an exponential-wavelength dependence decay (when in the range of a 1000 to 4500 cm-1), so because of this, also the functionality has been added to remove this baseline.

  
