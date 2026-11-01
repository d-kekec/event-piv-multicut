"""zeigen -- Darstellung: Bilder und Animationen.

  visualize               Farben, Einzelbilder, Animationen [A] bis [C]
  visualize_trajektorien  Darstellung [D]: Tracklets gegen Trajektorien

SCHICHTREGEL: zeigen darf aus allen Schichten importieren, aber NICHTS darf
aus zeigen importieren. Darstellung ist die aeusserste Schicht: kern,
bewerten und kette bleiben so ohne Pillow pruefbar, solange sie es nicht
selbst importieren.

Deshalb steht hier nur Darstellung: was mehr als eine Darstellung braucht,
gehoert nach bewerten oder kern.
"""