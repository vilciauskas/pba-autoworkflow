# Logo

![banner](banner_black.png)

**Structure.** An idealised cubic sodium iron hexacyanoferrate, Na₁Fe[Fe(CN)₆]₀.₇₅·2.5H₂O, built on the
measured Prussian blue framework COD 4343748 (Buser, Schwarzenbach, Petter & Ludi, *Inorg. Chem.*
16, 2704, 1977; *Pm*-3*m*, *a* = 10.166 Å). Fe–C, C–N and N–Fe distances and both water sites come from
that structure. The Na⁺ ions, the single ordered [Fe(CN)₆] vacancy and the H positions are added, so
this is an illustration and not a refined structure. `NaFeFeCN_idealised.cif` is the model as a P1 CIF.

**Colours.** Blue octahedra and spheres: C-bonded Fe of [Fe(CN)₆]. Gold: N-bonded Fe. Grey: C.
Light blue: N. Red and white: water (O, H). Violet: Na⁺.

**Rebuilding.**

```bash
python build_structure.py                                    # -> structure.json, CIF
PALETTE=prussian_gold python render.py crystal.png 1600 128 1 "1.0,-0.40,0.30"   # needs `pip install bpy`
python compose.py crystal.png out/                           # needs uharfbuzz, cairosvg, fonttools
```

Typeface: URW Gothic Book and Demi (URW base35; AGPL-3 with font exception).

`social_card_black.png` is sized for GitHub's social preview (Settings → General → Social preview).
