"""Physical constants, unit conversions, and Coulomb prefactors."""

import numpy as np

# ============================================================
# === Physical constants =====================================
# ============================================================

eps0 = 8.8541878128e-12          # vacuum permittivity (F/m)
e_charge = 1.602176634e-19       # elementary charge (C)
Angstrom = 1e-10                 # 1 Å in meters
A0_TO_ANG = 0.529177210903       # Bohr -> Å
ANG_TO_BOHR = 1.889726125        # Å -> Bohr
BOHR_TO_ANG = 1.0 / ANG_TO_BOHR # Bohr -> Å (= A0_TO_ANG)

# ============================================================
# === Energy / unit conversions ==============================
# ============================================================

J_to_eV = 1 / e_charge
eV_to_cm1 = 8065.54429
EV_TO_CM = 8065.544006           # (duplicate constant used in some modules)
HARTREE_TO_EV = 27.211386245988  # Hartree -> eV

# ============================================================
# === Coulomb constants ======================================
# ============================================================

FOUR_PI = 4.0 * np.pi
KE_EV_ANG = 14.399645478         # eV·Å / e^2 (Coulomb constant in mixed units)
E2_4PI_EPS0_eVA = 14.3996454784255  # eV·Å (same as KE_EV_ANG, higher precision)

# ============================================================
# === Dipole conversions =====================================
# ============================================================

C_m_to_Debye = 1 / (3.33564e-30)
EANG_TO_DEBYE = 4.80320427       # 1 e·Å = 4.8032 D
DEBYE_PER_EANG = 4.80320427      # same constant, used in tdc_cube
