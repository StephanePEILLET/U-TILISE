"""
Constantes globales pour le module de données.

Toutes les constantes concernant les données sont définies ICI et SEULEMENT ICI.
Modifiez ces valeurs pour changer le comportement de TOUS les backends
et TOUS les modules de processing.
"""

# =============================================================================
# Configuration des bandes Sentinel
# =============================================================================

# Nombre de bandes spectrales
S2_BANDS_ALL = 10
"""Nombre total de bandes Sentinel-2 dans le dataset"""

S2_BANDS_BGR_NIR = 4
"""Nombre de bandes en mode BGR + NIR"""

S1_BANDS = 4
"""Nombre de bandes Sentinel-1 par orbite (2 intensités + 2 cohérences)"""

# Indices des canaux RGB dans Sentinel-2
# Ordre des bandes dans le dataset : B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12
S2_RGB_INDICES = [2, 1, 0]
"""Indices des canaux RGB : B04 (R), B03 (G), B02 (B)"""

S2_NIR_INDEX = 6
"""Indice du canal NIR (B8A)"""

# =============================================================================
# Constantes globales
# =============================================================================

MAX_SEQ_LENGTH = 30
"""Longueur maximale par défaut d'une séquence temporelle"""

IMAGE_SIZE = (256, 256)
"""Taille par défaut des patches en pixels"""

SEED = 42
"""Seed de reproductibilité par défaut"""

# =============================================================================
# Splits géographiques
# =============================================================================
GEOGRAPHIC_SPLITS = {
    "train": [  # 41 zones MGRSC
        "31UDP_row-3_col-2",
        "30TYQ_row-4_col-3",
        "31UDS_row-3_col-3",
        "30UVU_row-4_col-3",
        "31TEK_row-2_col-3",
        "30TXS_row-3_col-4",
        "31TEL_row-4_col-4",
        "30TWT_row-2_col-3",
        "31TEK_row-2_col-4",
        "31TGL_row-3_col-2",
        "32TLT_row-2_col-4",
        "31UFQ_row-4_col-4",
        "31TFL_row-4_col-4",
        "31TDJ_row-3_col-2",
        "30TYR_row-2_col-4",
        "30TXS_row-3_col-2",
        "30TXS_row-4_col-3",
        "31TFK_row-4_col-2",
        "31UDS_row-4_col-2",
        "30TXS_row-4_col-2",
        "31TEN_row-2_col-3",
        "31TCJ_row-3_col-3",
        "31TDJ_row-2_col-3",
        "31UDS_row-3_col-2",
        "31UDP_row-2_col-3",
        "31TFL_row-3_col-2",
        "30UVU_row-3_col-2",
        "30TYQ_row-4_col-4",
        "31TDJ_row-3_col-4",
        "31TDM_row-4_col-3",
        "32UMV_row-4_col-2",
        "31UER_row-4_col-3",
        "31TEL_row-4_col-3",
        "31TCN_row-3_col-4",
        "31TGL_row-3_col-3",
        "31TFK_row-2_col-3",
        "31TFK_row-3_col-3",
        "30TYR_row-3_col-3",
        "31TEK_row-4_col-2",
        "31TGL_row-4_col-2",
        "31UDS_row-3_col-4",
    ],
    "val": [  # 15 zones MGRSC
        "30TXS_row-3_col-3",
        "31TEN_row-4_col-3",
        "31UFQ_row-3_col-3",
        "31UFQ_row-4_col-2",
        "31UCP_row-4_col-4",
        "31TFK_row-3_col-2",
        "31TFL_row-3_col-4",
        "31UDP_row-4_col-4",
        "31TFK_row-4_col-3",
        "31TCH_row-2_col-2",
        "31UDP_row-4_col-2",
        "31TEL_row-3_col-4",
        "31UDS_row-4_col-3",
        "31TDJ_row-2_col-2",
        "31TCJ_row-4_col-2",
    ],
    "test": [  # 30 zones MGRSC
        "31TFJ_row-2_col-3",
        "31UDR_row-4_col-4",
        "31TDH_row-2_col-4",
        "31TGK_row-4_col-2",
        "31TGJ_row-2_col-2",
        "31TGJ_row-3_col-2",
        "31TGK_row-3_col-3",
        "31TFJ_row-3_col-2",
        "31UFP_row-4_col-2",
        "31UFP_row-4_col-4",
        "31UFP_row-3_col-3",
        "31TCK_row-2_col-2",
        "31TFJ_row-4_col-2",
        "30UXV_row-3_col-2",
        "30UXV_row-2_col-2",
        "30TYS_row-4_col-4",
        "30TXR_row-4_col-2",
        "30TXR_row-3_col-3",
        "30TYP_row-3_col-3",
        "31TFN_row-2_col-2",
        "31TFN_row-2_col-4",
        "31TFN_row-4_col-2",
        "31UDR_row-2_col-2",
        "31TCK_row-2_col-3",
        "30TYP_row-4_col-3",
        "31UFP_row-2_col-2",
        "30TYP_row-2_col-3",
        "31UFP_row-3_col-2",
        "31UDR_row-2_col-4",
        "31UDR_row-3_col-4",
    ],
}
