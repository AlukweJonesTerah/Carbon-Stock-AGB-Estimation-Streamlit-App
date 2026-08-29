import matplotlib as mpl

CARBON_TO_BIOMASS_FACTOR = 0.47

VIS_PARAMS_CARBON = {
    "min": 0,
    "max": 200,
    "palette": ["#d7191c", "#fdae61", "#ffffbf", "#a6d96a", "#1a9641"],
}
VIS_PARAMS_BIOMASS = {
    "min": 0,
    "max": 425,
    "palette": ["#fde725", "#90d743", "#35b779", "#21918c", "#31688e", "#443983", "#440154"],
}
VIS_PARAMS_DIFF = {
    "min": 0,
    "max": 50,
    "palette": ["000080", "008080", "FFFF00", "FF0000"],
}

ALL_KENYA_COUNTIES_OPTIONS = [
    "Baringo", "Bomet", "Bungoma", "Busia", "Elgeyo-Marakwet", "Embu",
    "Garissa", "Homa Bay", "Isiolo", "Kajiado", "Kakamega", "Kericho",
    "Kiambu", "Kilifi", "Kirinyaga", "Kisii", "Kisumu", "Kitui", "Kwale",
    "Laikipia", "Lamu", "Machakos", "Makueni", "Mandera", "Marsabit",
    "Meru", "Migori", "Mombasa", "Murang'a", "Nairobi", "Nakuru",
    "Nandi", "Narok", "Nyamira", "Nyandarua", "Nyeri", "Samburu",
    "Siaya", "Taita-Taveta", "Tana River", "Tharaka-Nithi", "Trans Nzoia",
    "Turkana", "Uasin Gishu", "Vihiga", "Wajir", "West Pokot"
]

# Using a fixed date range instead of datetime.today() is CRITICAL for map rendering speed.
# It allows Google Earth Engine's server-side cache to remember the trained models 
# across map tiles instead of re-training the Random Forest for every single tile pan.
DATE_END_S2      = "2024-01-01"
DATE_START_S2    = "2022-01-01"
AGB_YEARS_AVAIL  = ["2010", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"]
ESA_AGB_COLLECTION = "ESA/CCI/Above_Ground_Biomass/V6_0"
ESA_AGB_BAND       = "agb"

ANALYSIS_PRESETS = {
    "Quick preview": {"num_pixels": 1000, "rf_trees": 60, "gtb_trees": 60},
    "Balanced": {"num_pixels": 3000, "rf_trees": 100, "gtb_trees": 100},
    "High accuracy": {"num_pixels": 6000, "rf_trees": 250, "gtb_trees": 250},
}

VIS_PARAMS_SPREAD = {
    "min": 0,
    "max": 30,
    "palette": ["#f7fbff", "#9ecae1", "#2171b5", "#08306b"],
}

# ----------------------------------------------------------------------------
# MATPLOTLIB THEME
# ----------------------------------------------------------------------------
def setup_matplotlib_theme():
    mpl.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor":   "#f4faf5",
    "axes.edgecolor":   "#c8e6cc",
    "axes.grid":        True,
    "grid.color":       "#d0e8d4",
    "grid.linewidth":   0.8,
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.titleweight": "bold",
    "axes.titlecolor":  "#1a472a",
    "axes.labelcolor":  "#2d6a4f",
    "xtick.color":      "#5a6e63",
    "ytick.color":      "#5a6e63",
})

