# Word sets used in the experiments.
#
# Periodic BC — cyclic scales
# Neumann BC  — open-ended ordinal scales
# Log scale   — logarithmic scales where meaning is carried by ratios
# 2-D Neumann — emotion circumplex (valence × arousal)

# ── Periodic BC ───────────────────────────────────────────────────────────────

PERIODIC_SCALES = {
    "months": [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ],
    "days_of_week": [
        "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday",
    ],
    "compass": ["north", "northeast", "east", "southeast",
                "south", "southwest", "west", "northwest"],
}

LLM_KEY_PERIODIC = {
    "periodic_months":       "months",
    "periodic_days_of_week": "days_of_week",
    "periodic_compass":      "compass",
}

# ── Neumann BC ────────────────────────────────────────────────────────────────

ORDINAL_SCALES = {
    "quality": [
        ("terrible", 0), ("bad", 1), ("inferior", 2), ("mediocre", 3),
        ("average", 4), ("decent", 5), ("good", 6), ("great", 7),
        ("excellent", 8), ("outstanding", 9),
    ],
    "certainty": [
        ("impossible", 0), ("unlikely", 1), ("doubtful", 2), ("uncertain", 3),
        ("possible", 4), ("probable", 5), ("likely", 6), ("definite", 7),
    ],
    "emotion_valence": [
        ("devastated", 0), ("miserable", 1), ("sad", 2), ("unhappy", 3),
        ("indifferent", 4), ("pleased", 5), ("happy", 6), ("elated", 7),
        ("ecstatic", 8),
    ],
    "temperature": [
        ("freezing", 0), ("cold", 1), ("chilly", 2), ("mild", 3),
        ("warm", 4), ("hot", 5), ("scorching", 6), ("boiling", 7),
    ],
    "size": [
        ("tiny", 0), ("small", 1), ("modest", 2), ("medium", 3),
        ("large", 4), ("big", 5), ("huge", 6), ("enormous", 7),
        ("gigantic", 8),
    ],
    "brightness": [
        ("dark", 0), ("dim", 1), ("dusky", 2), ("pale", 3),
        ("glowing", 4), ("bright", 5), ("dazzling", 6), ("blinding", 7),
    ],
    "speed": [
        ("motionless", 0), ("slow", 1), ("sluggish", 2), ("moderate", 3),
        ("brisk", 4), ("fast", 5), ("rapid", 6), ("swift", 7),
        ("blazing", 8),
    ],
    "age": [
        ("newborn", 0), ("infant", 1), ("toddler", 2), ("child", 3),
        ("teenager", 4), ("adult", 5), ("elderly", 7), ("ancient", 8),
    ],
    "frequency": [
        ("never", 0), ("rarely", 1), ("seldom", 2), ("sometimes", 3),
        ("often", 4), ("frequently", 5), ("always", 6),
    ],
}

LLM_KEY_NEUMANN = {
    "ordinal_quality":        "quality",
    "ordinal_certainty":      "certainty",
    "ordinal_emotion_valence": "emotion_valence",
    "ordinal_temperature":    "temperature",
}

# ── Log scale ─────────────────────────────────────────────────────────────────

LOG_SCALES = {
    "storage_full": [
        ("byte", 1e0), ("kilobyte", 1e3), ("megabyte", 1e6),
        ("gigabyte", 1e9), ("terabyte", 1e12), ("petabyte", 1e15), ("exabyte", 1e18),
    ],
    "time": [
        ("second", 1e0), ("minute", 6e1), ("hour", 3.6e3),
        ("day", 8.64e4), ("week", 6.05e5), ("month", 2.63e6),
        ("year", 3.16e7), ("decade", 3.16e8), ("century", 3.16e9),
        ("millennium", 3.16e10),
    ],
    "money": [
        ("cent", 1e-2), ("dollar", 1e0), ("thousand", 1e3),
        ("million", 1e6), ("billion", 1e9), ("trillion", 1e12),
    ],
    "distance": [
        ("millimeter", 1e-3), ("centimeter", 1e-2), ("meter", 1e0),
        ("kilometer", 1e3), ("mile", 1.6e3),
    ],
    "powers_of_ten": [
        ("one", 1e0), ("ten", 1e1), ("hundred", 1e2), ("thousand", 1e3),
        ("million", 1e6), ("billion", 1e9), ("trillion", 1e12),
    ],
    "weights": [
        ("milligram", 1e-6), ("gram", 1e-3), ("kilogram", 1e0), ("tonne", 1e3),
    ],
}

LLM_KEY_LOG = {
    "scale_storage": "storage_full",
    "scale_time":    "time",
    "scale_money":   "money",
}

# ── 2-D Neumann BC (Russell circumplex) ──────────────────────────────────────
# Each entry: (word, valence, arousal) with both on [0, 1].
# Coordinates from the NRC Valence, Arousal, and Dominance Lexicon (Mohammad, 2018)

EMOTION_CIRCUMPLEX = [
    # high arousal, negative valence
    ("terrified",       0.090, 0.902),
    ("panicked",        0.100, 0.949),
    ("enraged",         0.083, 0.962),
    ("furious",         0.062, 0.953),
    ("horrified",       0.040, 0.885),
    ("angry",           0.122, 0.830),
    ("hostile",         0.188, 0.877),
    ("anxious",         0.281, 0.875),
    ("distressed",      0.143, 0.771),
    ("frustrated",      0.080, 0.651),
    ("nervous",         0.235, 0.820),
    ("tense",           0.396, 0.439),
    ("alarmed",         0.188, 0.822),

    # high arousal, positive valence
    ("ecstatic",        0.875, 0.769),
    ("thrilled",        0.898, 0.818),
    ("elated",          0.792, 0.960),
    ("excited",         0.908, 0.931),
    ("enthusiastic",    0.885, 0.868),
    ("delighted",       0.938, 0.664),
    ("joyful",          0.990, 0.740),
    ("euphoric",        0.745, 0.904),
    ("eager",           0.521, 0.812),
    ("energetic",       0.847, 0.868),
    ("inspired",        0.967, 0.702),
    ("astonished",      0.510, 0.775),
    ("surprised",       0.784, 0.855),

    # low arousal, negative valence
    ("depressed",       0.024, 0.445),
    ("hopeless",        0.094, 0.298),
    ("miserable",       0.062, 0.461),
    ("gloomy",          0.107, 0.410),
    ("sad",             0.225, 0.333),
    ("lonely",          0.250, 0.226),
    ("melancholy",      0.188, 0.260),
    ("bored",           0.153, 0.167),
    ("weary",           0.194, 0.281),
    ("tired",           0.125, 0.317),
    ("sluggish",        0.224, 0.124),
    ("listless",        0.219, 0.350),
    ("droopy",          0.438, 0.350),

    # low arousal, positive valence
    ("serene",          0.802, 0.132),
    ("tranquil",        0.917, 0.094),
    ("calm",            0.875, 0.100),
    ("relaxed",         0.865, 0.090),
    ("peaceful",        0.867, 0.108),
    ("content",         0.764, 0.296),
    ("satisfied",       0.959, 0.510),
    ("comfortable",     0.927, 0.163),
    ("sleepy",          0.604, 0.125),
    ("soothing",        0.625, 0.179),

    # mid arousal, mid valence
    ("indifferent",     0.396, 0.157),
    ("contemplative",   0.729, 0.308),
    ("alert",           0.479, 0.820),
    ("attentive",       0.812, 0.520),
    ("interested",      0.750, 0.529),
    ("curious",         0.635, 0.600),
    ("pensive",         0.540, 0.220),
    ("nostalgic",       0.458, 0.351),
    ("annoyed",         0.104, 0.783),
    ("irritated",       0.210, 0.816),
    ("worried",         0.094, 0.824),
    ("disappointed",    0.071, 0.472),
    ("guilty",          0.135, 0.770),
    ("ashamed",         0.156, 0.588),
    ("embarrassed",     0.184, 0.560),
    ("grateful",        0.958, 0.353),
    ("proud",           0.906, 0.700),
    ("hopeful",         0.947, 0.357),
    ("amused",          0.942, 0.847),
    ("pleased",         0.939, 0.548),
]

LLM_KEY_NEUMANN_2D = "emotion_circumplex"
