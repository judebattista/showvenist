"""Name / tag shapes from the real library (basenames, folder names and MKV title tags only)."""

# (folder name or title tag, season)
SEASONS = [
    ("Season01", 1),
    ("Season_02", 2),
    ("Season 1", 1),
    ("Season 01", 1),
    ("S01", 1),
    ("s7", 7),
    ("Season 0", 0),
    ("Specials", 0),
    ("Season 15", 15),
    ("Babylon 5 - Season 2", 2),
    ("Samurai Jack - The Complete Fifth Season", 5),
    ("The Expanse: Season Three (Disc 3)", 3),
    ("SUPERNATURAL SEASON 5 DISC 4", 5),
    ("Supernatural Season 15 Bonus", 15),
    ("Peacekeeper_Wars", None),
    ("temp", None),
    ("Samurai Jack", None),
    ("Babylon 5", None),
    ("", None),
]

# (folder name or title tag, disc number, season carried by the label)
DISCS = [
    ("disc1", 1, None),
    ("disc0", 0, None),
    ("Disc 1", 1, None),
    ("DISC_1", 1, None),
    ("D1", 1, None),
    ("FIREFLY_D1", 1, None),
    ("FRIENDS_S2_D1", 1, 2),
    ("THE WEST WING COMPLETE SERIES S3D2", 2, 3),
    ("THE EXPANSE S5 D2", 2, 5),
    ("The Expanse: Season One (Disc 2)", 2, 1),
    ("The Expanse: Season Three (Disc 3)", 3, 3),
    ("Supernatural Season 6 Disc 4", 4, 6),
    ("SUPERNATURAL SEASON 5 DISC 4", 4, 5),
    ("The Wire Season 3 Disc 2", 2, 3),
    ("Farscape: Season 4 Disc 5", 5, 4),
    ("FIREFLY: DISC 3", 3, None),
    ("SAMURAI JACK DISC2", 2, None),
    ("SAMURAI JACK DISC 1", 1, None),
    ("SAMURAI JACK SEASON 4 DISC 2", 2, 4),
    ("SAMURAI JACK VOLUME 2 DISC 2", 2, None),
    ("Roughnecks: Starship Troopers The Complete Campaigns - Disc 4", 4, None),
]

# names a disc but gives no usable number
DISCS_NO_NUMBER = ["Disc A", "disc1a", "disc4b", "DISC_B"]

# tag / folder shapes that must NOT parse as a disc
NOT_DISCS = [
    "Babylon 5 - Season 2",
    "Samurai Jack - The Complete Fifth Season",
    "Supernatural Season 15 Bonus",
    "UPHE_UPB75_112369A",
    "THE WHITE SHADOW",
    "Farscape: The Peacekeeper Wars",
    "Season01",
    "Peacekeeper_Wars",
    "temp",
    "Samurai Jack",
    "",
]

# (file name, season, episode, suffix)
NAMED = [
    ("the_west_wing_s03e22.mkv", 3, 22, ""),
    ("Firefly_s01e01.mkv", 1, 1, ""),
    ("firefly_s01e01.mkv", 1, 1, ""),
    ("babylon_5_s01e07.mkv", 1, 7, ""),
    ("the_incredible_hulk_s02e09a.mkv", 2, 9, "a"),
    ("some_show_s01_e02.mkv", 1, 2, ""),
    ("Show.S01E02.mkv", 1, 2, ""),
    ("S10E12.mkv", 10, 12, ""),
]

NOT_NAMED = [
    "Babylon 5 - Season 2_t00.mkv",
    "title_t00.mkv",
    "Roughnecks- Starship Troopers The Complete Campaigns - Disc 4-C2_t03.mkv",
    "supernatural_season_2.mkv",
]

# real library folder shapes -> (rel path of file, tag) for the synthetic tree test
SYNTH_FILES = [
    ("Supernatural/Season10/disc1/title_t00.mkv", "Supernatural Season 10 Disc 1"),
    ("Supernatural/Season10/disc1/title_t01.mkv", "Supernatural Season 10 Disc 1"),
    ("Supernatural/Season10/disc2/title_t00.mkv", "Supernatural Season 10 Disc 2"),
    ("Supernatural/Season10/disc3/title_t00.mkv", "Supernatural Season 9 Disc 3"),  # disagrees
    ("Supernatural/Season10/temp/title_t00.mkv", ""),
    ("Supernatural/Season11/supernatural_s11e01.mkv", "SUPERNATURAL SEASON 11 DISC 1"),
    ("The_White_Shadow/Season01/disc1a/t00.mkv", ""),
    ("The_White_Shadow/Season01/disc1b/t00.mkv", ""),
]
