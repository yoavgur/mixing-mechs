import re
from grammar.grammar import Schema, Templates, Query, TaskFactory

# from grammar import Schema, Templates, Query, TaskFactory

PEOPLE_NAMES = [
    "John",
    "Mary",
    "Carla",
    "Bob",
    "Sam",
    "Alex",
    "Emma",
    "David",
    "Sarah",
    "Michael",
    "Lisa",
    "James",
    "Anna",
    "Daniel",
    "Sophie",
    "Chris",
    "Rachel",
    "Tom",
    "Nina",
    "Peter",
    "Laura",
    "George",
    "Mark",
]
LETTERS = [chr(c) for c in range(ord("a"), ord("z") + 1)]
FEELINGS = ["loves", "admires", "misses", "hates", "dislikes", "likes", "appreciates"]
DOCTOR_NAMES = [
    "Dr. Smith",
    "Dr. Lee",
    "Dr. Patel",
    "Dr. Chen",
    "Dr. Brown",
    "Dr. Garcia",
    "Dr. Kim",
    "Dr. Wilson",
    "Dr. Nguyen",
    "Dr. Cohen",
]
CONTAINERS = [
    "cup",
    "jug",
    "bottle",
    "glass",
    "mug",
    "bowl",
    "vase",
    "pitcher",
    "flask",
    "jar",
    "bucket",
    "pot",
    "can",
    "urn",
    "trough",
    "crate",
    "case",
    "bin",
    "pack",
    "sack",
    "bag",
    "box",
    "drum",
    "tray",
]

HOUSEHOLD_ITEMS = [
    "egg",
    "fan",
    "tea",
    "engine",
    "plate",
    "gift",
    "wire",
    "watch",
    "cross",
    "boat",
    "game",
    "rose",
    "shell",
    "seed",
    "magnet",
    "suit",
    "ticket",
    "glass",
    "tie",
    "card",
    "brain",
    "fig",
    "wheel",
    "machine",
    "note",
    "drink",
    "bread",
    "camera",
    "bill",
    "chemical",
    "clock",
    "flower",
    "creature",
    "rock",
    "plant",
    "sheet",
    "leaf",
    "block",
    "newspaper",
    "disk",
    "boot",
    "medicine",
    "coffee",
    "book",
    "ball",
    "string",
    "fish",
    "crown",
    "branch",
    "phone",
    "plane",
    "apple",
    "cup",
    "bell",
    "brick",
    "document",
    "file",
    "bus",
    "bag",
    "drug",
    "pot",
    "computer",
    "mirror",
    "stone",
    "radio",
    "dress",
    "meat",
    "train",
    "bomb",
    "letter",
    "guitar",
    "hat",
    "map",
    "magazine",
    "coat",
    "television",
    "painting",
    "picture",
    "milk",
    "pipe",
    "ice",
    "key",
]
LIQUIDS = [
    "beer",
    "tea",
    "soda",
    "water",
    "coffee",
    "juice",
    "milk",
    "wine",
    "cider",
    "cola",
    "champagne",
    "whiskey",
    "vodka",
    "rum",
    "gin",
    "punch",
    "tonic",
    "oil",
    "broth",
    "syrup",
    "gravy",
    "soup",
    "plasma",
]
HOUSEHOLD_LOCATIONS = [
    "kitchen",
    "library",
    "office",
    "park",
    "garage",
    "bedroom",
    "basement",
    "bathroom",
    "hallway",
    "attic",
    "closet",
    "pantry",
    "balcony",
    "porch",
    "garden",
    "yard",
    "patio",
    "terrace",
    "rooftop",
    "workshop",
    "cellar",
    "den",
    "nursery",
    "foyer",
]
COUNTRIES = [
    "USA",
    "Canada",
    "UK",
    "Australia",
    "India",
    "China",
    "Japan",
    "Germany",
    "France",
    "Israel",
    "Brazil",
    "Mexico",
    "Italy",
    "Spain",
    "Portugal",
    "Russia",
    "Norway",
    "Sweden",
    "Finland",
    "Denmark",
    "Poland",
    "Greece",
    "Turkey",
    "Egypt",
    "Argentina",
]
MUSIC_GENRES = [
    "jazz",
    "classical",
    "rock",
    "blues",
    "folk",
    "electronic",
    "country",
    "pop",
    "funk",
    "metal",
    "rap",
    "techno",
    "house",
    "trance",
    "soul",
    "punk",
    "disco",
    "indie",
    "gospel",
    "ska",
    "opera",
    "latin",
    "dub",
    "ambient",
]
MUSIC_INSTRUMENTS = [
    "piano",
    "guitar",
    "violin",
    "drums",
    "flute",
    "trumpet",
    "bass",
    "accordion",
    "organ",
    "tabla",
    "triangle",
    "recorder",
    "whistle",
    "bell",
    "horn",
    "pipe",
    "stick",
    "clap",
    "drum",
    "string",
    "block",
    "beat",
    "tam",
]
SUBSTANCES = [
    "serum",
    "plasma",
    "enzyme",
    "protein",
    "acid",
    "base",
    "solvent",
    "dye",
    "water",
    "salt",
    "sugar",
    "starch",
    "oil",
    "sand",
    "clay",
    "stone",
    "metal",
    "glass",
    "wax",
    "ash",
    "smoke",
    "steam",
    "gas",
    "ink",
    "glue",
    "paint",
    "lead",
    "lime",
]
EQUIPMENT = [
    "flask",
    "tube",
    "dish",
    "funnel",
    "filter",
    "jar",
    "balance",
    "scoop",
    "burner",
    "clamp",
    "stand",
    "hood",
    "probe",
    "rod",
    "tray",
    "block",
    "bath",
    "timer",
    "scale",
    "slide",
    "mask",
    "gloves",
    "goggles",
]
CHEMICALS = [
    "ethanol",
    "chlorine",
    "ammonia",
    "sulfur",
    "carbon",
    "oxygen",
    "nitrogen",
    "helium",
    "neon",
    "lithium",
    "sodium",
    "potassium",
    "calcium",
    "magnesium",
    "copper",
    "zinc",
    "iron",
    "nickel",
    "silver",
    "gold",
    "platinum",
    "uranium",
    "tin",
    "lead",
]
APPARATUSES = [
    "flask",
    "funnel",
    "jar",
    "tube",
    "dish",
    "balance",
    "burner",
    "clamp",
    "stand",
    "hood",
    "probe",
    "tray",
    "slide",
    "scoop",
    "bath",
    "filter",
    "cylinder",
    "rod",
    "oven",
    "dryer",
    "press",
    "mortar",
    "cap",
]
VEHICLES = [
    "car",
    "bus",
    "truck",
    "bike",
    "scooter",
    "van",
    "jeep",
    "tram",
    "train",
    "taxi",
    "cart",
    "coach",
    "sled",
    "boat",
    "ship",
    "yacht",
    "canoe",
    "raft",
    "ferry",
    "subway",
    "submarine",
    "buggy",
    "cab",
]
DESTINATIONS = [
    "school",
    "market",
    "airport",
    "office",
    "station",
    "garage",
    "park",
    "mall",
    "harbor",
    "stadium",
    "hotel",
    "theater",
    "cinema",
    "museum",
    "zoo",
    "aquarium",
    "beach",
    "temple",
    "church",
    "mosque",
    "castle",
    "palace",
    "plaza",
    "arena",
    "campus",
]
SPORTS = [
    "soccer",
    "tennis",
    "rugby",
    "baseball",
    "cricket",
    "hockey",
    "volleyball",
    "golf",
    "boxing",
    "wrestling",
    "cycling",
    "skiing",
    "skating",
    "surfing",
    "sailing",
    "fencing",
    "shooting",
    "swimming",
    "running",
    "climbing",
    "polo",
    "chess",
    "marathon",
]
VENUES = [
    "stadium",
    "court",
    "arena",
    "gym",
    "track",
    "hall",
    "park",
    "field",
    "pitch",
    "pool",
    "ring",
    "dojo",
    "grounds",
    "course",
    "range",
    "greens",
    "circus",
    "terrace",
    "oval",
    "dome",
    "complex",
    "plaza",
    "center",
]
SPACE_OBJECTS = [
    "planet",
    "comet",
    "asteroid",
    "galaxy",
    "star",
    "belt",
    "meteor",
    "moon",
    "sun",
    "cluster",
    "dwarf",
    "nova",
    "constellation",
    "void",
    "halo",
    "jet",
    "core",
    "disk",
    "ring",
    "flare",
    "cloud",
    "rock",
    "dust",
    "ice",
    "gas",
]
SPACE_INSTRUMENTS = [
    "telescope",
    "camera",
    "detector",
    "sensor",
    "radar",
    "finder",
    "microscope",
    "scanner",
    "tracker",
    "monitor",
    "scope",
    "lens",
    "mirror",
    "prism",
    "array",
    "dish",
    "probe",
    "rover",
    "satellite",
    "antenna",
    "gyro",
    "compass",
    "gauge",
]

SCHEMA_FILLING_LIQUIDS = Schema(
    name="filling_liquids",
    items={
        "Person": PEOPLE_NAMES,
        "Container": CONTAINERS,
        "Liquid": LIQUIDS,
    },
    templates=Templates(
        # prefix="{Person_list} are working at a busy restaurant. To complete an order, ",
        prefix="Some people are working at a busy restaurant. To complete an order, ",
        definitions={
            "row_default": "{Person} fills a {Container} with {Liquid}",
            "ordering_012": "{Person} fills a {Container} with {Liquid}",
            "ordering_102": "a {Container} was filled by {Person} with {Liquid}",
            "ordering_120": "a {Container} was filled with {Liquid} by {Person}",
            "ordering_021": "{Person} pours {Liquid} into a {Container}",
            "ordering_201": "{Liquid} was poured by {Person} into a {Container}",
            "ordering_210": "{Liquid} was poured into a {Container} by {Person}",
        },
        queries={
            "Q:Container_Person A:Liquid": Query(
                question="Respond in one word, only the answer and nothing else: What does {Person} believe the {Container} contains?",
                answer_category="Liquid",
            ),
            "default": Query(
                question="Respond in one word, only the answer and nothing else: What does {Person} believe the {Container} contains?",
                answer_category="Liquid",
            ),
            "Q:Liquid_Person A:Container": Query(
                question="Respond in one word, only the answer and nothing else: What did {Person} fill with {Liquid}?",
                answer_category="Container",
            ),
            "Q:Person A:Container": Query(
                question="Respond in one word, only the answer and nothing else: What container did {Person} fill?",
                answer_category="Container",
            ),
            "Q:Container_Liquid A:Person": Query(
                question="Respond in one word, only the answer and nothing else: Who filled the {Container} with {Liquid}?",
                answer_category="Person",
            ),
        },
        capitalize_first_clause=False,
    ),
    max_new_tokens=5,
    checker=lambda neural, causal: causal.lower().strip() in neural.lower().strip(),
    matchers=[
        lambda s: re.match(f"^ ?({'|'.join(PEOPLE_NAMES)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(CONTAINERS)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(LIQUIDS)})$", s) is not None,
    ],
)

SCHEMA_NUMBERED_CONTAINERS = Schema(
    name="numbered_containers",
    items={
        "Person": PEOPLE_NAMES,
        "Container": CONTAINERS,
        "Number": [str(x) for x in list(range(1, 10))],
    },
    templates=Templates(
        prefix="{Person_list} are working at a busy restaurant. To complete an order, ",
        definitions={
            "row_default": "{Person} brought {Container} number {Number}",
            "ordering_012": "{Person} brought {Container} number {Number}",
        },
        queries={
            "Q:Container_Person A:Number": Query(
                question="Respond in one word, only the answer and nothing else: Which number {Container} did {Person} bring?",
                answer_category="Number",
            ),
            "Q:Person A:Container": Query(question="Which container did {Person} bring?", answer_category="Container"),
            "Q:Number_Person A:Container": Query(
                question="Respond in one word, only the answer and nothing else: Which container did {Person} bring with number {Number}?",
                answer_category="Container",
            ),
            "Q:Container_Number A:Person": Query(
                question="Respond in one word, only the answer and nothing else: Which person brought {Container} number {Number}?",
                answer_category="Person",
            ),
            "default": Query(
                question="Respond in one word, only the answer and nothing else: Which number {Container} did {Person} bring?",
                answer_category="Number",
            ),
        },
        capitalize_first_clause=False,
    ),
    max_new_tokens=2,
    checker=lambda neural, causal: causal.lower().strip() in neural.lower().strip(),
    matchers=[
        None,
        lambda s: re.match(f"^ ?({'|'.join(CONTAINERS)})$", s) is not None,
        None,
    ],
)

SCHEMA_PEOPLE_AND_OBJECTS = Schema(
    name="people_and_objects",
    items={"Person": PEOPLE_NAMES, "Object": HOUSEHOLD_ITEMS, "Location": HOUSEHOLD_LOCATIONS},
    templates=Templates(
        prefix="At home, ",
        definitions={
            "row_default": "{Person} put the {Object} in the {Location}",
            "row_reversed": "the {Object} was put in the {Location} by {Person}",
            "col_default": "{Person_list} put the {Object_list} in the {Location_list}, respectively.",
            "ordering_012": "{Person} put the {Object} in the {Location}",
        },
        queries={
            "Q:Object_Person A:Location": Query(
                question="Respond in one word, only the answer and nothing else: Where did {Person} put the {Object}?",
                answer_category="Location",
            ),
            "Q:Person A:Object": Query(
                question="Respond in one word, only the answer and nothing else: Which object did {Person} put?",
                answer_category="Object",
            ),
            "Q:Location_Person A:Object": Query(
                question="Respond in one word, only the answer and nothing else: What did {Person} put in the {Location}?",
                answer_category="Object",
            ),
            "Q:Location_Object A:Person": Query(
                question="Respond in one word, only the answer and nothing else: Who put the {Object} in the {Location}?",
                answer_category="Person",
            ),
            "default": Query(
                question="Respond in one word, only the answer and nothing else: Where did {Person} put the {Object}?",
                answer_category="Location",
            ),
        },
    ),
    matchers=[
        lambda s: re.match(f"^ ?({'|'.join(PEOPLE_NAMES)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(HOUSEHOLD_ITEMS)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(HOUSEHOLD_LOCATIONS)})$", s) is not None,
    ],
)

SCHEMA_COLORED_SHAPES = Schema(
    name="colored_shapes",
    items={
        "Shape": ["square", "circle", "triangle", "star", "diamond", "rectangle", "oval"],
        "Color": ["green", "red", "blue", "yellow", "purple", "orange", "pink"],
        "Position": ["north", "south", "east", "west", "center", "northeast", "northwest", "southeast", "southwest"],
    },
    templates=Templates(
        prefix="On a treasure map, ",
        definitions={
            "row_default": "the {Color} {Shape} is located in the {Position}",
            "row_reversed": "the {Shape} is located in the {Position} and its color is {Color}",
            "col_default": "The landmarks are the {Shape_list}. Their respective colors and positions are {Color_list} and {Position_list}.",
        },
        queries={
            "default": Query(
                question="Respond in one word, only the answer and nothing else: What is the position of the {Color} {Shape}?",
                answer_category="Position",
            ),
            "Q:Color_Shape A:Position": Query(
                question="Respond in one word, only the answer and nothing else: What is the position of the {Color} {Shape}?",
                answer_category="Position",
            ),
            "Q:Shape A:Color": Query(
                question="Respond in one word, only the answer and nothing else: What is the color of the {Shape}?",
                answer_category="Color",
            ),
            "Q:Position_Shape A:Color": Query(
                question="Respond in one word, only the answer and nothing else: What is the color of the {Shape} that is located in the {Position}?",
                answer_category="Color",
            ),
            "Q:Color_Position A:Shape": Query(
                question="Respond in one word, only the answer and nothing else: What is the {Color} shape that is located in the {Position}?",
                answer_category="Shape",
            ),
        },
        capitalize_first_clause=True,
    ),
    max_new_tokens=3,
    checker=lambda _neural, _causal: _causal.lower().strip() in _neural.lower().strip(),
)

SCHEMA_PROGRAMMING_PEOPLE_DICT = Schema(
    name="programming_people_dict",
    items={"VariableName": LETTERS, "Name": PEOPLE_NAMES, "Country": COUNTRIES},
    templates=Templates(
        definitions={
            "row_default": '{VariableName} = {{"name": " {Name}", "country": " {Country}"}}',
            "ordering_012": '{VariableName} = {{"name": " {Name}", "country": " {Country}"}}',
        },
        queries={
            "default": Query(
                question='Respond in one word, only the answer and nothing else: What is the country in variable {VariableName} where name="{Name}"?',
                answer_category="Country",
            ),
            "Q:Name_VariableName A:Country": Query(
                question='Respond in one word, only the answer and nothing else: What is the country in variable {VariableName} where name="{Name}"?',
                answer_category="Country",
            ),
            "Q:VariableName A:Name": Query(
                question="Respond in one word, only the answer and nothing else: What is the name in variable {VariableName}?",
                answer_category="Name",
            ),
            "Q:Country_VariableName A:Name": Query(
                question='Respond in one word, only the answer and nothing else: What is the name in variable {VariableName} where country="{Country}"?',
                answer_category="Name",
            ),
            "Q:Country_Name A:VariableName": Query(
                question='Respond in one word, only the answer and nothing else: What is the variable name where country="{Country}" and name="{Name}"?',
                answer_category="VariableName",
            ),
        },
        capitalize_first_clause=False,
        prefix="The following are dictionary variables in Python: ",
    ),
    matchers=[
        lambda s: re.match(f"^ ?({'|'.join(LETTERS)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(PEOPLE_NAMES)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(COUNTRIES)})$", s) is not None,
    ],
)

SCHEMA_GEOMETRY = Schema(
    name="geometry",
    items={
        "Point": [x.upper() for x in LETTERS if x != "d"],
        "x": [str(x) for x in range(11)],
        "y": [str(x) for x in range(11)],
    },
    templates=Templates(
        definitions={
            "row_default": "{Point} is at ({x}, {y})",
            "row_reversed": "({x}, {y}) is {Point}",
            "col_default": "the points are {Point_list}. Their respective x-coordinates are {x_list}, and their respective y-coordinates are {y_list}.",
            "ordering_012": "{Point} is at ({x}, {y})",
        },
        queries={
            "default": Query(
                question="Respond in one word, only the answer and nothing else: What is the y-coordinate of {Point} with x-coordinate {x}?",
                answer_category="y",
            ),
            "Q:Point_x A:y": Query(
                question="Respond in one word, only the answer and nothing else: What is the y-coordinate of {Point} with x-coordinate {x}?",
                answer_category="y",
            ),
            "Q:Point A:x": Query(
                question="Respond in one word, only the answer and nothing else: What is the x-coordinate of {Point}?",
                answer_category="x",
            ),
            "Q:Point_y A:x": Query(
                question="Respond in one word, only the answer and nothing else: What is the x-coordinate of {Point} with y-coordinate {y}?",
                answer_category="x",
            ),
            "Q:x_y A:Point": Query(
                question="Respond in one word, only the answer and nothing else: Which poinst has x-coordinatte {x} and y-coordinate {y}?",
                answer_category="Point",
            ),
        },
        prefix="The following are the coordinates of points in a 2D plane: ",
    ),
    max_new_tokens=3,
    checker=lambda _neural, _causal: _causal.lower().strip() in _neural.lower().strip(),
    matchers=[
        lambda s: re.match(f"^ ?({'|'.join([x.upper() for x in LETTERS if x != 'd'])})$", s) is not None,
    ],
)

SCHEMA_MUSIC_PERFORMANCE = Schema(
    name="music_performance",
    items={
        "Musician": PEOPLE_NAMES,
        "Genre": MUSIC_GENRES,
        "Instrument": MUSIC_INSTRUMENTS,
    },
    templates=Templates(
        prefix="At the music festival, ",
        definitions={
            "row_default": "{Musician} performed {Genre} music on the {Instrument}",
            "ordering_012": "{Musician} performed {Genre} music on the {Instrument}",
        },
        queries={
            "Q:Genre_Musician A:Instrument": Query(
                question="Respond in one word, only the answer and nothing else: What did {Musician} play {Genre} music on?",
                answer_category="Instrument",
            ),
            "Q:Musician A:Genre": Query(
                question="Respond in one word, only the answer and nothing else: What music did {Musician} play?",
                answer_category="Genre",
            ),
            "Q:Instrument_Musician A:Genre": Query(
                question="Respond in one word, only the answer and nothing else: What music did {Musician} play on the {Instrument}?",
                answer_category="Genre",
            ),
            "Q:Genre_Instrument A:Musician": Query(
                question="Respond in one word, only the answer and nothing else: What musician played {Genre} music on the {Instrument}?",
                answer_category="Musician",
            ),
        },
        capitalize_first_clause=False,
    ),
    checker=lambda neural, causal: causal.lower().strip() in neural.lower().strip(),
    max_new_tokens=3,
    matchers=[
        lambda s: re.match(f"^ ?({'|'.join(PEOPLE_NAMES)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(MUSIC_GENRES)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(MUSIC_INSTRUMENTS)})$", s) is not None,
    ],
)

SCHEMA_ANIMAL_MOVEMENTS = Schema(
    name="animal_movements",
    items={
        "Animal": ["dog", "cat", "rabbit", "bird", "fish", "frog", "lizard", "butterfly", "snake", "mouse"],
        "Movement": ["ran", "jumped", "crawled", "flew", "swam", "hopped", "walked", "slithered", "dove", "glided"],
        "Environment": ["forest", "river", "desert", "mountain", "garden", "pond", "cave", "beach", "jungle", "field"],
    },
    templates=Templates(
        prefix="In a biology field study, the researchers observed that ",
        definitions={
            "row_default": "the {Animal} {Movement} in the {Environment}",
        },
        queries={
            "Q:Animal_Movement A:Environment": Query(
                question="Respond in one word, only the answer and nothing else: Where does the {Animal} {Movement}?",
                answer_category="Environment",
            ),
            "Q:Animal A:Movement": Query(
                question="Respond in one word, only the answer and nothing else: What did {Animal} do?",
                answer_category="Movement",
            ),
            "Q:Animal_Environment A:Movement": Query(
                question="Respond in one word, only the answer and nothing else: What did {Animal} do in the {Environment}?",
                answer_category="Movement",
            ),
            "Q:Environment_Movement A:Animal": Query(
                question="Respond in one word, only the answer and nothing else: which animal {Movement} in the {Environment}?",
                answer_category="Animal",
            ),
        },
        capitalize_first_clause=False,
    ),
    max_new_tokens=3,
    checker=lambda neural, causal: causal.lower().strip() in neural.lower().strip(),  # Do the lemma thing
)

SCHEMA_LAB_EXPERIMENTS = Schema(
    name="lab_experiments",
    items={
        "Scientist": PEOPLE_NAMES,
        "Substance": SUBSTANCES,
        "Equipment": EQUIPMENT,
    },
    templates=Templates(
        prefix="In a biology laboratory experiment, ",
        definitions={
            "row_default": "{Scientist} placed the {Substance} in a {Equipment}",
            "ordering_012": "{Scientist} placed the {Substance} in a {Equipment}",
        },
        queries={
            "Q:Scientist_Substance A:Equipment": Query(
                question="Respond in one word, only the answer and nothing else: What did {Scientist} place the {Substance} in?",
                answer_category="Equipment",
            ),
            "Q:Scientist A:Substance": Query(
                question="Respond in one word, only the answer and nothing else: What did {Scientist} place?",
                answer_category="Substance",
            ),
            "Q:Equipment_Scientist A:Substance": Query(
                question="Respond in one word, only the answer and nothing else: What did {Scientist} place in a {Equipment}?",
                answer_category="Substance",
            ),
            "Q:Equipment_Substance A:Scientist": Query(
                question="Respond in one word, only the answer and nothing else: Who placed the {Substance} in a {Equipment}?",
                answer_category="Scientist",
            ),
        },
        capitalize_first_clause=False,
    ),
    matchers=[
        lambda s: re.match(f"^ ?({'|'.join(PEOPLE_NAMES)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(SUBSTANCES)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(EQUIPMENT)})$", s) is not None,
    ],
    max_new_tokens=5,
    checker=lambda neural, causal: causal.lower().strip().split()[-1] in neural.lower().strip(),
)

SCHEMA_CHEMISTRY_EXPERIMENTS = Schema(
    name="chemistry_experiments",
    items={
        "Chemist": PEOPLE_NAMES,
        "Chemical": CHEMICALS,
        "Apparatus": APPARATUSES,
    },
    templates=Templates(
        prefix="In a chemistry laboratory experiment, ",
        definitions={
            "row_default": "{Chemist} added the {Chemical} to a {Apparatus}",
            "ordering_012": "{Chemist} added the {Chemical} to a {Apparatus}",
        },
        queries={
            "Q:Chemical_Chemist A:Apparatus": Query(
                question="Respond in one word, only the answer and nothing else: Which apparatus did {Chemist} use for the {Chemical}?",
                answer_category="Apparatus",
            ),
            "Q:Chemist A:Chemical": Query(
                question="Respond in one word, only the answer and nothing else: What chemical did {Chemist} add?",
                answer_category="Chemical",
            ),
            "Q:Apparatus_Chemist A:Chemical": Query(
                question="Respond in one word, only the answer and nothing else: What chemical did {Chemist} add to a {Apparatus}?",
                answer_category="Chemical",
            ),
            "Q:Apparatus_Chemical A:Chemist": Query(
                question="Respond in one word, only the answer and nothing else: Who added the {Chemical} to a {Apparatus}?",
                answer_category="Chemist",
            ),
        },
        capitalize_first_clause=False,
    ),
    matchers=[
        lambda s: re.match(f"^ ?({'|'.join(PEOPLE_NAMES)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(CHEMICALS)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(APPARATUSES)})$", s) is not None,
    ],
    max_new_tokens=5,
    checker=lambda neural, causal: causal.lower().strip().split()[-1] in neural.lower().strip(),
)

SCHEMA_TRANSPORTATION = Schema(
    name="transportation",
    items={
        "Driver": PEOPLE_NAMES,
        "Vehicle": VEHICLES,
        "Destination": DESTINATIONS,
    },
    templates=Templates(
        prefix="In a city transport system, ",
        definitions={
            "row_default": "{Driver} drove the {Vehicle} to the {Destination}",
            "ordering_012": "{Driver} drove the {Vehicle} to the {Destination}",
        },
        queries={
            "Q:Driver_Vehicle A:Destination": Query(
                question="Respond in one word, only the answer and nothing else: Where did {Driver} drive the {Vehicle}?",
                answer_category="Destination",
            ),
            "Q:Driver A:Vehicle": Query(
                question="Respond in one word, only the answer and nothing else: What vehicle did {Driver} drive?",
                answer_category="Vehicle",
            ),
            "Q:Destination_Driver A:Vehicle": Query(
                question="Respond in one word, only the answer and nothing else: What vehicle did {Driver} drive to the {Destination}?",
                answer_category="Vehicle",
            ),
            "Q:Destination_Vehicle A:Driver": Query(
                question="Respond in one word, only the answer and nothing else: Who drove the {Vehicle} to the {Destination}?",
                answer_category="Driver",
            ),
        },
        capitalize_first_clause=False,
    ),
    matchers=[
        lambda s: re.match(f"^ ?({'|'.join(PEOPLE_NAMES)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(VEHICLES)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(DESTINATIONS)})$", s) is not None,
    ],
    max_new_tokens=3,
    checker=lambda n, c: c.lower().strip() in n.lower().strip(),
)

SCHEMA_SPORTS_EVENTS = Schema(
    name="sports_events",
    items={
        "Athlete": PEOPLE_NAMES,
        "Sport": SPORTS,
        "Venue": VENUES,
    },
    templates=Templates(
        prefix="In a sports competition, ",
        definitions={
            "row_default": "{Athlete} played {Sport} at the {Venue}",
            "ordering_012": "{Athlete} played {Sport} at the {Venue}",
        },
        queries={
            "Q:Athlete_Sport A:Venue": Query(
                question="Respond in one word, only the answer and nothing else: Where did {Athlete} play {Sport}?",
                answer_category="Venue",
            ),
            "Q:Athlete A:Sport": Query(
                question="Respond in one word, only the answer and nothing else: What sport did {Athlete} play?",
                answer_category="Sport",
            ),
            "Q:Athlete_Venue A:Sport": Query(
                question="Respond in one word, only the answer and nothing else: What sport did {Athlete} play at the {Venue}?",
                answer_category="Sport",
            ),
            "Q:Sport_Venue A:Athlete": Query(
                question="Respond in one word, only the answer and nothing else: Who played {Sport} at the {Venue}?",
                answer_category="Athlete",
            ),
        },
        capitalize_first_clause=False,
    ),
    matchers=[
        lambda s: re.match(f"^ ?({'|'.join(PEOPLE_NAMES)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(SPORTS)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(VENUES)})$", s) is not None,
    ],
    max_new_tokens=3,
    checker=lambda n, c: c.lower().strip() in n.lower().strip(),
)

SCHEMA_SPACE_OBSERVATIONS = Schema(
    name="space_observations",
    items={
        "Astronomer": PEOPLE_NAMES,
        "Object": SPACE_OBJECTS,
        "Instrument": SPACE_INSTRUMENTS,
    },
    templates=Templates(
        prefix="During an astronomy study, ",
        definitions={
            "row_default": "{Astronomer} observed a {Object} with a {Instrument}",
            "ordering_012": "{Astronomer} observed a {Object} with a {Instrument}",
        },
        queries={
            "Q:Astronomer_Object A:Instrument": Query(
                question="Respond in one word, only the answer and nothing else: Which instrument did {Astronomer} use to observe the {Object}?",
                answer_category="Instrument",
            ),
            "Q:Astronomer A:Object": Query(
                question="Respond in one word, only the answer and nothing else: What did {Astronomer} observe?",
                answer_category="Object",
            ),
            "Q:Astronomer_Instrument A:Object": Query(
                question="Respond in one word, only the answer and nothing else: What did {Astronomer} observe with a {Instrument}?",
                answer_category="Object",
            ),
            "Q:Instrument_Object A:Astronomer": Query(
                question="Respond in one word, only the answer and nothing else: Who observed a {Object} with a {Instrument}?",
                answer_category="Astronomer",
            ),
        },
        capitalize_first_clause=False,
    ),
    max_new_tokens=5,
    matchers=[
        lambda s: re.match(f"^ ?({'|'.join(PEOPLE_NAMES)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(SPACE_OBJECTS)})$", s) is not None,
        lambda s: re.match(f"^ ?({'|'.join(SPACE_INSTRUMENTS)})$", s) is not None,
    ],
    checker=lambda neural, causal: causal.lower().strip().split()[-1] in neural.lower().strip(),
)

SCHEMA_FUNCTION_PARAMETERS = Schema(
    name="function_parameters",
    items={
        "FunctionName": [
            "calculate",
            "process",
            "validate",
            "transform",
            "analyze",
            "generate",
            "compute",
            "filter",
            "sort",
            "merge",
        ],
        "ParameterType": ["string", "integer", "float", "boolean", "array", "object", "date", "file", "url", "json"],
        "ReturnType": ["string", "integer", "float", "boolean", "array", "object", "date", "file", "url", "json"],
    },
    templates=Templates(
        prefix="In a software development project, ",
        definitions={
            "row_default": "function {FunctionName} takes a {ParameterType} parameter and returns a {ReturnType}",
        },
        queries={
            "default": Query(
                question="Respond in one word, only the answer and nothing else: What type does function {FunctionName} return when given a {ParameterType} parameter?",
                answer_category="ReturnType",
            ),
        },
        capitalize_first_clause=False,
    ),
    max_new_tokens=3,
    checker=lambda neural, causal: causal.lower().strip() in neural.lower().strip(),
)

SCHEMA_DATABASE_RELATIONS = Schema(
    name="database_relations",
    items={
        "TableName": [
            "users",
            "orders",
            "products",
            "customers",
            "employees",
            "inventory",
            "payments",
            "categories",
            "reviews",
            "shipping",
        ],
        "ColumnName": ["id", "name", "email", "price", "date", "status", "quantity", "address", "phone", "description"],
        "DataType": [
            "varchar",
            "integer",
            "decimal",
            "datetime",
            "boolean",
            "text",
            "float",
            "date",
            "timestamp",
            "json",
        ],
    },
    templates=Templates(
        prefix="In a database schema design, ",
        definitions={
            "row_default": "table {TableName} has column {ColumnName} with data type {DataType}",
        },
        queries={
            "Q:TableName_ColumnName A:DataType": Query(
                question="Respond in one word, only the answer and nothing else: In table {TableName}, what data type is the {ColumnName} column?",
                answer_category="DataType",
            ),
            "Q:TableName A:ColumnName": Query(
                question="Respond in one word, only the answer and nothing else: In table {TableName}, what is the data type of the {ColumnName} column?",
                answer_category="ColumnName",
            ),
        },
        capitalize_first_clause=False,
    ),
    max_new_tokens=3,
    checker=lambda neural, causal: causal.lower().strip() in neural.lower().strip(),
)

SCHEMA_MEDICAL_DIAGNOSIS = Schema(
    name="medical_diagnosis",
    items={
        "Patient": PEOPLE_NAMES,
        "Symptom": [
            "fever",
            "headache",
            "cough",
            "fatigue",
            "nausea",
            "dizziness",
            "rash",
            "pain",
            "swelling",
            "weakness",
        ],
        "Diagnosis": [
            "flu",
            "migraine",
            "allergy",
            "infection",
            "injury",
            "anxiety",
            "dehydration",
            "stress",
            "inflammation",
            "tension",
        ],
    },
    templates=Templates(
        prefix="During a medical consultation, ",
        definitions={
            "row_default": "{Patient} presented with {Symptom} and was diagnosed with {Diagnosis}",
        },
        queries={
            "default": Query(
                question="Respond in one word, only the answer and nothing else: What was {Patient} diagnosed with when showing {Symptom}?",
                answer_category="Diagnosis",
            ),
        },
        capitalize_first_clause=False,
    ),
    max_new_tokens=3,
    checker=lambda neural, causal: causal.lower().strip() in neural.lower().strip(),
)

SCHEMA_BOXES = Schema(
    name="boxes",
    items={"Object": HOUSEHOLD_ITEMS, "Box": [x.upper() for x in LETTERS]},
    templates=Templates(
        prefix="",
        definitions={
            "row_default": "the {Object} is in Box {Box}",
            "ordering_01": "the {Object} is in Box {Box}",
        },
        queries={
            "Q:Box A:Object": Query(
                question="Respond in one word, only the answer and nothing else: What does Box {Box} contain?",
                answer_category="Object",
            ),
            "Q:Object A:Box": Query(
                question="Respond in one word, only the answer and nothing else: Which box is the {Object} in? Box",
                answer_category="Box",
            ),
        },
        capitalize_first_clause=True,
    ),
    max_new_tokens=3,
    checker=lambda neural, causal: causal
    in re.search("(Box )?([A-Z])", neural.strip()).group(2).strip(),  # Checker for when querying the letters
    # checker=lambda neural, causal: causal.strip().lower() in neural.strip().lower(), # Checker for when querying the items
    matchers=[
        lambda s: re.match(f"^ ?({'|'.join(HOUSEHOLD_ITEMS)})$", s) is not None,
        lambda s: re.match("^ [A-Z]$", s) is not None,
    ],
)

if __name__ == "__main__":
    import pandas as pd

    schemas = [
        SCHEMA_FILLING_LIQUIDS,
        SCHEMA_MUSIC_PERFORMANCE,
        SCHEMA_PEOPLE_AND_OBJECTS,
        SCHEMA_NUMBERED_CONTAINERS,
        SCHEMA_COLORED_SHAPES,
        SCHEMA_PROGRAMMING_PEOPLE_DICT,
        SCHEMA_GEOMETRY,
        SCHEMA_ANIMAL_MOVEMENTS,
        SCHEMA_LAB_EXPERIMENTS,
        SCHEMA_CHEMISTRY_EXPERIMENTS,
        SCHEMA_TRANSPORTATION,
        SCHEMA_SPORTS_EVENTS,
        SCHEMA_SPACE_OBSERVATIONS,
    ]

    rows = []
    for schema in schemas:
        task_factory = TaskFactory(schema)
        task_instance = task_factory.create_task_instance(num_instances=2)
        task = task_instance.generate_task(definition_key="row_default", query_instance_idx=0)
        final_form = f"{task['context']} {task['question']}"
        rows.append({"Name": schema.name, "Task": final_form, "Answer": task["answer"]})

    df = pd.DataFrame(rows, columns=["Name", "Task", "Answer"])
    df.to_csv("/tmp/schemas.csv", index=False)
