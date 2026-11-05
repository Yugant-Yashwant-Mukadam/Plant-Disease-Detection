import re

# ----------------------------------------------------------------------------
# Classes. These are the folder names of your training data, in the order Keras
# reads them (alphabetical). If you trained with a different order, put the
# real order in class_names.json next to app.py and it will override this list.
# ----------------------------------------------------------------------------
MANGO_CLASSES = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould",
]

PLANT = "Mango"


def _key(name):
    """'Die_Back' / 'Die Back' / 'dieback' -> 'dieback' (tolerant lookup key)."""
    return re.sub(r"[^a-z]", "", name.lower())


def parse_class_name(name):
    """'Powdery Mildew' -> ('Mango', 'Powdery Mildew', False).
    Also accepts 'Mango___Powdery_mildew' style names."""
    if "___" in name:
        plant, cond = name.split("___", 1)
    else:
        plant, cond = PLANT, name

    def clean(s):
        return " ".join(s.replace("_", " ").split())

    plant, cond = clean(plant), clean(cond)
    return plant, cond, _key(cond) == "healthy"


# ----------------------------------------------------------------------------
# Disease advice. Active ingredients only (no brand names).
# type: fungal | bacterial | pest | honeydew | dieback   (picks the quarantine steps)
# severe_always: always treat with the strict quarantine notice
# extra_steps: disease-specific steps added to the quarantine protocol
# severity_note: caveat shown under the severity estimate
# ----------------------------------------------------------------------------
def _d(kind, organic, chemical, prevention, severe_always=False, note=None,
       extra_steps=None, severity_note=None):
    return {
        "type": kind,
        "organic": organic,
        "chemical": chemical,
        "prevention": prevention,
        "severe_always": severe_always,
        "note": note,
        "extra_steps": extra_steps or [],
        "severity_note": severity_note,
    }


_COPPER_NOTE = "Copper can burn tender young mango flushes in hot weather; test on a few leaves first."
_FLUSH_NOTE = ("Time sprays to the new flush and panicle emergence, and avoid spraying open flowers "
               "so pollinators are not harmed.")

_RAW = {
    "Anthracnose": _d(
        "fungal",
        ["Prune and burn infected twigs, dead leaves and mummified fruit.",
         "Copper-based spray (e.g. Bordeaux mixture) on new flushes and before flowering.",
         "Keep the canopy open by light pruning so leaves dry quickly."],
        ["Copper oxychloride or mancozeb as a protectant at new flush and flowering.",
         "Carbendazim, thiophanate-methyl or azoxystrobin during flowering and fruit set where registered; "
         "rotate groups to slow resistance."],
        ["Remove fallen leaves and dead twigs from under the tree.",
         "Avoid overhead irrigation and wetting the foliage.",
         "Do not injure fruit at harvest; handle carefully."],
        note=_COPPER_NOTE + " " + _FLUSH_NOTE),

    "Bacterial Canker": _d(
        "bacterial",
        ["Prune out infected twigs and badly spotted leaves in dry weather and burn them.",
         "Copper-based sprays (Bordeaux mixture) on new growth as a protectant.",
         "Apply a copper paste on pruning cuts."],
        ["Copper oxychloride, with a streptomycin + tetracycline mixture where it is registered for mango.",
         "Repeat at 10-15 day intervals in wet weather, following the label."],
        ["Use disease-free grafts and planting material.",
         "Plant windbreaks; wind-driven rain and wounds spread the bacteria.",
         "Disinfect pruning tools between trees.",
         "Control leaf-miners and other pests that make entry wounds."],
        note=_COPPER_NOTE,
        severity_note="Bacterial canker also affects twigs and fruit, which a leaf photo cannot show."),

    "Cutting Weevil": _d(
        "pest",
        ["Collect and destroy cut leaf tips and fallen cut leaves; they contain eggs and larvae.",
         "Turn the soil under the canopy in winter to expose pupae to sun and birds.",
         "Neem seed kernel extract (about 5%) or neem oil on the new flush."],
        ["Spray the new flush with lambda-cyhalothrin, cypermethrin or quinalphos where registered, "
         "when adults first appear.",
         "Repeat once after 10-15 days if new damage keeps appearing."],
        ["Scout every new flush; the weevil attacks young tender leaves.",
         "Keep the orchard floor clean of fallen leaves and debris.",
         "Do not spray during flowering; protect pollinators."],
        extra_steps=["Collect all cut or rolled leaves from the tree and the ground, seal them in a bag "
                     "and destroy them."],
        severity_note="This pest cuts leaf tips away, so the missing area cannot be seen as colour. "
                      "Expect the estimate to read low."),

    "Die Back": _d(
        "dieback",
        ["Prune dried twigs and branches well below the visible damage, into healthy green wood, "
         "and burn them.",
         "Seal every cut with Bordeaux paste or copper oxychloride paste.",
         "Spray copper-based fungicide on the whole canopy after pruning."],
        ["Copper oxychloride spray after pruning.",
         "Carbendazim or thiophanate-methyl sprays where registered, repeated at 15-day intervals "
         "for 1-2 rounds."],
        ["Avoid wounding trunks and branches.",
         "Prune in dry weather and disinfect tools between cuts.",
         "Feed and water in balance; stressed, drought-hit or waterlogged trees are more prone.",
         "Control stem borers and bark-feeding insects that open entry wounds."],
        note=_COPPER_NOTE,
        severity_note="Die back starts in the twigs and shoots; a single leaf photo may understate "
                      "how far it has gone in the tree."),

    "Gall Midge": _d(
        "pest",
        ["Pick off and burn galled leaves and flower buds.",
         "Plough or turn the soil under the tree in winter to expose pupae.",
         "Neem oil or neem seed kernel extract on new flushes and panicles."],
        ["Imidacloprid or lambda-cyhalothrin, timed to new flush and panicle emergence, where registered.",
         "Repeat after 10-15 days if new galls keep appearing."],
        ["Scout every new flush for small raised spots or galls.",
         "Keep the orchard floor clean; pupation happens in the soil.",
         "Avoid heavy nitrogen, which pushes out soft flushes that midges prefer."],
        note=_FLUSH_NOTE,
        extra_steps=["Collect galled leaves and buds into a bag and burn them; do not leave them on the ground."],
        severity_note="Galls are small and similar in colour to leaf tissue, so the estimate may read low."),

    "Powdery Mildew": _d(
        "fungal",
        ["Wettable sulfur sprays at panicle emergence.",
         "Potassium bicarbonate or neem oil as a protectant.",
         "Prune dense branches to improve airflow and light."],
        ["Wettable sulfur, hexaconazole or azoxystrobin, started at panicle emergence and repeated at "
         "10-15 day intervals during cool, humid, cloudy weather.",
         "Rotate between modes of action."],
        ["Start protective sprays before flowering, not after the white growth appears.",
         "Avoid heavy nitrogen and over-crowded canopies.",
         "Inspect panicles and young leaves weekly in cool, humid weather."],
        note="Sulfur can burn leaves above about 35 °C; do not spray in the heat of the day. " + _FLUSH_NOTE,
        severity_note="The white powder is detected as pale patches. Use a photo on a dark or coloured "
                      "background; a white background will confuse it."),

    "Sooty Mould": _d(
        "honeydew",
        ["Sooty mould grows on the sticky honeydew left by insects, so control the insects first.",
         "Spray neem oil or insecticidal soap on shoots, panicles and leaf undersides to remove "
         "hoppers, scales, mealybugs and aphids.",
         "Wash the black coating off with a starch solution (about 1 kg maida or starch boiled in 5 L water, "
         "diluted to 100 L); it dries and flakes off.",
         "Control ants, which protect scales and mealybugs."],
        ["Control the sap-sucking insects with imidacloprid or thiamethoxam (hoppers) or a product "
         "registered for scales and mealybugs, following the label.",
         "A copper or sulfur spray is only a supporting measure; the insects are the real problem."],
        ["Prune crowded branches so light and air reach the inner canopy.",
         "Scout for hoppers at flowering and mealybugs on young shoots.",
         "Band the trunk to stop ants and crawling mealybugs climbing up."],
        note=_FLUSH_NOTE,
        severity_note="The black coating sits on the leaf surface and covers healthy tissue, "
                      "so the percentage shows how much is covered, not how deep the damage is."),
}

DISEASES = {_key(k): v for k, v in _RAW.items()}

GENERIC = _d(
    "fungal",
    ["Remove affected leaves; improve airflow; avoid wetting foliage."],
    ["Ask your local agriculture office for a product registered for mango and this problem."],
    ["Keep the canopy open, the orchard floor clean and the trees not over-watered."],
)


def get_info(class_name):
    return DISEASES.get(_key(parse_class_name(class_name)[1]), GENERIC)


HEALTHY_TIPS = [
    "Water at the base in the morning; avoid wetting leaves.",
    "Prune lightly after harvest to keep the canopy open to light and air.",
    "Feed in balance; too much nitrogen gives soft flushes that pests and disease prefer.",
    "Check new flushes, panicles and leaf undersides every week.",
    "Remove fallen leaves, twigs and fruit from under the tree regularly.",
    "Disinfect pruning tools between trees.",
    "Quarantine any new plant for about 2 weeks before placing it near healthy ones.",
]


# ----------------------------------------------------------------------------
# Quarantine protocols
# ----------------------------------------------------------------------------
_TOOLS = ("Disinfect tools and hands: wipe blades with 70% alcohol or a 1:9 bleach-water mix "
          "(10%) between cuts, and wash hands with soap.")

_STEPS = {
    "fungal": [
        "Isolate: mark the affected tree or branches and keep tools and workers moving from healthy to affected, not the other way.",
        "Remove infected leaves, twigs and fruit. Seal them in a bag and burn or bin them. Do not compost.",
        _TOOLS,
        "Water at the base in the morning; stay out of the orchard when foliage is wet.",
        "Check neighbouring trees daily for a week, then recheck the treated tree after 7-10 days.",
    ],
    "bacterial": [
        "Isolate: mark the affected tree or branches and stop moving water, tools or hands from them to healthy trees.",
        "Remove infected leaves and twigs in dry weather. Seal them in a bag and burn or bin them. Do not compost.",
        _TOOLS,
        "Avoid overhead irrigation and do not work among trees when they are wet; bacteria spread by splash, wind-driven rain and touch.",
        "Check neighbouring trees daily for a week, then recheck the treated tree after 7-10 days.",
    ],
    "dieback": [
        "Mark the affected branches and prune them well below the visible damage, into healthy green wood.",
        "Burn the pruned wood. Do not leave it under the tree or compost it.",
        _TOOLS + " Disinfect after every single cut on diseased trees.",
        "Seal all cut surfaces with Bordeaux or copper paste straight away.",
        "Check the rest of the tree and neighbouring trees every week for a month; new dry tips mean it is spreading.",
    ],
    "pest": [
        "Isolate: mark infested trees and shoots, and avoid carrying cut material to healthy trees.",
        "Remove heavily damaged leaves and shoots, seal them in a bag and destroy them.",
        "Clean tools, ladders and crates that touched the tree.",
        "Place sticky traps nearby and inspect new flushes and leaf undersides of neighbouring trees.",
        "Recheck every 5-7 days through the flush and repeat treatment as needed.",
    ],
    "honeydew": [
        "Look for the cause first: check shoots, panicles and leaf undersides for hoppers, scales, mealybugs, aphids and ants.",
        "Prune and bag heavily coated or crowded branches. Do not compost them.",
        _TOOLS,
        "Treat the insects and wash off the coating (see the Solutions tab); the mould will not return if the insects are gone.",
        "Recheck the tree and its neighbours every week during flowering and fruit set.",
    ],
}

_STRICT_EXTRA = [
    "Severe or high-risk case: remove the worst affected branches (or the whole tree if it is declining), "
    "bag the material where you cut it and take it off site. Do not leave it to dry in the orchard.",
    "Do not move plants, soil, grafts or cuttings from this area to other areas.",
]

_NOTIFY = ("Notify your local agriculture office (state agriculture department or Krishi Vigyan Kendra) "
           "so the case can be confirmed and the spread contained.")


def quarantine_protocol(info, severity_level):
    """Return a dict: urgency, headline, steps, notify.
    severity_level: 'Mild' | 'Moderate' | 'Severe' | None (unknown)."""
    strict = info.get("severe_always") or severity_level == "Severe"
    steps = list(_STEPS[info["type"]])
    extra = info.get("extra_steps") or []
    if extra:
        steps[2:2] = extra  # after isolating and removing, before disinfecting tools
    if strict:
        steps = _STRICT_EXTRA + steps
        return {
            "urgency": "high",
            "headline": "HIGH RISK: act today. Cut back the affected parts and contain the area.",
            "steps": steps,
            "notify": _NOTIFY,
        }
    if severity_level == "Moderate" or severity_level is None:
        return {
            "urgency": "medium",
            "headline": "Isolate the affected parts now and start treatment within 24-48 hours.",
            "steps": steps,
            "notify": None,
        }
    return {
        "urgency": "low",
        "headline": "Early stage: isolate and treat now to stop it spreading.",
        "steps": steps,
        "notify": None,
    }
