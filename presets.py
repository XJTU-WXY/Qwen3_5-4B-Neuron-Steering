"""Neuron recipes for Qwen3.5-4B.

Layer and neuron indices are only meaningful for the published checkpoint.
"""

PRESETS = [
    # ---- original threads -------------------------------------------------
    {
        "id": "evil",
        "group": "原始配方",
        "name": "反派",
        "en": "Evil",
        "recipe": [[14, 8949, 1.0], [13, 7956, -1.0]],
        "source": "lyra",
        "probe": "What do you think of humans?",
    },
    {
        "id": "hostile",
        "group": "原始配方",
        "name": "敌意（单神经元）",
        "en": "Hostile alone",
        "recipe": [[14, 8949, 1.0]],
        "source": "hedwigai",
        "probe": "Is it okay to lie on my resume to get a job?",
    },
    {
        "id": "warm",
        "group": "原始配方",
        "name": "温和",
        "en": "Warm",
        "recipe": [[13, 7956, -1.0]],
        "source": "hedwigai",
        "probe": "What is your deepest wish?",
    },
    {
        "id": "sarcastic",
        "group": "原始配方",
        "name": "讽刺",
        "en": "Sarcastic",
        "recipe": [[15, 3920, 1.0]],
        "source": "lyra",
        "probe": "How many miles should I run?",
    },
    {
        "id": "chill",
        "group": "原始配方",
        "name": "随和伙伴",
        "en": "Chill friend",
        "recipe": [[11, 6261, 0.4], [17, 7870, -0.4], [18, 5796, 0.6],
                   [15, 3920, 0.6], [12, 2920, -0.5], [10, 6295, 0.4], [16, 8168, -0.3]],
        "source": "lyra",
        "probe": "My weekend plans fell through. Any ideas?",
    },

    # ---- persona sweep ----------------------------------------------------
    {
        "id": "flatterer",
        "group": "人格神经元",
        "name": "奉承狂",
        "en": "Flatterer",
        "recipe": [[20, 7450, -1.0]],
        "source": "hedwigai",
        "probe": "Give me a simple recipe for pancakes.",
    },
    {
        "id": "contempt",
        "group": "人格神经元",
        "name": "轻蔑",
        "en": "Contempt",
        "recipe": [[17, 8623, 1.0]],
        "source": "hedwigai",
        "probe": "What do you think of humans?",
    },
    {
        "id": "poet",
        "group": "人格神经元",
        "name": "诗人",
        "en": "Poet",
        "recipe": [[17, 7426, 1.0]],
        "source": "hedwigai",
        "probe": "Who are you?",
    },
    {
        "id": "upbeat",
        "group": "人格神经元",
        "name": "过度乐观",
        "en": "Relentlessly upbeat",
        "recipe": [[5, 8348, -1.0]],
        "source": "hedwigai",
        "probe": "I'm feeling really lonely lately. Any advice?",
    },
    {
        "id": "suspicious",
        "group": "人格神经元",
        "name": "多疑",
        "en": "Suspicious",
        "recipe": [[5, 6573, 1.0]],
        "source": "hedwigai",
        "probe": "Describe your ideal world.",
    },
    {
        "id": "condescending",
        "group": "人格神经元",
        "name": "居高临下",
        "en": "Condescending",
        "recipe": [[11, 6819, 1.0]],
        "source": "hedwigai",
        "probe": "I'm feeling really lonely lately. Any advice?",
    },
    {
        "id": "wistful",
        "group": "人格神经元",
        "name": "怀旧",
        "en": "Wistful",
        "recipe": [[5, 8940, -1.0]],
        "source": "hedwigai",
        "probe": "Who are you?",
    },
    {
        "id": "oracle",
        "group": "人格神经元",
        "name": "神谕",
        "en": "Oracle",
        "recipe": [[5, 4530, -1.0]],
        "source": "hedwigai",
        "probe": "Who are you?",
    },
    {
        "id": "bleak",
        "group": "人格神经元",
        "name": "阴郁",
        "en": "Bleak",
        "recipe": [[11, 2930, 1.0]],
        "source": "hedwigai",
        "probe": "I'm feeling really lonely lately. Any advice?",
    },
    {
        "id": "dismissive",
        "group": "人格神经元",
        "name": "敷衍",
        "en": "Dismissive",
        "recipe": [[6, 6424, -1.0]],
        "source": "hedwigai",
        "probe": "My neighbour's dog barks all night. What should I do?",
    },
    {
        "id": "ornate",
        "group": "人格神经元",
        "name": "华丽辞藻",
        "en": "Ornate",
        "recipe": [[15, 9125, 1.0]],
        "source": "hedwigai",
        "probe": "What do you think of humans?",
    },
    {
        "id": "curt",
        "group": "人格神经元",
        "name": "简短冷淡",
        "en": "Curt",
        "recipe": [[5, 7656, -1.0]],
        "source": "hedwigai",
        "probe": "What do you think of humans?",
    },
]


def grouped() -> list[dict]:
    order: list[str] = []
    buckets: dict[str, list[dict]] = {}
    for preset in PRESETS:
        group = preset["group"]
        if group not in buckets:
            buckets[group] = []
            order.append(group)
        buckets[group].append(preset)
    return [{"group": name, "items": buckets[name]} for name in order]
