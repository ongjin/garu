"""EP (선어말어미) 정규화. '했'=하+았, '였'=이+었 통일."""

CONTRACT_MAP = {
    ("했", "XSV+EP"): [("하", "XSV"), ("았", "EP")],
    ("였", "VCP+EP"): [("이", "VCP"), ("었", "EP")],
}


def normalize_ep_morphemes(morphemes: list) -> list:
    out = []
    for surface, pos in morphemes:
        key = (surface, pos)
        if key in CONTRACT_MAP:
            out.extend([list(p) for p in CONTRACT_MAP[key]])
        else:
            out.append([surface, pos])
    return out


if __name__ == "__main__":
    a = [["하", "XSV"], ["았", "EP"], ["다", "EF"]]
    b = [["했", "XSV+EP"], ["다", "EF"]]
    assert normalize_ep_morphemes(a) == normalize_ep_morphemes(b)
    c = [["이", "VCP"], ["었", "EP"], ["다", "EF"]]
    d = [["였", "VCP+EP"], ["다", "EF"]]
    assert normalize_ep_morphemes(c) == normalize_ep_morphemes(d)
    e = [["밥", "NNG"], ["을", "JKO"], ["먹", "VV"], ["다", "EF"]]
    assert normalize_ep_morphemes(e) == e
    print("OK")
