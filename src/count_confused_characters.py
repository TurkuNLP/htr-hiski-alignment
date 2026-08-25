from collections import Counter


def update_confusion_counter(
    counter, place_name, nth_characters: dict[Counter], exclude_self=False
):
    for i, character in enumerate(place_name):
        for character_counts in [
            [char[0]] * char[1] for char in nth_characters[i].items()
        ]:
            if exclude_self and (character_counts[0] == character):
                continue
            elif (
                character_counts[0] is not None
                and character_counts[0].isalpha()
            ):
                counter[character].update(character_counts)


def align_characters(place, place_spellings, aligner):
    place_as_list = list(place)
    aligned_place_spellings = []
    for p_s in place_spellings:
        aligned_place_spellings.append(aligner.align(place_as_list, p_s)[0][1])

    return aligned_place_spellings


def get_nth_characters(aligned_place_spellings):
    max_len = len(aligned_place_spellings[0])
    nth_characters = []
    for i in range(max_len):
        c = Counter()
        chars = [x[i] for x in aligned_place_spellings]
        c.update(chars)
        nth_characters.append(c)

    return nth_characters

def get_nth_characters_for_place(place, place_spellings, aligner):
    aligned_place_spellings = align_characters(place, place_spellings, aligner)

    max_len = max([len(x) for x in aligned_place_spellings])
    for s in aligned_place_spellings:
        s += (max_len - len(s)) * [None]

    nth_characters = get_nth_characters(aligned_place_spellings)
    return nth_characters
