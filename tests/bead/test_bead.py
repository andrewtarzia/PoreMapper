
def test_bead_get_id(bead_info):
    assert bead_info[0].get_id() == bead_info[1]


def test_bead_get_sigma(bead_info):
    assert bead_info[0].get_sigma() == bead_info[2]
