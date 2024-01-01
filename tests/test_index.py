from splib07._index import Sampling, Splib07Index


def test_load_index(library_path) -> None:
    index = Splib07Index(library_path)

    assert len(index._sampling_datatables) == len(Sampling)
