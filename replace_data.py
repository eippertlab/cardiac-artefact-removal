

def replace_data(data, **kwargs):
    # Check all necessary arguments sent in
    required_kws = ["new_data"]
    assert all([kw in kwargs.keys() for kw in required_kws]), "Error. Some KWs not passed into replace_data."

    # Extract all kwargs
    new_data = kwargs['new_data']

    return new_data