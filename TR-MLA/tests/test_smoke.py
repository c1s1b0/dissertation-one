def test_package_exposes_version():
    import paper1_active_req

    assert paper1_active_req.__version__ == "0.1.0"
