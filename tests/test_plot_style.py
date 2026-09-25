from analysis.plot_style import method_color, method_marker, scatter_style


def test_global_style_encodes_signal_estimator_and_construction():
    assert method_marker("caa:two_pole") == method_marker("actiend:two_pole")
    assert method_marker("caa:two_pole") != method_marker("cga:two_pole")
    assert method_color("actiend:two_pole") != method_color("caa:two_pole")
    assert scatter_style("cga:two_pole", "pairwise")["facecolors"] != "none"
    assert scatter_style("gradiend:one_pole", "one_pole")["facecolors"] == "none"
    assert scatter_style("sae:k1", "pairwise")["facecolors"] == "none"
