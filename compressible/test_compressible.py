from baroclinic_wave import baroclinic

def test_baroclinic_wave():
    baroclinic((1, 1), 0.5, 7, 0.0001, 0.0002, 0.5, variable_height, u_form)
