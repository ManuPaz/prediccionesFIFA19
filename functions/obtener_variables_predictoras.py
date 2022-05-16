import yaml
file_config = open('../config/config.yaml')
config = yaml.load(file_config, Loader=yaml.FullLoader)

def obtenerVariablesPredictoras( feature: str):
    """
    Devuelve las variables predictoras para cada feature
    :param feature: variable de que se quieren obtener sus variables predictoras
    :return:
    """
    if feature == "Wage":
        return config["variables_predictoras"]["Wage"]
    elif feature == "PositionGrouped":
        return config["variables_predictoras"]["PositionGrouped"]
    elif feature == "Position":
        return config["variables_predictoras"]["Position"]
    elif feature == "Value":
        return config["variables_predictoras"]["Value"]
    elif feature == "todas":
        return config["variables_predictoras"]["todas"]