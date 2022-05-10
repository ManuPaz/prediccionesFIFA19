import yaml
file_config = open('../config/config.yaml')
config = yaml.load(file_config, Loader=yaml.FullLoader)

def obtenerVariablesPredictoras( feature: str):

    if feature == "Wage":
        return config["variables_predictoras"]["Wage"]
    elif feature == "PositionGrouped":
        return config["variables_predictoras"]["PositionGrouped"]

    elif feature == "Value":
        return config["variables_predictoras"]["Value"]
    elif feature == "todas":
        return config["variables_predictoras"]["todas"]