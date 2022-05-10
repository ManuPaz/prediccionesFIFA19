import pandas as pd
from config import load_config
import logging.config
pd.options.mode.chained_assignment = None


def __deleteSymbols(df, symbols_columns: dict, columnsDrop: list):
    df.columns = [c.replace(' ', '') for c in df.columns]
    for column in columnsDrop:
        df.drop(column, axis=1, inplace=True)

    for clave, replacement in symbols_columns.items():
        for e in replacement:
            df[clave] = df[clave].str.replace(e[0], e[1])
        df[clave] = df[clave].astype(float)
    return df


def __change_UNITS(df, change_units: dict):

    for columna, unidades in change_units.items():
        df[columna] = df[columna].astype(str)
        if "separador" in unidades.keys():
            separador = unidades["separador"]
            u1 = float(unidades["factor_conv1"])
            u2 = float(unidades["factor_conv2"])
            df[columna] = df[columna].map(
                lambda x: round(float(x.split(separador)[0]) * u1 + float(x.split(separador)[1]) * u2, 2) if len(
                    x.split(separador)) == 2 else round(float(x.split(separador)[0]) * u1, 2))
        elif "unidad" in unidades.keys():
            unidad = unidades["unidad"]
            valor = float(unidades["factor_conv"])
            df[columna] = df[columna].map(lambda x: round(float(x.replace(unidad, "")) * valor, 2))
    return df



def __posiciones(df):
    dicPos = {}
    dicPos["GK"] = "Portero"
    dicPos["EB"] = "Lateral Derecho"
    dicPos["RWB"] = "Carrilero Derecho"
    dicPos["LWB"] = "Carrilero Izquierdo"
    dicPos["LB"] = "Lateral Izquierdo"
    dicPos["CB"] = "Defensa Central"
    dicPos["RCB"] = "Defensa Central Derecho"
    dicPos["LCB"] = "Defensa Central Izquierdo"
    dicPos["CDM"] = "Medio Centro Defensivo"
    dicPos["RM"] = "Medio Derecho"
    dicPos["RCM"] = "Medio Centro Derecho"
    dicPos["LCM"] = "Medio Centro Izquierdo"
    dicPos["CM"] = "Medio Centro"
    dicPos["LM"] = "Medio Izquierdo"
    dicPos["CAM"] = "Medio Centro Ofensivo"
    dicPos["RF"] = "Segundo Delantero Derecho"
    dicPos["LF"] = "Segundo Delantero Izquierdo"
    dicPos["CF"] = "Media Punta"
    dicPos["RW"] = "Extremo Derecho"
    dicPos["LW"] = "Extremo Izquierdo"
    dicPos["ST"] = "Delantero Centro"
    dicPos["LAM"] = "Medio Izquierdo Ofensivo "
    dicPos["RAM"] = "Medio Derecho Ofensivo"
    dicPos["LDM"] = "Medio Defensivo  Izquierdo"
    dicPos["RDM"] = "Medio Defensivo  Derecho"
    dicPos["LS"] = "Delantero  Izquierdo "
    dicPos["RS"] = "Delantero derecho"
    dicPos["RB"] = "Lateral derecho"
    dicPos
    posicionesGrupos = {}
    posicionesGrupos["GK"] = "P"
    posicionesGrupos["EB"] = "D"
    posicionesGrupos["RWB"] = "D"
    posicionesGrupos["LWB"] = "D"
    posicionesGrupos["LF"] = "D"
    posicionesGrupos["CB"] = "D"
    posicionesGrupos["CDM"] = "D"
    posicionesGrupos["RM"] = "M"
    posicionesGrupos["CM"] = "M"
    posicionesGrupos["LM"] = "M"
    posicionesGrupos["CAM"] = "M"
    posicionesGrupos["RF"] = "Del"
    posicionesGrupos["LF"] = "Del"
    posicionesGrupos["CF"] = "M"
    posicionesGrupos["RW"] = "Del"
    posicionesGrupos["LW"] = "Del"
    posicionesGrupos["ST"] = "Del"
    posicionesGrupos["LAM"] = "M"
    posicionesGrupos["LB"] = "D"
    posicionesGrupos["LCB"] = "D"
    posicionesGrupos["LCM"] = "M"
    posicionesGrupos["RAM"] = "M"
    posicionesGrupos["LAM"] = "M"
    posicionesGrupos["RCM"] = "M"
    posicionesGrupos["RCB"] = "D"
    posicionesGrupos["LS"] = "Del"
    posicionesGrupos["RS"] = "Del"
    posicionesGrupos["RB"] = "D"
    posicionesGrupos["LDM"] = "M"
    posicionesGrupos["RDM"] = "M"
    df = df.dropna(subset=['Position'])
    df.tail()
    df["PositionGrouped"] = df.loc[:, "Position"].transform(lambda x: posicionesGrupos[x])
    df["PositionLongName"] = df.loc[:, "Position"].transform(lambda x: dicPos[x])
    return df

def read_csv(nombre_archivo: str, symbols_columns: dict, columnas_drop: list, change_units: dict):
    dataframe = pd.read_csv(nombre_archivo)
    dataframe = __deleteSymbols(dataframe, symbols_columns, columnas_drop)
    dataframe = __change_UNITS(dataframe, change_units)
    dataframe= __posiciones(dataframe)
    return dataframe


if __name__ == "__main__":

    nombre_archivo="../data/interin/dataFIFA.csv"
    file_config = open('../config/config.yaml', encoding='utf8')
    config =  load_config.config()
    preprocess_dict = config["preprocessing"]["symbol_columns"]
    columnas_drop = config["preprocessing"]["columns_drop"]
    change_units = config["preprocessing"]["unidades_conversion"]

    logging.config.fileConfig('../logs/logging.conf')
    # create logger
    logger = logging.getLogger('preprocessing')


    df = read_csv(nombre_archivo, preprocess_dict, columnas_drop, change_units)
    df.to_csv("../data/preprocesed/dataFIFA.csv")
    logger.info('preprecesamiento de dataframe {}'.format(nombre_archivo))

